import bpy
import bmesh
import math
from enum import Enum
from typing import TYPE_CHECKING, Callable, List, Optional, Self, Tuple, NamedTuple, Type, Union
from typing_extensions import override
from mathutils import Vector, Matrix
import random

from .common import CurveLike, VectorLike, extract_vector
from .object import Object
from .location import Location

if TYPE_CHECKING:
    from .part import Part

class FillMode(Enum):
    """Enumeration for Blender's curve fill modes."""
    FULL = 'FULL'
    BACK = 'BACK'
    FRONT = 'FRONT'
    HALF = 'HALF'

class BBox(NamedTuple):
    min: Vector
    max: Vector

class CurvePoint(NamedTuple):
    """Represents an evaluated point on the curve with its distance from the start."""
    co: Vector
    tangent: Vector
    normal: Vector
    # Cumulative distance from the start of the first island
    distance: float

class BaseCurve(Object):
    def __init__(self, obj: Optional[bpy.types.Object] = None):
        super().__init__(obj)
        self._is_dirty: bool = True
        self._dependencies: List['Object'] = []

    @override
    def remove(self, physical=True):
        """Safely removes the curve object and its data from the Blender scene."""
        if self.obj and physical:
            curve_data = self.obj.data
            if curve_data and curve_data.users == 0:
                bpy.data.curves.remove(curve_data)
        super().remove(physical)

    def center(self) -> Vector:
        """Returns the geometric center of the curve's bounding box."""
        bbox = self.bbox
        return (bbox.min + bbox.max) / 2.0
    
    @property
    def resolution(self) -> int:
        return self.obj.data.resolution_u
        
    @resolution.setter
    def resolution(self, value: int):
        self.obj.data.resolution_u = value
        self._is_dirty = True

    @property
    def fill_mode(self) -> FillMode:
        return FillMode(self.obj.data.fill_mode)

    @fill_mode.setter
    def fill_mode(self, mode: FillMode):
        self.obj.data.fill_mode = mode.value

    def extrude(self, amount: float = 0.1) -> Self:
        """Extrudes the text geometry."""
        self.obj.data.extrude = amount
        self._is_dirty = True
        return self

    def bevel(self, depth: float = 0.1, resolution: int = 4, fill_caps: bool = True) -> Self:
        """Creates a geometric tube around the curve."""
        self.obj.data.bevel_mode = 'ROUND'
        self.obj.data.bevel_depth = depth
        self.obj.data.bevel_resolution = resolution
        self.obj.data.use_fill_caps = fill_caps
        self._is_dirty = True
        return self

    @property
    def part(self) -> 'Part':
        """Converts to mesh by temporarily linking deps to scene."""
        from .part import Part
        
        # Collect self and all curve dependencies
        temp_obs = {self.obj} | {d.obj for d in self._dependencies}
        scene_objects = bpy.context.scene.collection.objects
        
        # Link missing objects and track them for cleanup
        linked_temp = []
        for ob in temp_obs:
            if not ob.users_collection:
                scene_objects.link(ob)
                linked_temp.append(ob)

        try:
            dg = bpy.context.evaluated_depsgraph_get()
            dg.update()
            eval_obj = self.obj.evaluated_get(dg)
            
            mesh = bpy.data.meshes.new_from_object(eval_obj, depsgraph=dg)
            new_obj = bpy.data.objects.new("CurvePart", mesh)
            new_obj.matrix_world = self.obj.matrix_world.copy()

            # Preserve materials by copying them to the new object data
            for mat in self.obj.data.materials:
                mesh.materials.append(mat)

            return Part(obj=new_obj)
        finally:
            # Clean up only what we linked
            for ob in linked_temp:
                scene_objects.unlink(ob)

class Curve(BaseCurve, CurveLike):
    """
    An object representing a mathematical or poly-curve.
    Manages its own Blender CURVE object and provides precise evaluation methods.
    """
    def __init__(self, obj: Optional[bpy.types.Object] = None):
        super().__init__(obj)
        self._evaluated_points: List[List[CurvePoint]] = []
        self._total_length: float = 0.0

    @override
    def _create_empty_object(self):
        crv_data = bpy.data.curves.new(name="CurveData", type='CURVE')
        crv_data.dimensions = '3D'
        obj = bpy.data.objects.new("Curve", crv_data)
        return obj
    
    @override
    def copy(self) -> 'Curve':
        """Creates a copy of the Curve and its underlying Blender object."""
        if not self.is_valid:
            raise RuntimeError("Object is removed")
        
        # Copy object and data
        new_obj = self.obj.copy()
        new_obj.data = self.obj.data.copy()
        new_curve = Curve(new_obj)
        # Inherit dirty state to ensure first evaluation works
        new_curve._is_dirty = True
        return new_curve

    def _evaluate(self):
        """
        Fully evaluates the curve geometry, handling multiple splines, 
        cyclic paths, and accurate length calculations.
        """
        if not self._is_dirty:
            return

        # 1. Get the evaluated mesh (the tessellated "physical" version of the curve)
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        eval_obj = self.obj.evaluated_get(dg)
        
        # This gives us the line segments Blender actually uses
        mesh = bpy.data.meshes.new_from_object(eval_obj, depsgraph=dg)
        
        self._evaluated_points = []
        self._total_length = 0.0

        if not mesh.vertices:
            bpy.data.meshes.remove(mesh)
            self._is_dirty = False
            return

        # 2. Build adjacency map to traverse vertices in order
        # Since it's a curve, each vertex has max 2 neighbors.
        adj = {v.index: [] for v in mesh.vertices}
        for e in mesh.edges:
            adj[e.vertices[0]].append(e.vertices[1])
            adj[e.vertices[1]].append(e.vertices[0])

        visited = set()
        
        # 3. Extract Islands (Splines)
        for v_idx in range(len(mesh.vertices)):
            if v_idx in visited:
                continue
            
            # Find an endpoint to start traversal (vertex with 1 neighbor)
            # If it's a closed loop, all vertices have 2 neighbors; start anywhere.
            start_node = v_idx
            for node in adj:
                if node not in visited and len(adj[node]) == 1:
                    start_node = node
                    break
            
            island_raw_indices = []
            curr = start_node
            
            # Linear traversal of the chain
            while curr is not None and curr not in visited:
                visited.add(curr)
                island_raw_indices.append(curr)
                
                # Move to next unvisited neighbor
                next_node = None
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        next_node = neighbor
                        break
                curr = next_node

            # Handle closing the loop for cyclic splines
            is_cyclic = False
            if len(adj[island_raw_indices[-1]]) == 2 and island_raw_indices[0] in adj[island_raw_indices[-1]]:
                is_cyclic = True

            # 4. Process the island into CurvePoints
            island_points: List[CurvePoint] = []
            for i, idx in enumerate(island_raw_indices):
                v: bmesh.types.BMVert = mesh.vertices[idx]
                co = v.co.copy()
                
                # Tangent calculation
                if i < len(island_raw_indices) - 1:
                    next_co: Vector = mesh.vertices[island_raw_indices[i+1]].co
                    tangent = (next_co - co).normalized()
                    step_dist = (next_co - co).length
                elif is_cyclic:
                    next_co: Vector = mesh.vertices[island_raw_indices[0]].co
                    tangent = (next_co - co).normalized()
                    step_dist = (next_co - co).length
                else:
                    # End of open line: use previous tangent
                    prev_co: Vector = mesh.vertices[island_raw_indices[i-1]].co
                    tangent = (co - prev_co).normalized()
                    step_dist = 0

                # Normal calculation: vertex normals on wire meshes are zero/unreliable.
                # We calculate a stable reference frame instead.
                world_up = Vector((0, 0, 1))
                if abs(tangent.dot(world_up)) > 0.99:
                    world_up = Vector((0, 1, 0))
                
                # Calculate normal as perpendicular to tangent
                right = tangent.cross(world_up).normalized()
                normal = right.cross(tangent).normalized()
                
                island_points.append(CurvePoint(co, tangent, normal, self._total_length))
                self._total_length += step_dist

            # If cyclic, add the first point at the end to make t=1.0 work perfectly
            if is_cyclic:
                first = island_points[0]
                island_points.append(CurvePoint(first.co, first.tangent, first.normal, self._total_length))

            self._evaluated_points.append(island_points)

        # Cleanup
        bpy.data.meshes.remove(mesh)
        self._is_dirty = False

    def _get_point_at(self, t_or_tm: float, is_meters: bool = False) -> Tuple[Vector, Vector, Vector]:
        """
        High-precision interpolation across all islands.
        """
        self._evaluate()
        if not self._evaluated_points:
            return Vector((0,0,0)), Vector((1,0,0)), Vector((0,0,1))

        target_dist = t_or_tm if is_meters else t_or_tm * self._total_length
        target_dist = max(0.0, min(self._total_length, target_dist))

        # Find which island contains the target distance
        for island in self._evaluated_points:
            if not island: continue
            
            # Check if target is within this island's range
            island_start_dist = island[0].distance
            island_end_dist = island[-1].distance
            
            if island_start_dist <= target_dist <= island_end_dist:
                # Binary search could be used here for very dense curves, 
                # but linear search is usually fine for tessellated resolution.
                for i in range(len(island) - 1):
                    p1 = island[i]
                    p2 = island[i+1]
                    
                    if p1.distance <= target_dist <= p2.distance:
                        # Linear Interpolation
                        segment_dist = p2.distance - p1.distance
                        if segment_dist < 1e-6:
                            return p1.co, p1.tangent, p1.normal
                            
                        factor = (target_dist - p1.distance) / segment_dist
                        
                        pos = p1.co.lerp(p2.co, factor)
                        tan = p1.tangent.lerp(p2.tangent, factor).normalized()
                        norm = p1.normal.lerp(p2.normal, factor).normalized()
                        
                        return pos, tan, norm

        # Fallback (should not be reached due to clamping)
        last = self._evaluated_points[-1][-1]
        return last.co, last.tangent, last.normal

    # --- Properties ---
    @override
    def length(self) -> float:
        self._evaluate()
        return self._total_length

    @property
    def start(self) -> Vector:
        return self.position_at(0.0)

    @property
    def end(self) -> Vector:
        return self.position_at(1.0)

    def position_at(self, t: Optional[float] = None, t_m: Optional[float] = None) -> Vector:
        is_meters = t_m is not None
        val = t_m if is_meters else (t if t is not None else 0.0)
        pos, _, _ = self._get_point_at(val, is_meters)
        return pos

    def tangent_at(self, t: Optional[float] = None, t_m: Optional[float] = None) -> Vector:
        is_meters = t_m is not None
        val = t_m if is_meters else (t if t is not None else 0.0)
        _, tan, _ = self._get_point_at(val, is_meters)
        return tan

    def normal_at(self, t: Optional[float] = None, t_m: Optional[float] = None) -> Vector:
        is_meters = t_m is not None
        val = t_m if is_meters else (t if t is not None else 0.0)
        _, _, norm = self._get_point_at(val, is_meters)
        return norm

    @override
    def at(self, t: Optional[float] = None, t_m: Optional[float] = None) -> Location:
        """
        Returns a Location representing position and rotation along the path.
        The X-axis aligns with the tangent.
        """
        pos, tan, norm = self._get_point_at(t_m if t_m is not None else (t or 0.0), t_m is not None)
        
        # 1. Primary axis: Tangent (now X)
        x_axis = tan.normalized()
        
        # 2. Secondary guide (evaluated normal)
        secondary_guide = norm
        if abs(x_axis.dot(secondary_guide)) > 0.999:
            secondary_guide = Vector((0, 0, 1)) if abs(x_axis.z) < 0.9 else Vector((0, 1, 0))

        # 3. Construct right-handed system where X is tangent
        # Z = X cross Secondary
        z_axis = x_axis.cross(secondary_guide).normalized()
        # Y = Z cross X
        y_axis = z_axis.cross(x_axis).normalized()
        
        mat = Matrix((
            (x_axis.x, y_axis.x, z_axis.x, pos.x),
            (x_axis.y, y_axis.y, z_axis.y, pos.y),
            (x_axis.z, y_axis.z, z_axis.z, pos.z),
            (0.0,      0.0,      0.0,      1.0)
        ))
        return Location(mat)
    
    def bevel(self, depth: float = 0.1, resolution: int = 4, fill_caps: bool = True, limits: Tuple[float, float] = (0.0, 1.0)) -> Self:
        """Creates a geometric tube around the curve."""
        super().bevel(depth, resolution, fill_caps)
        self.obj.data.bevel_factor_mapping_start = 'SPLINE'
        self.obj.data.bevel_factor_mapping_end = 'SPLINE'
        self.obj.data.bevel_factor_start = limits[0]
        self.obj.data.bevel_factor_end = limits[1]
        return self

# ==========================================
# CONTEXT MANAGER
# ==========================================

class BuildCurve:
    """Context manager for constructing interconnected curves."""
    _context_stack: List['BuildCurve'] = []

    def __init__(self, curve: Optional[Curve] = None, merge: bool = True):
        self.curve = curve or Curve()
        self.merge = merge

    def __enter__(self) -> 'BuildCurve':
        BuildCurve._context_stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        BuildCurve._context_stack.pop()
        self.curve._is_dirty = True

    @classmethod
    def _get_context(cls) -> Optional['BuildCurve']:
        if not cls._context_stack:
            return None
        return cls._context_stack[-1]
    
    def length(self) -> float:
        return self.curve.length()
    
    @property
    def start(self) -> Vector:
        return self.curve.start
    
    @property
    def end(self) -> Vector:
        return self.curve.end
    
    def center(self) -> Vector:
        return self.curve.center()
    
    def position_at(self, t: Optional[float] = None, t_m: Optional[float] = None) -> Vector:
        return self.curve.position_at(t, t_m)

    def tangent_at(self, t: Optional[float] = None, t_m: Optional[float] = None) -> Vector:
        return self.curve.tangent_at(t, t_m)

    def normal_at(self, t: Optional[float] = None, t_m: Optional[float] = None) -> Vector:
        return self.curve.normal_at(t, t_m)
    
    def at(self, t: Optional[float] = None, t_m: Optional[float] = None) -> Location:
        return self.curve.at(t, t_m)

    @property
    def bbox(self):
        """Access the bounding box of the curve."""
        return self.curve.bbox

    @property
    def resolution(self) -> int:
        return self.curve.resolution
        
    @resolution.setter
    def resolution(self, value: int):
        self.curve.resolution = value

    @property
    def fill_mode(self) -> FillMode:
        return self.curve.fill_mode

    @fill_mode.setter
    def fill_mode(self, mode: FillMode):
        self.curve.fill_mode = mode

    def bevel(self, depth: float = 0.1, resolution: int = 4, fill_caps: bool = True, limits: Tuple[float, float] = (0.0, 1.0)) -> 'BuildCurve':
        self.curve.bevel(depth, resolution, fill_caps, limits)
        return self
    
    @property
    def part(self) -> 'Part':
        return self.curve.part

    @property
    def current_point(self) -> Optional[Vector]:
        """Returns the end point of the most recently added line segment."""
        if not self.curve.obj.data.splines:
            return None
        
        last_spline = self.curve.obj.data.splines[-1]
        if last_spline.type == 'BEZIER':
            return last_spline.bezier_points[-1].co
        else:
            return last_spline.points[-1].co.xyz

# ==========================================
# GEOMETRY PRIMITIVES
# ==========================================

class CurvePrimitive:
    """Base class for curve primitives that inject themselves into the active BuildCurve."""
    def __init__(self):
        self.ctx = BuildCurve._get_context()
        self.spline = None

    def _add_poly_spline(self, points: List[Vector], close: bool = False):
        if not self.ctx: return
        
        # Merging logic for POLY and NURBS (both use .points)
        if self.ctx.merge and self.ctx.curve.obj.data.splines:
            last = self.ctx.curve.obj.data.splines[-1]
            if last.type == 'POLY' and not last.use_cyclic_u:
                # Compare last point of existing spline with first point of new points
                if (last.points[-1].co.xyz - points[0]).length < 1e-4:
                    self.spline = last
                    start_idx = len(last.points)
                    last.points.add(len(points) - 1)
                    for i in range(1, len(points)):
                        last.points[start_idx + i - 1].co = (*points[i], 1.0)
                    last.use_cyclic_u = close
                    self.ctx.curve._is_dirty = True
                    return

        # Default: create new spline
        self.spline = self.ctx.curve.obj.data.splines.new(type='POLY')
        self.spline.points.add(len(points) - 1)
        for i, pt in enumerate(points):
            self.spline.points[i].co = (*pt, 1.0)
        self.spline.use_cyclic_u = close
        self.ctx.curve._is_dirty = True

    def _add_bezier_spline(self, coords: List[Vector], handles_left: List[Vector], handles_right: List[Vector], close: bool = False):
        if not self.ctx: return
        
        if self.ctx.merge and self.ctx.curve.obj.data.splines:
            last = self.ctx.curve.obj.data.splines[-1]
            if last.type == 'BEZIER' and not last.use_cyclic_u:
                if (last.bezier_points[-1].co - coords[0]).length < 1e-4:
                    self.spline = last
                    # Update handle of the existing shared point
                    last.bezier_points[-1].handle_right = handles_right[0]
                    
                    start_idx = len(last.bezier_points)
                    last.bezier_points.add(len(coords) - 1)
                    for i in range(1, len(coords)):
                        bp = last.bezier_points[start_idx + i - 1]
                        bp.co = coords[i]
                        bp.handle_left = handles_left[i]
                        bp.handle_right = handles_right[i]
                        bp.handle_left_type = 'FREE'
                        bp.handle_right_type = 'FREE'
                    last.use_cyclic_u = close
                    self.ctx.curve._is_dirty = True
                    return

        self.spline = self.ctx.curve.obj.data.splines.new(type='BEZIER')
        self.spline.bezier_points.add(len(coords) - 1)
        for i in range(len(coords)):
            bp = self.spline.bezier_points[i]
            bp.co = coords[i]
            bp.handle_left = handles_left[i]
            bp.handle_right = handles_right[i]
            bp.handle_left_type = 'FREE'
            bp.handle_right_type = 'FREE'
        self.spline.use_cyclic_u = close
        self.ctx.curve._is_dirty = True

    def _add_nurbs_spline(self, points: List[Vector], close: bool = False):
        """Internal helper for NURBS to support merging."""
        if not self.ctx: return
        
        if self.ctx.merge and self.ctx.curve.obj.data.splines:
            last = self.ctx.curve.obj.data.splines[-1]
            if last.type == 'NURBS' and not last.use_cyclic_u:
                if (last.points[-1].co.xyz - points[0]).length < 1e-4:
                    self.spline = last
                    start_idx = len(last.points)
                    last.points.add(len(points) - 1)
                    for i in range(1, len(points)):
                        last.points[start_idx + i - 1].co = (*points[i], 1.0)
                    last.use_cyclic_u = close
                    self.ctx.curve._is_dirty = True
                    return

        self.spline = self.ctx.curve.obj.data.splines.new(type='NURBS')
        self.spline.points.add(len(points) - 1)
        for i, pt in enumerate(points):
            self.spline.points[i].co = (*pt, 1.0)
        self.spline.use_cyclic_u = close
        self.spline.use_endpoint_u = True
        self.ctx.curve._is_dirty = True

class Line(CurvePrimitive):
    """A straight line segment."""
    def __init__(self, start: Optional[VectorLike] = None, end: VectorLike = (0, 0, 0)):
        super().__init__()
        
        self.end = extract_vector(end)
        if start is None:
            self.start = self.ctx.current_point if self.ctx else Vector((0, 0, 0))
        else:
            self.start = extract_vector(start)

        self._add_poly_spline([self.start, self.end])

class Polyline(CurvePrimitive):
    """A series of connected straight lines."""
    def __init__(self, *pts: VectorLike, close: bool = False):
        super().__init__()
        # If the first argument is a list or tuple of vectors, use it directly
        if len(pts) == 1 and isinstance(pts[0], (list, tuple)) and not isinstance(pts[0][0], (float, int)):
            pts = pts[0]
        points = [extract_vector(p) for p in pts]
        self._add_poly_spline(points, close=close)

class Spline(CurvePrimitive):
    """A smooth NURBS path through provided points."""
    def __init__(self, *pts: VectorLike, close: bool = False):
        super().__init__()
        # If the first argument is a list or tuple of vectors, use it directly
        if len(pts) == 1 and isinstance(pts[0], (list, tuple)) and not isinstance(pts[0][0], (float, int)):
            pts = pts[0]
        points = [extract_vector(p) for p in pts]
        self._add_nurbs_spline(points, close=close)

class BezierCurve(CurvePrimitive):
    """A standard Bezier curve using control points."""
    def __init__(self, start: VectorLike, handle1: VectorLike, handle2: VectorLike, end: VectorLike):
        super().__init__()
        s = extract_vector(start)
        h1 = extract_vector(handle1)
        h2 = extract_vector(handle2)
        e = extract_vector(end)
        
        self._add_bezier_spline(
            coords=[s, e],
            handles_left=[s, h2],
            handles_right=[h1, e]
        )

class TangentArc(CurvePrimitive):
    """An arc that exits smoothly from the end tangent of the current curve."""
    def __init__(self, end: VectorLike):
        super().__init__()
        if not self.ctx: return
        
        end_vec = extract_vector(end)
        start_vec = self.ctx.current_point
        start_tangent = self.ctx.curve.tangent_at(1.0)
        
        # Approximate the arc with a Bezier curve to maintain tangent continuity
        dist = (end_vec - start_vec).length
        handle_len = dist * 0.333
        
        h1 = start_vec + (start_tangent * handle_len)
        
        # We need a smooth entry to the end point. If just a generic arc, point handle towards start.
        end_tangent = (start_vec - end_vec).normalized()
        h2 = end_vec + (end_tangent * handle_len)

        self._add_bezier_spline(
            coords=[start_vec, end_vec],
            handles_left=[start_vec, h2],
            handles_right=[h1, end_vec]
        )

class RadiusArc(CurvePrimitive):
    """Creates an arc between two points given a specific radius."""
    def __init__(self, start: VectorLike, end: VectorLike, radius: float):
        super().__init__()
        # Simplified representation: 
        # In a full CAD implementation, you compute the circle center via intersection 
        # and generate poly/nurbs points along the arc. 
        # Here we approximate with subdivided polyline for stability.
        s = extract_vector(start)
        e = extract_vector(end)
        
        # Midpoint math to generate arc
        mid = (s + e) / 2
        dist = (s - e).length
        if radius < dist / 2: radius = dist / 2 # Prevent math domain errors
        
        sagitta = radius - math.sqrt(radius**2 - (dist/2)**2)
        normal = (e - s).cross(Vector((0,0,1))).normalized()
        if normal.length < 1e-6: normal = Vector((0,1,0))
        
        arc_mid = mid + (normal * sagitta)
        
        # 3-point Bezier approximation
        self._add_bezier_spline(
            coords=[s, e],
            handles_left=[s, arc_mid],
            handles_right=[arc_mid, e]
        )

class CenterArc(CurvePrimitive):
    """Draws an arc based on a center point, radius, and angles."""
    def __init__(self, center: VectorLike, radius: float, start_angle: float, end_angle: float):
        super().__init__()
        c = extract_vector(center)
        
        pts = []
        steps = 16 # Resolution of the arc segment
        angle_step = (end_angle - start_angle) / steps
        
        for i in range(steps + 1):
            theta = math.radians(start_angle + (i * angle_step))
            x = c.x + radius * math.cos(theta)
            y = c.y + radius * math.sin(theta)
            pts.append(Vector((x, y, c.z)))
            
        self._add_poly_spline(pts)

class Jiggle(CurvePrimitive):
    """An organic 'noisy' line between two points."""
    def __init__(self, start: VectorLike, end: VectorLike, noise_factor: float = 1.0, segments: int = 10):
        super().__init__()
        s = extract_vector(start)
        e = extract_vector(end)
        
        pts = [s]
        for i in range(1, segments):
            t = i / segments
            base_pt = s.lerp(e, t)
            
            # Add random noise orthogonal to the line
            noise = Vector((
                random.uniform(-noise_factor, noise_factor),
                random.uniform(-noise_factor, noise_factor),
                random.uniform(-noise_factor, noise_factor)
            ))
            pts.append(base_pt + noise)
            
        pts.append(e)
        self._add_poly_spline(pts)

def make_curve(
    rule: Callable[[float], Union[tuple[float, float, float], Vector]], 
    limit: float, 
    resolution: int = 50, 
    close: bool = False,
    curve_type: Union[Type[Polyline], Type[Spline]] = Spline
) -> Curve:
    """
    Generates a Curve based on a parametric function.
    
    Args:
        rule: A function taking t (0 to limit) and returning (x, y, z).
        limit: The maximum value of t.
        resolution: Number of segments (points = resolution + 1).
        curve_type: The build123d-style class to instantiate (Spline, Polyline, etc).
    """
    points = [
        Vector(rule((i / resolution) * limit)) 
        for i in range(resolution + 1)
    ]
    with BuildCurve() as bc:
        curve_type(points, close=close)
    return bc.curve
