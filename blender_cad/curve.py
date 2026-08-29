import math
import random
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    NamedTuple,
    Optional,
    Self,
    TypeAlias,
    Union,
)

import bmesh
import bpy
from mathutils import Matrix, Vector
from typing_extensions import override

from .common import (
    AbstractCurve,
    Axis,
    VectorLike,
    _flatten_items,
    extract_vector,
    match_tags,
    tag_to_list,
)
from .location import Location, Pos, Rot, Scale
from .object import AttributeDomainItems, Object

if TYPE_CHECKING:
    from .part import Part


class FillMode(Enum):
    """Enumeration for Blender's curve fill modes."""

    BOTH = "BOTH"
    BACK = "BACK"
    FRONT = "FRONT"
    HALF = "HALF"


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
        self._dependencies: list["Object"] = []

    @override
    def remove(self, physical=True):
        """Safely removes the curve object and its data from the Blender scene."""
        if physical and self.is_physically_valid:
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

    def bevel(
        self, depth: float = 0.1, resolution: int = 4, fill_caps: bool = True
    ) -> Self:
        """Creates a geometric tube around the curve."""
        self.obj.data.bevel_mode = "ROUND"
        self.obj.data.bevel_depth = depth
        self.obj.data.bevel_resolution = resolution
        self.obj.data.use_fill_caps = fill_caps
        self._is_dirty = True
        return self

    @property
    def part(self) -> "Part":
        """Converts to mesh by temporarily linking deps to scene."""
        from .part import Part

        # Curve and shrinkwrap modifiers only evaluate correctly when every
        # referenced object is linked, even if the caller keeps them temporary.
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

    @override
    def _domain_indices(
        self,
        domain: AttributeDomainItems,
    ) -> list[int]:
        """
        Returns all indices for the specified geometry domain.
        """
        data = self.obj.data
        assert isinstance(data, bpy.types.Curve)
        if domain == "CURVE":
            return list(range(len(data.splines)))
        if domain == "POINT":
            count = sum(
                len(s.bezier_points) if s.type == "BEZIER" else len(s.points)
                for s in data.splines
            )
            return list(range(count))
        return []


class Curve(BaseCurve, AbstractCurve):
    """
    An object representing a mathematical or poly-curve.
    Manages its own Blender CURVE object and provides precise evaluation methods.
    """

    TAG_POINT_INDEX = "curve:point_index"
    TAG_POINT_FIRST = "curve:point_index:0"
    TAG_POINT_LAST = "curve:point_index:-1"
    TAG_POINT_SMOOTH_FILLET_START = "curve:point_smooth_fillet:start"
    TAG_POINT_SMOOTH_FILLET_END = "curve:point_smooth_fillet:end"

    def __init__(self, obj: Optional[bpy.types.Object] = None):
        super().__init__(obj)
        self._evaluated_points: list[list[CurvePoint]] = []
        self._total_length: float = 0.0
        self.source_curve = self

    @property
    def points(self):
        self._evaluate()
        return [[pt.co for pt in island] for island in self._evaluated_points]

    @property
    def source_point_indices(self) -> list[int]:
        return self._get_eval_indices_in(self.source_curve)

    @override
    def curve(self) -> "Curve":
        return self

    @override
    def _create_empty_object(self):
        crv_data = bpy.data.curves.new(name="CurveData", type="CURVE")
        crv_data.dimensions = "3D"
        obj = bpy.data.objects.new("Curve", crv_data)
        return obj

    @override
    def copy(self) -> "Curve":
        """Creates a copy of the Curve and its underlying Blender object."""
        if not self.is_valid:
            raise RuntimeError("Object is removed")

        # Copy object and data
        new_obj = self.obj.copy()
        new_obj.data = self.obj.data.copy()
        new_curve = Curve(new_obj)
        # Inherit dirty state to ensure first evaluation works
        new_curve._is_dirty = True
        self._after_copy(new_curve)
        return new_curve

    @override
    def build_bvh(self):
        prev_extrude = self.obj.data.extrude
        prev_depth = self.obj.data.bevel_depth
        self.obj.data.extrude = 1
        self.obj.data.bevel_depth = 0.0
        bvh = super().build_bvh()
        self.obj.data.extrude = prev_extrude
        self.obj.data.bevel_depth = prev_depth
        return bvh

    def fill(self) -> Self:
        self.obj.data.dimensions = "2D"
        self.obj.data.fill_mode = "BOTH"
        return self

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
            if (
                len(adj[island_raw_indices[-1]]) == 2
                and island_raw_indices[0] in adj[island_raw_indices[-1]]
            ):
                is_cyclic = True

            # 4. Process the island into CurvePoints
            island_points: list[CurvePoint] = []
            for i, idx in enumerate(island_raw_indices):
                v: bmesh.types.BMVert = mesh.vertices[idx]
                co = v.co.copy()

                # Tangent calculation
                if i < len(island_raw_indices) - 1:
                    next_co: Vector = mesh.vertices[island_raw_indices[i + 1]].co
                    tangent = (next_co - co).normalized()
                    step_dist = (next_co - co).length
                elif is_cyclic:
                    next_co: Vector = mesh.vertices[island_raw_indices[0]].co
                    tangent = (next_co - co).normalized()
                    step_dist = (next_co - co).length
                else:
                    # End of open line: use previous tangent
                    prev_co: Vector = mesh.vertices[island_raw_indices[i - 1]].co
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

                island_points.append(
                    CurvePoint(co, tangent, normal, self._total_length)
                )
                self._total_length += step_dist

            # If cyclic, add the first point at the end to make t=1.0 work perfectly
            if is_cyclic:
                first = island_points[0]
                island_points.append(
                    CurvePoint(
                        first.co, first.tangent, first.normal, self._total_length
                    )
                )

            self._evaluated_points.append(island_points)

        # Cleanup
        bpy.data.meshes.remove(mesh)
        self._is_dirty = False

    def _get_point_at(
        self, t_or_tm: float, is_meters: bool = False
    ) -> tuple[Vector, Vector, Vector]:
        """
        High-precision interpolation across all islands.
        """
        self._evaluate()
        if not self._evaluated_points:
            return Vector((0, 0, 0)), Vector((1, 0, 0)), Vector((0, 0, 1))

        target_dist = t_or_tm if is_meters else t_or_tm * self._total_length
        target_dist = max(0.0, min(self._total_length, target_dist))

        # Find which island contains the target distance
        for island in self._evaluated_points:
            if not island:
                continue

            # Check if target is within this island's range
            island_start_dist = island[0].distance
            island_end_dist = island[-1].distance

            if island_start_dist <= target_dist <= island_end_dist:
                # Binary search could be used here for very dense curves,
                # but linear search is usually fine for tessellated resolution.
                for i in range(len(island) - 1):
                    p1 = island[i]
                    p2 = island[i + 1]

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

    def position_at(
        self, t: Optional[float] = None, t_m: Optional[float] = None
    ) -> Vector:
        is_meters = t_m is not None
        val = t_m if is_meters else (t if t is not None else 0.0)
        pos, _, _ = self._get_point_at(val, is_meters)
        return pos

    def tangent_at(
        self, t: Optional[float] = None, t_m: Optional[float] = None
    ) -> Vector:
        is_meters = t_m is not None
        val = t_m if is_meters else (t if t is not None else 0.0)
        _, tan, _ = self._get_point_at(val, is_meters)
        return tan

    def normal_at(
        self, t: Optional[float] = None, t_m: Optional[float] = None
    ) -> Vector:
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
        pos, tan, norm = self._get_point_at(
            t_m if t_m is not None else (t or 0.0), t_m is not None
        )

        # 1. Primary axis: Tangent (now X)
        x_axis = tan.normalized()

        # 2. Secondary guide (evaluated normal)
        secondary_guide = norm
        if abs(x_axis.dot(secondary_guide)) > 0.999:
            secondary_guide = (
                Vector((0, 0, 1)) if abs(x_axis.z) < 0.9 else Vector((0, 1, 0))
            )

        # 3. Construct right-handed system where X is tangent
        # Z = X cross Secondary
        z_axis = x_axis.cross(secondary_guide).normalized()
        # Y = Z cross X
        y_axis = z_axis.cross(x_axis).normalized()

        mat = Matrix(
            (
                (x_axis.x, y_axis.x, z_axis.x, pos.x),
                (x_axis.y, y_axis.y, z_axis.y, pos.y),
                (x_axis.z, y_axis.z, z_axis.z, pos.z),
                (0.0, 0.0, 0.0, 1.0),
            )
        )
        scaled_local_matrix = Scale(self.scale).matrix @ mat
        return Location(scaled_local_matrix, parent_loc=self.loc)

    def bevel(
        self,
        depth: float = 0.1,
        resolution: int = 4,
        fill_caps: bool = True,
        limits: tuple[float, float] = (0.0, 1.0),
    ) -> Self:
        """Creates a geometric tube around the curve."""
        super().bevel(depth, resolution, fill_caps)
        self.obj.data.bevel_factor_mapping_start = "SPLINE"
        self.obj.data.bevel_factor_mapping_end = "SPLINE"
        self.obj.data.bevel_factor_start = limits[0]
        self.obj.data.bevel_factor_end = limits[1]
        return self

    def _get_eval_indices_in(self, parent_curve: "Curve") -> list[int]:
        """
        Calculates and returns the evaluated point indices of parent_curve
        that correspond geometrically to the evaluated points of this curve.
        """
        self._evaluate()
        parent_curve._evaluate()

        # Flatten evaluated points for both curves
        parent_eval_pts = [
            pt.co for island in parent_curve._evaluated_points for pt in island
        ]
        self_eval_pts = [pt.co for island in self._evaluated_points for pt in island]

        if not parent_eval_pts or not self_eval_pts:
            return []

        matched_indices: list[int] = []

        # Find closest evaluated point in parent_curve for each point in self
        for self_pt in self_eval_pts:
            best_idx = 0
            best_dist_sq = float("inf")

            for parent_idx, parent_pt in enumerate(parent_eval_pts):
                dist_sq = (parent_pt - self_pt).length_squared
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_idx = parent_idx
                    if dist_sq < 1e-8:
                        break

            matched_indices.append(best_idx)

        # Preserve order while removing duplicate indices if needed, or return raw list
        return matched_indices

    def tagged(self, *tags: str, invert: bool = False) -> "Curve":
        """
        Extracts matching control points or entire splines into a new sub-Curve instance based on tags.
        """
        if not self.is_valid:
            raise RuntimeError("Object is removed")

        def _matches(applied_tags: Iterable[str]) -> bool:
            return match_tags(applied_tags, tags, invert)

        # Initialize empty sub-curve target
        sub_curve = Curve()
        sub_curve.obj.matrix_world = self.obj.matrix_world.copy()
        sub_curve.source_curve = (
            self.source_curve if self.source_curve is not None else self
        )
        sub_data = sub_curve.obj.data

        global_pt_offset = 0
        total_pt_count = sum(
            len(s.bezier_points) if s.type == "BEZIER" else len(s.points)
            for s in self.obj.data.splines
        )

        for spline_idx, spline in enumerate(self.obj.data.splines):
            # Check CURVE domain tags
            curve_tags = self._get_tags("CURVE", [spline_idx])
            curve_matched = _matches(curve_tags)

            pts = spline.bezier_points if spline.type == "BEZIER" else spline.points
            selected_items: list[
                tuple[int, int, set[str]]
            ] = []  # (local_idx, global_idx, pt_tags)

            # Check POINT domain tags for each control point
            for local_idx in range(len(pts)):
                pt_global_idx = global_pt_offset + local_idx
                pt_tags = self._get_tags("POINT", [pt_global_idx])
                pt_tags_src = pt_tags.copy()
                pt_tags.append(f"{self.TAG_POINT_INDEX}:{pt_global_idx}")
                pt_tags.append(
                    f"{self.TAG_POINT_INDEX}:{pt_global_idx - total_pt_count}"
                )

                if (
                    (curve_matched and _matches(pt_tags))
                    if invert
                    else (curve_matched or _matches(pt_tags))
                ):
                    selected_items.append((local_idx, pt_global_idx, pt_tags_src))

            global_pt_offset += len(pts)

            if not selected_items:
                continue

            # Create corresponding spline in the new sub-curve
            new_spline = sub_data.splines.new(type=spline.type)
            new_spline.use_smooth = spline.use_smooth
            new_spline.resolution_u = spline.resolution_u

            # Preserve cyclic state if all points from the original cyclic spline were selected
            if len(selected_items) == len(pts):
                new_spline.use_cyclic_u = spline.use_cyclic_u

            # Allocate control points FIRST (newly created splines start with 1 point by default)
            count = len(selected_items)
            if count > 1:
                if spline.type == "BEZIER":
                    new_spline.bezier_points.add(count - 1)
                else:
                    new_spline.points.add(count - 1)

            # Apply NURBS parameters AFTER points are allocated
            if spline.type == "NURBS":
                new_spline.use_endpoint_u = spline.use_endpoint_u
                new_spline.order_u = min(spline.order_u, count)

            # Global point offset for tag preservation in the new sub-curve
            new_spline_pt_offset = sum(
                len(s.bezier_points) if s.type == "BEZIER" else len(s.points)
                for s in sub_data.splines
                if s != new_spline
            )

            # Transfer point coordinates, handles, and properties
            for new_local_idx, (orig_local_idx, _, orig_pt_tags) in enumerate(
                selected_items
            ):
                if spline.type == "BEZIER":
                    src_bp = spline.bezier_points[orig_local_idx]
                    dst_bp = new_spline.bezier_points[new_local_idx]

                    # Assign coordinates first before lock/free handle types
                    dst_bp.co = src_bp.co.copy()
                    dst_bp.handle_left = src_bp.handle_left.copy()
                    dst_bp.handle_right = src_bp.handle_right.copy()
                    dst_bp.handle_left_type = src_bp.handle_left_type
                    dst_bp.handle_right_type = src_bp.handle_right_type
                    dst_bp.radius = src_bp.radius
                    dst_bp.tilt = src_bp.tilt
                else:
                    src_p = spline.points[orig_local_idx]
                    dst_p = new_spline.points[new_local_idx]

                    dst_p.co = src_p.co.copy()
                    dst_p.radius = src_p.radius
                    dst_p.tilt = src_p.tilt

                # Re-apply point tags onto the new sub-curve
                if orig_pt_tags:
                    new_global_pt_idx = new_spline_pt_offset + new_local_idx
                    sub_curve._add_tags("POINT", [new_global_pt_idx], orig_pt_tags)

            # Re-apply spline/curve tags onto the new sub-curve
            if curve_tags:
                new_spline_idx = len(sub_data.splines) - 1
                sub_curve._add_tags("CURVE", [new_spline_idx], curve_tags)

        sub_curve._is_dirty = True
        return sub_curve

    def untagged(self, *tags: str):
        return self.tagged(*tags, invert=True)


# ==========================================
# CONTEXT MANAGER
# ==========================================


class BuildCurve:
    """Context manager for constructing interconnected curves."""

    _context_stack: list["BuildCurve"] = []

    def __init__(self, curve: Optional[Curve] = None, merge: bool = True):
        self.curve = curve or Curve()
        self.merge = merge

    def __enter__(self) -> Self:
        BuildCurve._context_stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        BuildCurve._context_stack.pop()
        self.curve._is_dirty = True

    @classmethod
    def _get_context(cls) -> Optional["BuildCurve"]:
        if not cls._context_stack:
            return None
        return cls._context_stack[-1]

    def fill(self):
        return self.curve.fill()

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

    def position_at(
        self, t: Optional[float] = None, t_m: Optional[float] = None
    ) -> Vector:
        return self.curve.position_at(t, t_m)

    def tangent_at(
        self, t: Optional[float] = None, t_m: Optional[float] = None
    ) -> Vector:
        return self.curve.tangent_at(t, t_m)

    def normal_at(
        self, t: Optional[float] = None, t_m: Optional[float] = None
    ) -> Vector:
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

    def bevel(
        self,
        depth: float = 0.1,
        resolution: int = 4,
        fill_caps: bool = True,
        limits: tuple[float, float] = (0.0, 1.0),
    ) -> "BuildCurve":
        self.curve.bevel(depth, resolution, fill_caps, limits)
        return self

    @property
    def part(self) -> "Part":
        return self.curve.part

    @property
    def current_point(self) -> Optional[Vector]:
        """Returns the end point of the most recently added line segment."""
        if not self.curve.obj.data.splines:
            return None

        last_spline = self.curve.obj.data.splines[-1]
        if last_spline.type == "BEZIER":
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

    def _add_poly_spline(self, points: list[Vector], close: bool = False):
        if not self.ctx:
            return

        # Merging logic for POLY and NURBS (both use .points)
        if self.ctx.merge and self.ctx.curve.obj.data.splines:
            last = self.ctx.curve.obj.data.splines[-1]
            if last.type == "POLY" and not last.use_cyclic_u:
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
        self.spline = self.ctx.curve.obj.data.splines.new(type="POLY")
        self.spline.points.add(len(points) - 1)
        for i, pt in enumerate(points):
            self.spline.points[i].co = (*pt, 1.0)
        self.spline.use_cyclic_u = close
        self.ctx.curve._is_dirty = True

    def _add_bezier_spline(
        self,
        coords: list[Vector],
        handles_left: list[Vector],
        handles_right: list[Vector],
        close: bool = False,
    ):
        if not self.ctx:
            return

        if self.ctx.merge and self.ctx.curve.obj.data.splines:
            last = self.ctx.curve.obj.data.splines[-1]
            if last.type == "BEZIER" and not last.use_cyclic_u:
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
                        bp.handle_left_type = "FREE"
                        bp.handle_right_type = "FREE"
                    last.use_cyclic_u = close
                    self.ctx.curve._is_dirty = True
                    return

        self.spline = self.ctx.curve.obj.data.splines.new(type="BEZIER")
        self.spline.bezier_points.add(len(coords) - 1)
        for i in range(len(coords)):
            bp = self.spline.bezier_points[i]
            bp.co = coords[i]
            bp.handle_left = handles_left[i]
            bp.handle_right = handles_right[i]
            bp.handle_left_type = "FREE"
            bp.handle_right_type = "FREE"
        self.spline.use_cyclic_u = close
        self.ctx.curve._is_dirty = True

    def _add_nurbs_spline(self, points: list[Vector], close: bool = False):
        """Internal helper for NURBS to support merging."""
        if not self.ctx:
            return

        if self.ctx.merge and self.ctx.curve.obj.data.splines:
            last = self.ctx.curve.obj.data.splines[-1]
            if last.type == "NURBS" and not last.use_cyclic_u:
                if (last.points[-1].co.xyz - points[0]).length < 1e-4:
                    self.spline = last
                    start_idx = len(last.points)
                    last.points.add(len(points) - 1)
                    for i in range(1, len(points)):
                        last.points[start_idx + i - 1].co = (*points[i], 1.0)
                    last.use_cyclic_u = close
                    self.ctx.curve._is_dirty = True
                    return

        self.spline = self.ctx.curve.obj.data.splines.new(type="NURBS")
        self.spline.points.add(len(points) - 1)
        for i, pt in enumerate(points):
            self.spline.points[i].co = (*pt, 1.0)
        self.spline.use_cyclic_u = close
        self.spline.use_endpoint_u = True
        self.ctx.curve._is_dirty = True

    def _apply_tags(
        self,
        tags: Optional[str | Iterable[str]] = None,
        point_tags: Optional[dict[int, str | Iterable[str]]] = None,
        domain: AttributeDomainItems = "CURVE",
    ):
        """
        Applies tags to the spline itself (CURVE domain) or to individual control points (POINT domain).

        Args:
            tags: General tags to apply to the target domain/all indices.
            point_tags: Mapping of {local_point_index: tags} for fine-grained POINT domain tagging.
            domain: Target attribute domain ('CURVE' or 'POINT').
        """
        if not self.ctx or not self.spline:
            return

        curve_obj = self.ctx.curve
        splines = list(curve_obj.obj.data.splines)
        if self.spline not in splines:
            return

        if domain == "CURVE":
            spline_idx = splines.index(self.spline)
            curve_obj._add_tags(
                domain="CURVE", indices=[spline_idx], tags=tag_to_list(tags)
            )

        elif domain == "POINT":
            # Calculate global point offset for this spline within the Curve object data
            point_offset = 0
            for s in splines:
                if s == self.spline:
                    break
                point_offset += (
                    len(s.bezier_points) if s.type == "BEZIER" else len(s.points)
                )

            # Apply per-point tags using local-to-global index mapping
            if point_tags:
                for local_idx, p_tags in point_tags.items():
                    global_idx = point_offset + local_idx
                    curve_obj._add_tags(
                        domain="POINT", indices=[global_idx], tags=tag_to_list(p_tags)
                    )
            else:
                num_pts = (
                    len(self.spline.bezier_points)
                    if self.spline.type == "BEZIER"
                    else len(self.spline.points)
                )
                global_indices = [point_offset + i for i in range(num_pts)]
                curve_obj._add_tags(
                    domain="POINT", indices=global_indices, tags=tag_to_list(tags)
                )


class Line(CurvePrimitive):
    """A straight line segment."""

    def __init__(
        self,
        start: Optional[VectorLike] = None,
        end: VectorLike = (0, 0, 0),
        tag: Optional[str | Iterable[str]] = None,
    ):
        super().__init__()

        self.end = extract_vector(end)
        if start is None:
            self.start = self.ctx.current_point if self.ctx else Vector((0, 0, 0))
        else:
            self.start = extract_vector(start)

        self._add_poly_spline([self.start, self.end])
        self._apply_tags(tags=tag)


class Polyline(CurvePrimitive):
    """A series of connected straight lines."""

    def __init__(
        self,
        *pts: VectorLike,
        close: bool = False,
        tag: Optional[str | Iterable[str]] = None,
    ):
        super().__init__()
        # If the first argument is a list or tuple of vectors, use it directly
        if (
            len(pts) == 1
            and isinstance(pts[0], (list, tuple))
            and not isinstance(pts[0][0], (float, int))
        ):
            pts = pts[0]
        points = [extract_vector(p) for p in pts]
        self._add_poly_spline(points, close=close)
        self._apply_tags(tags=tag)


class Spline(CurvePrimitive):
    """A smooth NURBS path through provided points."""

    def __init__(
        self,
        *pts: VectorLike,
        close: bool = False,
        tag: Optional[str | Iterable[str]] = None,
    ):
        super().__init__()
        # If the first argument is a list or tuple of vectors, use it directly
        if (
            len(pts) == 1
            and isinstance(pts[0], (list, tuple))
            and not isinstance(pts[0][0], (float, int))
        ):
            pts = pts[0]
        points = [extract_vector(p) for p in pts]
        self._add_nurbs_spline(points, close=close)
        self._apply_tags(tags=tag)


class BezierCurve(CurvePrimitive):
    """A standard Bezier curve using control points."""

    def __init__(
        self,
        start: VectorLike,
        handle1: VectorLike,
        handle2: VectorLike,
        end: VectorLike,
        tag: Optional[str | Iterable[str]] = None,
    ):
        super().__init__()
        s = extract_vector(start)
        h1 = extract_vector(handle1)
        h2 = extract_vector(handle2)
        e = extract_vector(end)

        self._add_bezier_spline(
            coords=[s, e], handles_left=[s, h2], handles_right=[h1, e]
        )
        self._apply_tags(tags=tag)


class TangentArc(CurvePrimitive):
    """An arc that exits smoothly from the end tangent of the current curve."""

    def __init__(self, end: VectorLike, tag: Optional[str | Iterable[str]] = None):
        super().__init__()
        if not self.ctx:
            return

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
            handles_right=[h1, end_vec],
        )
        self._apply_tags(tags=tag)


class RadiusArc(CurvePrimitive):
    """Creates an arc between two points given a specific radius."""

    def __init__(
        self,
        start: VectorLike,
        end: VectorLike,
        radius: float,
        tag: Optional[str | Iterable[str]] = None,
    ):
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
        radius = max(radius, dist / 2)  # Prevent math domain errors

        sagitta = radius - math.sqrt(radius**2 - (dist / 2) ** 2)
        normal = (e - s).cross(Vector((0, 0, 1))).normalized()
        if normal.length < 1e-6:
            normal = Vector((0, 1, 0))

        arc_mid = mid + (normal * sagitta)

        # 3-point Bezier approximation
        self._add_bezier_spline(
            coords=[s, e], handles_left=[s, arc_mid], handles_right=[arc_mid, e]
        )
        self._apply_tags(tags=tag)


class CenterArc(CurvePrimitive):
    """Draws an arc based on a center point, radius, and angles."""

    def __init__(
        self,
        center: VectorLike,
        radius: float,
        start_angle: float,
        end_angle: float,
        tag: Optional[str | Iterable[str]] = None,
    ):
        super().__init__()
        c = extract_vector(center)

        pts = []
        steps = 16  # Resolution of the arc segment
        angle_step = (end_angle - start_angle) / steps

        for i in range(steps + 1):
            theta = math.radians(start_angle + (i * angle_step))
            x = c.x + radius * math.cos(theta)
            y = c.y + radius * math.sin(theta)
            pts.append(Vector((x, y, c.z)))

        self._add_poly_spline(pts)
        self._apply_tags(tags=tag)


class Jiggle(CurvePrimitive):
    """An organic 'noisy' line between two points."""

    def __init__(
        self,
        start: VectorLike,
        end: VectorLike,
        noise_factor: float = 1.0,
        segments: int = 10,
        tag: Optional[str | Iterable[str]] = None,
    ):
        super().__init__()
        s = extract_vector(start)
        e = extract_vector(end)

        pts = [s]
        for i in range(1, segments):
            t = i / segments
            base_pt = s.lerp(e, t)

            # Add random noise orthogonal to the line
            noise = Vector(
                (
                    random.uniform(-noise_factor, noise_factor),
                    random.uniform(-noise_factor, noise_factor),
                    random.uniform(-noise_factor, noise_factor),
                )
            )
            pts.append(base_pt + noise)

        pts.append(e)
        self._add_poly_spline(pts)
        self._apply_tags(tags=tag)


def make_curve(
    rule: Callable[[float], Union[tuple[float, float, float], Vector]],
    limit: float,
    resolution: int = 50,
    close: bool = False,
    curve_type: Union[type[Polyline], type[Spline]] = Spline,
    tag: Optional[str | Iterable[str]] = None,
) -> Curve:
    """
    Generates a Curve based on a parametric function.

    Args:
        rule: A function taking t (0 to limit) and returning (x, y, z).
        limit: The maximum value of t.
        resolution: Number of segments (points = resolution + 1).
        curve_type: The build123d-style class to instantiate (Spline, Polyline, etc).
    """
    points = [Vector(rule((i / resolution) * limit)) for i in range(resolution + 1)]
    with BuildCurve() as bc:
        curve_type(points, close=close, tag=tag)
    return bc.curve


class curve:
    class BuildContext:
        def __init__(
            self,
            pos: Pos,
            rot: Rot,
            axis: Axis,
            global_smooth: bool,
            global_radius: float,
            base_forward: Pos,
            active_tags: Optional[set[str]] = None,
        ):
            self.pos = pos
            self.rot = rot
            self.axis = axis
            self.global_smooth = global_smooth
            self.global_radius = global_radius
            self.base_forward = base_forward
            self.active_tags: set[str] = set(active_tags) if active_tags else set()
            # Path tracking nodes: (position_vector, radius, node_tags_set)
            self.path: list[tuple[Vector, float, set[str]]] = [
                (extract_vector(self.pos), 0.0, set(self.active_tags))
            ]

        def copy(self):
            """Creates a deep copy of the state for isolated branch execution."""
            return curve.BuildContext(
                pos=self.pos,
                rot=self.rot,
                axis=self.axis,
                global_smooth=self.global_smooth,
                global_radius=self.global_radius,
                base_forward=self.base_forward,
                active_tags=self.active_tags.copy(),
            )

    @dataclass
    class step:
        length: float
        angle: float = 0.0
        rot: Optional[Rot] = None
        smooth: Optional[bool] = None
        radius: Optional[float] = None
        tag: Optional[str | Iterable[str]] = None

    @dataclass
    class tag:
        tag: str | Iterable[str]

    @dataclass
    class axis:
        axis: Axis

    @dataclass
    class smooth:
        enabled: bool = True
        radius: float = 1.0

    @dataclass
    class rot:
        rot: Rot

    @dataclass
    class clear_rot:
        pass

    @dataclass
    class move_to:
        target: Pos

    @dataclass
    class move_to_X:
        pass

    @dataclass
    class move_to_Y:
        pass

    item_type: TypeAlias = Union[
        step,
        tag,
        axis,
        smooth,
        rot,
        clear_rot,
        move_to,
        move_to_X,
        move_to_Y,
        "curve",
    ]

    def __init__(
        self,
        *items: item_type,
        axis: Axis = Axis.Z,
        start_pos: Pos = Pos(),
        trim_ends: bool = False,
        close: bool = True,
        tag: Optional[str | Iterable[str]] = None,
    ):
        self.items: list[curve.item_type] = list(_flatten_items(items))
        self.axis = axis
        self.start_pos = start_pos
        self.trim_ends = trim_ends
        self.close = close
        self.tags = tag_to_list(tag)

    @property
    def part(self):
        return self.curve.part

    @property
    def curve(self):
        return self.build()

    def build(
        self,
        start_pos: Optional[Pos] = None,
        forward_dir: Pos = Pos(X=1),
        into_current_ctx: bool = False,
    ) -> "Curve":
        """Evaluates the entire declarative tree and generates geometry primitives."""
        with nullcontext() if into_current_ctx else BuildCurve():
            bc = BuildCurve._get_context()

            initial_pos = self.start_pos
            if start_pos is not None:
                initial_pos = start_pos
            elif bc.current_point is not None:
                initial_pos = Pos(bc.current_point)

            ctx = curve.BuildContext(
                pos=initial_pos,
                rot=Rot(),
                axis=Axis.Z,
                global_smooth=False,
                global_radius=1.0,
                base_forward=forward_dir,
                active_tags=set(self.tags) if self.tags else None,
            )
            self._execute_tree(self, ctx)
            self._flush_path(ctx)

            if self.tags and bc.curve:
                bc.curve.add_tags(self.tags)

            return bc.curve

    def _execute_tree(self, item: "curve.item_type", ctx: "curve.BuildContext"):
        """Recursive execution pass evaluating nodes and branches sequentially."""
        if isinstance(item, curve):
            for child in item.items:
                if isinstance(child, curve):
                    branch_ctx = ctx.copy()
                    self._execute_tree(child, branch_ctx)
                    child._flush_path(branch_ctx)
                else:
                    self._execute_tree(child, ctx)

        elif isinstance(item, curve.step):
            # Phase 1: Apply rotations via wrappers
            if item.rot is not None:
                ctx.rot *= item.rot
            elif item.angle != 0.0:
                ctx.rot *= Rot(ctx.axis.value * item.angle)

            # Phase 2: Compute heading tracking vectors
            current_dir = extract_vector(ctx.rot * ctx.base_forward)
            current_dir.normalize()
            new_pos = ctx.pos * Pos(current_dir * item.length)

            # Phase 3: Construct geometry using localized/global state overrides
            self._draw(ctx, new_pos, item.smooth, item.radius, item.tag)

        elif isinstance(item, curve.tag):
            new_tags = tag_to_list(item.tag)
            ctx.active_tags.update(new_tags)

            # Retroactively apply tag to the current/first point in the path if it exists
            if ctx.path:
                _, _, last_tags = ctx.path[-1]
                last_tags.update(new_tags)

        elif isinstance(item, curve.axis):
            ctx.axis = item.axis

        elif isinstance(item, curve.smooth):
            ctx.global_smooth = item.enabled
            ctx.global_radius = item.radius

        elif isinstance(item, curve.rot):
            ctx.rot *= item.rot

        elif isinstance(item, curve.clear_rot):
            ctx.rot = Rot()

        elif isinstance(item, curve.move_to):
            self._draw(ctx, item.target)

        elif isinstance(item, curve.move_to_X):
            self._draw(ctx, Pos(X=ctx.pos.x, Z=ctx.pos.z))
            ctx.rot = Rot(Z=-90)

        elif isinstance(item, curve.move_to_Y):
            self._draw(ctx, Pos(Y=ctx.pos.y, Z=ctx.pos.z))
            ctx.rot = Rot(Z=180)

    def _draw(
        self,
        ctx: "curve.BuildContext",
        new_pos: Pos,
        smooth: Optional[bool] = None,
        radius: Optional[float] = None,
        tag: Optional[str | Iterable[str]] = None,
    ):
        is_smooth = smooth if smooth is not None else ctx.global_smooth
        r_val = 0.0
        if is_smooth:
            r_val = radius if radius is not None else ctx.global_radius

        step_tags = set(ctx.active_tags)
        step_tags.update(tag_to_list(tag))

        # Ensure both endpoints of the newly created segment carry the segment's tags
        if ctx.path and step_tags:
            _, _, last_tags = ctx.path[-1]
            last_tags.update(step_tags)

        ctx.path.append((extract_vector(new_pos), r_val, step_tags))
        ctx.pos = new_pos

    def _flush_path(self, ctx: "curve.BuildContext"):
        """Processes accumulated tracking points, resolves fillet parameters, and tags individual control points."""
        points_data = ctx.path
        if len(points_data) < 2:
            return

        # Control knots and handles for continuous fillet interpolation
        coords: list[Vector] = []
        handles_left: list[Vector] = []
        handles_right: list[Vector] = []

        # Map generated knot indices to their respective point tags
        point_tags_map: dict[int, set[str]] = {}
        n = len(points_data)

        # 1. Insert start coordinate knot
        if not self.trim_ends:
            p0, _, t0 = points_data[0]
            coords.append(p0)
            handles_left.append(p0)
            handles_right.append(p0)
            if t0:
                point_tags_map[len(coords) - 1] = set(t0)

        # 2. Process intermediate corners and fillets
        for i in range(1, n - 1):
            p_curr, _, t_curr = points_data[i]
            p_prev, _, _ = points_data[i - 1]
            p_next, radius, _ = points_data[i + 1]

            v_prev = (p_prev - p_curr).normalized()
            v_next = (p_next - p_curr).normalized()

            dot = max(-1.0, min(1.0, v_prev.dot(v_next)))
            alpha = math.acos(dot)

            # Collinear safety check (sharp corner or straight pass)
            if radius <= 0.0 or alpha < 1e-4 or alpha > (math.pi - 1e-4):
                coords.append(p_curr)
                handles_left.append(p_curr)
                handles_right.append(p_curr)
                if t_curr:
                    point_tags_map[len(coords) - 1] = set(t_curr)
                continue

            # Fillet placement calculation
            theta = alpha / 2.0
            t_dist = radius / math.tan(theta)

            # Safety capping: guarantee fillet radius never exceeds available segment boundary (CSS model)
            max_t = min(
                0.45 * (p_curr - p_prev).length, 0.45 * (p_next - p_curr).length
            )
            t_dist = min(t_dist, max_t)

            # Pinpoint precise tangent entries and exits
            t1 = p_curr + v_prev * t_dist
            t2 = p_curr + v_next * t_dist

            # Perfect cubic Bezier circular arc weighting factor
            h_len = t_dist * (4.0 / 3.0) * math.tan((math.pi - alpha) / 4.0)

            # Knot 1: Fillet entrance point (inherits current tags + system fillet tag)
            coords.append(t1)
            handles_left.append(t1)
            handles_right.append(t1 - v_prev * h_len)
            t1_tags = set(t_curr) if t_curr else set()
            t1_tags.add(Curve.TAG_POINT_SMOOTH_FILLET_START)
            point_tags_map[len(coords) - 1] = t1_tags

            # Knot 2: Fillet exit point (inherits current tags + system fillet tag)
            coords.append(t2)
            handles_left.append(t2 - v_next * h_len)
            handles_right.append(t2)
            t2_tags = set(t_curr) if t_curr else set()
            t2_tags.add(Curve.TAG_POINT_SMOOTH_FILLET_END)
            point_tags_map[len(coords) - 1] = t2_tags

        # 3. Insert end coordinate knot
        if not self.trim_ends:
            pn, _, tn = points_data[-1]
            coords.append(pn)
            handles_left.append(pn)
            handles_right.append(pn)
            if tn:
                point_tags_map[len(coords) - 1] = set(tn)

        # Generate geometry primitive and apply point tags directly
        fillet_primitive = CurvePrimitive()
        fillet_primitive._add_bezier_spline(
            coords, handles_left, handles_right, self.close
        )
        fillet_primitive._apply_tags(point_tags=point_tags_map, domain="POINT")
