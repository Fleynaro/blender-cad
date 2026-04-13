from abc import ABC, abstractmethod
from enum import Enum
import math
import bpy
import bmesh
from typing import List, Optional, Union
from typing_extensions import override
from mathutils import Matrix, Quaternion, Vector

from .common import Axis
from .build_part import BuildPart, Mode
from .part import Part
from .curve import BaseCurve, BuildCurve
from .location import Location, Locations, Rot, Transform
from .shape_list import ShapeList
from .geometry import GeometryEntity

GeometryEntityLike = Optional[Union['ShapeList', 'GeometryEntity', 'Part', 'BuildPart']]

def _extract_shape_list(value: GeometryEntityLike) -> ShapeList:
    if isinstance(value, ShapeList):
        return value
    if isinstance(value, GeometryEntity):
        return ShapeList([value])
    if isinstance(value, Part):
        return value.faces()
    if isinstance(value, BuildPart):
        return _extract_shape_list(value.part)
    return _extract_shape_list(BuildPart._get_context())

class Falloff(Enum):
    SMOOTH = "SMOOTH"
    SPHERE = "SPHERE"
    LINEAR = "LINEAR"
    CONSTANT = "CONSTANT"
    SHARP = "SHARP"

class ProportionalEdit(ABC):
    """Base class for proportional editing calculations."""
    def __init__(self, falloff: Falloff = Falloff.SMOOTH):
        self.falloff = falloff

    def _get_k(self, t: float) -> float:
        """
        Returns the interpolation factor based on the falloff curve.
        Expects t in range [0, 1], where 0 is no effect and 1 is full effect.
        """
        t = max(0.0, min(1.0, t))
        
        if self.falloff == Falloff.CONSTANT:
            return 1.0
        elif self.falloff == Falloff.LINEAR:
            return t
        elif self.falloff == Falloff.SHARP:
            return t**2
        elif self.falloff == Falloff.SPHERE:
            # Circular arc from 0,0 to 1,1
            return 1.0 - math.sqrt(max(0.0, 1.0 - t**2))
        elif self.falloff == Falloff.SMOOTH:
            # Hermite interpolation (Smoothstep)
            return 3 * t**2 - 2 * t**3
        return 0.0
    
    def precalculate(self, verts: List[bmesh.types.BMVert]):
        pass

    @abstractmethod
    def calculate_weight(self, point: Vector) -> float:
        raise NotImplementedError

class RadialPropEdit(ProportionalEdit):
    """Influence based on distance to a specific point."""
    def __init__(self, origin: Union[Vector, Location] = Vector((0,0,0)), radius: float = 1.0, falloff: Falloff = Falloff.SMOOTH):
        super().__init__(falloff)
        self.origin = origin.position if isinstance(origin, Location) else origin
        self.radius = radius
    
    @override
    def calculate_weight(self, point: Vector) -> float:
        dist = (point - self.origin).length
        if dist > self.radius:
            return 0.0
        if self.radius == 0:
            return 1.0
        
        # For RadialPropEdit, we invert the factor: 
        # distance 0 should be k=1 (full effect)
        t = 1.0 - (dist / self.radius)
        return self._get_k(t)
    
class LinearPropEdit(ProportionalEdit):
    """
    Automatic gradient based on the selection's bounding box along a specific axis.
    Supports negative axis (e.g., -Axis.Z) to flip the gradient direction.
    """
    def __init__(self, axis: Union[Axis, Vector] = Axis.Z, falloff: Falloff = Falloff.SMOOTH):
        super().__init__(falloff=falloff)
        
        # Handle Axis enum or negated Vector
        self.direction_vec: Vector = axis.value if isinstance(axis, Axis) else axis
        # Determine which coordinate to track (0, 1, or 2)
        self.axis_idx = 0
        for i, val in enumerate(self.direction_vec):
            if abs(val) > 0.9:
                self.axis_idx = i
                break
        
        self.is_inverted = any(val < 0 for val in self.direction_vec)
        self._min = 0.0
        self._range = 0.0

    @override
    def precalculate(self, verts: List[bmesh.types.BMVert]):
        """Pre-calculates bounds from the actual bmesh vertices."""
        coords = [v.co[self.axis_idx] for v in verts]
        if not coords: return
        
        v_min, v_max = min(coords), max(coords)
        self._min = v_min
        self._range = v_max - v_min

    @override
    def calculate_weight(self, point: Vector) -> float:
        if self._range == 0:
            return 1.0
        
        # Normalized position 0.0 to 1.0
        t = (point[self.axis_idx] - self._min) / self._range
        
        # If axis is negated (-Axis.Z), the effect is strongest at the minimum coordinate
        factor = (1.0 - t) if self.is_inverted else t
        
        return self._get_k(factor)
    
def get_interpolated_matrix(op_matrix: Matrix, k: float) -> Matrix:
    """Blends between Identity matrix and the operation matrix based on k."""
    if k >= 1.0: return op_matrix
    if k <= 0.0: return Matrix.Identity(4)
    
    loc, rot, sca = op_matrix.decompose()
    
    # Interpolate components
    res_loc = Vector((0, 0, 0)).lerp(loc, k)
    res_rot = Quaternion().slerp(rot, k)
    res_sca = Vector((1, 1, 1)).lerp(sca, k)
    
    # Construct combined matrix
    return (Matrix.Translation(res_loc) @ 
            res_rot.to_matrix().to_4x4() @ 
            Matrix.Diagonal(res_sca.to_4d()))

def _apply_transform(
    bm: bmesh.types.BMesh, 
    verts: list[bmesh.types.BMVert], 
    op: Transform, 
    space: Location = Location(), 
    prop_edit: Optional[ProportionalEdit] = None
):
    """
    Internal helper to apply either a batch transform or a 
    proportional per-vertex transform.
    """
    if not verts:
        return

    if prop_edit is None:
        # Optimized batch transform
        bmesh.ops.transform(
            bm,
            verts=verts,
            matrix=op.matrix,
            space=space.matrix
        )
    else:
        # Per-vertex proportional transform
        space_mat = space.matrix
        space_inv = space_mat.inverted()
        op_mat = op.matrix

        prop_edit.precalculate(verts)
        for v in verts:
            # 1. Get the weight k for this specific vertex
            k = prop_edit.calculate_weight(v.co)
            
            if k <= 1e-6:
                continue
                
            # 2. Get the matrix specifically "diluted" by factor k
            # This ensures rotations/scales look natural
            prop_mat = get_interpolated_matrix(op_mat, k)
            
            # 3. Apply the transformation chain
            # Vertex is moved to the 'space' origin, transformed by prop_mat, then moved back
            v.co = space_mat @ prop_mat @ space_inv @ v.co

def transform(
    entities: GeometryEntityLike = None, 
    op: Transform = Transform(), 
    space: Transform = Location(),
    prop_edit: Optional[ProportionalEdit] = None
):
    """
    Applies a transformation matrix. If prop_edit is provided, applies per-vertex 
    interpolation based on proximity/influence.
    """
    ctx = BuildPart._get_context()
    part = ctx.part
    part._make_op_checkpoint()
 
    bm = part._ensure_bmesh(write=True)
    shape_list = _extract_shape_list(entities)
    verts, _ = part._get_actual_bmesh_verts(shape_list.bm_verts())
    
    _apply_transform(bm.native, [v.native for v in verts], op, space, prop_edit)
    
    part._write_bmesh()

def subdivide(entities: GeometryEntityLike = None, cuts: int = 1, faces: Optional[ShapeList] = None):
    """
    Subdivides the selected faces/edges into smaller parts.
    """
    # Handle recursive call for specific faces
    if faces is not None:
        for face in faces:
            # Call subdivide for each individual face entity
            subdivide(entities=face, cuts=cuts)
        return
    
    ctx = BuildPart._get_context()
    part = ctx.part
    part._make_op_checkpoint()

    # 1. Extract entities and get BMEdges
    bm = part._ensure_bmesh(write=True)
    shape_list = _extract_shape_list(entities)
    edges, _ = part._get_actual_bmesh_edges(shape_list.bm_edges())
    if not edges:
        return
            
    # 2. Apply the subdivision operator
    bmesh.ops.subdivide_edges(
        bm.native,
        edges=[e.native for e in edges],
        cuts=cuts,
        use_grid_fill=True
    )
    
    # 3. Write changes back
    part._write_bmesh()

def extrude(entities: GeometryEntityLike = None, op: Transform = Transform(), prop_edit: Optional[ProportionalEdit] = None):
    """
    Extrudes the provided entities (Face, Wire, Edge, or Vertex) and applies 
    the given transformation to the new geometry.
    """
    ctx = BuildPart._get_context()
    part = ctx.part
    part._make_op_checkpoint()
    bm = part._ensure_bmesh(write=True)
    shape_list = _extract_shape_list(entities)

    if not shape_list:
        return

    # 1. Determine entity type based on the first element
    # Assuming types are named Face, Wire, Edge, Vertex
    entity_type = shape_list.type

    new_geometry = []

    # 2. Extract specific BM entities and perform extrusion
    if entity_type == "Face":
        faces, _ = part._get_actual_bmesh_faces(shape_list.bm_faces())
        res = bmesh.ops.extrude_face_region(
            bm.native, 
            geom=[f.native for f in faces]
        )
        new_geometry = res["geom"]

    elif entity_type in ("Edge", "Wire"):
        # Wire is treated as Edges as requested
        edges, _ = part._get_actual_bmesh_edges(shape_list.bm_edges())
        res = bmesh.ops.extrude_edge_only(
            bm.native, 
            edges=[e.native for e in edges]
        )
        new_geometry = res["geom"]

    elif entity_type == "Vertex":
        verts, _ = part._get_actual_bmesh_verts(shape_list.bm_verts())
        res = bmesh.ops.extrude_vert_indiv(
            bm.native, 
            verts=[v.native for v in verts]
        )
        new_geometry = res["verts"]
    
    else:
        return

    # 3. Filter new vertices from the resulting geometry
    new_verts = [v for v in new_geometry if isinstance(v, bmesh.types.BMVert)]

    # 4. Apply transformation to the new vertices
    _apply_transform(bm.native, new_verts, op, prop_edit=prop_edit)

    # 5. Write changes back
    part._write_bmesh()

def delete(entities: GeometryEntityLike = None):
    """
    Deletes the provided entities (Face, Edge, Wire, or Vertex) from the Part.
    """
    ctx = BuildPart._get_context()
    part = ctx.part
    part._make_op_checkpoint()
    bm = part._ensure_bmesh(write=True)
    shape_list = _extract_shape_list(entities)

    if not shape_list:
        return

    entity_type = shape_list.type
    geom_to_delete = []
    del_context = 'VERTS' # Default context

    # 1. Collect native entities and set the appropriate deletion context
    if entity_type == "Face":
        faces, _ = part._get_actual_bmesh_faces(shape_list.bm_faces())
        geom_to_delete = [f.native for f in faces]
        # 'FACES' deletes faces but leaves edges/verts if they are shared
        del_context = 'FACES'

    elif entity_type in ("Edge", "Wire"):
        edges, _ = part._get_actual_bmesh_edges(shape_list.bm_edges())
        geom_to_delete = [e.native for e in edges]
        # 'EDGES' deletes edges and any faces that use them
        del_context = 'EDGES'

    elif entity_type == "Vertex":
        verts, _ = part._get_actual_bmesh_verts(shape_list.bm_verts())
        geom_to_delete = [v.native for v in verts]
        # 'VERTS' deletes vertices and all connected edges/faces
        del_context = 'VERTS'

    # 2. Execute the deletion operator
    if geom_to_delete:
        bmesh.ops.delete(
            bm.native,
            geom=geom_to_delete,
            context=del_context
        )

    # 3. Write changes back
    part._write_bmesh()

def _apply_modifiers(obj: bpy.types.Object):
    """
    Evaluates the object's dependency graph, applies all modifiers to the mesh,
    and cleans up the modifier stack.
    """
    # 1. Temporarily link to scene (required for Depsgraph to evaluate the object correctly)
    bpy.context.collection.objects.link(obj)
    
    # 2. Update dependency graph so Blender "sees" the modifiers
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update() 
    
    # 3. Get the mesh with modifiers applied
    eval_obj = obj.evaluated_get(dg)
    new_mesh = bpy.data.meshes.new_from_object(eval_obj, depsgraph=dg)
    
    # 4. Swap the mesh and unlink the object back to memory
    old_mesh = obj.data
    obj.data = new_mesh
    bpy.context.collection.objects.unlink(obj)
    
    # Cleanup old mesh data
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)
    
    # Clear the modifier stack as they are now baked into the new mesh
    obj.modifiers.clear()

def bevel(entities: GeometryEntityLike = None, radius: float = 0.1, segments: int = 10):
    """
    Applies a Bevel operation to the provided logical edges.
    Uses Bevel Weights to target specific edges within the polygonal model.
    """
    ctx = BuildPart._get_context()
    part = ctx.part
    part._make_op_checkpoint()
    
    # Layer for storing Bevel Weights
    bm = part._ensure_bmesh(write=True)
    bw_layer = bm.native.edges.layers.float.get("bevel_weight_edge")
    if bw_layer is None:
        raise RuntimeError("Edge weight layer (bevel_weight_edge) not found")
    
    # Ensure edges are valid
    shape_list = _extract_shape_list(entities)
    edges, all_edges = part._get_actual_bmesh_edges(shape_list.bm_edges())
        
    # 1. Reset weights on all physical edges of the mesh
    for e in all_edges:
        e.native[bw_layer] = 0.0
        
    # 2. Set weights for the provided edges
    for e in edges:
        e.native[bw_layer] = 1.0
        
    # 3. Write BMesh changes to the Blender object before applying the modifier
    part._write_bmesh(flush=True)
    
    # Setup and apply the Bevel modifier
    # Use a unique name to avoid conflicts with other modifiers
    mod: bpy.types.BevelModifier = part.obj.modifiers.new(name="BP_Bevel_Op" , type='BEVEL')
    mod.limit_method = 'WEIGHT'
    mod.offset_type = 'PERCENT'
    mod.width = radius * 100
    mod.segments = segments

    _apply_modifiers(part.obj)
    part._fix_topology()

def mirror(axis: Axis = Axis.X):
    """
    Applies a Mirror modifier to the current Part.
    """
    ctx = BuildPart._get_context()
    part = ctx.part
    part._make_op_checkpoint()

    # 1. Create the Mirror Modifier
    # We use a unique name to identify our library operations
    mod: bpy.types.MirrorModifier = part.obj.modifiers.new(name="BP_Mirror_Op", type='MIRROR')
    
    # Reset all axes first
    mod.use_axis = (False, False, False)
    mod.use_bisect_axis = (False, False, False)
    mod.use_bisect_flip_axis = (False, False, False)

    # 2. Configure the chosen axis
    axis_idx = 0
    if axis == Axis.Y:
        axis_idx = 1
    elif axis == Axis.Z:
        axis_idx = 2
    
    # Set the mirroring axis
    axes = [False, False, False]
    axes[axis_idx] = True
    mod.use_axis = axes

    # 4. Bake the modifier into the mesh data
    _apply_modifiers(part.obj)
    
    # Synchronize internal state
    ctx.part._flush_bmesh()

class DeformType(Enum):
    """Available types for the Simple Deform operation."""
    TWIST = "TWIST"
    BEND = "BEND"
    TAPER = "TAPER"
    STRETCH = "STRETCH"

def simple_deform(
    type: DeformType = DeformType.BEND,
    angle: float = 0.0, 
    origin: Location = Location(), 
    axis: Axis = Axis.X,
    limits: tuple[float, float] = (0.0, 1.0)
):
    """
    Applies a Simple Deform modifier (Twist, Bend, Taper, or Stretch) to the Part.
    """
    ctx = BuildPart._get_context()
    part = ctx.part
    part._make_op_checkpoint()

    # 1. Store original transformation and reset it to identity
    # This is crucial so the modifier applies in local space correctly
    old_transform = part.transform
    part.transform = Transform()
    
    # 2. Create a temporary empty object to act as the deformation origin
    temp_empty = bpy.data.objects.new("temp_origin", None)
    bpy.context.collection.objects.link(temp_empty)
    temp_empty.matrix_world = origin.matrix
    
    # 3. Setup the Simple Deform modifier
    mod: bpy.types.SimpleDeformModifier = part.obj.modifiers.new(name="BP_Deform_Op", type='SIMPLE_DEFORM')
    mod.deform_method = type.value
    mod.angle = math.radians(angle)
    mod.origin = temp_empty
    mod.deform_axis = axis.name
    mod.limits = limits

    # 4. Bake the modifier into the mesh
    _apply_modifiers(part.obj)

    # 5. Cleanup: remove the temporary empty and restore original transform
    bpy.data.objects.remove(temp_empty, do_unlink=True)
    part.transform = old_transform
    
    # Sync internal state
    ctx.part._flush_bmesh()

def _apply_simple_deform_op(
    deform_type: DeformType,
    angle: float,
    axis: Axis,
    segments: Optional[int],
    limits: tuple[float, float]
):
    """
    Internal helper to handle axis mapping, subdivision, and delegation 
    to simple_deform for specific deformation operations.
    """
    # 1. Pre-subdivide if requested
    if segments is not None:
        subdivide(cuts=segments)

    # 2. Axis mapping: X -> X, Y -> Z, Z -> Y
    axis_mapping = {
        Axis.X: Axis.X,
        Axis.Y: Axis.Z,
        Axis.Z: Axis.Y
    }
    target_axis = axis_mapping.get(axis, Axis.Z)

    # 3. Delegate to the main simple_deform function
    simple_deform(
        type=deform_type,
        angle=angle,
        origin=Rot(X=90),
        axis=target_axis,
        limits=limits
    )

def bend(
    angle: float, 
    axis: Axis = Axis.X, 
    segments: Optional[int] = None, 
    limits: tuple[float, float] = (0.0, 1.0)
):
    """
    Bends the Part along a specified axis. 
    Can optionally subdivide the mesh before bending.
    """
    _apply_simple_deform_op(DeformType.BEND, angle, axis, segments, limits)

def twist(
    angle: float, 
    axis: Axis = Axis.X, 
    segments: Optional[int] = None, 
    limits: tuple[float, float] = (0.0, 1.0)
):
    """
    Twists the Part along a specified axis.
    """
    _apply_simple_deform_op(DeformType.TWIST, angle, axis, segments, limits)

def _apply_boolean(target: bpy.types.Object, tool: bpy.types.Object, mode: Mode):
    """
    Performs a Boolean operation (Union, Difference, Intersect) 
    and updates the target mesh with the result.
    """
    if mode == Mode.ADD and target == tool:
        return

    # 1. Create a modifier on the target object
    mod: bpy.types.BooleanModifier = target.modifiers.new(name="TempBool", type='BOOLEAN')
    mod.object = tool
    mod.solver = 'EXACT'
    if mode == Mode.ADD:
        mod.operation = 'UNION'
    elif mode == Mode.SUBTRACT:
        mod.operation = 'DIFFERENCE'
    elif mode == Mode.INTERSECT:
        mod.operation = 'INTERSECT'

    _apply_modifiers(target)

def _join(target: bpy.types.Object, tool: bpy.types.Object):
    """
    Merges the tool object into the target using the Blender 'join' operator.
    This does not perform boolean cleanup, simply combines geometry data.
    """
    # Save current mode to return to it later
    original_mode = bpy.context.object.mode if bpy.context.object else 'OBJECT'
    
    if original_mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # Deselect everything
    bpy.ops.object.select_all(action='DESELECT')

    bpy.context.collection.objects.link(target)
    bpy.context.collection.objects.link(tool)

    # Select both target and tool
    target.select_set(True)
    tool.select_set(True)

    # Set target as active (everything will be merged into it)
    bpy.context.view_layer.objects.active = target

    # Perform the join
    bpy.ops.object.join()

    bpy.context.collection.objects.unlink(target)

    # Restore original mode
    if original_mode != 'OBJECT':
        bpy.ops.object.mode_set(mode=original_mode)

def add(to_add: Union['Part', BuildPart, 'BaseCurve', 'BuildCurve'], offset: Location = Location(), mode: Mode = Mode.ADD, _make_copy: bool = True):
    """
    Adds geometry to the current BuildPart context.
    Supports various modes including Boolean operations and simple Joining.
    """
    if mode == Mode.PRIVATE:
        return
    ctx = BuildPart._get_context()
    part = ctx.part
    part._make_op_checkpoint()
    
    # Extract Part
    part_to_add = to_add.part if isinstance(to_add, (BuildPart, BuildCurve, BaseCurve)) else to_add
    if not part_to_add.is_valid:
        raise RuntimeError("Cannot add an invalid object")
    if part.obj == part_to_add.obj:
        raise RuntimeError("Object cannot be added to itself")
    
    active_locs = Locations._get_active()
    need_copy = (_make_copy or len(active_locs) > 1) and not isinstance(to_add, (BuildCurve, BaseCurve))
    for loc in active_locs:
        # Clone part if making a copy or if multiple locations are active
        cloned_part = part_to_add.copy() if need_copy else part_to_add
        cloned_part.loc *= loc * offset

        if mode == Mode.JOIN:
            _join(part.obj, cloned_part.obj)
            cloned_part.remove(physical=False)
        else:
            _apply_boolean(part.obj, cloned_part.obj, mode)
            cloned_part.remove()
        # Synchronize internal BMesh state
        part._flush_bmesh()