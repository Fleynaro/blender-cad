from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import math
import uuid
import bpy
import bmesh
from typing import Callable, Iterable, List, Optional, Union
from typing_extensions import override
from mathutils import Matrix, Quaternion, Vector

from .common import Axis, GeometryEntityLike, PartLike, extract_part, extract_shape_list, extract_vector, tag_to_list
from .object import Object
from .build_part import BuildPart, Mode
from .part import BoxSetPart, Part
from .curve import BaseCurve, BuildCurve
from .material import mat
from .location import Location, Locations, Rot, Transform, TransformExpr
from .shape_list import ShapeList

@dataclass
class EvaluationContext:
    """Context container for proportional editing calculations."""
    verts: List[bmesh.types.BMVert]
    token: str

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
    
    @abstractmethod
    def calculate_weight(self, point: Vector, context: EvaluationContext) -> float:
        """Calculate weight for a given point using optional evaluation context."""
        raise NotImplementedError
    
    def _combine(
        self, 
        other: Union["ProportionalEdit", float, int], 
        op_func: Callable[[float, float], float]
    ) -> "LambdaPropEdit":
        """Helper to build LambdaPropEdit with dependency tracking."""
        is_prop = isinstance(other, ProportionalEdit)
        return LambdaPropEdit(
            func=lambda p, ctx: op_func(
                self.calculate_weight(p, ctx), 
                other.calculate_weight(p, ctx) if is_prop else other
            ),
        )

    def __add__(self, other: Union["ProportionalEdit", float, int]):
        return self._combine(other, lambda a, b: a + b)

    def __radd__(self, other: Union[float, int]):
        return self._combine(other, lambda a, b: b + a)
    
    def __sub__(self, other: Union["ProportionalEdit", float, int]):
        return self._combine(other, lambda a, b: a - b)

    def __rsub__(self, other: Union[float, int]):
        return self._combine(other, lambda a, b: b - a)

    def __mul__(self, other: Union["ProportionalEdit", float, int]):
        return self._combine(other, lambda a, b: a * b)

    def __rmul__(self, other: Union[float, int]):
        return self._combine(other, lambda a, b: b * a)
    
    def __truediv__(self, other: Union["ProportionalEdit", float, int]):
        return self._combine(other, lambda a, b: a / b if b != 0 else 0.0)

    def __rtruediv__(self, other: Union[float, int]):
        return self._combine(other, lambda a, b: b / a if a != 0 else 0.0)
    
    def __pow__(self, power: Union["ProportionalEdit", float, int]):
        return self._combine(power, lambda a, b: a ** b if a >= 0 else 0.0)

    def min(self, other: Union["ProportionalEdit", float, int]):
        return self._combine(other, min)

    def max(self, other: Union["ProportionalEdit", float, int]):
        return self._combine(other, max)
    
    def mix(self, other: "ProportionalEdit", factor: Union["ProportionalEdit", float]):
        """Linear interpolation (LERP) between two types of editing by factor."""
        is_factor_prop = isinstance(factor, ProportionalEdit)
        return LambdaPropEdit(
            func=lambda p, ctx: (
                lambda a, b, f: a + (b - a) * f
            )(
                self.calculate_weight(p, ctx), 
                other.calculate_weight(p, ctx), 
                factor.calculate_weight(p, ctx) if is_factor_prop else factor
            ),
        )
    
    def clamp(self, min_val: float = 0.0, max_val: float = 1.0):
        """Clamps the value between min_val and max_val."""
        return LambdaPropEdit(
            func=lambda p, ctx: max(min_val, min(max_val, self.calculate_weight(p, ctx))),
        )

    def invert(self):
        """Inverts the effect of the proportional edit."""
        return LambdaPropEdit(
            func=lambda p, ctx: 1.0 - self.calculate_weight(p, ctx),
        )
    
class LambdaPropEdit(ProportionalEdit):
    """Custom editing using a user-defined function or lambda with dependency tracking."""
    def __init__(
        self, 
        func: Callable[[Vector], float], 
    ):
        self.func = func

    @override
    def calculate_weight(self, point: Vector, context: EvaluationContext) -> float:
        return self.func(point, context)

class RadialPropEdit(ProportionalEdit):
    """Influence based on distance to a specific point."""
    def __init__(self, origin: Union[Vector, Location] = Vector((0,0,0)), radius: float = 1.0, falloff: Falloff = Falloff.SMOOTH):
        super().__init__(falloff)
        self.origin = origin.position if isinstance(origin, Location) else origin
        self.radius = radius
    
    @override
    def calculate_weight(self, point: Vector, context: EvaluationContext) -> float:
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
        self.direction_vec = extract_vector(axis)
        # Determine which coordinate to track (0, 1, or 2)
        self.axis, self.is_inverted = Axis.from_vector(axis)
        self.axis_idx = self.axis.index
        
        self._min = 0.0
        self._range = 0.0
        self._last_token: Optional[str] = None

    def _lazy_precalculate(self, context: EvaluationContext):
        """Internal helper to pre-calculate bounds only when token changes."""
        if self._last_token == context.token:
            return

        coords = [v.co[self.axis_idx] for v in context.verts]
        if not coords: 
            return
        
        v_min, v_max = min(coords), max(coords)
        self._min = v_min
        self._range = v_max - v_min
        self._last_token = context.token

    @override
    def calculate_weight(self, point: Vector, context: EvaluationContext) -> float:
        self._lazy_precalculate(context)

        if self._range == 0:
            return 1.0
        
        # Normalized position 0.0 to 1.0
        t = (point[self.axis_idx] - self._min) / self._range
        
        # If axis is negated (-Axis.Z), the effect is strongest at the minimum coordinate
        factor = (1.0 - t) if self.is_inverted else t
        
        return self._get_k(factor)

def make_box_sides_edit(
    neg_x: float = 1.0, 
    pos_x: float = 1.0, 
    neg_y: float = 1.0, 
    pos_y: float = 1.0,
    multiply = False
) -> LambdaPropEdit:
    """
    Creates a compound proportional editing object tailored for box shapes.
    Allows independent weighting [0.0 to 1.0] for all four lateral sides.
    
    :param neg_x: Weight multiplier for the Left (-X) face.
    :param pos_x: Weight multiplier for the Right (+X) face.
    :param neg_y: Weight multiplier for the Front (-Y) face.
    :param pos_y: Weight multiplier for the Back (+Y) face.
    """
    x_grad = LinearPropEdit(Axis.X)
    y_grad = LinearPropEdit(Axis.Y)

    mask_left  = x_grad          + x_grad.invert() * neg_x
    mask_right = x_grad.invert() + x_grad          * pos_x
    mask_front = y_grad          + y_grad.invert() * neg_y
    mask_back  = y_grad.invert() + y_grad          * pos_y

    if multiply:
        return mask_left * mask_right * mask_front * mask_back
    return mask_left.min(mask_right).min(mask_front).min(mask_back)

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
    op: TransformExpr, 
    space: Location = Location(), 
    prop_edit: Optional[ProportionalEdit] = None
):
    """
    Internal helper to apply either a batch transform or a 
    proportional per-vertex transform.
    """
    if not verts:
        return
    
    if not isinstance(op, Transform):
        with BuildPart(mode=Mode.PRIVATE) as bp:
            bp.part._add_vertices([v.co for v in verts], write=True)
            op = op.resolve(bp.part)

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

        eval_context = EvaluationContext(verts=verts, token=str(uuid.uuid4()))
        for v in verts:
            # 1. Get the weight k for this specific vertex
            k = prop_edit.calculate_weight(v.co, eval_context)
            
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
    op: TransformExpr = Transform(), 
    space: Transform = Location(),
    prop_edit: Optional[ProportionalEdit] = None
):
    """
    Applies a transformation matrix. If prop_edit is provided, applies per-vertex 
    interpolation based on proximity/influence.
    """
    ctx = BuildPart._get_context()
    part = ctx.part
    if isinstance(part, BoxSetPart) and entities is None and prop_edit is None:
        part.apply_transform(op, space)
        return

    part._make_op_checkpoint()
 
    shape_list = extract_shape_list(entities)
    bm = part._ensure_bmesh(write=True)
    part._inject_joint_markers()
    verts, all_verts = part._get_actual_bmesh_verts(shape_list.bm_verts())
    if entities is None or isinstance(entities, (Part, BuildPart)):
        # Include injected joint markers (because they are in all_verts)
        verts = all_verts
    
    _apply_transform(bm.native, [v.native for v in verts], op, space, prop_edit)
    
    part._sync_joint_markers()
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
    shape_list = extract_shape_list(entities)
    bm = part._ensure_bmesh(write=True)
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

def dissolve(entities: GeometryEntityLike = None, angle_limit: float = 5.0):
    """
    Dissolves planar faces and edges based on an angle threshold.
    Angle limit is expected in degrees.
    """
    ctx = BuildPart._get_context()
    part = ctx.part
    part._make_op_checkpoint()

    # 1. Access BMesh and extract selected entities
    shape_list = extract_shape_list(entities)
    bm = part._ensure_bmesh(write=True)
    
    # 2. Get actual BMesh faces from the shape list
    faces, _ = part._get_actual_bmesh_faces(shape_list.bm_faces())
    if not faces:
        return

    # 3. Collect all related geometry for the operator context
    native_faces = [f.native for f in faces]
    # We collect unique edges and vertices belonging to these faces
    native_edges = list({e for f in native_faces for e in f.edges})
    native_verts = list({v for f in native_faces for v in f.verts})

    # 4. Apply the dissolve_limit operator
    # This dissolves vertices/edges that are between faces with an angle < angle_limit
    bmesh.ops.dissolve_limit(
        bm.native,
        angle_limit=math.radians(angle_limit),
        use_dissolve_boundaries=False,
        verts=native_verts,
        edges=native_edges,
    )

    # 5. Write changes back to the part
    part._write_bmesh()

def extrude(
    entities: GeometryEntityLike = None,
    op: TransformExpr = Transform(),
    prop_edit: Optional[ProportionalEdit] = None,
    delete_source: bool = False,
    recalc_normals: bool = False,
    tag: Optional[str | Iterable[str]] = None
):
    """
    Extrudes the provided entities (Face, Wire, Edge, or Vertex) and applies 
    the given transformation to the new geometry.
    """
    ctx = BuildPart._get_context()
    part = ctx.part
    part._make_op_checkpoint()
    shape_list = extract_shape_list(entities)
    bm = part._ensure_bmesh(write=True)

    if not shape_list:
        return

    # 1. Determine entity type based on the first element
    # Assuming types are named Face, Wire, Edge, Vertex
    entity_type = shape_list.type

    new_geometry: List[Union[bmesh.types.BMFace, bmesh.types.BMEdge, bmesh.types.BMVert]] = []
    source_natives = []

    # 2. Extract specific BM entities and perform extrusion
    if entity_type == "Face":
        faces, _ = part._get_actual_bmesh_faces(shape_list.bm_faces())
        source_natives = [f.native for f in faces]
        res = bmesh.ops.extrude_face_region(
            bm.native, 
            geom=[f.native for f in faces]
        )
        new_geometry = res["geom"]

        # Add NEW side faces that are connected to the extruded faces
        all_linked_faces = set()
        for edge in new_geometry:
            if isinstance(edge, bmesh.types.BMEdge):
                all_linked_faces.update(edge.link_faces)
        new_side_faces = [f for f in all_linked_faces if f not in source_natives and f not in new_geometry]
        new_geometry.extend(new_side_faces)

        # Add NEW side edges that are connected to the extruded faces
        all_linked_edges = set()
        for vert in new_geometry:
            if isinstance(vert, bmesh.types.BMVert):
                all_linked_edges.update(vert.link_edges)
        source_edges = set()
        for face in source_natives:
            source_edges.update(face.edges)
        new_side_edges = [
            e for e in all_linked_edges 
            if e not in source_edges and e not in new_geometry
        ]
        new_geometry.extend(new_side_edges)

    elif entity_type in ("Edge", "Wire"):
        # Wire is treated as Edges as requested
        edges, _ = part._get_actual_bmesh_edges(shape_list.bm_edges())
        source_natives = [e.native for e in edges]
        res = bmesh.ops.extrude_edge_only(
            bm.native, 
            edges=[e.native for e in edges]
        )
        new_geometry = res["geom"]

        # Add NEW side edges that are connected to the extruded faces
        all_linked_edges = set()
        for vert in new_geometry:
            if isinstance(vert, bmesh.types.BMVert):
                all_linked_edges.update(vert.link_edges)
        new_side_edges = [
            e for e in all_linked_edges 
            if e not in source_natives and e not in new_geometry
        ]
        new_geometry.extend(new_side_edges)

    elif entity_type == "Vertex":
        verts, _ = part._get_actual_bmesh_verts(shape_list.bm_verts())
        res = bmesh.ops.extrude_vert_indiv(
            bm.native, 
            verts=[v.native for v in verts]
        )
        new_geometry = res["verts"] + res["edges"]
    
    else:
        return

    # 3. Filter new vertices from the resulting geometry
    new_verts = [v for v in new_geometry if isinstance(v, bmesh.types.BMVert)]

    # 4. Apply transformation to the new vertices
    _apply_transform(bm.native, new_verts, op, prop_edit=prop_edit)

    # 5. Delete source geometry if requested
    if delete_source and source_natives:
        bmesh.ops.delete(bm.native, geom=source_natives, context=("FACES" if entity_type == "Face" else "EDGES"))

    # 6. Recalculate normals
    if recalc_normals:
        bmesh.ops.recalc_face_normals(bm.native, faces=bm.native.faces)

    # 7. Write changes back
    part._write_bmesh()

    # 8. Add tags
    tags = [Object.TAG_OP_EXTRUDE]
    tags.extend(tag_to_list(tag))
    part._add_tags("POINT", [v.index for v in new_geometry if isinstance(v, bmesh.types.BMVert)], tags)
    part._add_tags("EDGE", [e.index for e in new_geometry if isinstance(e, bmesh.types.BMEdge)], tags)
    part._add_tags("FACE", [f.index for f in new_geometry if isinstance(f, bmesh.types.BMFace)], tags)


def _move_verts_along_normals(
    verts: List[bmesh.types.BMVert],
    distance: float,
):
    if abs(distance) <= 1e-9:
        return
    for v in verts:
        n = sum((f.normal for f in v.link_faces), Vector())
        if n.length > 0:
            v.co += n.normalized() * distance

def _copy_faces_to_bmesh(
    faces: List[bmesh.types.BMFace],
    dst_bm: bmesh.types.BMesh,
) -> List[bmesh.types.BMFace]:
    vert_map = {}
    for face in faces:
        for v in face.verts:
            vert_map.setdefault(
                v,
                dst_bm.verts.new(v.co.copy())
            )

    result = []
    for face in faces:
        try:
            result.append(
                dst_bm.faces.new(
                    [vert_map[v] for v in face.verts]
                )
            )
        except ValueError:
            pass

    dst_bm.verts.ensure_lookup_table()
    dst_bm.faces.ensure_lookup_table()
    return result

def solidify_faces(
    faces: GeometryEntityLike = None,
    height: float = 0.1,
    offset: float = 0.0,
):
    """
    Creates a closed clipping mesh from selected faces.
    """
    ctx = BuildPart._get_context()
    part = ctx.part

    shape_list = extract_shape_list(faces)
    bm = part._ensure_bmesh()

    bm_faces, _ = part._get_actual_bmesh_faces(shape_list.bm_faces())
    if not bm_faces:
        raise ValueError("No faces selected")
    src_faces = [f.native for f in bm_faces]

    # Duplicate inside source bmesh
    dup_geom = bmesh.ops.duplicate(
        bm.native,
        geom=src_faces
    )["geom"]

    new_faces = [
        g for g in dup_geom
        if isinstance(g, bmesh.types.BMFace)
    ]

    # Copy duplicated region into isolated clip bmesh
    clip_bm = bmesh.new()
    clip_faces = _copy_faces_to_bmesh(new_faces, clip_bm)
    clip_bm.normal_update()

    # Cleanup temporary duplicate
    bmesh.ops.delete(
        bm.native,
        geom=new_faces,
        context='FACES'
    )

    # Optional shell offset
    _move_verts_along_normals(
        list({
            v
            for f in clip_faces
            for v in f.verts
        }),
        offset
    )

    # Extrude whole region
    extruded_geom = bmesh.ops.extrude_face_region(
        clip_bm,
        geom=clip_faces
    )["geom"]
    original_verts = {
        v
        for f in clip_faces
        for v in f.verts
    }
    move_verts = [
        v for v in extruded_geom
        if isinstance(v, bmesh.types.BMVert)
        and v not in original_verts
    ]

    # Move new cap
    _move_verts_along_normals(
        move_verts,
        height
    )

    # Fix normals
    bmesh.ops.recalc_face_normals(
        clip_bm,
        faces=list(clip_bm.faces)
    )

    # Create object
    mesh = bpy.data.meshes.new("FaceClipper")
    clip_bm.to_mesh(mesh)
    clip_bm.free()
    mesh.update()

    clip_obj = bpy.data.objects.new("FaceClipper", mesh)
    clip_obj.matrix_world = part.obj.matrix_world.copy()
    return Part(clip_obj)

def delete(entities: GeometryEntityLike = None):
    """
    Deletes the provided entities (Face, Edge, Wire, or Vertex) from the Part.
    """
    ctx = BuildPart._get_context()
    part = ctx.part
    part._make_op_checkpoint()
    shape_list = extract_shape_list(entities)
    bm = part._ensure_bmesh(write=True)

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

    shape_list = extract_shape_list(entities)
    
    # Layer for storing Bevel Weights
    bm = part._ensure_bmesh(write=True)
    bw_layer = bm.native.edges.layers.float.get("bevel_weight_edge")
    if bw_layer is None:
        raise RuntimeError("Edge weight layer (bevel_weight_edge) not found")
    
    # Ensure edges are valid
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
    part._inject_joint_markers(write=True)

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
    part._sync_joint_markers(write=True)

def _apply_simple_deform_op(
    deform_type: DeformType,
    angle: float,
    origin: Location, 
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
        origin=Rot(X=90) * origin,
        axis=target_axis,
        limits=limits
    )

def bend(
    angle: float, 
    axis: Axis = Axis.X, 
    segments: Optional[int] = None, 
    origin: Location = Location(),
    limits: tuple[float, float] = (0.0, 1.0)
):
    """
    Bends the Part along a specified axis. 
    Can optionally subdivide the mesh before bending.
    """
    _apply_simple_deform_op(DeformType.BEND, angle, origin, axis, segments, limits)

def twist(
    angle: float, 
    axis: Axis = Axis.X, 
    segments: Optional[int] = None,
    origin: Location = Location(),
    limits: tuple[float, float] = (0.0, 1.0)
):
    """
    Twists the Part along a specified axis.
    """
    _apply_simple_deform_op(DeformType.TWIST, angle, origin, axis, segments, limits)

class WrapMode(Enum):
    NEAREST_SURFACEPOINT = "NEAREST_SURFACEPOINT"
    NEAREST_VERTEX = "NEAREST_VERTEX"

def wrap(
    target: PartLike,
    *,
    loc: Optional[Location] = None,
    mode: WrapMode = WrapMode.NEAREST_SURFACEPOINT,
    offset: float = 0.0,
    segments: int | None = None,
):
    ctx = BuildPart._get_context()
    part = ctx.part
    part._make_op_checkpoint()

    if loc:
        part.loc = loc

    if segments:
        subdivide(cuts=segments)

    mod: bpy.types.ShrinkwrapModifier = (
        part.obj.modifiers.new(
            name="BP_Wrap_Op",
            type='SHRINKWRAP'
        )
    )
    mod.target = extract_part(target).obj
    mod.wrap_method = mode.value
    mod.offset = offset

    _apply_modifiers(part.obj)
    part._flush_bmesh()

def _apply_boolean(target: bpy.types.Object, tool: bpy.types.Object, mode: Mode):
    """
    Performs a Boolean operation (Union, Difference, Intersect) 
    and updates the target mesh with the result.
    """
    # 1. Create a modifier on the target object
    mod: bpy.types.BooleanModifier = target.modifiers.new(name="TempBool", type='BOOLEAN')
    mod.object = tool
    mod.solver = 'FLOAT' if 'FAST' in mode.value else 'EXACT'
    mod.material_mode = 'TRANSFER'
    if mode == Mode.ADD or mode == Mode.ADD_FAST:
        mod.operation = 'UNION'
    elif mode == Mode.SUBTRACT or mode == Mode.SUBTRACT_FAST:
        mod.operation = 'DIFFERENCE'
    elif mode == Mode.INTERSECT or mode == Mode.INTERSECT_FAST:
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

def add(
    to_add: PartLike,
    offset: Location = Location(),
    transform: Optional[TransformExpr] = None,
    mode: Mode = Mode.ADD,
    mat: Optional[mat.Layer] = None,
    tag: Optional[str | Iterable[str]] = None,
    _make_copy: bool = True
):
    """
    Adds geometry to the current BuildPart context.
    Supports various modes including Boolean operations and simple Joining.
    """
    if mode == Mode.PRIVATE:
        return
    ctx = BuildPart._get_context()
    part = ctx.part
    if not isinstance(part, BoxSetPart):
        part._make_op_checkpoint()
    
    # Extract Part
    part_to_add = extract_part(to_add)
    if not part_to_add.is_physically_valid:
        raise RuntimeError("Cannot add an invalid object")
    
    active_locs = Locations._get_active()
    need_copy = (_make_copy or len(active_locs) > 1) and not isinstance(to_add, (BuildCurve, BaseCurve))
    for loc in active_locs:
        # Clone part if making a copy or if multiple locations are active
        cloned_part = part_to_add.copy() if need_copy else part_to_add
        cloned_part.loc = loc * cloned_part.loc * offset
        if transform:
            cloned_part.transform *= transform

        if mat:
            cloned_part.mat = mat

        cloned_part.add_tags(tag_to_list(tag), domain="FACE")

        is_add = mode in (Mode.ADD, Mode.ADD_FAST, Mode.JOIN)
        if is_add:
            cloned_part._transfer_registered_joints(part, propagate_only=True)

        if isinstance(part, BoxSetPart):
            if is_add:
                part.add_part(cloned_part)
        else:
            if not cloned_part.empty:
                if mode == Mode.JOIN or (mode == Mode.ADD and not cloned_part.has_polygons):
                    _join(part.obj, cloned_part.obj)
                else:
                    _apply_boolean(part.obj, cloned_part.obj, mode)
        cloned_part.remove(physical = (mode != Mode.JOIN))

    # Synchronize internal BMesh state
    if not isinstance(part, BoxSetPart):
        part._flush_bmesh()
