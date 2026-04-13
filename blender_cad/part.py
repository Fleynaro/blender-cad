from dataclasses import dataclass
import bpy
import bmesh
from typing import TYPE_CHECKING, Any, Generic, List, NamedTuple, Optional, TypeVar
from typing_extensions import override
import hashlib
import struct
from mathutils import Vector, Matrix

from .common import VectorLike, extract_vector
from .location import Location, Pos, Scale, Transform, TransformExpr
from .object import AttributeDomainItems, Object
from .material import bpy_material_hash, mat
from .bmesh_wrapper import BMFaceWrapper, BMEdgeWrapper, BMVertWrapper, BMeshWrapper
from .shape_list import ShapeList

if TYPE_CHECKING:
    from .geometry import Edge, Face, Topology, Vertex, Wire, TopologyConfig, GeometryCheckpoint

class Part(Object):
    """An object representing a part. Manages its own mesh and Blender object."""

    def __init__(self, obj: Optional[bpy.types.Object] = None, topology: Optional['TopologyConfig'] = None):
        super().__init__(obj)
        from .geometry import TopologyConfig
        self._bm_wrapper: Optional['BMeshWrapper'] = None
        self.topology_config = topology or TopologyConfig()
        self._topology: Optional['Topology'] = None
        self._last_checkpoint: Optional['GeometryCheckpoint'] = None
        self._last_op_checkpoint: Optional['GeometryCheckpoint'] = None

    @override
    def _create_empty_object(self):
        mesh = bpy.data.meshes.new("PartMesh")
        obj = bpy.data.objects.new("Part", mesh)
        return obj

    @override
    def copy(self) -> 'Part':
        """Creates a copy of the Part and its underlying Blender object."""
        if self.obj is None:
            raise RuntimeError("Object is removed")
        new_obj = self.obj.copy()
        new_obj.data = self.obj.data.copy()
        part = Part(new_obj)
        part._bm_wrapper = self._bm_wrapper
        part._topology = self._topology
        self._after_copy(part)
        return part

    @override
    def remove(self, physical=True):
        """Safely removes the object and its data from the Blender scene."""
        if physical and self.is_physically_valid:
            mesh_data = self.obj.data
            if mesh_data and mesh_data.users == 0:
                bpy.data.meshes.remove(mesh_data)
        super().remove(physical)
    
    def hash(self, precision=4, use_materials=False):
        """Generates a SHA256 hash based on geometry, materials, and world matrix."""
        if self.obj is None:
            raise RuntimeError("Object is removed")
        hash_m = hashlib.sha256()
        mesh = self.obj.data

        # 1. Get coordinates and ROUND them
        # Rounding mitigates float precision differences (e.g., 0.0000001)
        verts_coords = [0.0] * (len(mesh.vertices) * 3)
        mesh.vertices.foreach_get("co", verts_coords)
        
        # Group by (x,y,z) and round each component
        rounded_verts = [tuple(round(v, precision) for v in mesh.vertices[i].co) 
                        for i in range(len(mesh.vertices))]
        
        # SORT the vertex list. This is CRUCIAL:
        # It makes the hash independent of the vertex order in the mesh storage
        rounded_verts.sort()
        
        # Pack sorted and rounded coordinates
        for v in rounded_verts:
            hash_m.update(struct.pack('3f', *v))

        # 2. Face materials (if enabled)
        if use_materials:
            # 1. Pre-calculate and cache hashes for all material slots
            material_cache = []
            for slot in self.obj.material_slots:
                if slot.material:
                    material_cache.append(bpy_material_hash(slot.material))
                else:
                    material_cache.append("None")

            # 2. Iterate through polygons using the cache
            num_slots = len(material_cache)
            for p in mesh.polygons:
                slot_index = p.material_index
                
                # Retrieve from cache or fallback if index is out of bounds
                if 0 <= slot_index < num_slots:
                    mat_hash = material_cache[slot_index]
                else:
                    mat_hash = "None"
                
                # Update hash
                hash_m.update(mat_hash.encode('utf-8'))

        # 3. Transformation matrix
        matrix_flat = [round(val, precision) for row in self.transform.matrix for val in row]
        hash_m.update(struct.pack('16f', *matrix_flat))

        return hash_m.hexdigest()
    
    @property
    def has_polygons(self):
        """Checks if the part has any polygons."""
        if self.obj is None:
            raise RuntimeError("Object is removed")
        return len(self.obj.data.polygons) > 0

    @property
    def mat(self):
        """Access the material of the part."""
        pass

    @mat.setter
    def mat(self, material: Optional['mat.Layer']):
        """Sets the material for the selected faces (or all faces if none selected)."""
        self._set_material(material)

    @property
    def default_mat(self):
        """Access the default material (index 0) of the part."""
        pass

    @default_mat.setter
    def default_mat(self, material: Optional['mat.Layer']):
        """
        Sets the default material at index 0. 
        If specific face materials are removed (set to None), this material will be used.
        """
        self._set_material(material, replace=False, default=True)

    def _set_material(self, material: Optional['mat.Layer'], faces: List['BMFaceWrapper'] | None = None, replace: bool = True, default: bool = False):
        self._ensure_bmesh(write=True)
        faces, all_faces = self._get_actual_bmesh_faces(faces)
        assert not default or len(faces) == len(all_faces)
        idx = self._get_or_create_material_index(material, default)
        for f in faces:
            if not replace and f.native.material_index > 0:
                continue
            f.native.material_index = idx
        self._write_bmesh()

    def _fix_topology(self, remove_double_verts: bool = True, min_vert_dist = 1e-4):
        """Fixes the topology for the further correct topology analysis."""
        bm = self._ensure_bmesh(write=True)
        if remove_double_verts:
            bmesh.ops.remove_doubles(bm.native, verts=bm.native.verts, dist=min_vert_dist)
        loose_verts = [v for v in bm.native.verts if not v.link_edges]
        if loose_verts:
            bmesh.ops.delete(bm.native, geom=loose_verts, context='VERTS')
        bm.native.verts.index_update()
        self._write_bmesh()

    def _ensure_bmesh(self, write=False):
        """Creates or returns a BMesh wrapper."""
        if self.obj is None:
            raise RuntimeError("Object is removed")
        if write:
            self._flush_bmesh()
        if self._bm_wrapper is None:
            bm = bmesh.new()
            bm.from_mesh(self.obj.data)
            # Ensure index access via lookup tables
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            bm.edges.layers.float.new("bevel_weight_edge")
            self._bm_wrapper = BMeshWrapper(bm)
        return self._bm_wrapper
    
    def _get_actual_geometry(self, items: List, all_items: List, name: str, remove_duplicates: bool) -> List:
        """Internal helper maintaining original mapping logic."""
        if not items:
            return all_items
        
        item_map = {i: i for i in all_items}
        try:
            actual_items = [item_map[i] for i in items]
            if remove_duplicates:
                actual_items = list(dict.fromkeys(actual_items))
            return actual_items
        except KeyError:
            raise RuntimeError(f"{name.capitalize()} are not valid for this object anymore. Recall {name}().")

    def _get_actual_bmesh_faces(self, faces: List['BMEdgeWrapper'] | None = None, remove_duplicates = True):
        bm = self._ensure_bmesh()
        actual: List['BMFaceWrapper'] = self._get_actual_geometry(faces, bm.faces, "faces", remove_duplicates)
        return actual, bm.faces

    def _get_actual_bmesh_edges(self, edges: List['BMEdgeWrapper'] | None = None, remove_duplicates = True):
        bm = self._ensure_bmesh()
        actual: List['BMEdgeWrapper'] = self._get_actual_geometry(edges, bm.edges, "edges", remove_duplicates)
        return actual, bm.edges

    def _get_actual_bmesh_verts(self, verts: List['BMVertWrapper'] | None = None, remove_duplicates = True):
        bm = self._ensure_bmesh()
        actual: List['BMVertWrapper'] = self._get_actual_geometry(verts, bm.verts, "vertices", remove_duplicates)
        return actual, bm.verts
    
    def _write_bmesh(self, flush=False):
        """Writes the current BMesh data back to the Blender mesh."""
        if self.obj is None:
            raise RuntimeError("Object is removed")
        if self._bm_wrapper is None:
            raise RuntimeError("BMesh is not created")
        self._bm_wrapper.native.to_mesh(self.obj.data)
        self.obj.data.update()
        if flush:
            self._flush_bmesh()

    def _flush_bmesh(self):
        """Bakes the BMesh into the object and clears the current wrapper."""
        if self.obj is None:
            raise RuntimeError("Object is removed")
        if self._bm_wrapper is not None:
            self._bm_wrapper = None
            self._topology = None
            self._last_checkpoint = None

    def make_checkpoint(self) -> 'GeometryCheckpoint':
        """
        Fixes the current state as the last checkpoint. 
        If it already exists, returns the current one.
        """
        if self._last_checkpoint is None:
            from .geometry import GeometryCheckpoint
            bm = self._ensure_bmesh()
            self._last_checkpoint = GeometryCheckpoint(bm)
        return self._last_checkpoint
    
    def _make_op_checkpoint(self):
        self._last_op_checkpoint = self.make_checkpoint()
        return self._last_op_checkpoint
    
    def _add_vertices(
        self, 
        coords: List[Vector], 
        write: bool = False, 
    ) -> List[bmesh.types.BMVert]:
        """Adds a list of coordinates as loose vertices to the BMesh."""
        bm = self._ensure_bmesh(write)
        vertices: List[bmesh.types.BMVert] = []

        for co in coords:
            v = bm.native.verts.new(co)
            vertices.append(v)
        
        if write:
            self._write_bmesh()

        return vertices
    
    def get_vertices(self) -> List[Vector]:
        return [v.co for v in self._ensure_bmesh().verts]

    def _ensure_joint_layers(self):
        """Creates or reuses custom vertex layers used by deformable joints."""
        bm = self._ensure_bmesh()

        joint_id_layer = bm.native.verts.layers.int.get("bp_joint_id")
        if joint_id_layer is None:
            joint_id_layer = bm.native.verts.layers.int.new("bp_joint_id")

        joint_role_layer = bm.native.verts.layers.int.get("bp_joint_role")
        if joint_role_layer is None:
            joint_role_layer = bm.native.verts.layers.int.new("bp_joint_role")

        return joint_id_layer, joint_role_layer
    
    def _inject_joint_markers(self, write=False) -> List[bmesh.types.BMVert]:
        """Adds temporary loose vertices for all deformable joints."""
        if not self._deformable_joints:
            return []
        from .joint import JointRole
        bm = self._ensure_bmesh(write)
        joint_id_layer, joint_role_layer = self._ensure_joint_layers()

        markers: List[bmesh.types.BMVert] = []
        eps = 0.001

        for joint in self._deformable_joints:
            # Use local-space coordinates, not world-space coordinates.
            loc = joint._rel_loc.matrix
            o = loc.to_translation()
            x = (loc.col[0].to_3d().normalized() if loc.col[0].to_3d().length > 1e-9 else Vector((1, 0, 0))) * eps
            y = (loc.col[1].to_3d().normalized() if loc.col[1].to_3d().length > 1e-9 else Vector((0, 1, 0))) * eps

            payload = [
                (JointRole.ORIGIN, o),
                (JointRole.X_AXIS, o + x),
                (JointRole.Y_AXIS, o + y),
            ]

            for role, co in payload:
                v = bm.native.verts.new(co)
                v[joint_id_layer] = joint._joint_id
                v[joint_role_layer] = int(role)
                markers.append(v)

        bm.native.verts.ensure_lookup_table()
        if write:
            self._write_bmesh()
        return markers
    
    def _sync_joint_markers(self, write=False):
        """Reads temporary marker vertices back and updates attached joints."""
        if not self._deformable_joints:
            return
        from .joint import JointRole
        bm = self._ensure_bmesh(write)
        joint_id_layer, joint_role_layer = self._ensure_joint_layers()

        frames = {}
        to_delete: list[bmesh.types.BMVert] = []

        for v in bm.native.verts:
            joint_id = int(v[joint_id_layer])
            role = int(v[joint_role_layer])
            # Skip regular mesh vertices.
            if joint_id <= 0:
                continue

            frames.setdefault(joint_id, {})[role] = v.co.copy()
            to_delete.append(v)

        for joint_id, data in frames.items():
            joint = self._get_joint_by_id(joint_id)
            if joint is None:
                continue

            if (
                JointRole.ORIGIN in data and
                JointRole.X_AXIS in data and
                JointRole.Y_AXIS in data
            ):
                joint.sync_from_frame(
                    data[JointRole.ORIGIN],
                    data[JointRole.X_AXIS],
                    data[JointRole.Y_AXIS],
                )

        # remove markers
        bmesh.ops.delete(bm.native, geom=to_delete, context='VERTS')

        if write:
            self._write_bmesh()

    @override
    def _domain_indices(
        self,
        domain: AttributeDomainItems,
    ) -> list[int]:
        """
        Returns all indices for the specified geometry domain.
        """
        data = self.obj.data
        assert isinstance(data, bpy.types.Mesh)
        if domain == "FACE":
            return list(range(len(data.polygons)))
        if domain == "EDGE":
            return list(range(len(data.edges)))
        if domain == "POINT":
            return list(range(len(data.vertices)))
        return []

    def _get_topology(self):
        """Returns a cached or new topology graph."""
        from .geometry import Topology
        if self._topology is None or self.topology_config != self._topology.config:
            self._topology = Topology(self._ensure_bmesh(), self, self.topology_config)
        return self._topology

    # Selectors return specialized wrappers
    def faces(self) -> ShapeList['Face']:
        """Returns a ShapeList of Face objects."""
        return ShapeList(self._get_topology().faces)

    def wires(self) -> ShapeList['Wire']:
        """Returns a ShapeList of Wire objects."""
        return ShapeList(self._get_topology().wires)
    
    def edges(self) -> ShapeList['Edge']:
        """Returns a ShapeList of Edge objects."""
        return ShapeList(self._get_topology().edges)

    def vertices(self) -> ShapeList['Vertex']:
        """Returns a ShapeList of Vertex objects."""
        return ShapeList(self._get_topology().vertices)
    
    @staticmethod
    def from_any_mesh(mesh: bpy.types.Mesh, name: str = "unknown"):
        """Creates a Part from a Blender mesh (including evaluated meshes)."""
        physical_mesh = bpy.data.meshes.new(f"{name}_mesh")
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.to_mesh(physical_mesh)
        bm.free()
        return Part(obj=bpy.data.objects.new(name, physical_mesh))
    
    @staticmethod
    def from_any_object(obj: bpy.types.Object):
        """Creates a Part from a Blender object (including evaluated objects)."""
        dg = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(dg)
        return Part.from_any_mesh(eval_obj.to_mesh(), obj.name)
    
    @staticmethod
    def box_set_empty(topology: Optional['TopologyConfig'] = None):
        return BoxSetPart(topology=topology)

T = TypeVar('T', bound='Any')

class BoxSetPart(Part):
    """
    An optimized Part subclass representing a set of box primitives (primitive LOD).
    Does not hold a Blender mesh by default; maintains a list of BoxData items.
    The Blender mesh is constructed lazily upon accessing `self.obj`.
    """

    @dataclass
    class Box(Generic[T]):
        """Stores geometric metadata for a single box primitive."""
        transform: Transform
        size: Vector
        custom_data: Optional[T] = None

        @property
        def part(self):
            from .primitives import Box
            from .build_part import Mode
            return Box(
                self.size.x,
                self.size.y,
                self.size.z,
                mode=Mode.PRIVATE
            ).create_part()

    def __init__(
        self,
        boxes: Optional[List[Box]] = None,
        size: Optional[Vector] = None,
        topology: Optional['TopologyConfig'] = None,
        custom_data: Optional[T] = None,
        transform: Transform = Transform()
    ):
        self.boxes: List[BoxSetPart.Box] = boxes or []
        if size is not None:
            self.boxes.append(BoxSetPart.Box(transform=Transform(), size=Vector(size), custom_data=custom_data))
        self._transform: Transform = transform

        self._obj_cache: Optional[bpy.types.Object] = None
        self._vertices_cache: Optional[List[Vector]] = None
        self._bbox_cache: Optional[Part.BBox] = None
        super().__init__(obj=None, topology=topology)

    @property
    @override
    def transform(self) -> Transform:
        """Access stored world transformation without touching self.obj."""
        return self._transform

    @transform.setter
    @override
    def transform(self, value: 'TransformExpr'):
        """Sets transformation matrix directly in Python state."""
        self._transform = value.resolve(self)
        if self._obj_cache is not None:
            self._obj_cache.matrix_world = self._transform.matrix

    @property
    @override
    def scale(self) -> Vector:
        """Access scale without touching self.obj."""
        return self.transform.scale

    @scale.setter
    @override
    def scale(self, value: VectorLike):
        """Sets scale without touching self.obj."""
        self.transform = self.loc * Scale(value)

    @property
    @override
    def size(self) -> Vector:
        """
        Calculates total dimensions from local bounding box scaled by current object scale.
        """
        bbox = self.local_bbox
        local_dims = bbox.max - bbox.min
        return Vector((
            local_dims.x * self.scale.x,
            local_dims.y * self.scale.y,
            local_dims.z * self.scale.z
        ))

    @size.setter
    @override
    def size(self, value: VectorLike):
        """Adjusts scale based on desired overall dimensions."""
        target_size = extract_vector(value)
        orig = self.orig_size
        self.scale = Vector((
            target_size.x / orig.x if orig.x > 1e-9 else 1.0,
            target_size.y / orig.y if orig.y > 1e-9 else 1.0,
            target_size.z / orig.z if orig.z > 1e-9 else 1.0
        ))

    @override
    def _create_empty_object(self) -> Optional[bpy.types.Object]:
        # Defer creation of the Blender object until explicitly needed
        return None

    @property
    @override
    def is_valid(self) -> bool:
        return True

    @property
    @override
    def is_physically_valid(self) -> bool:
        if self._obj_cache is None:
            return True  # Valid logically even if mesh has not been built
        try:
            _ = self._obj_cache.name
            return True
        except ReferenceError:
            return False

    @property
    @override
    def obj(self) -> Optional[bpy.types.Object]:
        """
        Lazily generates and returns the Blender object when requested by client code.
        """
        if self._obj_cache is None:
            self._rebuild_mesh()
        return self._obj_cache
    
    @obj.setter
    def obj(self, value: Optional[bpy.types.Object]):
        self.invalidate_mesh_cache()
        self._obj_cache = value

    # Unit cube corner vectors centered at the origin [-0.5, 0.5]
    _UNIT_CORNERS = tuple(
        Vector((x, y, z))
        for x in (-0.5, 0.5)
        for y in (-0.5, 0.5)
        for z in (-0.5, 0.5)
    )

    @override
    def get_vertices(self) -> List[Vector]:
        """
        Calculates and caches all 8 local corner vertices for every box in the set.
        """
        if self._vertices_cache is not None:
            return self._vertices_cache

        vertices = []
        for box in self.boxes:
            # Scale matrix for box dimensions combined with box local transform matrix
            box_scale_mat = Matrix.Diagonal((*box.size, 1.0))
            box_matrix = box.transform.matrix @ box_scale_mat

            # Transform unit cube corners to box local coordinates
            for corner in self._UNIT_CORNERS:
                vertices.append(box_matrix @ corner)

        self._vertices_cache = vertices
        return self._vertices_cache

    @override
    def _bbox_from_matrix(self, matrix: Matrix = Matrix.Identity(4)) -> Part.BBox:
        """
        Fast mathematical evaluation of the bounding box using cached box corner points.
        Bypasses dependency graph evaluation and self.obj entirely.
        """
        if self._bbox_cache is not None:
            return self._bbox_cache

        vertices = self.get_vertices()
        if not vertices:
            return self.BBox()

        # Transform cached local vertices by target matrix
        transformed_verts = [matrix @ v for v in vertices]

        min_v = Vector((
            min(v.x for v in transformed_verts),
            min(v.y for v in transformed_verts),
            min(v.z for v in transformed_verts)
        ))
        max_v = Vector((
            max(v.x for v in transformed_verts),
            max(v.y for v in transformed_verts),
            max(v.z for v in transformed_verts)
        ))

        bbox = self.BBox(min=min_v, max=max_v)
        self._bbox_cache = bbox
        return bbox

    def add_part(self, part: Part):
        """
        Adds another Part to this box set.
        If `part` is a BoxSetPart, its boxes are absorbed.
        Otherwise, the part is converted to a box via its bounding box.
        """
        if isinstance(part, BoxSetPart):
            for box in part.boxes:
                combined_transform = part.transform * box.transform
                self.boxes.append(
                    BoxSetPart.Box(
                        transform=combined_transform,
                        size=box.size.copy(),
                        custom_data=box.custom_data
                    )
                )
        else:
            # Convert regular Part to a single box using local_bbox
            bbox = part.local_bbox
            box_size = bbox.max - bbox.min
            center = (bbox.max + bbox.min) / 2.0

            box_transform = part.transform * Pos(center)
            self.boxes.append(
                BoxSetPart.Box(
                    transform=box_transform,
                    size=box_size
                )
            )

        self.invalidate_mesh_cache()

    def _rebuild_mesh(self):
        """Builds a combined BMesh from all stored box primitives."""
        if self._obj_cache is None:
            mesh = bpy.data.meshes.new("BoxSetMesh")
            self._obj_cache = bpy.data.objects.new("BoxSetPart", mesh)
        else:
            mesh: bpy.types.Mesh = self._obj_cache.data

        self._obj_cache.matrix_world = self.transform.matrix

        bm = bmesh.new()
        for box in self.boxes:
            # Apply box dimensions scale matrix combined with box world transform
            world_mat = (box.transform * Scale(box.size)).matrix
            bmesh.ops.create_cube(bm, size=1.0, matrix=world_mat)

        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

    def apply_transform(
        self, 
        op: TransformExpr = Transform(), 
        space: Location = Location(), 
    ):
        """
        Applies transformation directly to internal BoxData primitives 
        without building or touching BMesh / Blender object.
        """
        if not self.boxes:
            return
        op_tr = op.resolve(self)
        delta_tr = space * op_tr * space.inverse
        for box in self.boxes:
            box.transform = delta_tr * box.transform

    @override
    def copy(self) -> 'BoxSetPart':
        """Deep copies box descriptors without immediately duplicating Blender mesh data."""
        boxes_copy = [
            BoxSetPart.Box(
                transform=b.transform.copy(),
                size=b.size.copy(),
                custom_data=b.custom_data
            )
            for b in self.boxes
        ]
        part = BoxSetPart(
            boxes=boxes_copy,
            topology=self.topology_config,
            transform=self.transform.copy()
        )
        part._obj_cache = self._obj_cache
        part._vertices_cache = self._vertices_cache
        return part

    @override
    def remove(self, physical=True):
        """Safely cleans up Blender object and mesh references."""
        if physical and self._obj_cache and self.is_physically_valid:
            mesh_data = self._obj_cache.data
            if mesh_data and mesh_data.users == 0:
                bpy.data.meshes.remove(mesh_data)
            if self._obj_cache.name in bpy.data.objects:
                bpy.data.objects.remove(self._obj_cache, do_unlink=True)
            self._obj_cache = None
        self.boxes.clear()

    def invalidate_mesh_cache(self):
        self._obj_cache = None
        self._vertices_cache = None
        self._bbox_cache = None
