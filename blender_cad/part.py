import bpy
import bmesh
from typing import TYPE_CHECKING, List, Optional
from typing_extensions import override
import hashlib
import struct

from .object import Object
from .material import mat, build_material
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
        return part

    @override
    def remove(self, physical=True):
        """Safely removes the object and its data from the Blender scene."""
        if self.obj and physical:
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
            for p in mesh.polygons:
                # Get the slot index
                slot_index = p.material_index
                
                # Check if there is a material in the slot (slot can contain None)
                if slot_index < len(self.obj.material_slots):
                    mat = self.obj.material_slots[slot_index].material
                    mat_name = mat.name.split(".")[0] if mat else "None"
                else:
                    mat_name = "None"
                
                # Update hash with material name
                hash_m.update(mat_name.encode('utf-8'))

        # 3. Transformation matrix
        matrix_flat = [round(val, precision) for row in self.transform.matrix for val in row]
        hash_m.update(struct.pack('16f', *matrix_flat))

        return hash_m.hexdigest()

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

    def _fix_topology(self):
        """Fixes the topology for the further correct topology analysis."""
        bm = self._ensure_bmesh(write=True)
        bmesh.ops.remove_doubles(bm.native, verts=bm.native.verts, dist=0.0001)
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
        return ShapeList([w for f in self.faces() for w in f.wires()])
    
    def edges(self) -> ShapeList['Edge']:
        """Returns a ShapeList of Edge objects."""
        return ShapeList(self._get_topology().edges)

    def vertices(self) -> ShapeList['Vertex']:
        """Returns a ShapeList of Vertex objects."""
        return ShapeList(self._get_topology().vertices)
    