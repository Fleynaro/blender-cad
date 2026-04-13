import bmesh
from mathutils import Vector

class BMVertWrapper:
    """
    A wrapper for bmesh.types.BMVert that provides consistent ordering, 
    type safety, and high-level access to vertex properties.
    """
    def __init__(self, edge: bmesh.types.BMVert):
        self.native = edge

    @property
    def index(self):
        """The index of the vertex in the BMesh."""
        return self.native.index

    @property
    def co(self):
        """The coordinate (Vector) of the vertex."""
        return self.native.co
    
    @property
    def normal(self):
        """The normal vector of the vertex."""
        return self.native.normal
    
    @property
    def link_edges(self):
        """
        Returns a list of BMEdgeWrapper objects connected to this vertex, 
        sorted by their geometric centers for deterministic access.
        """
        return sorted([BMEdgeWrapper(e) for e in self.native.link_edges], key=lambda e: tuple(e.center))
    
    @property
    def link_faces(self):
        """
        Returns a list of BMFaceWrapper objects connected to this vertex, 
        sorted by their geometric centers for deterministic access.
        """
        return sorted([BMFaceWrapper(f) for f in self.native.link_faces], key=lambda f: tuple(f.center))
    
    def __repr__(self):
        return f"BMVert({self.co})"
    
    def __eq__(self, other):
        if isinstance(other, BMVertWrapper):
            return self.index == other.index
        return False
    
    def __hash__(self):
        return hash(self.index)

class BMEdgeWrapper:
    """
    A wrapper for bmesh.types.BMEdge providing geometric properties 
    and sorted access to linked vertices and faces.
    """
    def __init__(self, edge: bmesh.types.BMEdge):
        self.native = edge

    @property
    def index(self):
        """The index of the edge in the BMesh."""
        return self.native.index

    @property
    def is_valid(self):
        """Checks if the underlying BMesh edge is still valid (not deleted)."""
        return self.native.is_valid

    @property
    def length(self):
        """Calculated length of the edge."""
        return self.native.calc_length()
    
    @property
    def center(self) -> Vector:
        """Calculates the geometric center of the edge."""
        return sum((v.co for v in self.verts), Vector()) / len(self.verts)

    @property
    def verts(self):
        """
        Returns a list of BMVertWrapper objects forming this edge, 
        sorted by their coordinates.
        """
        return sorted([BMVertWrapper(v) for v in self.native.verts], key=lambda v: tuple(v.co))
    
    @property
    def link_faces(self):
        """
        Returns a list of BMFaceWrapper objects sharing this edge, 
        sorted by their geometric centers.
        """
        return sorted([BMFaceWrapper(f) for f in self.native.link_faces], key=lambda f: tuple(f.center))
    
    def other_vert(self, v: BMVertWrapper):
        """Returns the vertex at the opposite end of the edge relative to the provided vertex."""
        return BMVertWrapper(self.native.other_vert(v.native))
    
    def __repr__(self):
        return f"BMEdge({self.center})"
    
    def __eq__(self, other):
        if isinstance(other, BMEdgeWrapper):
            return self.index == other.index
        return False
    
    def __hash__(self):
        return hash(self.index)

class BMFaceWrapper:
    """
    A wrapper for bmesh.types.BMFace that exposes geometric data 
    and provides sorted access to its component edges and vertices.
    """
    def __init__(self, face: bmesh.types.BMFace):
        self.native = face

    @property
    def index(self):
        """The index of the face in the BMesh."""
        return self.native.index

    @property
    def smooth(self):
        """Boolean indicating if the face is set to smooth shading."""
        return self.native.smooth

    @property
    def area(self):
        """Calculated surface area of the face."""
        return self.native.calc_area()
    
    @property
    def center(self) -> Vector:
        """The median center point of the face."""
        return self.native.calc_center_median()
    
    @property
    def normal(self) -> Vector:
        """The normal vector of the face."""
        return self.native.normal

    @property
    def edges(self):
        """
        Returns a list of BMEdgeWrapper objects defining the face boundary, 
        sorted by their geometric centers.
        """
        return sorted([BMEdgeWrapper(e) for e in self.native.edges], key=lambda e: tuple(e.center))
    
    @property
    def verts(self):
        """Returns a list of BMVertWrapper objects that make up the face corners."""
        return [BMVertWrapper(v) for v in self.native.verts]
    
    def __repr__(self):
        return f"BMFace({self.center})"
    
    def __eq__(self, other):
        if isinstance(other, BMFaceWrapper):
            return self.index == other.index
        return False
    
    def __hash__(self):
        return hash(self.index)

class BMeshWrapper:
    """
    A high-level wrapper for the BMesh object, providing a entry point 
    for accessing faces and managing BMesh memory.
    """
    def __init__(self, bm: bmesh.types.BMesh):
        self._bm = bm

    @property
    def native(self) -> bmesh.types.BMesh:
        """Access to the underlying Blender bmesh object."""
        return self._bm
    
    @property
    def faces_raw(self):
        """
        Returns all faces in the mesh as BMFaceWrapper objects 
        in their original BMesh order.
        """
        return [BMFaceWrapper(f) for f in self._bm.faces]

    @property
    def edges_raw(self):
        """
        Returns all edges in the mesh as BMEdgeWrapper objects 
        in their original BMesh order.
        """
        return [BMEdgeWrapper(e) for e in self._bm.edges]

    @property
    def verts_raw(self):
        """
        Returns all vertices in the mesh as BMVertWrapper objects 
        in their original BMesh order.
        """
        return [BMVertWrapper(v) for v in self._bm.verts]

    @property
    def faces(self):
        """
        Returns all faces sorted by their geometric centers.
        """
        return sorted(self.faces_raw, key=lambda f: tuple(f.center))
    
    @property
    def edges(self):
        """
        Returns all edges sorted by their geometric centers.
        """
        return sorted(self.edges_raw, key=lambda e: tuple(e.center))
    
    @property
    def verts(self):
        """
        Returns all vertices sorted by their coordinates.
        """
        return sorted(self.verts_raw, key=lambda v: tuple(v.co))

    def __del__(self):
        """Ensures the native BMesh is freed from memory when the wrapper is destroyed."""
        if self._bm and self._bm.is_valid:
            self._bm.free()