from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Iterable, Self, Set, Tuple, List, Optional, Type, Union, overload
from typing_extensions import override
import math
from mathutils import Vector, Matrix, geometry
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from .common import CurveLike
from .location import Location, SurfaceLocation, Pos, Scale
from .shape_list import ShapeList
from .curve import BuildCurve, Curve, Polyline, Spline, make_curve

if TYPE_CHECKING:
    from .part import Part
    from .material import MaterialLayer
    from .bmesh_wrapper import BMEdgeWrapper, BMFaceWrapper, BMVertWrapper, BMeshWrapper
    
class GeomType(Enum):
    """Supported geometric primitive types for face analysis."""
    PLANE = "PLANE"
    CYLINDER = "CYLINDER"
    CONE = "CONE"
    SPHERE = "SPHERE"
    UNKNOWN = "UNKNOWN"

class UVProjection(Enum):
    """Methods for mapping 3D coordinates to 2D UV space."""
    PLANAR = "PLANAR"
    CYLINDRICAL = "CYLINDRICAL"
    SPHERICAL = "SPHERICAL"

class GeometryEntity(ABC):
    """Base class for all CAD-like geometric entities (Vertex, Edge, Wire, Face)."""
    def __init__(self, owner: 'Topology'):
        self._owner = owner

    def owner_loc(self, loc: Union['Location', 'Vector']) -> 'Location':
        owner_part = self._owner._part
        local_loc = loc if isinstance(loc, Location) else Pos(loc)
        scaled_local_matrix = (Scale(owner_part.scale) * local_loc).matrix        
        return Location(scaled_local_matrix, parent_loc=owner_part.loc)

    @abstractmethod
    def center(self) -> 'Location':
        """Returns the geometric center of the entity."""
        raise NotImplementedError
    
    @abstractmethod
    def is_new(self, checkpoint: Optional['GeometryCheckpoint'] = None) -> bool:
        """Determines if this entity was created after the given (or last) checkpoint."""
        raise NotImplementedError
    
    def bm_faces(self) -> List['BMFaceWrapper']:
        """Returns a list of BMesh faces associated with the entity."""
        return []
    
    def bm_edges(self) -> List['BMEdgeWrapper']:
        """Returns a list of BMesh edges associated with the entity."""
        return []
    
    def bm_verts(self) -> List['BMVertWrapper']:
        """Returns a list of BMesh vertices associated with the entity."""
        return []
    
    def faces(self) -> 'ShapeList[Face]':
        """Returns a list of Faces associated with the entity."""
        return ShapeList()
    
    def wires(self) -> 'ShapeList[Wire]':
        """Returns a list of Wires associated with the entity."""
        return ShapeList()
    
    def edges(self) -> 'ShapeList[Edge]':
        """Returns a list of Edges associated with the entity."""
        return ShapeList()
    
    def vertices(self) -> 'ShapeList[Vertex]':
        """Returns a list of Vertices associated with the entity."""
        return ShapeList()
    
    def all_single_faces(self) -> 'ShapeList[Face]':
        """Returns a list of single Faces associated with the entity. Essentially wraps bm_faces() into a ShapeList."""
        return ShapeList(Face(self._owner, faces=[f], wires=[Wire(self._owner,
                 edges=[Edge(self._owner, edges=[e],
                 start_v=e.verts[0]) for e in f.edges])]) for f in
                 self.bm_faces())
    
    def all_single_edges(self) -> 'ShapeList[Edge]':
        """Returns a list of single Edges associated with the entity. Essentially wraps bm_edges() into a ShapeList."""
        return ShapeList(Edge(self._owner, edges=[e], start_v=e.verts[0])
                 for e in self.bm_edges())
    
    def all_single_vertices(self) -> 'ShapeList[Vertex]':
        """Returns a list of single Vertices associated with the entity. Essentially wraps bm_verts() into a ShapeList."""
        return ShapeList((Vertex(self._owner, vert = v) for v in self.bm_verts()))
    
    @abstractmethod
    def split(self) -> 'ShapeList[Self]':
        """Splits (ungroups) the entity into a list of single entities."""
        raise NotImplementedError
    
    @abstractmethod
    def part(self) -> 'Part':
        """Creates a new Part containing a copy of this entity."""
        raise NotImplementedError

class Vertex(GeometryEntity):
    """Represents a single point in 3D space, wrapping a BMesh vertex."""
    def __init__(self, owner: 'Topology', vert: 'BMVertWrapper'):
        super().__init__(owner)
        self.element = vert

    @override
    def bm_verts(self):
        return [self.element]
    
    @override
    def vertices(self) -> ShapeList['Vertex']:
        return ShapeList([self])

    @override
    def center(self) -> 'Location':
        """Returns a copy of the vertex coordinates."""
        return self.owner_loc(self.element.co.copy())
    
    @override
    def is_new(self, checkpoint: Optional['GeometryCheckpoint'] = None) -> bool:
        cp = checkpoint or self._owner._part._last_op_checkpoint
        if not cp: return False
        new_v, _, _ = cp.get_new_entities(self._owner.bw)
        return self.element in new_v
    
    @override
    def split(self):
        return super().all_single_vertices()
    
    @override
    def part(self) -> 'Part':
        """Creates a new Part containing a copy of this vertex."""
        from .build_part import BuildPart, Mode
        with BuildPart(mode=Mode.PRIVATE) as bp:
            bm = bp.part._ensure_bmesh(write=True)
            # Create a new vertex at the same local coordinates
            bm.native.verts.new(self.element.co)
            bp.part._write_bmesh()
        return bp.part
    
    def __eq__(self, other):
        if not isinstance(other, Vertex): return False
        return self.element == other.element

    def __hash__(self):
        return hash(self.element)
    
    def __repr__(self):
        return f"Vertex({self.center().position})"

class Edge(GeometryEntity, CurveLike):
    """
    Represents a CAD Edge consisting of one or more BMesh edges.
    In CAD terms, an Edge is defined by its start and end vertices.
    """
    def __init__(self, owner: 'Topology', edges: List['BMEdgeWrapper'], start_v: 'BMVertWrapper'):
        super().__init__(owner)
        self.elements = edges
        self.start_v = start_v
        self._vertices: List[Vertex] = []

    @override
    def bm_verts(self):
        return [v for e in self.elements for v in e.verts]
    
    @override
    def bm_edges(self):
        return self.elements

    @override
    def center(self) -> 'Location':
        """Calculates the average center of all component BMesh edges."""
        return self.owner_loc(sum([e.center for e in self.elements], Vector()) / len(self.elements))

    @override
    def length(self) -> float:
        """Total length of the edge chain."""
        return sum(e.length for e in self.elements)
    
    def is_new(self, checkpoint: Optional['GeometryCheckpoint'] = None) -> bool:
        cp = checkpoint or self._owner._part._last_op_checkpoint
        if not cp: return False
        _, new_e, _ = cp.get_new_entities(self._owner.bw)
        return any(e in new_e for e in self.elements)

    @override
    def vertices(self) -> ShapeList[Vertex]:
        """A CAD Edge always returns two vertices: start and end."""
        # End is the opposite vertex of the last BMesh edge in the chain
        curr_v = self.start_v
        for e in self.elements:
            curr_v = e.other_vert(curr_v)
        
        # If the edge is a closed ring, start and end will coincide
        v_start = Vertex(self._owner, self.start_v)
        v_end = Vertex(self._owner, curr_v)
        return ShapeList([v_start, v_end])
    
    @override
    def edges(self) -> ShapeList['Edge']:
        return ShapeList([self])
    
    @override
    def split(self):
        return super().all_single_edges()
    
    @override
    def part(self) -> 'Part':
        """Creates a new Part containing a copy of this edge (via Curve)."""
        return self.curve().part

    @override
    def at(self, t: Optional[float] = None, t_m: Optional[float] = None) -> Location:
        """Returns the Location (position + orientation) at parameter t (0.0 to 1.0)."""
        assert t is not None or t_m is not None
        return self._at_dist((self.length() * max(0.0, min(1.0, t))) if t is not None else t_m)

    def _at_dist(self, target_len: float) -> Location:
        """Internal helper to calculate Location at a specific linear distance."""
        current_len = 0.0
        curr_v = self.start_v 
        
        for e in self.elements:
            l = e.length
            # Determine the 'exit' vertex of the current BMesh edge
            next_v = e.other_vert(curr_v)
            assert next_v is not None
            
            if current_len + l >= target_len or e == self.elements[-1]:
                local_t = (target_len - current_len) / l if l > 0 else 0
                
                # Use strict vertex order for consistent interpolation and direction
                p_start = curr_v.co
                p_end = next_v.co
                
                pt = p_start.lerp(p_end, local_t)
                direction = (p_end - p_start).normalized()
                
                # Construct rotation matrix where X aligns with the edge direction
                # to_track_quat(track_axis, up_axis)
                rot = direction.to_track_quat('X', 'Z').to_matrix().to_4x4()
                return self.owner_loc(Location(Matrix.Translation(pt) @ rot))
                
            current_len += l
            curr_v = next_v # Move to the next joint in the chain
            
        raise RuntimeError("Invalid edge location")
    
    def curve(self) -> 'Curve':
        """Converts the CAD Edge into a Polyline Curve using its BMesh segments."""
        # Start with the initial vertex coordinate
        pts = [self.start_v.co]
        curr_v = self.start_v
        
        # Traverse the chain of BMesh edges to get all intermediate points
        for e in self.elements:
            curr_v = e.other_vert(curr_v)
            pts.append(curr_v.co)
        
        # Check if the last vertex is the same as the start vertex
        is_closed = (curr_v == self.start_v)
        if is_closed and len(pts) > 1:
            pts.pop()

        with BuildCurve() as bc:
            Polyline(*pts, close=is_closed)
        return bc.curve
    
    def __eq__(self, other):
        if not isinstance(other, Edge): return False
        return self.elements == other.elements

    def __hash__(self):
        return hash(tuple(self.elements))

    def __repr__(self):
        return f"Edge({self.center().position})"

class Wire(GeometryEntity):
    """A collection of Edges forming a continuous path or loop."""
    def __init__(self, owner: 'Topology', edges: List[Edge], is_outer: bool = True):
        super().__init__(owner)
        self._edges = edges
        self.is_outer = is_outer

    @override
    def bm_verts(self):
        return [v for e in self._edges for v in e.bm_verts()]
    
    @override
    def bm_edges(self):
        return [bm_e for e in self._edges for bm_e in e.bm_edges()]

    @override
    def center(self) -> 'Location':
        """Average center of all component edges."""
        return self.owner_loc(sum([e.center().position for e in self._edges], Vector()) / len(self._edges))

    @override
    def length(self) -> float:
        """Total perimeter length of the wire."""
        return sum(e.length() for e in self._edges)
    
    def is_new(self, checkpoint: Optional['GeometryCheckpoint'] = None) -> bool:
        # A wire is new if its component edges are new
        return any(edge.is_new(checkpoint) for edge in self._edges)

    @override
    def wires(self) -> ShapeList['Wire']:
        return ShapeList([self])

    @override
    def edges(self) -> ShapeList[Edge]:
        """Returns the list of CAD edges in this wire."""
        return ShapeList(self._edges)

    @override
    def vertices(self) -> ShapeList[Vertex]:
        """
        Returns vertices in traversal order.
        In a closed loop, count equals the number of edges.
        In an open wire, count is number of edges + 1.
        """
        res = []
        if not self._edges: return ShapeList([])
        
        for i, edge in enumerate(self._edges):
            verts = edge.vertices()
            res.append(verts[0]) # Add the start of each edge
            
            # If this is the last edge and the wire is open, add the final vertex
            if i == len(self._edges) - 1:
                # Closure check: does the end of the last edge match the start of the first?
                if verts[1].element != self._edges[0].start_v:
                    res.append(verts[1])
                    
        return ShapeList(res)
    
    @override
    def at(self, t: Optional[float] = None, t_m: Optional[float] = None) -> Location:
        """Calculates Location at parameter t (0.0 to 1.0) along the entire wire."""
        assert t is not None or t_m is not None
        total_len = self.length()
        assert total_len > 0
        
        target_dist: float = (total_len * max(0.0, min(1.0, t))) if t is not None else t_m
        accumulated_dist = 0.0
        
        for edge in self._edges:
            edge_len = edge.length()
            # Find the specific edge containing the target distance
            if accumulated_dist + edge_len >= target_dist or edge == self._edges[-1]:
                local_dist = target_dist - accumulated_dist
                return edge._at_dist(local_dist)
            accumulated_dist += edge_len
            
        raise RuntimeError("Invalid wire location")
    
    @override
    def split(self):
        return super().all_single_edges()
    
    @override
    def part(self) -> 'Part':
        """Creates a new Part containing a copy of this wire (via Curve)."""
        return self.curve().part
    
    def curve(self) -> 'Curve':
        """Converts the Wire into a Polyline Curve by traversing its component Edges."""            
        pts = []
        # We start with the start_v of the first CAD edge
        start_v = self._edges[0].start_v
        curr_v = start_v
        pts.append(curr_v.co)
        
        # Traverse through each CAD edge and their internal BMesh elements
        for edge in self._edges:
            for bm_e in edge.elements:
                curr_v = bm_e.other_vert(curr_v)
                pts.append(curr_v.co)
        
        # Determine if the wire forms a closed loop
        is_closed = (curr_v == start_v)
        if is_closed and len(pts) > 1:
            pts.pop()

        with BuildCurve() as bc:
            Polyline(*pts, close=is_closed)
        return bc.curve
    
    def __eq__(self, other):
        if not isinstance(other, Wire): return False
        return self._edges == other._edges

    def __hash__(self):
        return hash(tuple(self._edges))

    def __repr__(self):
        return f"Wire({self.center().position})"

class UVSelector:
    """
    A fluent API for selecting a Location on a face based on geometric 
    criteria or specific UV coordinates.
    """
    def __init__(
        self, 
        criteria: List[Tuple[Callable[[Location], float], bool]] = None, 
        u_fixed: Optional[float] = None, 
        v_fixed: Optional[float] = None,
        pref_u=0.5,
        pref_v=0.5,
        u_offset: float = 0.0,
        v_offset: float = 0.0,
        u_offset_m: float = 0.0,
        v_offset_m: float = 0.0,
        projection: Optional[UVProjection] = None,
        tolerance_val: float = 1e-4,
        is_local: bool = False,
        final_offset: Location = Location()
    ):
        self.criteria = criteria if criteria is not None else []
        self.u_fixed = u_fixed
        self.v_fixed = v_fixed
        self.pref_u = pref_u
        self.pref_v = pref_v
        self.u_offset = u_offset
        self.v_offset = v_offset
        self.u_offset_m = u_offset_m
        self.v_offset_m = v_offset_m
        self.projection = projection
        self.tolerance_val = tolerance_val
        self.is_local = is_local
        self.final_offset = final_offset

    def _clone(self, **kwargs):
        """Internal helper to maintain immutability in the fluent chain."""
        return UVSelector(
            criteria=kwargs.get('criteria', self.criteria.copy()),
            u_fixed=kwargs.get('u_fixed', self.u_fixed),
            v_fixed=kwargs.get('v_fixed', self.v_fixed),
            pref_u=kwargs.get('pref_u', self.pref_u),
            pref_v=kwargs.get('pref_v', self.pref_v),
            u_offset=kwargs.get('u_offset', self.u_offset),
            v_offset=kwargs.get('v_offset', self.v_offset),
            u_offset_m=kwargs.get('u_offset_m', self.u_offset_m),
            v_offset_m=kwargs.get('v_offset_m', self.v_offset_m),
            projection=kwargs.get('projection', self.projection),
            tolerance_val=kwargs.get('tolerance_val', self.tolerance_val),
            is_local=kwargs.get('is_local', self.is_local),
            final_offset=kwargs.get('final_offset', self.final_offset)
        )
    
    def tolerance(self, value: float):
        """Sets the tolerance for grouping coordinates."""
        return self._clone(tolerance_val=value)
    
    def local(self, value: bool = True):
        """Switches coordinate evaluation to local space."""
        return self._clone(is_local=value)

    def set(self, u: Optional[float] = None, v: Optional[float] = None, projection: Optional[UVProjection] = None): 
        """Fixes the U coordinate for selection."""
        return self._clone(u_fixed=u, v_fixed=v, projection=projection)
    
    def offset(self, u: float = 0.0, v: float = 0.0):
        """Adds a UV offset that wraps around the [0, 1] boundary."""
        return self._clone(u_offset=u, v_offset=v)
    
    def set_m(self, u: float = 0.0, v: float = 0.0, projection: Optional[UVProjection] = None): 
        """Fixes the UV coordinate to absolute metric values (relative to 0,0)."""
        return self._clone(u_fixed=0.0, v_fixed=0.0, u_offset_m=u, v_offset_m=v, projection=projection)
    
    def offset_m(self, u: float = 0.0, v: float = 0.0):
        """Adds an absolute metric offset to the current selection."""
        return self._clone(u_offset_m=u, v_offset_m=v)
    
    def offset_final(self, loc: Location):
        """Adds a final offset to the current selection."""
        return self._clone(final_offset=loc)

    def with_projection(self, projection):
        """Sets a specific UVProjection to be used during sampling."""
        return self._clone(projection=projection)
    
    def min_u(self):
        return self._clone(pref_u=0)
    
    def max_u(self):
        return self._clone(pref_u=1)

    def min_v(self):
        return self._clone(pref_v=0)
    
    def max_v(self):
        return self._clone(pref_v=1)

    def min_x(self):
        return self._clone(criteria=self.criteria + [(lambda loc: loc.position.x, False)])
    
    def max_x(self):
        return self._clone(criteria=self.criteria + [(lambda loc: loc.position.x, True)])
    
    def min_y(self):
        return self._clone(criteria=self.criteria + [(lambda loc: loc.position.y, False)])
    
    def max_y(self):
        return self._clone(criteria=self.criteria + [(lambda loc: loc.position.y, True)])
    
    def min_z(self):
        return self._clone(criteria=self.criteria + [(lambda loc: loc.position.z, False)])
    
    def max_z(self):
        return self._clone(criteria=self.criteria + [(lambda loc: loc.position.z, True)])
    
    def top(self):
        """Alias for max_z()."""
        return self.max_z()
    
    def bottom(self):
        """Alias for min_z()."""
        return self.min_z()

    def select(self, at_uv_fn: Callable[[float, float, UVProjection], Location]) -> 'Location':
        """
        Samples the surface and selects the best Location using hierarchical 
        tolerance-based grouping and center-proximity tie-breaking.
        """
        # 1. Generate candidate samples with their UV coordinates
        u_samples = [self.u_fixed] if self.u_fixed is not None else [0.0, 0.5, 1.0]
        v_samples = [self.v_fixed] if self.v_fixed is not None else [0.0, 0.5, 1.0]
        
        # Candidates list stores tuples: (Location, u, v)
        candidates: List[Tuple[Location, float, float]] = []
        for u_val in u_samples:
            for v_val in v_samples:
                loc = at_uv_fn(u_val, v_val, 0, 0, self.projection)
                candidates.append((loc, u_val, v_val))
        
        if not candidates:
            raise ValueError("Failed to generate UV samples.")

        # 2. Process criteria sequentially (Filtering/Grouping)
        # We use a reductive approach: each criterion narrows down the candidate pool
        current_pool = candidates
        for key_func, reverse in self.criteria:
            actual_key_func: Callable[[Location], float] = (lambda loc: key_func(loc.local)) if self.is_local else key_func
            # Sort to find the "best" value (min or max)
            current_pool.sort(key=lambda item: actual_key_func(item[0]), reverse=reverse)
            best_val = actual_key_func(current_pool[0][0])
            
            # Keep all candidates that fall within the tolerance of the best value
            current_pool = [
                item for item in current_pool
                if abs(actual_key_func(item[0]) - best_val) <= self.tolerance_val
            ]

        # 3. Final Tie-breaker: Sort by proximity to the center (0.5, 0.5)
        # Distance squared: (u-0.5)^2 + (v-0.5)^2
        current_pool.sort(key=lambda item: (item[1] - self.pref_u)**2 + (item[2] - self.pref_v)**2)
            
        # Identify the winning coordinates before offset
        winner_loc, final_u, final_v = current_pool[0]
        if self.u_offset == 0.0 and self.v_offset == 0.0 and self.u_offset_m == 0.0 and self.v_offset_m == 0.0:
            return winner_loc * self.final_offset

        # 4. Apply UV Offset with wrap-around logic (0.0 to 1.0)
        # Using modulo 1.0 ensures 0.9 + 0.2 = 0.1
        offset_u = (final_u + self.u_offset) % 1.0
        offset_v = (final_v + self.v_offset) % 1.0
            
        # Return the final location sampled at the offset coordinates
        return at_uv_fn(offset_u, offset_v, self.u_offset_m, self.v_offset_m, self.projection) * self.final_offset

# Global instance for the fluent entry point
uv = UVSelector()

class Face(GeometryEntity):
    """
    Represents a CAD Face consisting of a group of smoothed BMesh faces.
    Includes logic for primitive detection and surface UV mapping.
    """
    def __init__(self, owner: 'Topology', faces: List['BMFaceWrapper'], wires: List[Wire]):
        super().__init__(owner)
        self.elements = faces
        self._wires = wires
        self._geom_type: Optional[GeomType] = None
        self._main_axis: Optional[Vector] = None
        self._best_uv_proj: Optional[tuple[UVProjection, list[Vector], dict[int, int]]] = None
        self._last_uv_proj: Optional[tuple[UVProjection, list[Vector], dict[int, int]]] = None

    @property
    def mat(self):
        """Material access for the face."""
        pass

    @mat.setter
    def mat(self, material: Optional['MaterialLayer']):
        self._owner._part._set_material(material, faces=self.elements)

    @property
    def geom_type(self) -> GeomType:
        """Lazy-loaded geometric primitive type."""
        if self._geom_type is None:
            self._geom_type, self._main_axis = self._detect_geom_type()
        return self._geom_type
    
    @property
    def main_axis(self) -> Vector:
        """The primary axis of symmetry or rotation for the face."""
        if self._geom_type is None:
            self._geom_type, self._main_axis = self._detect_geom_type()
        return self._main_axis

    @override
    def bm_verts(self):
        return [v for f in self.elements for v in f.verts]
    
    @override
    def bm_edges(self):
        return [e for f in self.elements for e in f.edges]
    
    @override
    def bm_faces(self):
        return self.elements

    @override
    def center(self) -> Location:
        return self.owner_loc(sum([f.center for f in self.elements], Vector()) / len(self.elements))
    
    def is_new(self, checkpoint: Optional['GeometryCheckpoint'] = None) -> bool:
        cp = checkpoint or self._owner._part._last_op_checkpoint
        if not cp: return False
        _, _, new_f = cp.get_new_entities(self._owner.bw)
        return any(f in new_f for f in self.elements)

    def area(self) -> float:
        return sum(f.area for f in self.elements)
    
    @override
    def faces(self) -> ShapeList['Face']:
        return ShapeList([self])

    @override
    def wires(self) -> ShapeList[Wire]:
        return ShapeList(self._wires)

    def inner_wires(self) -> ShapeList[Wire]:
        return ShapeList([w for w in self._wires if not w.is_outer])

    def outer_wires(self) -> ShapeList[Wire]:
        return ShapeList([w for w in self._wires if w.is_outer])

    @override
    def edges(self) -> ShapeList[Edge]:
        return ShapeList([e for w in self._wires for e in w.edges()])

    @override
    def vertices(self) -> ShapeList[Vertex]:
        return ShapeList(set(v for e in self.edges() for v in e.vertices()))
    
    @override
    def split(self):
        return super().all_single_faces()
    
    @override
    def part(self) -> 'Part':
        """Creates a new Part containing a copy of all BMFace elements in this entity."""
        from .build_part import BuildPart, Mode
        with BuildPart(mode=Mode.PRIVATE) as bp:
            dest_bm = bp.part._ensure_bmesh(write=True)
            v_map = {} # To not duplicate vertices

            for f in self.elements:
                new_verts = []
                for v in f.verts:
                    if v not in v_map:
                        # Create vertex in the new BMesh if not already copied
                        v_map[v] = dest_bm.native.verts.new(v.co)
                    new_verts.append(v_map[v])
                
                # Create the face in the new part
                dest_bm.native.faces.new(new_verts)
            
            bp.part._write_bmesh()
        return bp.part
    
    def _bm_verts_co(self):
        all_verts = self.bm_verts()
        v_indices = {v.index: i for i, v in enumerate(all_verts)}
        verts_co = [v.co.copy() for v in all_verts]
        return verts_co, v_indices
    
    def _detect_geom_type(self) -> Tuple[GeomType, Vector]:
        """Analyzes mesh normals and curvature to determine the primitive type."""
        default_axis = Vector((0, 0, 1))
        if not self.elements:
            return GeomType.UNKNOWN, default_axis

        faces = self.elements
        normals = [f.normal.normalized() for f in faces]
        f_data = [(f.center, f.normal) for f in faces]
        mesh_center = sum((d[0] for d in f_data), Vector()) / len(f_data)

        # 1. Plane check: do all face normals point in the same direction?
        ref_norm = f_data[0][1]
        if all(d[1].dot(ref_norm) > 0.999 for d in f_data):
            axis = sum(normals, Vector()).normalized()
            return GeomType.PLANE, axis

        # 2. Axis search for revolve-primitives (Cylinder/Cone)
        # We look for a vector that is most perpendicular to the differences between normals.
        # This identifies the rotation axis.
        
        # Collect 'chords' on the Gaussian sphere (normal space)
        diff_vectors: List[Vector] = []
        for i in range(len(normals) - 1):
            d = normals[i+1] - normals[i]
            if d.length > 0.001:
                diff_vectors.append(d.normalized())
        
        if len(diff_vectors) < 2:
            return GeomType.UNKNOWN, default_axis

        # Axis is approximately the cross product of non-parallel chords
        axis = diff_vectors[0].cross(diff_vectors[len(diff_vectors)//2]).normalized()
        
        # Ensure axis aligns with the general outward normal direction
        avg_n = sum(normals, Vector()) / len(normals)
        if axis.dot(avg_n) < 0:
            axis = -axis

        # 3. Cone and Cylinder test
        # Calculate the angle between each normal and the candidate axis
        angles = [n.angle(axis) for n in normals]
        avg_angle = sum(angles) / len(angles)
        
        # Variance check: are the angles nearly identical?
        variance = sum(abs(a - avg_angle) for a in angles) / len(angles)
        
        if variance < 0.02: # Threshold roughly ~1 degree
            # If angles are ~90 degrees, it's a cylinder; otherwise, it's a cone
            if abs(avg_angle - math.pi/2) < 0.05:
                return GeomType.CYLINDER, axis
            else:
                return GeomType.CONE, axis
            
        # 4. Sphere test: normals must point directly away from the geometric center
        sphere_score = []
        for f_center, f_norm in f_data:
            vec_from_center = (f_center - mesh_center).normalized()
            sphere_score.append(f_norm.dot(vec_from_center))
        
        # If all normals point 'outward' from the center (dot > 0.98)
        if sum(sphere_score) / len(sphere_score) > 0.98:
            # Find poles (vertices with highest valence/edge count)
            axis = Vector((0, 0, 1))
            verts = sorted(self.bm_verts(), key=lambda v: len(v.link_edges), reverse=True)
            if len(verts) >= 2:
                pole1 = verts[0].co
                pole2 = verts[1].co
                axis = (pole1 - pole2).normalized()
            return GeomType.SPHERE, axis

        return GeomType.UNKNOWN, default_axis

    def _detect_best_uv_projection(self):
        """Analyzes which UV projection produces the least distortion for this face."""
        if self._best_uv_proj is None:
            verts_co, v_indices = self._bm_verts_co()
            
            best_projection = UVProjection.PLANAR
            best_norm_uvs = []
            min_score = float('inf')
            for p_type in UVProjection:
                uvs = project_coords(verts_co, self.main_axis, p_type)
                norm_uvs, _ = normalize_uvs(uvs)
                score = get_projection_score(verts_co, norm_uvs, self.elements, v_indices, p_type, self.main_axis)
                if score < min_score:
                    best_projection = p_type
                    best_norm_uvs = norm_uvs
                    min_score = score
            self._best_uv_proj = (best_projection, best_norm_uvs, v_indices)
        return self._best_uv_proj
    
    def surface(self, selector: Optional['UVSelector'] = None) -> SurfaceLocation:
        return SurfaceLocation(self, selector)
    
    def curve(
        self, 
        rule: Callable[[float], tuple[float, float]], 
        limit: float = 1.0, 
        selector: UVSelector = uv,
        resolution: int = 50,
        metrics: bool = False,
        close: bool = False,
        curve_type: Union[Type[Polyline], Type[Spline]] = Spline
    ) -> Curve:
        """
        Generates a Curve on the surface of the face using UV or Metric offsets.
        Examples:
        1) face.curve(selector=uv.max_z(), rule=lambda v: (sin(v), v), limit=0.5)
        2) face.curve(rule=lambda u: (u, u), limit=10.0, metrics=True)
        """
        def surface_rule(t: float) -> Vector:
            coords = rule(t) # returns (u_val, v_val)
            if metrics:
                # Use offset in meters
                target_selector = selector.offset_m(coords[0], coords[1])
            else:
                # Use normalized UV offset
                target_selector = selector.offset(coords[0], coords[1])
            # Evaluate the 3D position on the face using the existing .at() logic
            return self.at(selector=target_selector).position

        return make_curve(
            rule=surface_rule, 
            limit=limit, 
            resolution=resolution,
            close=close,
            curve_type=curve_type
        )

    @overload
    def at(self, u: float, v: float, u_offset_m: float = 0.0, v_offset_m: float = 0.0, projection: Optional['UVProjection'] = None) -> Location:
        """Get Location by precise U and V coordinates."""
        ...

    @overload
    def at(self, selector: 'UVSelector') -> Location:
        """Get Location using a fluent UVSelector."""
        ...

    def at(self, *args, **kwargs) -> Location:
        """
        Universal positioning method.
        Usage:
            face.at(0.5, 0.5)
            face.at(uv.max_z())
            face.at(u=1.0, v=0.0, projection=my_proj)
            face.at(u_offset_m=10.0, v_offset_m=-20.0)
            fact.at(selector=UVSelector(u=0.5, v=0.5))
        """
        # 1. Handle UVSelector case (either as first positional or keyword)
        selector = kwargs.get('selector')
        if not selector and args and isinstance(args[0], UVSelector):
            selector = args[0]
        
        if selector:
            return selector.select(self._at_uv)

        # 2. Handle positional arguments (u, v, projection)
        # We extract values from args or kwargs to support any call style
        u = kwargs.get('u', args[0] if len(args) > 0 else None)
        v = kwargs.get('v', args[1] if len(args) > 1 else None)
        u_offset_m = kwargs.get('u_offset_m', args[2] if len(args) > 2 else 0.0)
        v_offset_m = kwargs.get('v_offset_m', args[3] if len(args) > 3 else 0.0)
        projection = kwargs.get('projection', args[4] if len(args) > 4 else None)

        if u is not None and v is not None:
            return self._at_uv(u, v, u_offset_m, v_offset_m, projection)

        raise ValueError(
            "Invalid arguments for .at(). "
            "Provide either a UVSelector or both 'u' and 'v' coordinates."
        )
    
    def _at_uv(self, u: float, v: float, u_offset_m: float = 0.0, v_offset_m: float = 0.0, projection: UVProjection | None = None) -> Location:
        """Finds the Location at surface coordinates (u, v) with metric offsets."""        
        u = max(1e-6, min(1.0 - 1e-6, u))
        v = max(1e-6, min(1.0 - 1e-6, v))

        if projection is None and self.geom_type != GeomType.UNKNOWN:
            projection = {
                GeomType.PLANE: UVProjection.PLANAR,
                GeomType.SPHERE: UVProjection.SPHERICAL,
                GeomType.CYLINDER: UVProjection.CYLINDRICAL,
                GeomType.CONE: UVProjection.CYLINDRICAL
            }.get(self.geom_type)

        # 1. Projection Selection
        if projection is None:
            projection, uvs, v_indices = self._detect_best_uv_projection()
        else:
            if self._last_uv_proj and self._last_uv_proj[0] == projection:
                uvs, v_indices = self._last_uv_proj[1], self._last_uv_proj[2]
            else:
                verts_co, v_indices = self._bm_verts_co()
                uvs_raw = project_coords(verts_co, self.main_axis, projection)
                uvs, _ = normalize_uvs(uvs_raw, projection)
                self._last_uv_proj = (projection, uvs, v_indices)
        
        res_u, res_v = u, v
        # 2. Apply Metric Offsets by walking through polygons
        if v_offset_m != 0.0:
            res_u, res_v = self._walk_metric(res_u, res_v, v_offset_m, False, projection, uvs, v_indices)
        if u_offset_m != 0.0:
            res_u, res_v = self._walk_metric(res_u, res_v, u_offset_m, True, projection, uvs, v_indices)

        # 3. Final Evaluation
        return self._calc_uv_location(res_u, res_v, projection, uvs, v_indices)

    DEBUG_MODE = False
    def _debug_visualize(self, curr_uv, move_dir_uv, v_tri, uv_tri, obj_scale):
        """Helper for debugging. Visualizes a triangle and its UVs."""
        import bpy
        p = [v.co * obj_scale for v in v_tri]
        
        # Triangle 3D
        mesh_3d = bpy.data.meshes.new("DEBUG_tri_3d_mesh")
        obj_3d = bpy.data.objects.new("DEBUG_tri_3d", mesh_3d)
        bpy.context.collection.objects.link(obj_3d)
        mesh_3d.from_pydata(p, [[0, 1], [1, 2], [2, 0]], [[0, 1, 2]])
        obj_3d.color = (1, 0.5, 0, 1)

        uv_offset = Vector((10.0, 0.0, 0.0)) 
        
        def to_2d_vec(uv_vec):
            return Vector((uv_vec.x, uv_vec.y, 0.0)) + uv_offset

        # Triangle UV
        uv_p = [to_2d_vec(uv) for uv in uv_tri]
        mesh_uv = bpy.data.meshes.new("DEBUG_tri_uv_mesh")
        obj_uv = bpy.data.objects.new("DEBUG_tri_uv", mesh_uv)
        bpy.context.collection.objects.link(obj_uv)
        mesh_uv.from_pydata(uv_p, [[0, 1], [1, 2], [2, 0]], [[0, 1, 2]])

        # Point curr_uv
        curr_p = to_2d_vec(curr_uv)
        marker = bpy.data.objects.new("DEBUG_curr_uv", None)
        marker.location = curr_p
        marker.empty_display_type = 'SINGLE_ARROW'
        marker.empty_display_size = 0.1
        bpy.context.collection.objects.link(marker)

        # Vector move_dir_uv
        dir_p = to_2d_vec(curr_uv + move_dir_uv * 0.2)
        line_mesh = bpy.data.meshes.new("DEBUG_dir_mesh")
        line_obj = bpy.data.objects.new("DEBUG_move_dir", line_mesh)
        bpy.context.collection.objects.link(line_obj)
        line_mesh.from_pydata([curr_p, dir_p], [[0, 1]], [])

        # Update scene
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

    def _walk_metric(self, u: float, v: float, offset_m: float, is_u: bool, 
                     projection: UVProjection, uvs: list[Vector], v_indices: dict) -> Tuple[float, float]:
        """
        Iteratively traverses adjacent polygons in UV space to cover a metric distance,
        correctly handling non-uniform object scaling.
        """
        curr_uv = Vector((u, v))
        rem_dist = abs(offset_m)
        direction = 1.0 if offset_m > 0 else -1.0
        move_dir_uv = Vector((direction, 0.0)) if is_u else Vector((0.0, direction))
        obj_scale = self._owner._part.scale
        
        # Start by finding the initial triangle
        current_hit = self._find_face_and_uv_tri(curr_uv, projection, uvs, v_indices)
        if not current_hit:
            raise ValueError("Failed to find initial triangle for metric walk.")
        
        for _ in range(500): # Safety limit for very dense meshes
            if rem_dist < 1e-7 or not current_hit: break
            
            v_tri, bm_face, uv_tri, _ = current_hit
            
            if self.DEBUG_MODE:
                self._debug_visualize(curr_uv, move_dir_uv, v_tri, uv_tri, obj_scale)
            
            # Apply scale to vertex coordinates
            # This ensures tangents T_u/T_v account for the flattened/stretched geometry.
            p = [v.co * obj_scale for v in v_tri]
            uv0, uv1, uv2 = uv_tri
            
            # Edges in scaled 3D and UV space
            e1_3d, e2_3d = p[1] - p[0], p[2] - p[0]
            e1_uv, e2_uv = uv1 - uv0, uv2 - uv0
            
            # Solve for tangents dP/du and dP/dv (The Jacobian)
            det = e1_uv.x * e2_uv.y - e1_uv.y * e2_uv.x
            if abs(det) < 1e-11: break
            
            inv_det = 1.0 / det
            t_u_m = (e1_3d * e2_uv.y - e2_3d * e1_uv.y) * inv_det
            t_v_m = (e1_3d * -e2_uv.x + e2_3d * e1_uv.x) * inv_det
            
            # Metric speed: how many meters we gain per 1.0 UV unit in current direction
            meters_per_uv_unit = t_u_m.length if is_u else t_v_m.length
            if meters_per_uv_unit < 1e-9: break
            
            # Intersection test with triangle edges in UV
            t_to_edge, edge_idx = float('inf'), -1
            for i in range(3):
                intersect = geometry.intersect_line_line_2d(curr_uv, curr_uv + move_dir_uv, uv_tri[i], uv_tri[(i+1)%3])
                if intersect:
                    vec = intersect - curr_uv
                    if vec.dot(move_dir_uv) > 1e-8:
                        if vec.length < t_to_edge:
                            t_to_edge, edge_idx = vec.length, i

            # If we are stuck on an edge and can't find an exit in current tri
            if edge_idx == -1:
                # Nudge curr_uv slightly and retry
                curr_uv += move_dir_uv * 1e-6
                current_hit = self._find_face_and_uv_tri(curr_uv, projection, uvs, v_indices)
                continue

            dist_to_edge_m = t_to_edge * meters_per_uv_unit
            if dist_to_edge_m >= rem_dist:
                # Finished walking within this triangle
                curr_uv += move_dir_uv * (rem_dist / meters_per_uv_unit)
                rem_dist = 0
                break
            else:
                # Cross the edge and move to the next triangle
                rem_dist -= dist_to_edge_m
                # Nudge curr_uv slightly into the next triangle
                curr_uv += move_dir_uv * (t_to_edge + 1e-6)
                
                # Identify the vertices of the edge we are crossing
                v1, v2 = v_tri[edge_idx], v_tri[(edge_idx+1)%3]
                
                # Find the adjacent triangle (could be in the same BMFace or neighbor BMFace)
                current_hit = self._find_next_triangle(v1, v2, bm_face, curr_uv, projection, uvs, v_indices)
                    
        return curr_uv.x % 1.0, curr_uv.y # Wrap X for cylindrical logic
    
    def _iter_face_triangles(self, face: 'BMFaceWrapper', projection: UVProjection, uvs: list[Vector], v_indices: dict) -> Iterable[Tuple[tuple['BMVertWrapper'], list[Vector]]]:
        """Generator that yields (v_tri, uv_tri) for a face, handling triangulation and UV seams."""
        f_verts = face.verts
        for i in range(1, len(f_verts) - 1):
            v_tri = (f_verts[0], f_verts[i], f_verts[i+1])
            uv_tri = [uvs[v_indices[v.index]].copy() for v in v_tri]

            # Seam Fix for spherical/cylindrical wrapping
            if projection in {UVProjection.SPHERICAL, UVProjection.CYLINDRICAL}:
                # Find average X to determine where the mass of the triangle is
                avg_x = sum(uv.x for uv in uv_tri) / 3
                
                for uv in uv_tri:
                    # If vertex is too far from average (seam), 
                    # pull it back to 0..1 via remainder of division
                    if abs(uv.x - avg_x) > 0.5:
                        if uv.x > avg_x:
                            uv.x -= 1.0
                        else:
                            uv.x += 1.0
                    
                    uv.x = max(0.0, min(1.0, uv.x))
            yield v_tri, uv_tri

    def _find_face_and_uv_tri(self, target_uv: Vector, projection: UVProjection, uvs: list[Vector], v_indices: dict, faces: list['BMFaceWrapper'] = None, use_closest: bool = False):
        """Helper to find which triangle of which BMFace contains the UV point."""
        closest_data, min_dist_2d = None, float('inf')
        search_set = faces if faces is not None else self.elements

        for f in search_set:
            for v_tri, uv_tri in self._iter_face_triangles(f, projection, uvs, v_indices):
                # Check if UV point is within triangle
                if geometry.intersect_point_tri_2d(target_uv, *uv_tri):
                    return v_tri, f, uv_tri, get_barycentric_2d(target_uv, *uv_tri)
                
                # Fallback: search for the closest triangle if exact hit is not found
                if use_closest:
                    cp = geometry.closest_point_on_tri(target_uv.to_3d(), uv_tri[0].to_3d(), uv_tri[1].to_3d(), uv_tri[2].to_3d())
                    dist = (cp.to_2d() - target_uv).length
                    if dist < min_dist_2d:
                        min_dist_2d = dist
                        closest_data = (v_tri, f, uv_tri, get_barycentric_2d(cp.to_2d(), *uv_tri))
        return closest_data

    def _find_next_triangle(self, v1: 'BMVertWrapper', v2: 'BMVertWrapper', current_face: 'BMFaceWrapper', target_uv: Vector, projection: UVProjection, uvs: list[Vector], v_indices: dict):
        """Finds the next triangle in the walking sequence by checking linked faces."""
        # 1. Same face (n-gons)
        hit = self._find_face_and_uv_tri(target_uv, projection, uvs, v_indices, [current_face])
        if hit: return hit
        
        # 2. Shared edges, then fallback to shared verts
        common_faces = set(v1.link_faces).intersection(v2.link_faces)
        fallback_faces = set(v1.link_faces).union(v2.link_faces)
        
        for faces in [common_faces, fallback_faces]:
            for f in faces:
                if f != current_face and f in self.elements:
                    hit = self._find_face_and_uv_tri(target_uv, projection, uvs, v_indices, [f])
                    if hit: return hit
        return None
    
    def _calc_uv_location(self, u: float, v: float, projection: UVProjection, uvs: list[Vector], v_indices: dict) -> Location:
        """Internal helper to find exact 3D Location from normalized UV coordinates."""
        hit = self._find_face_and_uv_tri(Vector((u, v)), projection, uvs, v_indices, use_closest=True)
        
        if hit:
            v_tri, f, _, bary = hit
            # Interpolated 3D Position
            p_3d = v_tri[0].co * bary.x + v_tri[1].co * bary.y + v_tri[2].co * bary.z
            # Interpolated Normal
            normals = [v.normal if f.smooth else f.normal for v in v_tri]
            n_3d = (normals[0] * bary.x + normals[1] * bary.y + normals[2] * bary.z).normalized()
            
            rotation_matrix = n_3d.to_track_quat('Z', 'Y').to_matrix().to_4x4()
            return self.owner_loc(Location(Matrix.Translation(p_3d) @ rotation_matrix))

        raise Exception("Point not found")
    
    def __eq__(self, other):
        if not isinstance(other, Face): return False
        return frozenset(self.elements) == frozenset(other.elements)

    def __hash__(self):
        return hash(frozenset(self.elements))
    
    def __repr__(self):
        return f"Face({self.center().position})"


# --- 3. Topology Analyzer ---

@dataclass(frozen=True)
class TopologyConfig:
    """Thresholds and parameters for geometry analysis."""
    smooth_angle: float = 15.0
    edge_break_angle: float = 15.0

class Topology:
    """
    Engine that reconstructs a CAD topology from a raw BMesh.
    Groups bmesh polygons into CAD Faces, and boundary loops into CAD Edges/Wires.
    """
    def __init__(self, bmesh_wrapper: 'BMeshWrapper', part: 'Part', config = TopologyConfig()):
        self.bw = bmesh_wrapper
        self._part = part
        self.config = config
        self.faces: List[Face] = []
        self.edges: List[Edge] = []
        self.vertices: List[Vertex] = []
        self._build_topology()

    def _build_topology(self):
        visited_faces: set['BMFaceWrapper'] = set()
        
        # 1. Grouping faces by smoothness
        for f in self.bw.faces:
            if f in visited_faces: continue
            
            group = self._grow_face_group(f, visited_faces)
            f_set = set(group)
            
            # 2. Identify boundary edges for the group
            boundary_edges: List['BMEdgeWrapper'] = []
            for face in group:
                for e in face.edges:
                    # An edge is a boundary if only one neighbor face belongs to this group
                    if sum(1 for lf in e.link_faces if lf in f_set) == 1:
                        boundary_edges.append(e)

            # 3. Assemble Wires and segment them into Edges
            wires = self._build_wires(boundary_edges)
            
            # 4. Create high-level CAD Face
            cad_face = Face(self, group, wires)
            self.faces.append(cad_face)

        # Collect unique edges and vertices at the end
        self.edges = list(set(e for w in self.faces for e in w.edges()))
        self.vertices = list(set(v for e in self.edges for v in e.vertices()))

    def _grow_face_group(self, start_face: 'BMFaceWrapper', visited_faces: set['BMFaceWrapper']):
        """BFS algorithm to collect all polygons that belong to a smoothed surface group."""
        group: List['BMFaceWrapper'] = []
        queue = [start_face]
        visited_faces.add(start_face)
        
        while queue:
            curr = queue.pop(0)
            group.append(curr)

            curr_n = curr.normal
            if curr_n.length <= 1e-6:
                continue

            for e in curr.edges:
                for adj in e.link_faces:
                    if adj not in visited_faces:
                        adj_n = adj.normal
                        # Compare normals: if angle is small enough, they belong to the same CAD face
                        if adj_n.length > 1e-6:
                            angle = curr.normal.angle(adj.normal)
                            if angle > math.pi / 2:
                                # see test test_topology_reconstruction_complex
                                angle = abs(math.pi - angle)
                            if angle <= math.radians(self.config.smooth_angle):
                                visited_faces.add(adj)
                                queue.append(adj)
                        else:
                            visited_faces.add(adj)
        return group

    def _build_wires(self, boundary_edges: list['BMEdgeWrapper']) -> List[Wire]:
        """Assembles boundary edges into ordered loops (Wires)."""
        wires = []
        while boundary_edges:
            start_e = next(iter(boundary_edges))
            boundary_edges.remove(start_e)
            
            # Build an ordered chain (Loop)
            ordered_loop = [start_e]
            curr_v = start_e.verts[1]
            start_v = start_e.verts[0]
            
            while True:
                next_e: Optional['BMEdgeWrapper'] = None
                for e in curr_v.link_edges:
                    if e in boundary_edges:
                        next_e = e
                        break
                
                if next_e:
                    boundary_edges.remove(next_e)
                    ordered_loop.append(next_e)
                    curr_v = next_e.other_vert(curr_v)
                    if curr_v == start_v: # Loop closure
                        break
                else:
                    break

            # Segment the Loop into individual CAD Edges based on sharp angles
            cad_edges = self._segment_loop(ordered_loop, start_v)
            
            # Simple heuristic for outer contour identification
            is_outer = (len(wires) == 0)
            wire = Wire(self, cad_edges, is_outer)
            wires.append(wire)
        return wires

    def _segment_loop(self, loop: List['BMEdgeWrapper'], wire_start_v: 'BMVertWrapper') -> List[Edge]:
        """Splits a BMesh edge loop into CAD Edges wherever a sharp break angle is found."""
        if not loop: return []
        
        cad_edges_in_wire: List[Edge] = []
        current_segment = [loop[0]]
        edge_start_v = wire_start_v
        
        for i in range(1, len(loop)):
            e_prev = loop[i-1]
            e_curr = loop[i]
            
            # Find the common vertex (joining point)
            common_v = list(set(e_prev.verts) & set(e_curr.verts))[0]
            
            # Vectors pointing AWAY from the shared vertex
            v1 = (e_prev.other_vert(common_v).co - common_v.co).normalized()
            v2 = (e_curr.other_vert(common_v).co - common_v.co).normalized()
            
            # Dot product clamp for acos safety
            dot = max(-1.0, min(1.0, v1.dot(v2)))
            angle = math.acos(dot)
            
            # Check for a sharp break (deviation from 180-degree straight line)
            if abs(math.pi - angle) > math.radians(self.config.edge_break_angle):
                new_edge = Edge(self, current_segment, start_v=edge_start_v)
                cad_edges_in_wire.append(new_edge)
                edge_start_v = common_v
                current_segment = [e_curr]
            else:
                current_segment.append(e_curr)
            
            common_v = e_curr.other_vert(common_v)

        # Close and store the final edge segment
        if current_segment:
            last_edge = Edge(self, current_segment, start_v=edge_start_v)
            cad_edges_in_wire.append(last_edge)
            
        return cad_edges_in_wire

class GeometryCheckpoint:
    """
    Stores a spatial snapshot of BMesh geometry for comparison.
    Trees (BVH/KD) are built lazily. Results are cached per BMesh.
    """
    def __init__(self, bm: 'BMeshWrapper'):
        self.bm = bm
        self._bvh: Optional[BVHTree] = None
        self._kd_verts: Optional[KDTree] = None
        self._kd_edges: Optional[KDTree] = None
        
        # Cache to store comparison results for a specific BMesh pointer
        self._cache_bm_ptr: int = 0
        self._cache_res: Tuple[Set[BMVertWrapper], Set[BMEdgeWrapper], Set[BMFaceWrapper]] = (set(), set(), set())

    def _ensure_trees(self):
        """Lazy initialization of spatial search structures."""
        if self._bvh is None:
            if len(self.bm.faces_raw) > 0:
                self._bvh = BVHTree.FromBMesh(self.bm.native)
            
            self._kd_verts = KDTree(max(1, len(self.bm.verts_raw)))
            for i, v in enumerate(self.bm.verts_raw):
                self._kd_verts.insert(v.co, i)
            self._kd_verts.balance()

            self._kd_edges = KDTree(max(1, len(self.bm.edges_raw)))
            for i, e in enumerate(self.bm.edges_raw):
                self._kd_edges.insert(e.center, i)
            self._kd_edges.balance()

    def get_new_entities(self, bm: 'BMeshWrapper'):
        """
        Compares the given BMesh with this checkpoint and returns indices of new entities.
        Uses caching if the same BMesh object is passed again.
        """
        # Check if we have a cached result for this specific BMesh instance
        if self._cache_bm_ptr == id(bm):
            return self._cache_res

        self._ensure_trees()
        new_verts: Set[BMVertWrapper] = set()
        new_edges: Set[BMEdgeWrapper] = set()
        new_faces: Set[BMFaceWrapper] = set()
        
        # Check Vertices
        kd_v = self._kd_verts
        for v in bm.verts_raw:
            _, _, dist = kd_v.find(v.co)
            if dist > 1e-4:
                new_verts.add(v)
                
        # Check Faces (Surface proximity check)
        bvh = self._bvh
        for f in bm.faces_raw:
            is_old = False
            if bvh:
                loc, norm, idx, dist = bvh.find_nearest(f.center)
                if loc and dist < 1e-4:
                    # Even if geometry changed (boolean cut), if the face 
                    # lies on the old plane with the same normal, it's 'old'.
                    idx: int
                    old_f = self.bm.faces_raw[idx]
                    if f.normal.dot(old_f.normal) > 0.99:
                        is_old = True
            if not is_old:
                new_faces.add(f)
                
        # Check Edges
        kd_e = self._kd_edges
        for e in bm.edges_raw:
            _, idx, dist = kd_e.find(e.center)
            is_old = False
            if dist < 1e-4:
                old_e = self.bm.edges_raw[idx]
                # Length check to ensure the edge hasn't been split/resized
                if abs(e.length - old_e.length) < 1e-4:
                    is_old = True
            if not is_old:
                new_edges.add(e)
                
        # Update cache
        self._cache_bm_ptr = id(bm)
        self._cache_res = (new_verts, new_edges, new_faces)
        return self._cache_res

def get_barycentric_2d(p: Vector, a: Vector, b: Vector, c: Vector):
    """Calculates 2D barycentric coordinates for point p in triangle abc."""
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-8:
        return Vector((1, 0, 0))
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return Vector((u, v, w))

def _estimate_geometry_center(local_verts: list[Vector], p_type: UVProjection) -> Vector:
    """Estimates the curvature center using geometric properties of the shell."""
    min_v = Vector((min(v.x for v in local_verts), min(v.y for v in local_verts), min(v.z for v in local_verts)))
    max_v = Vector((max(v.x for v in local_verts), max(v.y for v in local_verts), max(v.z for v in local_verts)))
    bb_center = (min_v + max_v) / 2
    return bb_center

def project_coords(verts_co: list[Vector], normal: Vector, p_type: UVProjection):
    """Projects 3D vertex coordinates into 2D UV space based on projection type."""
    uvs: list[Vector] = []
    # Matrix that aligns 'normal' with Z, ensuring poles align with the face orientation
    rot_quat = normal.to_track_quat('Z', 'Y')
    inv_rot = rot_quat.inverted()
    local_verts = [inv_rot @ co for co in verts_co]
    center = _estimate_geometry_center(local_verts, p_type)

    for co in local_verts:
        # Move to local space relative to center
        rel = co - center
        if p_type == UVProjection.PLANAR:
            u, v = rel.x, rel.y
        elif p_type == UVProjection.SPHERICAL:
            r = rel.length
            if r < 1e-8:
                u, v = 0.5, 0.5
            else:
                u = 0.5 + math.atan2(rel.y, rel.x) / (2 * math.pi)
                v = 0.5 + math.asin(max(-1, min(1, rel.z / r))) / math.pi
        elif p_type == UVProjection.CYLINDRICAL:
            u = 0.5 + math.atan2(rel.y, rel.x) / (2 * math.pi)
            v = rel.z
        uvs.append(Vector((u, v)))
    return uvs

def normalize_uvs(uvs: list[Vector], p_type: "UVProjection"):
    """Remaps UV coordinates to a normalized [0, 1] range with seam awareness."""
    if not uvs:
        return uvs, (0, 1, 0, 1)

    # 1. Handle wrap-around for circular projections (Cylinder/Sphere)
    is_circular = p_type in {UVProjection.CYLINDRICAL, UVProjection.SPHERICAL}
    
    if is_circular and len(uvs) > 1:
        u_coords = sorted([p.x for p in uvs])
        max_gap = 0
        split_threshold = 0
        
        # Find the largest gap between adjacent U coordinates
        for i in range(len(u_coords) - 1):
            gap = u_coords[i+1] - u_coords[i]
            if gap > max_gap:
                max_gap = gap
                split_threshold = u_coords[i+1]
        
        # Gap across the 1.0 -> 0.0 boundary
        wrap_gap = (1.0 - u_coords[-1]) + u_coords[0]
        
        # If the internal gap is much larger than the wrap gap, 
        # it means the mesh is split by the 0/1 seam.
        if max_gap > wrap_gap:
            for p in uvs:
                if p.x < split_threshold:
                    p.x += 1.0

    # 2. Standard normalization
    min_u = min(p.x for p in uvs)
    max_u = max(p.x for p in uvs)
    min_v = min(p.y for p in uvs)
    max_v = max(p.y for p in uvs)
    
    du = max_u - min_u if max_u != min_u else 1.0
    dv = max_v - min_v if max_v != min_v else 1.0
    
    normalized = [Vector(((p.x - min_u) / du, (p.y - min_v) / dv)) for p in uvs]
    return normalized, (min_u, max_u, min_v, max_v)

def get_projection_score(verts_co: list[Vector], uvs: list[Vector], test_faces: list['BMFaceWrapper'], v_indices: dict[int, int], p_type: UVProjection, avg_norm: Vector):
    """Calculates distortion score for a UV projection. Lower is better."""
    ratios = []
    distortion_penalty = 0
    
    for f in test_faces:
        # Planar check: if a face normal is perpendicular to projection axis, it 'collapses'
        if p_type == UVProjection.PLANAR:
            angle_diff = f.normal.angle(avg_norm)
            # Apply penalty if angle exceeds ~60 degrees
            if angle_diff > 1.05: 
                distortion_penalty += angle_diff * 10 

        for edge in f.edges:
            i1, i2 = v_indices[edge.verts[0].index], v_indices[edge.verts[1].index]
            len_3d = (verts_co[i1] - verts_co[i2]).length
            
            delta_uv: Vector = uvs[i1] - uvs[i2]
            # Handle seam wrapping for spherical/cylindrical
            if p_type in [UVProjection.SPHERICAL, UVProjection.CYLINDRICAL] and abs(delta_uv.x) > 0.5:
                delta_uv.x = 1.0 - abs(delta_uv.x)
            
            len_2d = delta_uv.length
            if len_3d > 1e-6:
                ratios.append(len_2d / len_3d)

    if not ratios: return float('inf')

    avg_ratio = sum(ratios) / len(ratios)
    variance = sum((r - avg_ratio) ** 2 for r in ratios) / len(ratios)
    
    # Final score combines edge length variance and collapse penalties
    return variance + distortion_penalty