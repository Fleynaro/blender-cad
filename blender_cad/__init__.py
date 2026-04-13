import importlib
import sys

# Hot-reload logic: automatically reloads all submodules when the main package is reloaded.
# This is essential for Blender addon development to see changes without restarting Blender.
if __package__ in sys.modules:
    for name in list(sys.modules.keys()):
        if name.startswith(f"{__package__}"):
            importlib.reload(sys.modules[name])

# Common utilities
from .common import Axis, CurveLike
# Low-level BMesh API wrappers for easier interaction with Blender's mesh data
from .bmesh_wrapper import BMEdgeWrapper, BMFaceWrapper, BMVertWrapper, BMeshWrapper
# Core context manager and global selection/state functions (build123d-like workflow)
from .build_part import BuildPart, make_checkpoint, faces, wires, edges, vertices, set_mat, set_default_mat, set_topology, set_mode
# High-level geometric abstractions and topology management
from .geometry import (
    Edge, Face, GeomType, GeometryEntity, 
    Topology, TopologyConfig, Vertex, Wire, uv
)
# Spatial positioning, orientation, and object distribution patterns
from .location import Transform, Location, SurfaceLocation, Pos, Rot, Scale, FlipX, FlipY, FlipZ, Locations, GridLocations, PolarLocations, HexLocations, CurveLocations, align
# Constraint and solver systems
from .solver import Solver, sm, solver
# Joint-like connections between parts
from .joint import Joint
# Material management system for polygonal faces
from .material import mat, BlendMode, build_material, bpy_material_hash
# Boolean operations and geometric modifiers
from .modifiers import Mode, RadialPropEdit, LinearPropEdit, Falloff, add, transform, subdivide, extrude, delete, bevel, mirror, simple_deform, bend, twist
# The main Part class representing a geometric object in the scene
from .part import Part
# Curves and splines
from .curve import Curve, BuildCurve, Line, Polyline, Spline, BezierCurve, TangentArc, RadiusArc, CenterArc, Jiggle, make_curve
# Text
from .text import Text, t
# Advanced collection handling for filtering, sorting, and grouping shapes
from .shape_list import ShapeList, SortBy, GroupBy
# Procedural primitive generators
from .primitives import Box, Plane, Sphere, IcoSphere, Cone, Cylinder, Torus, Grid, Monkey
# Debugging and helper components
from .component import Component, BoxComp, Marker
# Useful helper functions
from .helpers import clear_scene

__all__ = [
    # common
    "Axis",
    "CurveLike",

    # bmesh_wrapper
    "BMEdgeWrapper",
    "BMFaceWrapper",
    "BMVertWrapper",
    "BMeshWrapper",

    # build_part
    "BuildPart",
    "make_checkpoint",
    "faces",
    "wires",
    "edges",
    "vertices",
    "set_mat",
    "set_default_mat",
    "set_topology",
    "set_mode",

    # geometry
    "Edge",
    "Face",
    "GeomType",
    "GeometryEntity",
    "Topology",
    "TopologyConfig",
    "Vertex",
    "Wire",
    "uv",

    # location
    "Transform",
    "Location",
    "SurfaceLocation",
    "Pos",
    "Rot",
    "Scale",
    "FlipX",
    "FlipY",
    "FlipZ",
    "Locations",
    "GridLocations",
    "PolarLocations",
    "HexLocations",
    "CurveLocations",
    "align",

    # solver
    "Solver",
    "sm",
    "solver",

    # joint
    "Joint",

    # material
    "mat",
    "BlendMode",
    "build_material",
    "bpy_material_hash",

    # modifiers
    "Mode",
    "RadialPropEdit",
    "LinearPropEdit",
    "Falloff",
    "add",
    "transform",
    "subdivide",
    "extrude",
    "delete",
    "bevel",
    "mirror",
    "simple_deform",
    "bend",
    "twist",
    
    # part
    "Part",

    # curve
    "Curve",
    "BuildCurve",
    "Line",
    "Polyline",
    "Spline",
    "BezierCurve",
    "TangentArc",
    "RadiusArc",
    "CenterArc",
    "Jiggle",
    "make_curve",

    # text
    "Text",
    "t",

    # shape_list
    "ShapeList",
    "SortBy",
    "GroupBy",

    # primitives
    "Box",
    "Plane",
    "Sphere",
    "IcoSphere",
    "Cone",
    "Cylinder",
    "Torus",
    "Grid",
    "Monkey",

    # component
    "Component",
    "BoxComp",
    "Marker",

    # helpers
    "clear_scene",
]