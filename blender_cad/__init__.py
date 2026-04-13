import importlib
import sys

# Hot-reload logic: automatically reloads all submodules when the main package is reloaded.
# This is essential for Blender addon development to see changes without restarting Blender.
if __package__ in sys.modules:
    for name in list(sys.modules.keys()):
        if name.startswith(f"{__package__}"):
            importlib.reload(sys.modules[name])

# Common utilities
from .common import Axis, AbstractCurve
# Low-level BMesh API wrappers for easier interaction with Blender's mesh data
from .bmesh_wrapper import BMEdgeWrapper, BMFaceWrapper, BMVertWrapper, BMeshWrapper
# Core context manager and global selection/state functions (build123d-like workflow)
from .build_part import BuildPart, make_checkpoint, faces, wires, edges, vertices, set_mat, set_default_mat, set_topology, set_mode, build_part_context, get_tags, add_tags, set_tags, remove_tags
# High-level geometric abstractions and topology management
from .geometry import (
    Edge, Face, GeomType, GeometryEntity, 
    Topology, TopologyConfig, Vertex, Wire, uv
)
# Spatial positioning, orientation, and object distribution patterns
from .location import SVector as Vector, Transform, Location, SurfaceLocation, CurveLocation, Pos, Rot, Scale, Origin, Size, ScaleAlongAxis, SizeAlongAxis, FlipX, FlipY, FlipZ, Locations, GridLocations, PolarLocations, HexLocations, CurveLocations, align
# Constraint and solver systems
from .solver import Solver, sm, solver
# Joint-like connections between parts
from .joint import Joint
# Material management system for polygonal faces
from .material import mat, BlendMode, build_material, bpy_material_hash
# Boolean operations and geometric modifiers
from .modifiers import Mode, WrapMode, LambdaPropEdit, RadialPropEdit, LinearPropEdit, Falloff, make_box_sides_edit, add, transform, subdivide, extrude, solidify_faces, delete, bevel, mirror, simple_deform, bend, twist, wrap
# The main Part class representing a geometric object in the scene
from .part import Part, BoxSetPart
# Curves and splines
from .curve import FillMode, Curve, curve, BuildCurve, Line, Polyline, Spline, BezierCurve, TangentArc, RadiusArc, CenterArc, Jiggle, make_curve
# Text
from .text import Text, t
# Advanced collection handling for filtering, sorting, and grouping shapes
from .shape_list import ShapeList, SortBy, GroupBy
# Procedural primitive generators
from .primitives import Point, Box, Plane, Sphere, IcoSphere, Cone, Cylinder, Torus, Grid, Monkey
# Markup language for 2d modeling
from .ml import ml, MLStyle as style
# Rule-based layout
from .rbl import rl
# Chain-based modeling
from .chain import chain
# Debugging and helper components
from .component import Component, BoxComp, Marker
# Useful helper functions
from .helpers import clear_scene, purge_orphaned_data

__all__ = [
    # common
    "Axis",
    "AbstractCurve",

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
    "build_part_context",
    "get_tags",
    "add_tags",
    "set_tags",
    "remove_tags",

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
    "Vector",
    "Transform",
    "Location",
    "SurfaceLocation",
    "CurveLocation",
    "Pos",
    "Rot",
    "Scale",
    "Origin",
    "Size",
    "ScaleAlongAxis",
    "SizeAlongAxis",
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
    "WrapMode",
    "LambdaPropEdit",
    "RadialPropEdit",
    "LinearPropEdit",
    "Falloff",
    "make_box_sides_edit",
    "add",
    "transform",
    "subdivide",
    "extrude",
    "solidify_faces",
    "delete",
    "bevel",
    "mirror",
    "simple_deform",
    "bend",
    "twist",
    "wrap",
    
    # part
    "Part",
    "BoxSetPart",

    # curve
    "FillMode",
    "Curve",
    "curve",
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
    "Point",
    "Box",
    "Plane",
    "Sphere",
    "IcoSphere",
    "Cone",
    "Cylinder",
    "Torus",
    "Grid",
    "Monkey",

    # ml
    "ml",
    "style",

    # rbl
    "rl",

    # chain
    "chain",

    # component
    "Component",
    "BoxComp",
    "Marker",

    # helpers
    "clear_scene",
    "purge_orphaned_data",
]