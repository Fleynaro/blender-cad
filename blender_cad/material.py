from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, auto
from hashlib import blake2b
import hashlib
import math
import os
from typing import Literal, Mapping, Sequence, Tuple

import bpy
from mathutils import Vector, Color, Euler
from typing_extensions import override

from .location import Location, Rot, Transform


# ---------------------------------------------------------------------------
# Enums & constants
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    """Blender shader node types used by the builder."""
    BSDF = "ShaderNodeBsdfPrincipled"
    MIX = "ShaderNodeMix"
    TEX_IMAGE = "ShaderNodeTexImage"
    MAPPING = "ShaderNodeMapping"
    TEX_COORD = "ShaderNodeTexCoord"
    MATH = "ShaderNodeMath"
    RGB = "ShaderNodeRGB"
    VALUE = "ShaderNodeValue"
    NORMAL_MAP = "ShaderNodeNormalMap"
    COMBINE_XYZ = "ShaderNodeCombineXYZ"
    OUTPUT_MATERIAL = "ShaderNodeOutputMaterial"
    TEX_NOISE = "ShaderNodeTexNoise"
    # Note: In Blender 4.x Musgrave is merged into Noise, but still available as separate node for compatibility
    TEX_MUSGRAVE = "ShaderNodeTexMusgrave"
    TEX_VORONOI = "ShaderNodeTexVoronoi"
    COLOR_RAMP = "ShaderNodeValToRGB"
    MAP_RANGE = "ShaderNodeMapRange"
    LAYER_WEIGHT = "ShaderNodeLayerWeight"
    VAL_TO_RGB = "ShaderNodeValToRGB"
    BUMP = "ShaderNodeBump"
    TEX_WAVE = "ShaderNodeTexWave"
    TEX_BRICK = "ShaderNodeTexBrick"


class SocketType(str, Enum):
    """Supported node socket data types."""
    FLOAT = "FLOAT"
    VECTOR = "VECTOR"
    RGBA = "RGBA"


class ValueKind(str, Enum):
    """Unified value kinds used across expressions and channels."""
    FLOAT = "FLOAT"
    VECTOR = "VECTOR"
    RGBA = "RGBA"


class Channel(Enum):
    BASE_COLOR = auto()
    METALLIC = auto()
    ROUGHNESS = auto()
    SPECULAR = auto()
    NORMAL = auto()
    EMISSION_COLOR = auto()
    EMISSION_STRENGTH = auto()
    ALPHA = auto()
    AO = auto()


class BlendMode(Enum):
    # Standard Mix
    MIX = auto()
    # Darken
    DARKEN = auto()
    MULTIPLY = auto()
    BURN = auto()
    LINEAR_BURN = auto()
    # Lighten
    LIGHTEN = auto()
    SCREEN = auto()
    DODGE = auto()
    ADD = auto()
    # Contrast
    OVERLAY = auto()
    SOFT_LIGHT = auto()
    LINEAR_LIGHT = auto()
    VIVID_LIGHT = auto()
    HARD_LIGHT = auto()
    PIN_LIGHT = auto()
    # Inversion/Difference
    DIFFERENCE = auto()
    EXCLUSION = auto()
    SUBTRACT = auto()
    DIVIDE = auto()
    # Color/HSL
    HUE = auto()
    SATURATION = auto()
    COLOR = auto()
    VALUE = auto()


class CoordType(Enum):
    UV = auto()
    OBJECT = auto()
    GENERATED = auto()
    NORMAL = auto()
    CAMERA = auto()
    WINDOW = auto()
    TRIPLANAR = auto()


BPY_MAT_HASH_PROP = "pbr_layer_hash"
FLOAT_EPS = 1e-6


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    socket_name: str
    kind: ValueKind


CHANNEL_SPECS: dict[str, ChannelSpec] = {
    "base_color": ChannelSpec("Base Color", ValueKind.RGBA),
    "metallic": ChannelSpec("Metallic", ValueKind.FLOAT),
    "roughness": ChannelSpec("Roughness", ValueKind.FLOAT),
    "specular": ChannelSpec("Specular IOR Level", ValueKind.FLOAT),
    "ior": ChannelSpec("IOR", ValueKind.FLOAT),
    "transmission": ChannelSpec("Transmission Weight", ValueKind.FLOAT),
    "normal": ChannelSpec("Normal", ValueKind.RGBA),
    "emission_color": ChannelSpec("Emission Color", ValueKind.RGBA),
    "emission_strength": ChannelSpec("Emission Strength", ValueKind.FLOAT),
    "alpha": ChannelSpec("Alpha", ValueKind.FLOAT),
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _kind_to_socket_type(kind: ValueKind) -> SocketType:
    if kind == ValueKind.FLOAT:
        return SocketType.FLOAT
    if kind == ValueKind.VECTOR:
        return SocketType.VECTOR
    return SocketType.RGBA


def _is_socket(value: object) -> bool:
    return isinstance(value, bpy.types.NodeSocket)


def _as_vector3(value: object) -> Vector:
    """Convert a value to a 3D mathutils.Vector."""
    if isinstance(value, Euler):
        return _as_vector3([a for a in value])
    
    if isinstance(value, Vector):
        if len(value) >= 3:
            return Vector((float(value[0]), float(value[1]), float(value[2])))
        if len(value) == 1:
            v = float(value[0])
            return Vector((v, v, v))
        raise TypeError(f"Cannot convert Vector of length {len(value)} to Vector3")

    if isinstance(value, (tuple, list)):
        if len(value) >= 3:
            return Vector((float(value[0]), float(value[1]), float(value[2])))
        if len(value) == 1:
            v = float(value[0])
            return Vector((v, v, v))

    if isinstance(value, (int, float)):
        v = float(value)
        return Vector((v, v, v))

    raise TypeError(f"Cannot convert to Vector3: {value!r}")


def _as_vector4(value: object) -> Vector:
    """Convert a value to a 4D vector so colors and RGBA stay unified."""
    if isinstance(value, Vector):
        if len(value) >= 4:
            return Vector((float(value[0]), float(value[1]), float(value[2]), float(value[3])))
        if len(value) == 3:
            return Vector((float(value[0]), float(value[1]), float(value[2]), 1.0))
        if len(value) == 1:
            v = float(value[0])
            return Vector((v, v, v, 1.0))
        raise TypeError(f"Cannot convert Vector of length {len(value)} to Vector4")

    if isinstance(value, (tuple, list)):
        if len(value) >= 4:
            return Vector((float(value[0]), float(value[1]), float(value[2]), float(value[3])))
        if len(value) == 3:
            return Vector((float(value[0]), float(value[1]), float(value[2]), 1.0))
        if len(value) == 1:
            v = float(value[0])
            return Vector((v, v, v, 1.0))

    if isinstance(value, (int, float)):
        v = float(value)
        return Vector((v, v, v, 1.0))

    raise TypeError(f"Cannot convert to Vector4: {value!r}")


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Vector) and len(value) > 0:
        return float(value[0])
    if isinstance(value, (tuple, list)) and len(value) > 0:
        return float(value[0])
    raise TypeError(f"Cannot convert to float: {value!r}")


def _stable_any(value: object) -> object:
    """Create a stable, hashable representation for arbitrary values."""
    if hasattr(value, "stable_key"):
        return value.stable_key()  # type: ignore[no-any-return]
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    if isinstance(value, Vector):
        return tuple(round(float(v), 8) for v in value)
    if isinstance(value, Transform):
        return _stable_any(value.values)
    if isinstance(value, (tuple, list)):
        return tuple(_stable_any(v) for v in value)
    if isinstance(value, Mapping):
        return tuple(sorted((k, _stable_any(v)) for k, v in value.items()))
    if hasattr(value, "name") and isinstance(getattr(value, "name"), str):
        return f"REF_{getattr(value, 'name')}"
    rep = repr(value)
    if " at 0x" in rep:
        return type(value).__name__
    return rep


def _is_plain_value(value: object) -> bool:
    return isinstance(value, (int, float, tuple, list, Vector))


def _scalarize(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Vector):
        return float(value[0]) if len(value) else 0.0
    if isinstance(value, (tuple, list)) and value:
        return float(value[0])
    return 0.0


# ---------------------------------------------------------------------------
# Unified value expression system
# ---------------------------------------------------------------------------

class MaterialValue(ABC):
    """Base contract for values that can be used in material channels."""

    kind: ValueKind

    @abstractmethod
    def stable_key(self) -> tuple[object, ...]:
        raise NotImplementedError

    @abstractmethod
    def _build_impl(self, ctx: "BuildMaterial") -> bpy.types.NodeSocket | float | Vector:
        raise NotImplementedError

    def build(self, ctx: "BuildMaterial") -> bpy.types.NodeSocket | float | Vector:
        key = self.stable_key()
        cached = ctx._expr_cache.get(key)
        if cached is not None:
            return cached

        result = self._build_impl(ctx)
        ctx._expr_cache[key] = result
        return result
    
    def __mul__(self, other: object):
        return BinaryExpr(self, MathOp.MULTIPLY, other)
    
    __rmul__ = __mul__

    def __truediv__(self, other: object): return BinaryExpr(self, MathOp.DIVIDE, other)

    def __add__(self, other: object):
        return BinaryExpr(self, MathOp.ADD, other)
    
    __radd__ = __add__

    def __sub__(self, other: object):
        return BinaryExpr(self, MathOp.SUBTRACT, other)
    
    def __rsub__(self, other: object):
        return BinaryExpr(other, MathOp.SUBTRACT, self)
    
    def __lt__(self, other: object):
        return BinaryExpr(self, MathOp.LESS_THAN, other)

    def __gt__(self, other: object):
        return BinaryExpr(self, MathOp.GREATER_THAN, other)

    def max(self, other: object):
        """Returns the maximum of self and other."""
        return BinaryExpr(self, MathOp.MAXIMUM, other)

    def min(self, other: object):
        """Returns the minimum of self and other."""
        return BinaryExpr(self, MathOp.MINIMUM, other)


class MathOp(str, Enum):
    """Blender ShaderNodeMath operations."""
    # Arithmetic
    ADD = "ADD"
    SUBTRACT = "SUBTRACT"
    MULTIPLY = "MULTIPLY"
    DIVIDE = "DIVIDE"
    MULTIPLY_ADD = "MULTIPLY_ADD"
    # Comparison
    LESS_THAN = "LESS_THAN"
    GREATER_THAN = "GREATER_THAN"
    COMPARE = "COMPARE"
    # Functions
    POWER = "POWER"
    LOGARITHM = "LOGARITHM"
    SQRT = "SQRT"
    ABSOLUTE = "ABSOLUTE"
    EXPONENT = "EXPONENT"
    # Trig
    SINE = "SINE"
    COSINE = "COSINE"
    TANGENT = "TANGENT"
    # Rounding
    FLOOR = "FLOOR"
    CEIL = "CEIL"
    FRACTION = "FRACTION"
    MODULO = "MODULO"
    WRAP = "WRAP"
    SNAP = "SNAP"
    # Clamping/Selection
    MINIMUM = "MINIMUM"
    MAXIMUM = "MAXIMUM"
    PINGPONG = "PINGPONG"
    SMOOTH_MIN = "SMOOTH_MIN"
    SMOOTH_MAX = "SMOOTH_MAX"


@dataclass(frozen=True, slots=True)
class BinaryExpr(MaterialValue):
    """Binary expression for scalar math and factor stacking."""

    left: object
    op: MathOp
    right: object
    kind: ValueKind = ValueKind.FLOAT

    @override
    def stable_key(self) -> tuple[object, ...]:
        return ("expr", self.kind.value, self.op, _stable_any(self.left), _stable_any(self.right))

    @override
    def _build_impl(self, ctx: "BuildMaterial") -> bpy.types.NodeSocket | float | Vector:
        left_val = ctx.resolve(self.left, self.kind)
        right_val = ctx.resolve(self.right, self.kind)

        if _is_socket(left_val) or _is_socket(right_val):
            node: bpy.types.ShaderNodeMath = ctx.new_node(NodeType.MATH, label=f"Expr {self.op.value}")
            node.operation = self.op.value
            ctx._apply_to_socket(node.inputs[0], left_val, self.kind)
            ctx._apply_to_socket(node.inputs[1], right_val, self.kind)
            result = node.outputs[0]
        else:
            result = _python_math(left_val, right_val, self.op)
        return result


@dataclass(frozen=True, slots=True)
class VariableExpr(MaterialValue):
    """A user-tweakable node-backed variable with a default value."""

    name: str
    default: object
    kind: ValueKind = ValueKind.FLOAT

    @override
    def stable_key(self) -> tuple[object, ...]:
        return ("var", self.name, self.kind.value, _stable_any(self.default))

    @override
    def _build_impl(self, ctx: "BuildMaterial") -> bpy.types.NodeSocket | float | Vector:
        if self.kind == ValueKind.FLOAT:
            node: bpy.types.ShaderNodeValue = ctx.new_node(NodeType.VALUE, label=f"Var {self.name}")
            node.outputs[0].default_value = _as_float(self.default)
            result = node.outputs[0]
        elif self.kind == ValueKind.VECTOR:
            node: bpy.types.ShaderNodeCombineXYZ = ctx.new_node(NodeType.COMBINE_XYZ, label=f"Var {self.name}")
            vec = _as_vector3(self.default)
            node.inputs[0].default_value = vec.x
            node.inputs[1].default_value = vec.y
            node.inputs[2].default_value = vec.z
            result = node.outputs[0]
        else:
            node: bpy.types.ShaderNodeRGB = ctx.new_node(NodeType.RGB, label=f"Var {self.name}")
            node.outputs[0].default_value = _as_vector4(self.default)
            result = node.outputs[0]
        return result


@dataclass(frozen=True, slots=True)
class SocketExpr(MaterialValue):
    """
    Wraps a raw Blender NodeSocket to allow using it back in expressions.
    Caching is disabled by using a unique ID in the stable_key.
    """
    socket: bpy.types.NodeSocket
    kind: ValueKind = ValueKind.FLOAT

    @override
    def stable_key(self) -> tuple[object, ...]:
        # Using id(self) to ensure this specific wrapper instance is unique 
        # and doesn't collide in the ctx._expr_cache
        return ("socket_wrapper", id(self))

    @override
    def _build_impl(self, ctx: "BuildMaterial") -> bpy.types.NodeSocket:
        return self.socket


@dataclass(frozen=True, slots=True)
class MixExpr(MaterialValue):
    """Expression wrapper for Blender's Mix Node (Blender 4.x+)."""
    a: object
    b: object
    factor: object
    mode: BlendMode = BlendMode.MIX
    kind: ValueKind = ValueKind.FLOAT

    @override
    def stable_key(self) -> tuple[object, ...]:
        return ("mix_expr", _stable_any(self.a), _stable_any(self.b), _stable_any(self.factor), self.mode.name, self.kind.value)

    @override
    def _build_impl(self, ctx: "BuildMaterial") -> bpy.types.NodeSocket:
        # This is where the actual Blender node is created
        mix: bpy.types.ShaderNodeMix = ctx.new_node(NodeType.MIX, label=f"Mix {self.mode.name}")
        mix.data_type = _kind_to_socket_type(self.kind).value
        if hasattr(mix, "blend_type"):
            mix.blend_type = self.mode.name

        factor = self.factor
        if self.mode == BlendMode.MIX and isinstance(self.b, BaseTexture):
            factor *= self.b.alpha

        ctx._apply_to_socket(mix.inputs["A"], self.a, self.kind)
        ctx._apply_to_socket(mix.inputs["B"], self.b, self.kind)
        ctx._apply_to_socket(mix.inputs["Factor"], factor, ValueKind.FLOAT)
        return mix.outputs["Result"]


@dataclass(frozen=True, slots=True)
class MappingSettings:
    """Texture mapping settings used by image-based inputs."""

    transform: Transform = field(default_factory=lambda: Transform())
    coord_type: CoordType = CoordType.OBJECT
    use_triplanar: bool = False
    normal_space: str = "OBJECT"

    def stable_key(self) -> tuple[object, ...]:
        return (
            self.coord_type.name,
            _stable_any(self.transform),
            self.use_triplanar,
            self.normal_space,
        )


@dataclass(frozen=True, slots=True)
class MappingExpr(MaterialValue):
    """Expression for generating and transforming texture coordinates."""
    settings: MappingSettings = field(default_factory=MappingSettings)

    @override
    def stable_key(self) -> tuple[object, ...]:
        return ("mapping", self.settings.stable_key())

    @override
    def _build_impl(self, ctx: "BuildMaterial") -> bpy.types.NodeSocket:
        # 1. Coordinate Source
        coord_node: bpy.types.ShaderNodeTexCoord = ctx.new_node(NodeType.TEX_COORD)
        coord_output_map = {
            CoordType.UV: "UV",
            CoordType.OBJECT: "Object",
            CoordType.GENERATED: "Generated",
            CoordType.NORMAL: "Normal",
            CoordType.CAMERA: "Camera",
            CoordType.WINDOW: "Window",
        }
        output_name = coord_output_map.get(self.settings.coord_type, "Object")
        
        # 2. Mapping Transformation
        mapping_node: bpy.types.ShaderNodeMapping = ctx.new_node(NodeType.MAPPING)
        ctx.links.new(coord_node.outputs[output_name], mapping_node.inputs["Vector"])
        
        tr = self.settings.transform
        ctx._apply_to_socket(mapping_node.inputs["Location"], tr.position, kind=ValueKind.VECTOR)
        ctx._apply_to_socket(mapping_node.inputs["Rotation"], tr.euler_rad, kind=ValueKind.VECTOR)
        ctx._apply_to_socket(mapping_node.inputs["Scale"], tr.scale, kind=ValueKind.VECTOR)

        return mapping_node.outputs["Vector"]
    

@dataclass(frozen=True)
class BaseTexture(ABC):
    """Base class for texture descriptors."""
    mapping: MappingSettings | None = None
    label: str | None = None

    @property
    def alpha(self):
        """Helper property to get the alpha channel (mask) of a texture."""
        return TextureExpr(self, kind=ValueKind.FLOAT)

    def stable_key(self) -> tuple[object, ...]:
        return (self.mapping.stable_key() if self.mapping else None, self.label)
    
    @abstractmethod
    def build_image(self) -> bpy.types.Image:
        raise NotImplementedError
    
    @abstractmethod
    def get_label(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class Texture(BaseTexture):
    """Image texture descriptor."""
    image_path: str = ''

    @override
    def stable_key(self) -> tuple[object, ...]:
        return ("texture", self.image_path, super().stable_key())
    
    @override
    def build_image(self) -> bpy.types.Image:
        """Loads image from disk."""
        return _build_or_get_image(self.image_path)
    
    @override
    def get_label(self) -> str:
        return self.label or self.image_path.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class CameraTexture(BaseTexture):
    """
    A texture generated by rendering the scene from a temporary camera.
    Uses a Location object to define the camera's transform.
    """
    location: 'Location' = field(default_factory=lambda: Location())
    fov: float = 39.6
    resolution: tuple[int, int] = (1024, 1024)

    _cached_image: bpy.types.Image | None = field(default=None, init=False, repr=False, compare=False)

    @override
    def stable_key(self) -> tuple[object, ...]:
        return (
            "camera_texture",
            _stable_any(self.location),
            self.fov,
            self.resolution,
            super().stable_key(),
        )
    
    @override
    def build_image(self) -> bpy.types.Image:
        """Renders image from camera."""
        if self._cached_image is not None:
            try:
                self._cached_image.name
                return self._cached_image
            except ReferenceError:
                object.__setattr__(self, "_cached_image", None)
        image = _render_camera_to_image(self.location, self.fov, self.resolution, self.get_label())
        object.__setattr__(self, "_cached_image", image)
        return image
    
    @override
    def get_label(self) -> str:
        return self.label or f"camera_{_material_hash(self)[:8]}"


@dataclass(frozen=True, slots=True)
class TextureExpr(MaterialValue):
    """Texture-backed expression used in channel blending."""

    texture: Texture
    kind: ValueKind = ValueKind.RGBA

    @override
    def stable_key(self) -> tuple[object, ...]:
        return ("texexpr", self.texture.stable_key(), self.kind.value)

    @override
    def _build_impl(self, ctx: "BuildMaterial") -> bpy.types.NodeSocket:
        key = ("texexpr", self.texture.stable_key())
        img_node = ctx._expr_cache.get(key)
        if img_node is None:
            img_node: bpy.types.ShaderNodeTexImage = ctx.new_node(NodeType.TEX_IMAGE, label=self.texture.get_label())
            img_node.image = self.texture.build_image()

            if self.texture.mapping is not None:
                ctx._apply_to_socket(img_node.inputs["Vector"], self.texture.mapping)
            ctx._expr_cache[key] = img_node

        result = img_node.outputs["Color"] if self.kind != ValueKind.FLOAT else img_node.outputs["Alpha"]
        return result


@dataclass(frozen=True, slots=True)
class NoiseExpr(MaterialValue):
    """Procedural Noise Texture expression."""
    scale: object = 5.0
    detail: object = 2.0
    roughness: object = 0.5
    distortion: object = 0.0
    mapping: MappingSettings | MappingExpr | None = None
    kind: ValueKind = ValueKind.FLOAT # FLOAT for factor output, RGBA for color output

    @override
    def stable_key(self) -> tuple[object, ...]:
        return ("noise", _stable_any(self.scale), _stable_any(self.detail), _stable_any(self.roughness), _stable_any(self.distortion), _stable_any(self.mapping), self.kind.value)

    @override
    def _build_impl(self, ctx: "BuildMaterial") -> bpy.types.NodeSocket:
        node: bpy.types.ShaderNodeTexNoise = ctx.new_node(NodeType.TEX_NOISE)
        ctx._apply_to_socket(node.inputs["Scale"], self.scale)
        ctx._apply_to_socket(node.inputs["Detail"], self.detail)
        ctx._apply_to_socket(node.inputs["Roughness"], self.roughness)
        ctx._apply_to_socket(node.inputs["Distortion"], self.distortion)
        if self.mapping:
            ctx._apply_to_socket(node.inputs["Vector"], self.mapping)
        return node.outputs["Fac"] if self.kind == ValueKind.FLOAT else node.outputs["Color"]


@dataclass(frozen=True, slots=True)
class VoronoiExpr(MaterialValue):
    """Voronoi Texture for cracks (Distance to Edge)."""
    scale: object = 5.0
    randomness: object = 1.0
    mapping: MappingSettings | MappingExpr | None = None
    kind: ValueKind = ValueKind.FLOAT

    @override
    def stable_key(self) -> tuple[object, ...]:
        return ("voronoi", _stable_any(self.scale), _stable_any(self.randomness), _stable_any(self.mapping), self.kind.value)

    @override
    def _build_impl(self, ctx: "BuildMaterial") -> bpy.types.NodeSocket:
        node: bpy.types.ShaderNodeTexVoronoi = ctx.new_node(NodeType.TEX_VORONOI)
        # Blender 4.x: Distance to Edge is the standard for cracks
        node.feature = 'DISTANCE_TO_EDGE' 
        ctx._apply_to_socket(node.inputs["Scale"], self.scale)
        ctx._apply_to_socket(node.inputs["Randomness"], self.randomness)
        if self.mapping:
            ctx._apply_to_socket(node.inputs["Vector"], self.mapping)
        return node.outputs["Distance"]


@dataclass(frozen=True, slots=True)
class WaveExpr(MaterialValue):
    """Wave Texture for wood rings and fabric patterns."""
    wave_type: str = 'BANDS'  # 'BANDS' or 'RINGS'
    scale: object = 5.0
    distortion: object = 0.0
    detail: object = 2.0
    detail_scale: object = 1.0
    mapping: MappingSettings | MappingExpr | None = None
    kind: ValueKind = ValueKind.FLOAT

    @override
    def stable_key(self) -> tuple[object, ...]:
        return ("wave", self.wave_type, _stable_any(self.scale), _stable_any(self.distortion), _stable_any(self.detail), _stable_any(self.detail_scale), _stable_any(self.mapping), self.kind.value)

    @override
    def _build_impl(self, ctx: "BuildMaterial") -> bpy.types.NodeSocket:
        node: bpy.types.ShaderNodeTexWave = ctx.new_node(NodeType.TEX_WAVE)
        node.wave_type = self.wave_type
        ctx._apply_to_socket(node.inputs["Scale"], self.scale)
        ctx._apply_to_socket(node.inputs["Distortion"], self.distortion)
        ctx._apply_to_socket(node.inputs["Detail"], self.detail)
        ctx._apply_to_socket(node.inputs["Detail Scale"], self.detail_scale)
        if self.mapping:
            ctx._apply_to_socket(node.inputs["Vector"], self.mapping)
        return node.outputs["Fac"] if self.kind == ValueKind.FLOAT else node.outputs["Color"]


@dataclass(frozen=True, slots=True)
class BrickExpr(MaterialValue):
    """Brick Texture for walls and paths."""
    color1: object = (0.5, 0.25, 0.15, 1.0)
    color2: object = (0.2, 0.1, 0.05, 1.0)
    mortar: object = (0.8, 0.8, 0.8, 1.0)
    scale: object = 5.0
    mortar_size: object = 0.02
    mortar_smooth: object = 0.1
    mapping: MappingSettings | MappingExpr | None = None
    kind: ValueKind = ValueKind.RGBA

    @override
    def stable_key(self) -> tuple[object, ...]:
        return ("brick", _stable_any(self.color1), _stable_any(self.color2), _stable_any(self.mortar), _stable_any(self.scale), _stable_any(self.mortar_size), _stable_any(self.mapping), self.kind.value)

    @override
    def _build_impl(self, ctx: "BuildMaterial") -> bpy.types.NodeSocket:
        node: bpy.types.ShaderNodeTexBrick = ctx.new_node(NodeType.TEX_BRICK)
        ctx._apply_to_socket(node.inputs["Color1"], self.color1)
        ctx._apply_to_socket(node.inputs["Color2"], self.color2)
        ctx._apply_to_socket(node.inputs["Mortar"], self.mortar)
        ctx._apply_to_socket(node.inputs["Scale"], self.scale)
        ctx._apply_to_socket(node.inputs["Mortar Size"], self.mortar_size)
        ctx._apply_to_socket(node.inputs["Mortar Smooth"], self.mortar_smooth)
        if self.mapping:
            ctx._apply_to_socket(node.inputs["Vector"], self.mapping)
        return node.outputs["Color"] if self.kind == ValueKind.RGBA else node.outputs["Fac"]


@dataclass(frozen=True, slots=True)
class ColorRampExpr(MaterialValue):
    """ColorRamp (ValToRGB) expression for remapping values."""
    input_value: object
    # Stops as tuple of (position, color_rgba_tuple)
    stops: tuple[tuple[float, tuple[float, float, float, float]], ...] = ((0.0, (0,0,0,1)), (1.0, (1,1,1,1)))
    interpolation: str = 'LINEAR'
    kind: ValueKind = ValueKind.RGBA

    @override
    def stable_key(self) -> tuple[object, ...]:
        return ("colorramp", _stable_any(self.input_value), self.stops, self.interpolation, self.kind.value)

    @override
    def _build_impl(self, ctx: "BuildMaterial") -> bpy.types.NodeSocket:
        node: bpy.types.ShaderNodeValToRGB = ctx.new_node(NodeType.COLOR_RAMP)
        elements = node.color_ramp.elements
        elements.remove(elements[1]) # Clear default stops
        elements[0].position = self.stops[0][0]
        elements[0].color = self.stops[0][1]
        
        for pos, color in self.stops[1:]:
            stop = elements.new(pos)
            stop.color = color
            
        node.color_ramp.interpolation = self.interpolation
        ctx._apply_to_socket(node.inputs[0], self.input_value)
        return node.outputs["Color"] if self.kind == ValueKind.RGBA else node.outputs["Alpha"]


@dataclass(frozen=True, slots=True)
class MapRangeExpr(MaterialValue):
    """Map Range node for precision control over procedural masks."""
    input_value: object
    from_min: float = 0.0
    from_max: float = 1.0
    to_min: float = 0.0
    to_max: float = 1.0
    kind: ValueKind = ValueKind.FLOAT

    @override
    def stable_key(self) -> tuple[object, ...]:
        return ("map_range", _stable_any(self.input_value), self.from_min, self.from_max, self.to_min, self.to_max, self.kind.value)

    @override
    def _build_impl(self, ctx: "BuildMaterial") -> bpy.types.NodeSocket:
        node: bpy.types.ShaderNodeMapRange = ctx.new_node(NodeType.MAP_RANGE)
        ctx._apply_to_socket(node.inputs[0], self.input_value)
        node.inputs[1].default_value = self.from_min
        node.inputs[2].default_value = self.from_max
        node.inputs[3].default_value = self.to_min
        node.inputs[4].default_value = self.to_max
        return node.outputs[0]


@dataclass(frozen=True, slots=True)
class BumpExpr(MaterialValue):
    """Converts a height value (crack mask) into a Normal vector using Bump node."""
    height: object
    strength: object = 1.0
    distance: object = 0.1
    kind: ValueKind = ValueKind.VECTOR

    @override
    def stable_key(self) -> tuple[object, ...]:
        return ("bump", _stable_any(self.height), _stable_any(self.strength), _stable_any(self.distance), self.kind.value)

    @override
    def _build_impl(self, ctx: "BuildMaterial") -> bpy.types.NodeSocket:
        node: bpy.types.ShaderNodeBump = ctx.new_node(NodeType.BUMP)
        ctx._apply_to_socket(node.inputs["Height"], self.height)
        ctx._apply_to_socket(node.inputs["Strength"], self.strength)
        ctx._apply_to_socket(node.inputs["Distance"], self.distance)
        return node.outputs["Normal"]

# ---------------------------------------------------------------------------
# Material layers
# ---------------------------------------------------------------------------

class MaterialExpr(ABC):
    """Base class for composable material expressions."""
    
    def stable_key(self) -> tuple[object, ...]:
        if is_dataclass(self):
            return (self.__class__.__name__,) + tuple(_stable_any(getattr(self, f.name)) for f in fields(self))
        return (self.__class__.__name__,) + tuple(
            _stable_any(v) for k, v in self.__dict__.items() if not k.startswith("_")
        )

    @abstractmethod
    def build(self, ctx: "BuildMaterial") -> None:
        raise NotImplementedError

    def __call__(self, ctx: "BuildMaterial") -> "MaterialExpr":
        self.build(ctx)
        return self

    def __add__(self, other: "MaterialExpr") -> "MaterialCompositeLayer":
        return MaterialCompositeLayer.from_terms([
            MaterialWeightedLayer(self, 1.0),
            MaterialWeightedLayer(other, 1.0),
        ])

    def __sub__(self, other: "MaterialExpr") -> "MaterialCompositeLayer":
        return MaterialCompositeLayer.from_terms([
            MaterialWeightedLayer(self, 1.0),
            MaterialWeightedLayer(other, -1.0),
        ])

    def __mul__(self, factor: object) -> "MaterialWeightedLayer":
        return MaterialWeightedLayer(self, factor)

    def __rmul__(self, factor: object) -> "MaterialWeightedLayer":
        return MaterialWeightedLayer(self, factor)


class MaterialLayer(MaterialExpr):
    """Base class for all material layers."""

    @override
    def build(self, ctx: "BuildMaterial") -> None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class MaterialWeightedLayer(MaterialLayer):
    """A layer multiplied by a factor."""

    layer: MaterialExpr
    factor: object = 1.0

    @override
    def build(self, ctx: "BuildMaterial") -> None:
        ctx.push_factor(self.factor)
        try:
            self.layer.build(ctx)
        finally:
            ctx.pop_factor()


@dataclass(frozen=True, slots=True)
class MaterialCompositeLayer(MaterialLayer):
    """An additive composition of layers with canonicalization."""

    terms: tuple[MaterialWeightedLayer, ...]
    op: str = "add"

    @classmethod
    def from_terms(cls, terms: Sequence[MaterialWeightedLayer], op: str = "add") -> "MaterialCompositeLayer":
        flat: list[MaterialWeightedLayer] = []
        for term in terms:
            if isinstance(term.layer, MaterialCompositeLayer):
                flat.extend(term.layer.terms)
            else:
                flat.append(term)
        return cls(tuple(flat), op=op)

    @override
    def __add__(self, other: MaterialExpr) -> "MaterialCompositeLayer":
        return MaterialCompositeLayer.from_terms(self.terms + (_as_weighted(other, 1.0),))

    @override
    def __sub__(self, other: MaterialExpr) -> "MaterialCompositeLayer":
        return MaterialCompositeLayer.from_terms(self.terms + (_as_weighted(other, -1.0),))

    @override
    def __mul__(self, factor: object) -> "MaterialCompositeLayer":
        return MaterialCompositeLayer(
            tuple(MaterialWeightedLayer(MaterialWeightedLayer(term.layer, term.factor), factor) for term in self.terms),
            op=self.op,
        )

    @override
    def build(self, ctx: "BuildMaterial") -> None:
        for term in self.terms:
            term.build(ctx)


# ---------------------------------------------------------------------------
# Channel containers
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PBRChannels:
    """Typed PBR channel container."""

    base_color: object | None = Vector((1.0, 1.0, 1.0, 1.0))
    metallic: object | None = 0.0
    roughness: object | None = 0.5
    specular: object | None = 0.5
    ior: object | None = 1.5
    transmission: object | None = 0.0
    normal: object | None = None
    emission_color: object | None = Vector((0.0, 0.0, 0.0, 1.0))
    emission_strength: object | None = 0.0
    alpha: object | None = 1.0
    ao: object | None = 1.0

    def as_key(self) -> tuple[object, ...]:
        return (
            _stable_any(self.base_color),
            _stable_any(self.metallic),
            _stable_any(self.roughness),
            _stable_any(self.specular),
            _stable_any(self.ior),
            _stable_any(self.transmission),
            _stable_any(self.normal),
            _stable_any(self.emission_color),
            _stable_any(self.emission_strength),
            _stable_any(self.alpha),
            _stable_any(self.ao),
        )


@dataclass(slots=True)
class MaterialMeta:
    """Mutable metadata collected from layers."""

    name: str | None = None
    custom: dict[str, object] = field(default_factory=dict)

    def as_key(self) -> tuple[object, ...]:
        custom_items = tuple(sorted((k, _stable_any(v)) for k, v in self.custom.items()))
        return (self.name, custom_items)


# ---------------------------------------------------------------------------
# Build context
# ---------------------------------------------------------------------------

class BuildMaterial:
    """Material build context."""

    var = VariableExpr
    socket = SocketExpr
    mapping = MappingSettings
    tex = TextureExpr
    noise = NoiseExpr
    voronoi = VoronoiExpr
    color_ramp = ColorRampExpr
    map_range = MapRangeExpr
    bump = BumpExpr
    wave = WaveExpr
    brick = BrickExpr

    def __init__(self, material: bpy.types.Material) -> None:
        self.material = material
        self.material.use_nodes = True
        self.node_tree = material.node_tree
        assert self.node_tree is not None

        self.nodes = self.node_tree.nodes
        self.links = self.node_tree.links
        self.nodes.clear()

        self.channels = PBRChannels()
        self.meta = MaterialMeta()
        
        self._factor_stack: list[object] = []
        self._expr_cache: dict[tuple[object, ...], bpy.types.Node, bpy.types.NodeSocket | float | Vector] = {}

        self._node_x = -300
        self._node_y = 0
        self._bsdf_node: bpy.types.Node | None = None

    @property
    def factor(self) -> object:
        """Return aggregated factor with constant folding across the stack."""
        if not self._factor_stack:
            return 1.0

        # Separate items into groups by type to maximize folding potential
        # We use a dictionary where key is the type and value is the accumulated product
        folded_groups: dict[type, object] = {}
        non_foldable: list[object] = []

        for val in self._factor_stack:
            val_type = type(val)

            if isinstance(val, (int, float, complex)) or hasattr(val, "__mul__") or hasattr(val, "__rmul__"):
                folded_groups[val_type] = folded_groups[val_type] * val if val_type in folded_groups else val
            else:
                non_foldable.append(val)

        result_val: object = None

        for folded_val in folded_groups.values():
            if folded_val == 1.0:
                continue
            if result_val is None:
                result_val = folded_val
            else:
                result_val = BinaryExpr(left=result_val, op=MathOp.MULTIPLY, right=folded_val)

        for expr in non_foldable:
            if result_val is None:
                result_val = expr
            else:
                result_val = BinaryExpr(left=result_val, op=MathOp.MULTIPLY, right=expr)

        if result_val is None:
            return 1.0

        return result_val

    def push_factor(self, factor: object) -> None:
        self._factor_stack.append(factor)

    def pop_factor(self) -> None:
        self._factor_stack.pop()

    def new_node(
        self,
        node_type: NodeType | str,
        label: str | None = None,
        *,
        x: int | None = None,
        y: int | None = None,
    ) -> bpy.types.Node:
        """Create a new node with a simple grid-like auto layout."""
        type_str = node_type.value if isinstance(node_type, NodeType) else node_type
        node = self.nodes.new(type_str)

        if label is not None:
            node.label = label

        if x is not None and y is not None:
            node.location = (x, y)
        else:
            node.location = (self._node_x, self._node_y)
            self._node_y -= 300
            if self._node_y < -1500:
                self._node_y = 0
                self._node_x -= 300
        return node

    def resolve(self, value: object, kind: ValueKind = ValueKind.FLOAT) -> bpy.types.NodeSocket | float | Vector:
        """Resolve literals, expressions, and Blender-backed values."""
        if value is None:
            return 0.0 if kind == ValueKind.FLOAT else Vector((0.0, 0.0, 0.0, 1.0))
        if isinstance(value, MaterialValue):
            return value.build(self)
        if isinstance(value, MappingSettings):
            return MappingExpr(value).build(self)
        if isinstance(value, BaseTexture):
            return TextureExpr(value, kind=kind).build(self)
        if _is_socket(value):
            return value

        if kind == ValueKind.FLOAT:
            return _as_float(value)
        if kind == ValueKind.VECTOR:
            return _as_vector3(value)
        return _as_vector4(value)

    def _socket_default_kind(self, socket: bpy.types.NodeSocket) -> ValueKind:
        if hasattr(socket, "default_value"):
            default = socket.default_value
            if isinstance(default, (float, int)):
                return ValueKind.FLOAT
            if hasattr(default, "__len__") and len(default) == 3:
                return ValueKind.VECTOR
        return ValueKind.RGBA

    def _apply_to_socket(
        self,
        socket: bpy.types.NodeSocket,
        value: object,
        kind: ValueKind | None = None,
    ) -> None:
        """Apply a value to a socket, linking when possible and otherwise setting defaults."""
        resolved_kind = kind or self._socket_default_kind(socket)
        resolved = self.resolve(value, resolved_kind)

        if _is_socket(resolved):
            self.links.new(resolved, socket)
            return

        if not hasattr(socket, "default_value"):
            return

        if isinstance(resolved, (int, float)):
            socket.default_value = float(resolved)
            return

        if isinstance(resolved, Vector):
            if len(socket.default_value) == 4:
                socket.default_value = _as_vector4(resolved)
            elif len(socket.default_value) == 3:
                socket.default_value = _as_vector3(resolved)
            else:
                socket.default_value = tuple(float(v) for v in resolved)
            return

        if isinstance(resolved, (tuple, list)):
            values = tuple(float(v) for v in resolved)
            if len(socket.default_value) == 4 and len(values) == 3:
                socket.default_value = (*values, 1.0)
            elif len(socket.default_value) == 3 and len(values) >= 3:
                socket.default_value = values[:3]
            else:
                socket.default_value = values

    def blend(
        self,
        a: object,
        b: object,
        *,
        factor: object | None = None,
        mode: BlendMode = BlendMode.MIX,
        kind: ValueKind | None = None,
    ) -> object:
        """Blend two values or sockets, with constant folding for plain Python inputs."""
        if b is None:
            return a
        if a is None:
            return b
        if factor is None:
            factor = self.factor

        resolved_kind = kind or _infer_kind(a, b)

        if _is_plain_value(a) and _is_plain_value(b) and _is_plain_value(factor):
            return _python_blend(a, b, _scalarize(factor), mode, kind=resolved_kind)

        return MixExpr(a=a, b=b, factor=factor, mode=mode, kind=resolved_kind)

    def finish(self) -> bpy.types.Material:
        """Create the final node tree with exactly one Principled BSDF node."""
        output: bpy.types.ShaderNodeOutputMaterial = self.new_node(NodeType.OUTPUT_MATERIAL, x=300, y=0)
        bsdf: bpy.types.ShaderNodeBsdfPrincipled = self.new_node(NodeType.BSDF, x=0, y=0)
        self._bsdf_node = bsdf

        for attr, spec in CHANNEL_SPECS.items():
            value = getattr(self.channels, attr)
            if value is None:
                continue
            socket_in = bsdf.inputs[spec.socket_name]
            if attr == "normal":
                resolved = self.resolve(value, spec.kind)
                
                # If it's already a Vector/Normal output (e.g. from Bump node)
                if _is_socket(resolved) and resolved.type == 'VECTOR':
                    self.links.new(resolved, socket_in)
                else:
                    # It's an RGB texture, needs a Normal Map wrapper
                    nmap_node: bpy.types.ShaderNodeNormalMap = self.new_node(NodeType.NORMAL_MAP)
                    self._apply_to_socket(nmap_node.inputs["Color"], resolved, spec.kind)
                    self.links.new(nmap_node.outputs["Normal"], socket_in)
            else:
                self._apply_to_socket(socket_in, value, spec.kind)

        self.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

        if self.meta.name:
            self.material.name = self.meta.name
        return self.material


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_weighted(layer: MaterialExpr, factor: object = 0.5) -> MaterialWeightedLayer:
    if isinstance(layer, MaterialWeightedLayer):
        return layer
    if isinstance(layer, MaterialLayer):
        return MaterialWeightedLayer(layer, factor)
    raise TypeError(f"Unsupported layer type: {type(layer)!r}")


def _infer_kind(a: object, b: object) -> ValueKind:
    """
    Selects the higher priority type between two operands.
    Priority order: FLOAT < VECTOR < RGBA.
    """
    def get_kind(value: object) -> ValueKind:
        if hasattr(value, "kind"): # Handles TextureExpr, Variable
            return value.kind
        if isinstance(value, (Vector, Color, tuple, list)):
            return ValueKind.RGBA if len(value) >= 4 else ValueKind.VECTOR
        if isinstance(value, bpy.types.NodeSocket):
            # Blender sockets are usually treated as full data types
            if value.type == 'VALUE': return ValueKind.FLOAT
            if value.type == 'VECTOR': return ValueKind.VECTOR
            return ValueKind.RGBA
        if isinstance(value, (int, float)):
            return ValueKind.FLOAT
        return ValueKind.RGBA # Default fallback

    kind_a = get_kind(a)
    kind_b = get_kind(b)
    
    # Priority mapping: higher value wins
    priority = {
        ValueKind.FLOAT: 0,
        ValueKind.VECTOR: 1,
        ValueKind.RGBA: 2
    }
    
    return kind_a if priority[kind_a] >= priority[kind_b] else kind_b


def _python_blend(a: object, b: object, factor: float, mode: BlendMode, kind: ValueKind):
    """
    Blends two Python values by promoting them to RGBA vectors, 
    performing mathutils operations, and casting back to the target kind.
    """
    factor = max(0.0, min(1.0, float(factor)))
    
    # Internal helper to ensure we always work with a 4-component vector
    def to_vec4(val: object) -> Vector:
        if isinstance(val, (int, float)):
            return Vector((float(val), float(val), float(val), 1.0))
        if isinstance(val, (Vector, Color, tuple, list)):
            l = len(val)
            if l == 1: return Vector((val[0], val[0], val[0], 1.0))
            if l == 3: return Vector((val[0], val[1], val[2], 1.0))
            if l >= 4: return Vector((val[0], val[1], val[2], val[3]))
        return Vector((0.0, 0.0, 0.0, 1.0))

    va = to_vec4(a)
    vb = to_vec4(b)
    res = Vector((0.0, 0.0, 0.0, 1.0))

    # Core math using mathutils.Vector operations
    if mode == BlendMode.MIX:
        res = va.lerp(vb, factor)
    elif mode == BlendMode.ADD:
        res = va + (vb * factor)
    elif mode == BlendMode.SUBTRACT:
        res = va - (vb * factor)
    elif mode == BlendMode.MULTIPLY:
        # Per-element multiplication with factor interpolation
        # Blender's multiply blend: A * ((1-f) + f * B)
        blend_factor = Vector((1.0, 1.0, 1.0, 1.0)).lerp(vb, factor)
        res = Vector(tuple(va[i] * blend_factor[i] for i in range(4)))
    elif mode == BlendMode.MAX:
        # Interpolate between A and max(A, B)
        vmax = Vector(tuple(max(va[i], vb[i]) for i in range(4)))
        res = va.lerp(vmax, factor)
    elif mode == BlendMode.MIN:
        # Interpolate between A and min(A, B)
        vmin = Vector(tuple(min(va[i], vb[i]) for i in range(4)))
        res = va.lerp(vmin, factor)
    else:
        raise NotImplementedError(f"Blend mode {mode} not supported for constant folding.")

    # Cast back to the requested kind
    if kind == ValueKind.FLOAT:
        return float(res.x) # Usually R or grayscale intensity
    if kind == ValueKind.VECTOR:
        return Vector((res.x, res.y, res.z))
    
    return res # Return full Vector4 for RGBA


def _python_math(left: object, right: object, op: str) -> object:
    """A tiny constant-folding backend for binary scalar math."""
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    raise NotImplementedError(f"Unsupported math operation: {op}")


def _build_or_get_image(image_ref: str) -> bpy.types.Image:
    path = os.path.abspath(image_ref)
    filename = os.path.basename(path)

    img = bpy.data.images.get(filename)
    if img is not None and bpy.path.abspath(getattr(img, "filepath", "")).endswith(filename):
        return img

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    return bpy.data.images.load(path, check_existing=True)


def _render_camera_to_image(location: Location, fov: float, resolution: Tuple[int, int], label = 'camera_render') -> bpy.types.Image:
    """Internal utility to perform a temporary render and return a Blender Image."""
    scene: bpy.types.Scene = bpy.context.scene
    
    # Store original settings
    orig_camera = scene.camera
    orig_res_x = scene.render.resolution_x
    orig_res_y = scene.render.resolution_y
    orig_filepath = scene.render.filepath
    orig_transparent = scene.render.film_transparent
    orig_color_mode = scene.render.image_settings.color_mode
    orig_file_format = scene.render.image_settings.file_format

    # Create temporary camera
    cam_data = bpy.data.cameras.new(name="TempCameraData")
    cam_data.angle = fov * (3.14159265 / 180.0)
    cam_obj = bpy.data.objects.new("TempCamera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    
    # Apply transform from Location class
    cam_obj.matrix_world = location.matrix
    
    # Setup render environment
    scene.camera = cam_obj
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]

    # Makes the world background transparent
    scene.render.film_transparent = True
    # Sets output to Red, Green, Blue, and Alpha
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.file_format = 'PNG'
    
    img_name = label
    temp_path = os.path.join(bpy.app.tempdir, f"{img_name}.png")
    scene.render.filepath = temp_path
    
    # Perform the render (using EEVEE/Cycles depending on current settings)
    bpy.ops.render.render(write_still=True)
    
    # Check if image already exists in the project
    img: bpy.types.Image = bpy.data.images.get(img_name)
    
    if img:
        # REPLACE LOGIC:
        # Update the file path of the existing image and reload it.
        # This keeps all Material nodes pointing to this 'img' object alive.
        img.filepath = temp_path
        img.reload()
    else:
        # CREATE LOGIC:
        # Load for the first time
        img = bpy.data.images.load(temp_path, check_existing=False)
        img.name = img_name

    # Ensure the image is packed into the .blend file
    img.pack()
    
    # Cleanup
    bpy.data.objects.remove(cam_obj, do_unlink=True)
    bpy.data.cameras.remove(cam_data, do_unlink=True)
    
    # Restore
    scene.camera = orig_camera
    scene.render.resolution_x = orig_res_x
    scene.render.resolution_y = orig_res_y
    scene.render.filepath = orig_filepath
    scene.render.film_transparent = orig_transparent
    scene.render.image_settings.color_mode = orig_color_mode
    scene.render.image_settings.file_format = orig_file_format

    # Delete temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return img


class NodeTreeError(Exception):
    """Custom exception for material node tree traversal errors."""


def _find_existing_material(layer_hash: str) -> bpy.types.Material | None:
    """Find an existing material with the given hash."""
    for mat in bpy.data.materials:
        if BPY_MAT_HASH_PROP in mat and mat[BPY_MAT_HASH_PROP] == layer_hash:
            return mat
    return None


def _material_hash(value: object) -> str:
    """Generate a stable hash for a material value."""
    key = value.stable_key() if hasattr(value, "stable_key") else repr(value)
    payload = repr(key).encode("utf8")
    return blake2b(payload, digest_size=4).hexdigest().upper()


# ---------------------------------------------------------------------------
# Material graph hashing
# ---------------------------------------------------------------------------

def bpy_material_hash(material: bpy.types.Material, hash_image_pixels = False) -> str:
    """Generate a stable hash of the material node graph."""
    if not material.use_nodes or not material.node_tree:
        raise NodeTreeError(f"Material '{material.name}' does not use nodes.")

    root_node: bpy.types.Node | None = next(
        (n for n in material.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"),
        next((n for n in material.node_tree.nodes if n.type == "OUTPUT_MATERIAL"), None),
    )

    if root_node is None:
        raise NodeTreeError(f"No suitable root node found in '{material.name}'.")

    visited_nodes: set[str] = set()
    ignored_props = {
        "name",
        "location",
        "location_absolute",
        "width",
        "width_hidden",
        "select",
        "hide",
        "height",
        "label",
        "parent",
        "warning_propagation",
        "use_custom_color",
        "color",
        "show_options",
        "show_preview",
        "mute",
        "show_texture",
    }

    def format_value(value: object) -> str:
        if hash_image_pixels and isinstance(value, bpy.types.Image):
            image = value
            import numpy as np
            pixels = np.empty(image.size[0] * image.size[1] * 4, dtype=np.float32)
            image.pixels.foreach_get(pixels)
            quantized_pixels = (np.clip(pixels, 0, 1) * 255).astype(np.uint8)
            img_hash = hashlib.sha256(quantized_pixels.tobytes()).hexdigest()
            return f"Image:{value.name}:{img_hash}"
        if hasattr(value, "name") and isinstance(value, bpy.types.ID):
            return f"ID:{value.name}"
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            try:
                return str([round(float(v), 6) for v in value])
            except (ValueError, TypeError):
                return str([str(v) for v in value])
        if hasattr(value, "bl_rna"):
            return f"RNA:{value.bl_rna.identifier}"
        if isinstance(value, float):
            return str(round(value, 6))
        return str(value)

    def hash_node(node: bpy.types.Node) -> str:
        if node.name in visited_nodes:
            return f"ref:{node.bl_idname}"

        visited_nodes.add(node.name)
        data: list[str] = [node.bl_idname]

        for prop in node.bl_rna.properties:
            if prop.is_readonly:
                continue
            if prop.identifier in ignored_props or prop.identifier.startswith("bl_"):
                continue
            data.append(f"p:{prop.identifier}:{format_value(getattr(node, prop.identifier))}")

        for inp in node.inputs:
            if inp.is_linked and inp.links:
                from_node = inp.links[0].from_node
                data.append(f"in:{inp.identifier}->{hash_node(from_node)}")
            elif hasattr(inp, "default_value"):
                data.append(f"val:{inp.identifier}:{format_value(inp.default_value)}")

        data.sort()
        return hashlib.sha256("|".join(data).encode()).hexdigest()

    return hash_node(root_node)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_material(layer: MaterialLayer, rebuild: bool = False) -> bpy.types.Material:
    """Build or reuse a material from the given layer graph."""
    mat_hash = _material_hash(layer)
    material = _find_existing_material(mat_hash)

    if material is not None and not rebuild:
        return material

    if material is None:
        material = bpy.data.materials.new("Mat_" + mat_hash)

    material[BPY_MAT_HASH_PROP] = mat_hash
    ctx = BuildMaterial(material)
    layer.build(ctx)
    return ctx.finish()


# ---------------------------------------------------------------------------
# Public layer implementations
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PBRMaterial(MaterialLayer):
    """PBR material layer."""

    base_color: object | None = None
    metallic: object | None = None
    roughness: object | None = None
    specular: object | None = None
    ior: object | None = None
    transmission: object | None = None
    normal: object | None = None
    emission_color: object | None = None
    emission_strength: object | None = None
    alpha: object | None = None
    ao: object | None = None
    mode: BlendMode | None = None

    @override
    def build(self, ctx: BuildMaterial) -> None:
        attrs = (
            "base_color", "metallic", "roughness", "specular", 
            "ior", "transmission", "normal", 
            "emission_color", "emission_strength", "alpha", "ao"
        )
        for attr in attrs:
            current = getattr(ctx.channels, attr)
            incoming = getattr(self, attr)
            if incoming is None:
                continue
            setattr(ctx.channels, attr, ctx.blend(current, incoming, mode=self.mode) if self.mode is not None else incoming)

@dataclass(frozen=True, slots=True)
class Meta(MaterialLayer):
    """Material metadata layer."""

    name: str | None = None
    custom: Mapping[str, object] = field(default_factory=dict)

    @override
    def build(self, ctx: BuildMaterial) -> None:
        if self.name is not None:
            ctx.meta.name = self.name
        ctx.meta.custom.update(dict(self.custom))

@dataclass(frozen=True, slots=True)
class Glass(MaterialLayer):
    """
    Realistic Glass layer.
    """
    color: object = field(default_factory=lambda: Vector((1.0, 1.0, 1.0, 1.0)))
    ior: float = 1.5
    roughness: float = 0.05
    transmission: float = 1.0

    @override
    def build(self, ctx: BuildMaterial) -> None:
        ctx.channels.base_color = self.color
        ctx.channels.transmission = self.transmission
        ctx.channels.ior = self.ior
        ctx.channels.roughness = self.roughness
        ctx.channels.metallic = 0.0
        ctx.channels.alpha = 1.0

@dataclass(frozen=True, slots=True)
class Metal(MaterialLayer):
    """Procedural Metal with surface imperfections."""
    color: object = field(default_factory=lambda: Vector((0.8, 0.8, 0.8, 1.0)))
    roughness_min: float = 0.1
    roughness_max: float = 0.3
    smudge_scale: float = 10.0

    @override
    def build(self, ctx: BuildMaterial) -> None:
        noise = ctx.noise(scale=self.smudge_scale, detail=15.0, roughness=0.6)
        roughness_map = ctx.map_range(noise, from_min=0.3, from_max=0.7, to_min=self.roughness_min, to_max=self.roughness_max)
        
        ctx.channels.base_color = self.color
        ctx.channels.metallic = 1.0
        ctx.channels.roughness = roughness_map

@dataclass(frozen=True, slots=True)
class Wood(MaterialLayer):
    """Procedural Wood using distorted waves."""
    color_dark: object = field(default_factory=lambda: Vector((0.3, 0.15, 0.05, 1.0)))
    color_light: object = field(default_factory=lambda: Vector((0.6, 0.35, 0.15, 1.0)))
    scale: float = 3.0
    distortion: float = 7.0

    @override
    def build(self, ctx: BuildMaterial) -> None:
        wave = ctx.wave(scale=self.scale, distortion=self.distortion, detail=10.0)
        wood_color = ctx.blend(self.color_dark, self.color_light, factor=wave)
        
        ctx.channels.base_color = wood_color
        ctx.channels.metallic = 0.0
        ctx.channels.roughness = ctx.map_range(wave, to_min=0.4, to_max=0.7)
        ctx.channels.normal = ctx.bump(height=wave, distance=0.05)

@dataclass(frozen=True, slots=True)
class Concrete(MaterialLayer):
    """Procedural Concrete/Stone with pitting and noise."""
    color: object = field(default_factory=lambda: Vector((0.5, 0.5, 0.5, 1.0)))
    scale: float = 15.0

    @override
    def build(self, ctx: BuildMaterial) -> None:
        noise = ctx.noise(scale=self.scale, detail=15.0)
        voronoi = ctx.voronoi(scale=self.scale * 2.0)
        
        # Pits/dents mapped from Voronoi
        pits = ctx.map_range(voronoi, from_min=0.0, from_max=0.3, to_min=1.0, to_max=0.0)
        surface = ctx.blend(noise, pits, factor=0.5, mode=BlendMode.SUBTRACT)
        
        ctx.channels.base_color = ctx.blend(self.color, Vector((0.2, 0.2, 0.2, 1.0)), factor=pits)
        ctx.channels.metallic = 0.0
        ctx.channels.roughness = ctx.map_range(noise, to_min=0.7, to_max=1.0)
        ctx.channels.normal = ctx.bump(height=surface, strength=0.4, distance=0.1)

@dataclass(frozen=True, slots=True)
class Brick(MaterialLayer):
    """Procedural Brick Wall."""
    color1: object = field(default_factory=lambda: Vector((0.6, 0.2, 0.1, 1.0)))
    color2: object = field(default_factory=lambda: Vector((0.4, 0.1, 0.05, 1.0)))
    scale: float = 8.0

    @override
    def build(self, ctx: BuildMaterial) -> None:
        brick_color = ctx.brick(color1=self.color1, color2=self.color2, scale=self.scale, kind=ValueKind.RGBA)
        brick_mask = ctx.brick(scale=self.scale, kind=ValueKind.FLOAT) # Returns mortar mask via fac
        
        # Add high frequency noise to the bump map for realistic brick surface
        noise = ctx.noise(scale=self.scale * 10.0, detail=15.0)
        height = ctx.blend(brick_mask, noise, factor=0.1)

        ctx.channels.base_color = brick_color
        ctx.channels.metallic = 0.0
        ctx.channels.roughness = ctx.map_range(brick_mask, to_min=0.8, to_max=0.95)
        ctx.channels.normal = ctx.bump(height=height, strength=1.0, distance=0.1)

@dataclass(frozen=True, slots=True)
class Sand(MaterialLayer):
    """Procedural Sand with dunes."""
    color: object = field(default_factory=lambda: Vector((0.76, 0.65, 0.45, 1.0)))
    scale: float = 25.0

    @override
    def build(self, ctx: BuildMaterial) -> None:
        # Fine grain
        grain = ctx.noise(scale=self.scale * 10.0, detail=15.0)
        # Dunes (large waves)
        dunes = ctx.wave(scale=self.scale * 0.1, distortion=2.0)
        
        height = ctx.blend(grain, dunes, factor=0.3, mode=BlendMode.ADD)
        sand_color = ctx.blend(self.color, Vector((0.6, 0.5, 0.3, 1.0)), factor=grain)

        ctx.channels.base_color = sand_color
        ctx.channels.metallic = 0.0
        ctx.channels.roughness = 0.95
        ctx.channels.normal = ctx.bump(height=height, strength=0.5, distance=0.2)

@dataclass(frozen=True, slots=True)
class Fabric(MaterialLayer):
    """Procedural Fabric (Woven)."""
    color: object = field(default_factory=lambda: Vector((0.1, 0.2, 0.6, 1.0))) # Denim default
    scale: float = 50.0

    @override
    def build(self, ctx: BuildMaterial) -> None:
        # High freq noise acts as a good cheap fabric weave pattern
        weave = ctx.noise(scale=self.scale, detail=5.0)
        
        ctx.channels.base_color = ctx.blend(self.color, Vector((0.0, 0.0, 0.0, 1.0)), factor=weave * 0.2)
        ctx.channels.metallic = 0.0
        ctx.channels.roughness = 0.9 # Fabric is very rough
        ctx.channels.normal = ctx.bump(height=weave, strength=0.2, distance=0.01)

@dataclass(frozen=True, slots=True)
class Leather(MaterialLayer):
    """Procedural Leather (non-organic look, good for upholstery)."""
    color: object = field(default_factory=lambda: Vector((0.3, 0.15, 0.05, 1.0)))
    scale: float = 20.0

    @override
    def build(self, ctx: BuildMaterial) -> None:
        # Voronoi distance creates the classic pebbled leather look
        pebbles = ctx.voronoi(scale=self.scale, randomness=0.8)
        
        ctx.channels.base_color = ctx.blend(self.color, Vector((0.15, 0.05, 0.02, 1.0)), factor=pebbles)
        ctx.channels.metallic = 0.0
        ctx.channels.roughness = ctx.map_range(pebbles, to_min=0.4, to_max=0.6)
        ctx.channels.normal = ctx.bump(height=pebbles, strength=0.6, distance=0.05)

@dataclass(frozen=True, slots=True)
class Plastic(MaterialLayer):
    """Procedural Plastic (clean, adjustable roughness)."""
    color: object = field(default_factory=lambda: Vector((0.8, 0.1, 0.1, 1.0)))
    roughness: float = 0.2

    @override
    def build(self, ctx: BuildMaterial) -> None:
        noise = ctx.noise(scale=50.0) # Tiny imperfections
        
        ctx.channels.base_color = self.color
        ctx.channels.metallic = 0.0
        ctx.channels.roughness = ctx.blend(self.roughness, noise, factor=0.05)
        # Almost no bump, just pure clean plastic
        ctx.channels.normal = ctx.bump(height=noise, strength=0.05, distance=0.01)


@dataclass(frozen=True, slots=True)
class Glow(MaterialLayer):
    """Emission layer."""

    emission_color: object = field(default_factory=lambda: Vector((1.0, 1.0, 1.0, 1.0)))
    emission_strength: object = 1.0

    @override
    def build(self, ctx: BuildMaterial) -> None:
        ctx.channels.emission_color = ctx.blend(ctx.channels.emission_color, self.emission_color)
        ctx.channels.emission_strength = ctx.blend(ctx.channels.emission_strength, self.emission_strength)


@dataclass(frozen=True, slots=True)
class Dirt(MaterialLayer):
    """Procedural Dirt layer using noise and smart masking."""
    color: object = field(default_factory=lambda: Vector((0.05, 0.03, 0.02, 1.0)))
    scale: float = 10.0
    
    @override
    def build(self, ctx: BuildMaterial) -> None:
        # Create a "patchy" noise mask for dirt
        mask = ctx.noise(scale=self.scale, detail=8.0, roughness=0.7)
        mask = ctx.color_ramp(mask, stops=((0.3, (0,0,0,1)), (0.6, (1,1,1,1))))
        
        # Apply factor scaling if exists (e.g. Dirt() * 0.5)
        combined_mask = ctx.blend(0.0, mask, factor=ctx.factor)
        
        ctx.channels.base_color = ctx.blend(ctx.channels.base_color, self.color, factor=combined_mask)
        ctx.channels.roughness = ctx.blend(ctx.channels.roughness, 0.9, factor=combined_mask)


@dataclass(frozen=True, slots=True)
class DirtImg(MaterialLayer):
    """Dirt layer."""

    texture: Texture = field(default_factory=lambda: Texture(image_path="./assets/test/dirt.png"))

    @override
    def build(self, ctx: BuildMaterial) -> None:
        ctx.channels.base_color = ctx.blend(ctx.channels.base_color, self.texture)
        ctx.channels.roughness = ctx.blend(ctx.channels.roughness, 0.9)


@dataclass(frozen=True, slots=True)
class Rust(MaterialLayer):
    """Procedural Rust layer with complex noise structure."""
    
    @override
    def build(self, ctx: BuildMaterial) -> None:
        # Base rust noise
        noise = ctx.noise(scale=15.0, detail=12.0, roughness=0.85)
        
        # Rust color gradient (orange to dark brown)
        rust_colors = ctx.color_ramp(noise, stops=(
            (0.4, (0.1, 0.02, 0.0, 1.0)), # Dark
            (0.5, (0.4, 0.1, 0.02, 1.0)), # Mid
            (0.7, (0.6, 0.2, 0.05, 1.0))  # Light orange
        ))
        
        # Mask for where rust appears
        rust_mask = ctx.map_range(noise, from_min=0.4, from_max=0.6)
        
        final_factor = ctx.blend(0.0, rust_mask, factor=ctx.factor)

        ctx.channels.base_color = ctx.blend(ctx.channels.base_color, rust_colors, factor=final_factor)
        ctx.channels.metallic = ctx.blend(ctx.channels.metallic, 0.0, factor=final_factor)
        ctx.channels.roughness = ctx.blend(ctx.channels.roughness, 0.95, factor=final_factor)

@dataclass(frozen=True, slots=True)
class RustImg(MaterialLayer):
    """Rust layer."""

    texture: Texture = field(default_factory=lambda: Texture(image_path="./assets/test/rust.png"))

    @override
    def build(self, ctx: BuildMaterial) -> None:
        ctx.channels.base_color = ctx.blend(ctx.channels.base_color, self.texture)
        ctx.channels.metallic = ctx.blend(ctx.channels.metallic, 0.0)
        ctx.channels.roughness = ctx.blend(ctx.channels.roughness, 1.0)

@dataclass(frozen=True, slots=True)
class Dust(MaterialLayer):
    """
    Procedural residue/dust layer.
    Adds a thin, rough, whitish layer based on noise.
    """
    color: object = field(default_factory=lambda: Vector((0.8, 0.8, 0.8, 1.0)))
    scale: float = 20.0
    density: float = 0.5
    roughness: float = 0.9

    @override
    def build(self, ctx: BuildMaterial) -> None:
        # Generate fine grain noise for dust particles
        noise = ctx.noise(scale=self.scale, detail=15.0, roughness=0.6)
        
        # Create a mask: higher density = more dust coverage
        mask = ctx.map_range(
            noise, 
            from_min=1.0 - self.density, 
            from_max=1.1 - self.density, 
            to_min=0.0, 
            to_max=1.0
        )
        
        # Apply factor scaling (DustProc() * factor)
        final_factor = ctx.blend(0.0, mask, factor=ctx.factor)

        # Blend base color with dust color
        ctx.channels.base_color = ctx.blend(ctx.channels.base_color, self.color, factor=final_factor)
        # Dust is never metallic and very rough
        ctx.channels.metallic = ctx.blend(ctx.channels.metallic, 0.0, factor=final_factor)
        ctx.channels.roughness = ctx.blend(ctx.channels.roughness, self.roughness, factor=final_factor)
        # Dust blocks transmission (transparency)
        if ctx.channels.transmission is not None:
             ctx.channels.transmission = ctx.blend(ctx.channels.transmission, 0.0, factor=final_factor)


@dataclass(frozen=True, slots=True)
class Cracks(MaterialLayer):
    """
    Highly realistic procedural cracks.
    Features: wobbly paths, thickness control, and dirt-filled crevices.
    """
    scale: float = 3.0
    distortion: float = 0.5  # How much the cracks 'wiggle'
    thickness: float = 0.02  # Thickness of the crack line
    dirt_color: object = field(default_factory=lambda: Vector((0.02, 0.01, 0.0, 1.0)))
    bump_strength: float = 0.5

    @override
    def build(self, ctx: BuildMaterial) -> None:
        # 1. Distort coordinates to make cracks organic
        # We mix generated/object coords with noise
        noise_distort = ctx.noise(scale=self.scale * 2.0, detail=4.0, roughness=0.5, distortion=self.distortion)
        
        # 2. Generate the Voronoi crack base
        voronoi_base = ctx.voronoi(scale=self.scale + noise_distort)
        
        # 3. Process the mask
        # Distance to edge gives us 0 at the edge, >0 inside. We invert and tighten it.
        crack_mask = ctx.map_range(
            voronoi_base,
            from_min=0.0,
            from_max=self.thickness,
            to_min=1.0,
            to_max=0.0
        )
        # Apply factor scaling (CracksProc() * factor)
        final_mask = ctx.blend(0.0, crack_mask, factor=ctx.factor)

        # 4. Apply to PBR Channels
        # Base Color: Fill cracks with dirt/darkness
        ctx.channels.base_color = ctx.blend(ctx.channels.base_color, self.dirt_color, factor=final_mask)
        
        # Roughness: Cracks are usually very rough (dusty/dry)
        ctx.channels.roughness = ctx.blend(ctx.channels.roughness, 1.0, factor=final_mask)
        
        # Metallic: No metal inside cracks
        ctx.channels.metallic = ctx.blend(ctx.channels.metallic, 0.0, factor=final_mask)

        # 5. Normal/Bump Depth
        # We use the crack mask as a height map for the Bump node
        # Invert it for 'indentation'
        height_map = final_mask * -1.0 
        ctx.channels.normal = ctx.blend(ctx.channels.normal, BumpExpr(height=height_map, strength=self.bump_strength), factor=final_mask)

@dataclass(frozen=True, slots=True)
class LaserGrid(MaterialLayer):
    """
    Refactored Laser Grid using MappingExpr for procedural coordination.
    """
    color: object = field(default_factory=lambda: Vector((0.0, 1.0, 0.5, 1.0)))
    scale: float = 5.0
    thickness: float = 0.02
    glow_intensity: float = 10.0

    @override
    def build(self, ctx: BuildMaterial) -> None:
        # X-Axis Lines
        map_x = ctx.mapping(coord_type=CoordType.OBJECT)
        wave_x = ctx.wave(scale=self.scale, mapping=map_x)

        # Y-Axis Lines (Rotated on Z)
        map_y = ctx.mapping(coord_type=CoordType.OBJECT, transform=Rot(Z=90))
        wave_y = ctx.wave(scale=self.scale, mapping=map_y)

        # Z-Axis Lines (Rotated on Y)
        map_z = ctx.mapping(coord_type=CoordType.OBJECT, transform=Rot(Y=90))
        wave_z = ctx.wave(scale=self.scale, mapping=map_z)

        # Combine all 3 axes using MAX logic
        grid_mask = wave_x.max(wave_y).max(wave_z)

        # Sharpening
        sharp_mask = ctx.map_range(
            grid_mask, 
            from_min=1.0 - self.thickness, from_max=1.0, 
            to_min=0.0, to_max=1.0
        )
        
        final_mask = sharp_mask * ctx.factor

        # Standard assignment
        ctx.channels.alpha = final_mask
        ctx.channels.emission_color = self.color
        ctx.channels.emission_strength = final_mask * self.glow_intensity
        ctx.channels.base_color = Vector((0.0, 0.0, 0.0, 1.0))

# ---------------------------------------------------------------------------
# Some pre-defined materials
# ---------------------------------------------------------------------------

class mat:
    Var = VariableExpr
    Mapping = MappingSettings
    Tex = Texture
    CameraTex = CameraTexture

    Layer = MaterialLayer
    PBR = PBRMaterial
    Meta = Meta
    Name = Meta
    Glass = Glass
    Metal = Metal
    Wood = Wood
    Concrete = Concrete
    Brick = Brick
    Sand = Sand
    Fabric = Fabric
    Leather = Leather
    Plastic = Plastic
    Glow = Glow
    Dirt = Dirt
    DirtImg = DirtImg
    Rust = Rust
    RustImg = RustImg
    Dust = Dust
    Cracks = Cracks
    LaserGrid = LaserGrid

    red = PBR(base_color=(1.0, 0.0, 0.0, 1.0)) + Name("RedMat")
    green = PBR(base_color=(0.0, 1.0, 0.0, 1.0)) + Name("GreenMat")
    blue = PBR(base_color=(0.0, 0.0, 1.0, 1.0)) + Name("BlueMat")
    yellow = PBR(base_color=(1.0, 1.0, 0.0, 1.0)) + Name("YellowMat")

    iron = Metal(color=Vector((0.3, 0.3, 0.3, 1.0)), roughness_min=0.2, roughness_max=0.5) + Name("IronMat")
    gold = Metal(color=Vector((1.0, 0.75, 0.1, 1.0)), roughness_min=0.1, roughness_max=0.2) + Name("GoldMat")
    copper = Metal(color=Vector((0.8, 0.3, 0.2, 1.0)), roughness_min=0.15, roughness_max=0.4) + Name("CopperMat")

    wood_oak = Wood(color_dark=Vector((0.25, 0.12, 0.04, 1.0)), color_light=Vector((0.55, 0.35, 0.15, 1.0))) + Name("WoodOakMat")
    wood_pine = Wood(color_dark=Vector((0.6, 0.4, 0.2, 1.0)), color_light=Vector((0.8, 0.6, 0.3, 1.0)), scale=2.0) + Name("WoodPineMat")

    concrete = Concrete(color=Vector((0.4, 0.4, 0.4, 1.0))) + Name("ConcreteMat")
    brick_red = Brick() + Name("BrickRedMat")

    sand_desert = Sand() + Name("SandMat")

    denim = Fabric() + Name("DenimMat")
    leather_brown = Leather() + Name("LeatherMat")
    plastic_glossy_red = Plastic(color=Vector((0.8, 0.05, 0.05, 1.0)), roughness=0.1) + Name("PlasticGlossyRedMat")
    plastic_matte_grey = Plastic(color=Vector((0.3, 0.3, 0.3, 1.0)), roughness=0.6) + Name("PlasticMatteGreyMat")
