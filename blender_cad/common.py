import fnmatch
from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Union

from mathutils import Vector

if TYPE_CHECKING:
    from .build_part import BuildPart
    from .chain import chain
    from .curve import BaseCurve, BuildCurve, Curve, curve
    from .geometry import GeometryEntity
    from .joint import Joint
    from .location import CurveLocation, Location, Transform
    from .ml import ml
    from .object import Object
    from .part import Part
    from .shape_list import ShapeList

VectorLike = Union[
    Vector,
    tuple[Union[float, int]],
    list[Union[float, int]],
    float,
    int,
    "Axis",
    "Transform",
]


def try_extract_vector(value: Any) -> Optional[Vector]:
    from .location import Transform

    if isinstance(value, Axis):
        return value.value
    if isinstance(value, Transform):
        return value.position
    if isinstance(value, (float, int)):
        return Vector((value, value, value))
    if isinstance(value, (Vector, tuple, list)):
        v = Vector(value)
        # Ensure 3D if 2D was provided
        if len(v) == 2:
            return Vector((v[0], v[1], 0))
        return v
    return None


def extract_vector(value: VectorLike) -> Vector:
    v = try_extract_vector(value)
    assert v is not None, f"Cannot extract vector from {value!r}"
    return v


PartLike = Union[
    "Part", "BuildPart", "BaseCurve", "BuildCurve", "curve", "chain", "ml", "Joint"
]


def try_extract_part(
    value: Any,
    to_loc: Optional["Location"] = None,
    ensure_copy=False,
    width: Optional[float] = None,
    height: Optional[float] = None,
) -> Optional["Part"]:
    from .build_part import BuildPart, Mode
    from .chain import chain
    from .curve import BaseCurve, BuildCurve, curve
    from .joint import Joint
    from .location import Location
    from .ml import ml
    from .part import Part

    if isinstance(value, Joint):
        value.to(to_loc or Location(), mode=Mode.PRIVATE)
        return extract_part(value.object)
    if isinstance(value, (BuildPart, BaseCurve, BuildCurve, curve, chain)):
        return value.part
    if isinstance(value, ml):
        return value.to_part(width=width, height=height)
    if isinstance(value, Part):
        return value.copy() if ensure_copy else value
    return None


def extract_part(
    part_like: PartLike,
    to_loc: Optional["Location"] = None,
    ensure_copy=False,
    width: Optional[float] = None,
    height: Optional[float] = None,
) -> "Part":
    part = try_extract_part(part_like, to_loc, ensure_copy, width, height)
    assert part is not None, f"Unsupported type for part extraction: {type(part_like)}"
    return part


def extract_object(obj: PartLike) -> Optional["Object"]:
    from .build_part import BuildPart
    from .curve import BuildCurve
    from .joint import Joint
    from .object import Object

    if isinstance(obj, BuildPart):
        return obj.part
    if isinstance(obj, BuildCurve):
        return obj.curve
    if isinstance(obj, Joint):
        return obj.object
    return obj if isinstance(obj, Object) else None


GeometryEntityLike = Optional[Union["ShapeList", "GeometryEntity", "Part", "BuildPart"]]


def extract_shape_list(value: GeometryEntityLike) -> "ShapeList":
    from .build_part import BuildPart
    from .geometry import GeometryEntity, ShapeList
    from .part import Part

    if isinstance(value, ShapeList):
        return value
    if isinstance(value, GeometryEntity):
        return ShapeList([value])
    if isinstance(value, Part):
        return value.faces()
    if isinstance(value, BuildPart):
        return extract_shape_list(value.part)
    return extract_shape_list(BuildPart._get_context())


class Axis(Enum):
    """Enumeration for standard 3D axes as mathutils Vectors."""

    X = Vector((1, 0, 0))
    Y = Vector((0, 1, 0))
    Z = Vector((0, 0, 1))

    @staticmethod
    def all() -> set["Axis"]:
        return {Axis.X, Axis.Y, Axis.Z}

    def __neg__(self):
        return -self.value

    @property
    def value(self) -> Vector:
        return super().value

    @property
    def index(self):
        return self._get_index(self.value)[0]

    @staticmethod
    def _get_index(vector_like: VectorLike):
        vector = extract_vector(vector_like)
        for i, val in enumerate(vector):
            if abs(val) > 0.9:
                return i, bool(val < 0.0)
        raise ValueError(f"Unsupported vector: {vector}")

    @staticmethod
    def from_vector(vector_like: VectorLike):
        idx, neg = Axis._get_index(vector_like)
        for axis in Axis.all():
            if axis.index == idx:
                return axis, neg
        raise ValueError(f"Unsupported vector: {vector_like}")


class AbstractCurve(ABC):
    """Abstract base class for all objects that can be evaluated as a path."""

    @abstractmethod
    def curve(self) -> "Curve":
        raise NotImplementedError

    @abstractmethod
    def at(self, t: Optional[float] = None, t_m: Optional[float] = None) -> "Location":
        """Returns a Location object at the given parameter or distance."""
        raise NotImplementedError

    @abstractmethod
    def length(self) -> float:
        """Returns the total length of the curve."""
        raise NotImplementedError

    def location(self) -> "CurveLocation":
        """Returns a CurveLocation object at the start of the curve."""
        from .location import CurveLocation

        return CurveLocation(self)


CurveLike = Optional[Union["AbstractCurve", "BuildCurve", "curve"]]


def extract_curve(curve_like: CurveLike) -> "Curve":
    from .curve import BuildCurve, curve

    if isinstance(curve_like, (BuildCurve, curve)):
        return curve_like.curve
    return curve_like.curve()


class DualMethod:
    def __init__(self, instance_fn, class_fn):
        self.instance_fn = instance_fn
        self.class_fn = class_fn

    def __get__(self, obj, cls):
        if obj is None:
            return self.class_fn.__get__(cls, cls)
        return self.instance_fn.__get__(obj, cls)


def _flatten_items(items: Iterable[object | None]) -> Iterable[object]:
    """Flatten nested lists while leaving tuples intact."""
    for item in items:
        if item is None:
            continue
        if isinstance(item, (list, tuple, set)):
            yield from _flatten_items(item)
        else:
            yield item


def tag_to_list(tag: Optional[str | Iterable[str]]) -> Iterable[str]:
    return [tag] if isinstance(tag, str) else (tag or [])


def match_tags(
    applied_tags: Iterable[str], target_tags: Iterable[str], invert: bool = False
) -> bool:
    if not target_tags:
        return invert

    has_match = False
    for target in target_tags:
        if "*" in target:
            if any(
                fnmatch.fnmatchcase(applied_tag, target) for applied_tag in applied_tags
            ):
                has_match = True
                break
        else:
            if target in applied_tags:
                has_match = True
                break

    return not has_match if invert else has_match
