from collections.abc import Callable, Iterable
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    TypeVar,
    Union,
    overload,
)

from mathutils import Vector
from typing_extensions import override

from .common import Axis, match_tags

if TYPE_CHECKING:
    from .bmesh_wrapper import BMEdgeWrapper, BMFaceWrapper, BMVertWrapper
    from .geometry import Edge, Face, GeometryEntity, GeomType, Vertex, Wire
    from .material import MaterialLayer
    from .part import Part


class SortBy(Enum):
    """Criteria for sorting geometry entities."""

    AREA = "AREA"
    LENGTH = "LENGTH"
    GEOM_TYPE = "GEOM_TYPE"


class GroupBy(Enum):
    """Criteria for grouping geometry entities."""

    AREA = "AREA"
    LENGTH = "LENGTH"
    GEOM_TYPE = "GEOM_TYPE"


T = TypeVar("T", bound="GeometryEntity")


class ShapeList(list[T]):
    """
    An extended list providing selection, filtering, and set-like operations
    for geometry entities, mimicking the build123d selection style.
    """

    def __init__(
        self, iterable: Iterable[T] = (), full_list: Optional["ShapeList"] = None
    ):
        # Ensure only unique items are stored to behave like a set
        unique_items = list(dict.fromkeys(iterable).keys())
        super().__init__(unique_items)
        self.full_list = full_list or self

    @property
    def type(self) -> Literal["Face", "Wire", "Edge", "Vertex"]:
        """Returns the type of the first item in the list."""
        items = self
        if len(items) == 0:
            items = self.full_list
        if len(items) == 0:
            raise Exception("ShapeList is empty, cannot determine type")
        return type(self[0]).__name__

    @property
    def _owner(self):
        items = self
        if len(items) == 0:
            raise Exception("ShapeList is empty, cannot determine owner")
        return self[0]._owner

    @property
    def source_part(self) -> "Part":
        return self._owner.part

    @property
    def tags(self):
        result: list[str] = []
        for item in self:
            for tag in item.tags:
                if tag not in result:
                    result.append(tag)
        return result

    @tags.setter
    def tags(self, value: Iterable[str]):
        for item in self:
            item.tags = value

    def add_tags(self, *tags: str | Iterable[str]):
        for item in self:
            item.add_tags(*tags)

    def remove_tags(self, *tags: str | Iterable[str]):
        for item in self:
            item.remove_tags(*tags)

    @property
    def mat(self):
        """Getter for material (returns None as materials are set-only on lists)."""
        pass

    @mat.setter
    def mat(self, material: Optional["MaterialLayer"]):
        """Applies a material to every item in the ShapeList."""
        if len(self) == 0:
            return
        if not hasattr(self[0], "mat"):
            raise Exception("This ShapeList does not allow to set materials")
        for item in self:
            item.mat = material

    def bm_faces(self) -> list["BMFaceWrapper"]:
        """Flattens all BMesh Faces from the entities in this list."""
        return [f for item in self for f in item.bm_faces()]

    def bm_edges(self) -> list["BMEdgeWrapper"]:
        """Flattens all BMesh Edges from the entities in this list."""
        return [e for item in self for e in item.bm_edges()]

    def bm_verts(self) -> list["BMVertWrapper"]:
        """Flattens all BMesh Vertices from the entities in this list."""
        return [v for item in self for v in item.bm_verts()]

    def faces(self) -> "ShapeList[Face]":
        """Flattens all Faces from the entities in this list into a new ShapeList."""
        return ShapeList([f for item in self for f in item.faces()])

    def wires(self) -> "ShapeList[Wire]":
        """Flattens all Wires from the entities in this list into a new ShapeList."""
        return ShapeList([w for item in self for w in item.wires()])

    def edges(self) -> "ShapeList[Edge]":
        """Flattens all Edges from the entities in this list into a new ShapeList."""
        return ShapeList([e for item in self for e in item.edges()])

    def vertices(self) -> "ShapeList[Vertex]":
        """Flattens all Vertices from the entities in this list into a new ShapeList."""
        return ShapeList([v for item in self for v in item.vertices()])

    def all_single_faces(self) -> "ShapeList[Face]":
        """Flattens all single Faces from the entities in this list into a new ShapeList."""
        return ShapeList([f for item in self for f in item.all_single_faces()])

    def all_single_edges(self) -> "ShapeList[Edge]":
        """Flattens all single Edges from the entities in this list into a new ShapeList."""
        return ShapeList([e for item in self for e in item.all_single_edges()])

    def all_single_vertices(self) -> "ShapeList[Vertex]":
        """Flattens all single Vertices from the entities in this list into a new ShapeList."""
        return ShapeList([v for item in self for v in item.all_single_vertices()])

    def split(self) -> "ShapeList[T]":
        """Splits (ungroups) the entities into a list of single entities."""
        return ShapeList([item for item in self for item in item.split()])

    def part(self) -> "Part":
        """Creates a new Part containing a copy of this entities."""
        from .build_part import BuildPart, Mode
        from .modifiers import add

        with BuildPart(mode=Mode.PRIVATE) as bp:
            for item in self:
                add(item.part(), mode=Mode.JOIN, _make_copy=False)
        return bp.part

    def to_wire(self):
        from .geometry import Edge, Wire

        if len(self) == 0 or not isinstance(self[0], Edge):
            raise Exception("This ShapeList does not contain Edges")
        return Wire(self._owner, self)

    def _new_child_list(self, iterable: Iterable[T]) -> "ShapeList[T]":
        """Creates a new ShapeList maintaining a reference to the full parent set."""
        return ShapeList(iterable, self.full_list)

    @override
    def __add__(self, other: Iterable[T]) -> "ShapeList[T]":
        """Union operation: Returns a new list with unique elements from both collections."""
        return self._new_child_list(list(self) + list(other))

    def __sub__(self, other: Iterable[T]) -> "ShapeList[T]":
        """Difference operation: Removes elements found in 'other' from 'self'."""
        other_set = set(other)
        return self._new_child_list(item for item in self if item not in other_set)

    def __and__(self, other: Iterable[T]) -> "ShapeList[T]":
        """Intersection operation: Returns only elements present in both lists."""
        other_set = set(other)
        return self._new_child_list(item for item in self if item in other_set)

    def __invert__(self) -> "ShapeList[T]":
        """Inversion operation: Returns everything in the parent set that is NOT in this list."""
        return self.full_list - self

    @overload
    def __getitem__(self, item: int) -> T: ...

    @overload
    def __getitem__(self, item: slice) -> "ShapeList[T]": ...

    @override
    def __getitem__(self, item: Union[int, slice]) -> Union[T, "ShapeList[T]"]:
        """
        Overrides the default list indexing.
        If a slice is requested, returns a new ShapeList instead of a standard list.
        """
        result = super().__getitem__(item)
        if isinstance(item, slice):
            return self._new_child_list(result)
        return result

    def filter_by(
        self, predicate_or_type: Union[Callable[[T], bool], "GeomType"]
    ) -> "ShapeList[T]":
        """Filters the list by a custom predicate function or a specific Geometry Type."""
        from .geometry import GeomType

        if isinstance(predicate_or_type, GeomType):
            return self._new_child_list(
                filter(lambda e: e.geom_type == predicate_or_type, self)
            )
        return self._new_child_list(filter(predicate_or_type, self))

    def planes_only(self):
        """Returns a new ShapeList containing only Planes."""
        from .geometry import GeomType

        return self.filter_by(GeomType.PLANE)

    def cylinders_only(self):
        """Returns a new ShapeList containing only Cylinders."""
        from .geometry import GeomType

        return self.filter_by(GeomType.CYLINDER)

    def cones_only(self):
        """Returns a new ShapeList containing only Cones."""
        from .geometry import GeomType

        return self.filter_by(GeomType.CONE)

    def spheres_only(self):
        """Returns a new ShapeList containing only Spheres."""
        from .geometry import GeomType

        return self.filter_by(GeomType.SPHERE)

    def tagged(self, *tags: str):
        """
        Returns a new ShapeList containing only entities with the specified tag(s).
        Supports wildcards inside tags (e.g., 'my_tag_*') and multiple tags via positional arguments.
        """
        return self.filter_by(lambda entity: match_tags(entity.tags, tags))

    def untagged(self, *tags: str):
        """
        Returns a new ShapeList containing only entities without the specified tag(s).
        Supports wildcards inside tags (e.g., 'my_tag_*') and multiple tags via positional arguments.
        """
        return self - self.tagged(*tags)

    def sort_by(
        self,
        key: Union[Callable[[T], Any], Vector, Axis, SortBy],
        reverse: bool = False,
    ) -> "ShapeList[T]":
        """
        Sorts entities based on various criteria:
        - Vector/Axis: Sorts by the dot product of the entity center and the direction.
        - SortBy: Sorts by specific geometric properties (Area, Length, etc).
        - Callable: Standard python sort key.
        """
        if isinstance(key, Vector):
            key_func = lambda e: e.center().position.dot(key)
        elif isinstance(key, Axis):
            axis_vec = key.value
            key_func = lambda e: e.center().position.dot(axis_vec)
        elif isinstance(key, SortBy):
            key_func = {
                SortBy.AREA: lambda e: e.area(),
                SortBy.LENGTH: lambda e: e.length(),
                SortBy.GEOM_TYPE: lambda e: (
                    e.geom_type.value
                ),  # Sort by enum string value
            }[key]
        else:
            key_func = key
        return self._new_child_list(sorted(self, key=key_func, reverse=reverse))

    def group_by(
        self,
        key: Union[Callable[[T], Any], Vector, Axis, GroupBy],
        tolerance: float = 1e-4,
    ) -> list["ShapeList[T]"]:
        """
        Groups elements into multiple ShapeLists based on a criteria.
        Numerical values (like area or position) are grouped if they fall within the 'tolerance'.
        """
        if not self:
            return []

        key_func: Callable[[T], Any]

        if isinstance(key, Vector):
            key_func = lambda e: e.center().position.dot(key)
        elif isinstance(key, Axis):
            axis_vec = key.value
            key_func = lambda e: e.center().position.dot(axis_vec)
        elif isinstance(key, GroupBy):
            key_func = {
                GroupBy.AREA: lambda e: e.area(),
                GroupBy.LENGTH: lambda e: e.length(),
                GroupBy.GEOM_TYPE: lambda e: e.geom_type.value,
            }[key]
        else:
            key_func = key

        # Sort and group items based on the tolerance threshold
        sorted_items = sorted(self, key=key_func)
        groups = []
        current_group = [sorted_items[0]]
        current_val = key_func(sorted_items[0])

        for item in sorted_items[1:]:
            val = key_func(item)
            # Check if value is float-like or enum-like for grouping logic
            is_numeric = isinstance(val, (int, float))

            if (is_numeric and abs(val - current_val) <= tolerance) or (
                not is_numeric and val == current_val
            ):
                current_group.append(item)
            else:
                groups.append(self._new_child_list(current_group))
                current_group = [item]
                current_val = val
        groups.append(self._new_child_list(current_group))
        return groups

    def max_x(self, tolerance: float = 1e-4):
        """Returns the entities with the highest X value."""
        return self.group_by(Axis.X, tolerance)[-1]

    def min_x(self, tolerance: float = 1e-4):
        """Returns the entities with the lowest X value."""
        return self.group_by(Axis.X, tolerance)[0]

    def max_y(self, tolerance: float = 1e-4):
        """Returns the entities with the highest Y value."""
        return self.group_by(Axis.Y, tolerance)[-1]

    def min_y(self, tolerance: float = 1e-4):
        """Returns the entities with the lowest Y value."""
        return self.group_by(Axis.Y, tolerance)[0]

    def max_z(self, tolerance: float = 1e-4):
        """Returns the entities with the highest Z value."""
        return self.group_by(Axis.Z, tolerance)[-1]

    def min_z(self, tolerance: float = 1e-4):
        """Returns the entities with the lowest Z value."""
        return self.group_by(Axis.Z, tolerance)[0]

    def top(self, tolerance: float = 1e-4):
        """Alias for max_z()."""
        return self.max_z(tolerance)

    def bottom(self, tolerance: float = 1e-4):
        """Alias for min_z()."""
        return self.min_z(tolerance)

    def side(self, tolerance: float = 1e-4):
        """Alias for ~(top() + bottom())."""
        return self - (self.bottom(tolerance) + self.top(tolerance))
