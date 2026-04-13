from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Optional, Union
from mathutils import Vector

if TYPE_CHECKING:
    from .location import Location

VectorLike = Union[Vector, tuple[Union[float, int]], list[Union[float, int]], float, int]

def extract_vector(value: VectorLike):
    if isinstance(value, (float, int)):
        return Vector((value, value, value))
    if isinstance(value, (Vector, tuple, list)):
        v = Vector(value)
        # Ensure 3D if 2D was provided
        if len(v) == 2:
            return Vector((v[0], v[1], 0))
        return v
    raise TypeError(f"Unsupported type for vector extraction: {type(value)}")

class Axis(Enum):
    """Enumeration for standard 3D axes as mathutils Vectors."""
    X = Vector((1, 0, 0))
    Y = Vector((0, 1, 0))
    Z = Vector((0, 0, 1))

    def __neg__(self):
        return -self.value

class CurveLike(ABC):
    """Abstract base class for all objects that can be evaluated as a path."""
    
    @abstractmethod
    def at(self, t: Optional[float] = None, t_m: Optional[float] = None) -> 'Location':
        """Returns a Location object at the given parameter or distance."""
        raise NotImplementedError

    @abstractmethod
    def length(self) -> float:
        """Returns the total length of the curve."""
        raise NotImplementedError