from enum import Enum
from typing import TYPE_CHECKING, List, Any, Optional, Union
from mathutils import Vector

from .part import Part
from .location import Transform, Location, VectorLike, Scale

if TYPE_CHECKING:
    from .common import Axis
    from .geometry import TopologyConfig
    from .material import MaterialLayer

class Mode(Enum):
    """
    Defines how the current geometry interacts with the existing Part.
    
    ADD: Boolean Union operation. Merges geometry and removes internal/occluded vertices.
    SUBTRACT: Boolean Difference operation.
    INTERSECT: Boolean Intersection operation.
    JOIN: Blender-style Join. Merges objects into one mesh without performing 
          boolean calculations (keeps internal geometry and separate materials).
    PRIVATE: Geometry is created but not automatically added to the main Part.
    """
    ADD = "ADD"
    SUBTRACT = "SUBTRACT"
    INTERSECT = "INTERSECT"
    JOIN = "JOIN"
    PRIVATE = "PRIVATE"

class BuildPart:
    """
    Context manager for part construction with stack-based nesting support.
    Manages the lifecycle of a Part and its integration into parent contexts.
    """
    # Stack of active contexts (class-level variable)
    _context_stack: List['BuildPart'] = []

    def __init__(self, part: Optional['Part'] = None, mat: Optional['MaterialLayer'] = None, topology: Optional['TopologyConfig'] = None, offset: Location = Location(), mode: Mode = Mode.ADD):
        """
        Initialize a new construction context.
        
        :param mat: The material to apply to the part.
        :param topology: Configuration for topology handling.
        :param mode: The interaction mode (ADD, SUBTRACT, etc.) when the context exits.
        """
        self.part = part or Part()
        if topology:
            self.part.topology_config = topology
        self._mat = mat
        self._mode = mode
        self._locations_stack: List[List[Location]] = [[offset]]

    def __enter__(self) -> 'BuildPart':
        """Push the current context onto the stack."""
        BuildPart._context_stack.append(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        """Pop the context from the stack and integrate the result into the parent context."""
        BuildPart._context_stack.pop()
        
        if exc_type is None and self.part:
            if self._mat:
                self.default_mat = self._mat
            # If there is a parent context, automatically add this part to it
            if BuildPart._context_stack:
                from .modifiers import add
                add(self, mode=self._mode, _make_copy=False)

    @classmethod
    def _get_context(cls) -> 'BuildPart':
        """Retrieve the currently active context from the top of the stack."""
        if not cls._context_stack:
            raise RuntimeError("Operation must be performed within a 'with BuildPart()' context")
        ctx = cls._context_stack[-1]
        if not ctx.is_valid:
            raise RuntimeError("BuildPart context is invalid")
        return ctx
    
    @property
    def is_valid(self):
        """Checks if the current part is valid (not removed)."""
        return self.part.is_valid

    @property
    def transform(self):
        """Access the transformation of the current part."""
        return self.part.transform
    
    @transform.setter
    def transform(self, value: 'Transform'):
        """Set the transformation of the current part."""
        self.part.transform = value

    @property
    def loc(self):
        """Access the location of the current part."""
        return self.part.loc
    
    @loc.setter
    def loc(self, loc: 'Location'):
        """Set the location of the current part."""
        self.part.loc = loc

    @property
    def scale(self):
        """Access the scale of the current part."""
        return self.part.scale
    
    @scale.setter
    def scale(self, value: Union['VectorLike', 'Scale']):
        """Set the scale of the current part."""
        self.part.scale = value
    
    @property
    def size(self):
        """Access the size of the current part."""
        return self.part.size
    
    @size.setter
    def size(self, value: 'VectorLike'):
        """Set the size of the current part."""
        self.part.size = value
    
    @property
    def bbox(self):
        """Access the bounding box of the current part."""
        return self.part.bbox

    @property
    def mat(self):
        """Get or set the material of the current part."""
        return self.part.mat

    @mat.setter
    def mat(self, material: Optional['MaterialLayer']):
        """Set the material for the current part."""
        self.part.mat = material

    @property
    def default_mat(self):
        """Get or set the default material of the current part."""
        return self.part.default_mat

    @default_mat.setter
    def default_mat(self, material: Optional['MaterialLayer']):
        """Set the default material for the current part."""
        self.part.default_mat = material

    @property
    def topology_config(self):
        """Access the topology configuration for the current part."""
        return self.part.topology_config

    @default_mat.setter
    def topology_config(self, topology: 'TopologyConfig'):
        """Set the topology configuration for the current part."""
        self.part.topology_config = topology

    def set_scale(self, value: float, axis: Union['Axis', 'Vector']):
        """Set scale along axis for the current part. Use -Axis to anchor the opposite side."""
        self.part.set_scale(value, axis)

    def set_size(self, value: float, axis: Union['Axis', 'Vector']):
        """Set size along axis for the current part. Use -Axis to anchor the opposite side."""
        self.part.set_size(value, axis)

    def make_checkpoint(self):
        """Make a checkpoint of the current part."""
        return self.part.make_checkpoint()
    
    def shoot(
        self,
        camera_loc: Location,
        fov: float = 39.6,
        resolution: tuple[int, int] = (1024, 1024),
        offset: Location = Location(),
        label: str | None = None
    ):
        """Temporarily shows the current part, captures it with a camera, and returns a CameraTexture."""
        return self.part.shoot(camera_loc, fov, resolution, offset, label)

    # --- Geometry Selectors ---
    def faces(self): 
        """Return faces of the current part."""
        return self.part.faces()
    
    def wires(self): 
        """Return wires of the current part."""
        return self.part.wires()
    
    def edges(self): 
        """Return edges of the current part."""
        return self.part.edges()
    
    def vertices(self): 
        """Return vertices of the current part."""
        return self.part.vertices()
    
def make_checkpoint(): 
    """Make a checkpoint of the current active BuildPart context."""
    return BuildPart()._get_context().make_checkpoint()

def faces(): 
    """Retrieve faces from the current active BuildPart context."""
    return BuildPart()._get_context().faces()

def wires(): 
    """Retrieve wires from the current active BuildPart context."""
    return BuildPart()._get_context().wires()

def edges(): 
    """Retrieve edges from the current active BuildPart context."""
    return BuildPart()._get_context().edges()

def vertices(): 
    """Retrieve vertices from the current active BuildPart context."""
    return BuildPart()._get_context().vertices()

def set_mat(mat: 'MaterialLayer'): 
    """Set the material for the current active BuildPart context."""
    BuildPart()._get_context().mat = mat

def set_default_mat(mat: 'MaterialLayer'): 
    """Set the default material for the current active BuildPart context."""
    BuildPart()._get_context().default_mat = mat

def set_topology(topology: 'TopologyConfig'): 
    """Set the topology config for the current active BuildPart context."""
    BuildPart()._get_context().topology_config = topology

def set_mode(mode: Mode): 
    """Set the interaction mode for the current active BuildPart context."""
    BuildPart()._get_context()._mode = mode