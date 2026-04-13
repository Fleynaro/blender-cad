from typing import Optional, Union

from .modifiers import add
from .build_part import BuildPart, Mode
from .location import Location, Transform, align
from .part import Part

class Joint:
    """Represents a joint for connecting two parts."""
    def __init__(self, loc: Location, part: Optional[Part] = None):
        self.part = part or BuildPart._get_context().part
        self._rel_loc = self.part.loc.inverse * loc

    @property
    def loc(self):
        """Access the global location of the joint."""
        return Location(self._rel_loc.matrix, parent_loc=self.part.loc)
    
    def offset(self, value: Location):
        return Joint(self.loc * value, self.part)

    def to(self, joint: Union['Joint', 'Location'], op: Optional[Transform] = Transform(), twist: Optional[float] = None, move_only: bool = False, mode: Mode = Mode.ADD):
        """Moves the current part to align with the specified joint or location and adds it to the current context with the specified mode."""
        from_port=self._rel_loc
        to_port=joint.loc if isinstance(joint, Joint) else joint
        rot=self.part.loc.quaternion if move_only else None
        self.part.loc = align(from_port, to_port, twist, rot)
        self.part.transform *= op
        add(self.part, mode=mode)
