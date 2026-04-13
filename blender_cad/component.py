from typing import Optional, Self, Union

from .geometry import uv
from .material import Meta, PBRMaterial, mat
from .build_part import BuildPart, Mode, faces
from .joint import Joint
from .location import Location, Locations, Scale
from .modifiers import transform
from .primitives import Box, Cone
from .common import Axis

class Component(BuildPart):
    def __init__(self, mode: Mode = Mode.PRIVATE):
        super().__init__(mode=mode)

    def _joint(self, loc: Location):
        return Joint(loc, part=self.part)
    
    def __enter__(self) -> Self:
        return super().__enter__()

class BoxComp(Component):
    def __init__(self, length: float, width: float, height: float, taper: float = 1, freeze_joints: bool = True):
        super().__init__()
        with self:
            Box(length, width, height)
            transform(faces().top(), op=Scale(taper))
            self.freezed_faces = faces() if freeze_joints else None

    def freeze_joints(self):
        self.freezed_faces = self.faces()

    def unfreeze_joints(self):
        self.freezed_faces = None

    def update_freezed_joints(self):
        if not self.freezed_faces:
            raise RuntimeError("Freezed joints not set.")
        self.freezed_faces = self.faces()

    def _faces(self):
        return self.freezed_faces or self.faces()

    def j_face(self, axis: Axis, selector = uv):
        return self._joint(self._faces().sort_by(axis)[-1].at(selector))

    def j_top(self, selector = uv):
        return self.j_face(Axis.Z, selector)
    
    def j_bottom(self, selector = uv):
        return self.j_face(-Axis.Z, selector)
    
class Marker(Component):
    def __init__(self, loc: Optional[Union['Joint', 'Location']] = Location(), size = 1, mode: Mode = Mode.JOIN):
        super().__init__(mode=mode)
        with Locations(loc if isinstance(loc, Location) else loc.loc):
            with self:
                Cone(0.1, 0.01, 0.1)
                self.scale = size
                self.mat = mat.red + PBRMaterial(alpha=0.5) + Meta(name="MarkerMat")