from collections.abc import Iterable
from enum import IntEnum
from typing import Optional, Union

from mathutils import Matrix, Vector

from .build_part import BuildPart, Mode
from .location import Location, Transform, align
from .material import mat
from .object import Object


class JointRole(IntEnum):
    """Marker roles used to reconstruct the joint local frame."""

    ORIGIN = 0
    X_AXIS = 1
    Y_AXIS = 2


class Joint:
    """Represents a joint for connecting two parts."""

    def __init__(
        self, loc: Location, object: Optional[Object] = None, deformable: bool = False
    ):
        self.object = object or BuildPart._get_context().part
        # Store the frame in owner-local space so object transforms keep the
        # joint attached without rewriting its definition.
        self._rel_loc = self.object.loc.inverse * loc
        self.deformable = deformable
        self._joint_id = self.object._alloc_joint_id()
        self.object._register_joint(self)

    @property
    def loc(self):
        """Access the global location of the joint."""
        return Location(self._rel_loc.matrix, parent_loc=self.object.loc)

    def offset(self, value: Location):
        return Joint(self.loc * value, self.object, self.deformable)

    def to(
        self,
        joint: Union["Joint", "Location"],
        op: Optional[Transform] = Transform(),
        twist: Optional[float] = None,
        move_only: bool = False,
        mode: Mode = Mode.ADD,
        mat: Optional[mat.Layer] = None,
        tag: Optional[str | Iterable[str]] = None,
    ):
        """Moves the current part to align with the specified joint or location and adds it to the current context with the specified mode."""
        from .modifiers import add

        from_port = self._rel_loc
        to_port = joint.loc if isinstance(joint, Joint) else joint
        # move_only intentionally supplies the existing world rotation while
        # still translating the source port onto the target port.
        rot = self.object.loc.quaternion if move_only else None
        self.object.loc = align(from_port, to_port, twist, rot)
        self.object.transform *= op
        add(self.object, mode=mode, mat=mat, tag=tag)

    def sync_from_frame(self, origin: Vector, x_marker: Vector, y_marker: Vector):
        """Updates the joint frame from three reconstructed marker vertices."""
        self._rel_loc = self._frame_to_local_location(origin, x_marker, y_marker)

    def _frame_to_local_location(self, o: Vector, px: Vector, py: Vector) -> Location:
        """Converts three marker points into a rigid local Location."""
        eps = 1e-9

        x = px - o
        y_raw = py - o

        if x.length < eps:
            raise RuntimeError("Joint frame X axis collapsed during deformation")

        x.normalize()

        # Remove X projection from Y to keep the frame orthogonal.
        y = y_raw - x * x.dot(y_raw)
        if y.length < eps:
            raise RuntimeError("Joint frame Y axis collapsed during deformation")

        y.normalize()
        z = x.cross(y)
        if z.length < eps:
            raise RuntimeError("Joint frame became degenerate during deformation")
        z.normalize()

        # Rebuild Y to guarantee orthonormality.
        y = z.cross(x).normalized()

        # Blender matrices are easiest to build from the basis vectors.
        mat = Matrix(
            (
                (x.x, y.x, z.x, o.x),
                (x.y, y.y, z.y, o.y),
                (x.z, y.z, z.z, o.z),
                (0.0, 0.0, 0.0, 1.0),
            )
        )
        return Location(mat)
