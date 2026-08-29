from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional, TypeAlias, Union

from mathutils import Vector

from .build_part import BuildPart, Mode
from .common import (
    Axis,
    PartLike,
    VectorLike,
    _flatten_items,
    extract_part,
    extract_vector,
    tag_to_list,
)
from .geometry import UVSelector
from .joint import Joint
from .location import Location, Pos, Rot, Transform, TransformExpr
from .modifiers import add, bend, extrude
from .part import Part


class ChainJoint(Enum):
    TO = "TO"
    FROM = "FROM"


def _chain_joint_prefix(joint_name: str) -> str:
    return f"__chain_joint_{joint_name}__"


def _chain_joint_name(joint: ChainJoint) -> str:
    return _chain_joint_prefix(joint.value.lower())


def _chain_joint_axis_name(axis_like: VectorLike) -> str:
    axis, neg = Axis.from_vector(axis_like)
    return _chain_joint_prefix(f"-{axis.name.lower()}" if neg else axis.name.lower())


@dataclass(slots=True)
class ChainBuildContext:
    branch: "chain"
    branch_part: "Part" = field(default_factory=Part)
    parent: Optional["ChainBuildContext"] = None
    item_part: Optional["Part"] = None
    prev_item_part: Optional["Part"] = None
    rot_transform: Rot = field(default_factory=Rot)
    initial_transforms: dict["Part", "Transform"] = field(default_factory=dict)
    item_width: Optional[float] = None
    item_height: Optional[float] = None

    deferred_children: list[tuple["chain", Optional["Part"], Rot]] = field(
        default_factory=list
    )
    active_attach: Optional["chain.attach"] = None
    active_operations: list["chain._operation"] = field(default_factory=list)


class chain:
    class clear_rotation:
        pass

    @dataclass(slots=True)
    class attach:
        """A transient modifier placed in branch.items to temporarily override
        the connection joints for the single subsequent item.
        """

        to_joint: Optional[
            Union[Axis, Vector, Callable[["Part"], Optional["Joint"]]]
        ] = None
        from_joint: Optional[
            Union[Axis, Vector, Callable[["Part"], Optional["Joint"]]]
        ] = None
        move_only: bool = False
        twist: Optional[float] = None

    class _operation(ABC):
        @abstractmethod
        def _execute(self, ctx: ChainBuildContext): ...

    class bend(_operation):
        """Bends the Part along a specified axis. Can optionally subdivide the mesh before bending."""

        def __init__(
            self,
            angle: float,
            axis: Optional[Axis] = None,
            segments: Optional[int] = None,
        ):
            self.angle = angle
            self.axis = axis
            self.segments = segments

        def _execute(self, ctx: ChainBuildContext):
            branch = ctx.branch
            part = ctx.item_part
            rot_axis = self.axis or branch.resolve_rot_axis(ctx)
            part.register_joint(
                _chain_joint_name(ChainJoint.FROM),
                branch.resolve_item_joint(part, ChainJoint.FROM, ctx),
            )
            part.register_joint(
                _chain_joint_name(ChainJoint.TO),
                branch.resolve_item_joint(part, ChainJoint.TO, ctx),
            )
            branch_axis = extract_vector(branch.resolve_axis(ctx))

            with BuildPart(part=part, mode=Mode.PRIVATE):
                angle = self.angle
                origin = Location()
                if branch_axis in (Axis.X.value, -Axis.Y.value) or rot_axis == Axis.Z:
                    angle = -angle
                if rot_axis == Axis.Z:
                    if branch_axis in (Axis.X.value, -Axis.X.value):
                        origin = (
                            Rot(X=-90) * Rot(X=90, Z=-90) * Pos(X=-part.size.y * 0.5)
                        )
                    else:
                        origin = Pos(X=part.size.x * 0.5)
                    if branch_axis in (-Axis.X.value, -Axis.Y.value):
                        origin *= Rot(Z=180)
                bend(angle=angle, axis=rot_axis, segments=self.segments, origin=origin)

            cur_rot_transform = Rot(rot_axis.value * self.angle)
            if branch_axis in (Axis.X.value, Axis.Y.value):
                part.transform *= cur_rot_transform.inverse
            ctx.rot_transform *= cur_rot_transform

    ItemType: TypeAlias = Union[PartLike, Rot, int, float, attach, _operation]

    def __init__(
        self,
        *items: ItemType,
        axis: Optional[Axis | Vector] = None,
        rot_axis: Optional[Axis] = None,
        from_joint: Optional[Callable[["Part"], Optional["Joint"]]] = None,
        to_joint: Optional[Callable[["Part"], Optional["Joint"]]] = None,
        item_width: Optional[float] = None,
        item_height: Optional[float] = None,
        clip_by_parent: bool | Mode = False,
        tag: Optional[str | Iterable[str]] = None,
        transform: Optional[TransformExpr] = None,
        side: Optional[Literal["left", "right", "top", "bottom"]] = None,
    ):
        if side is not None:
            axis = {"left": -Axis.X, "right": Axis.X, "top": -Axis.Y, "bottom": Axis.Y}[
                side
            ]
            rot_axis = Axis.X if side in ("bottom", "top") else Axis.Y

        self.items: list[chain.ItemType] = list(_flatten_items(items))
        self.axis = axis
        self.rot_axis = rot_axis
        self.from_joint = from_joint
        self.to_joint = to_joint
        self.item_width = item_width
        self.item_height = item_height
        self.clip_by_parent = clip_by_parent
        self.tag = tag
        self.transform = transform

    @property
    def part(self):
        return self.build(mode=Mode.PRIVATE)

    def bbox_joint(self, axis: Axis | Vector, selector: Optional["UVSelector"] = None):
        return self.part.bbox_joint(axis, selector)

    def resolve_item_joint(
        self, part: "Part", joint: ChainJoint, ctx: ChainBuildContext
    ) -> "Joint":
        custom_joint = self.to_joint if joint == ChainJoint.TO else self.from_joint
        if custom_joint:
            j = custom_joint(part)
            if j:
                return j
        resolved_axis = self.resolve_axis(ctx)
        axis = resolved_axis if joint == ChainJoint.TO else -resolved_axis
        reg_joints = part.joints_by_name(
            _chain_joint_axis_name(axis), _chain_joint_name(joint)
        )
        if reg_joints:
            return reg_joints[0]
        return part.bbox_joint(axis, deformable=True)

    def resolve_axis(self, ctx: ChainBuildContext) -> Axis | Vector:
        if self.axis is not None:
            return self.axis
        if ctx.parent:
            return ctx.parent.branch.resolve_axis(ctx.parent)
        return Axis.X

    def resolve_rot_axis(self, ctx: ChainBuildContext) -> Axis:
        if self.rot_axis is not None:
            return self.rot_axis
        if ctx.parent:
            return ctx.parent.branch.resolve_rot_axis(ctx.parent)
        return Axis.Y

    def resolve_part(self, part_like: PartLike, ctx: ChainBuildContext):
        return extract_part(
            part_like,
            ensure_copy=True,
            width=self.item_width or ctx.item_width,
            height=self.item_height or ctx.item_height,
        )

    def build(self, mode: Mode = Mode.ADD, min_vert_dist=1e-4):
        def _build(ctx: ChainBuildContext):
            branch = ctx.branch
            # Build a branch privately. Nested branches need the completed parent mesh for
            # clipping and dimension propagation, so they are merged only after this phase.
            with BuildPart(part=ctx.branch_part, mode=Mode.PRIVATE):
                prev_rot_transform = ctx.rot_transform

                # PHASE 1: Build the parent/current branch level completely
                for item in branch.items:
                    if isinstance(item, (int, float)):
                        ctx.rot_transform *= Rot(
                            branch.resolve_rot_axis(ctx).value * item
                        )
                        continue

                    if isinstance(item, Rot):
                        ctx.rot_transform *= item
                        continue

                    if isinstance(item, chain.clear_rotation):
                        ctx.rot_transform = Rot()
                        continue

                    if isinstance(item, chain.attach):
                        ctx.active_attach = item
                        continue

                    if isinstance(item, branch._operation):
                        ctx.active_operations.append(item)
                        continue

                    if isinstance(item, chain):
                        # Defer child chain evaluation until the current branch geometry is fully realized
                        ctx.deferred_children.append(
                            (item, ctx.prev_item_part, ctx.rot_transform)
                        )
                        continue

                    part = branch.resolve_part(item, ctx)
                    ctx.item_part = part

                    if ctx.active_operations:
                        # Operations are one-shot modifiers: they affect this next concrete part
                        # rather than every remaining item in the branch.
                        for op in ctx.active_operations:
                            op._execute(ctx)
                        ctx.active_operations.clear()

                    if part not in ctx.initial_transforms:
                        ctx.initial_transforms[part] = part.transform

                    part.transform = ctx.initial_transforms[part]

                    from_joint = None
                    if (
                        ctx.active_attach is not None
                        and ctx.active_attach.from_joint is not None
                    ):
                        spec = ctx.active_attach.from_joint
                        if callable(spec):
                            from_joint = spec(part)
                        elif isinstance(spec, (Axis, Vector)):
                            from_joint = part.bbox_joint(spec, deformable=True)
                    if from_joint is None:
                        from_joint = branch.resolve_item_joint(
                            part, ChainJoint.FROM, ctx
                        )

                    part.transform = ctx.rot_transform * ctx.initial_transforms[part]

                    if ctx.prev_item_part is not None:
                        current_prev_transform = ctx.prev_item_part.transform
                        ctx.prev_item_part.transform = ctx.initial_transforms[
                            ctx.prev_item_part
                        ]

                        prev_to_joint = None
                        move_only = True
                        twist = None
                        if ctx.active_attach is not None:
                            move_only = ctx.active_attach.move_only
                            twist = ctx.active_attach.twist
                            if ctx.active_attach.to_joint is not None:
                                spec = ctx.active_attach.to_joint
                                if callable(spec):
                                    prev_to_joint = spec(ctx.prev_item_part)
                                elif isinstance(spec, (Axis, Vector)):
                                    prev_to_joint = ctx.prev_item_part.bbox_joint(
                                        spec, deformable=True
                                    )
                        if prev_to_joint is None:
                            prev_to_joint = branch.resolve_item_joint(
                                ctx.prev_item_part, ChainJoint.TO, ctx
                            )

                        # Resolve the previous joint in its unrotated local frame, then restore
                        # its accumulated placement before aligning the current part to it.
                        ctx.prev_item_part.transform = current_prev_transform
                        from_joint.to(
                            prev_to_joint,
                            move_only=move_only,
                            twist=twist,
                            mode=Mode.JOIN,
                        )
                        if not move_only:
                            ctx.rot_transform *= (
                                prev_rot_transform.inverse * part.transform.rotation_loc
                            )
                    else:
                        add(part, mode=Mode.JOIN)

                    ctx.active_attach = None
                    ctx.prev_item_part = part
                    prev_rot_transform = ctx.rot_transform

                # PHASE 2: Apply clipping to the current branch if requested and parent exists
                if branch.clip_by_parent and ctx.parent is not None:
                    # Generate the clipping boundary from the parent's convex hull projected to 2D and extruded
                    convex_hull = ctx.parent.branch_part.convex_hull_part
                    plane, _ = Axis.from_vector(branch.resolve_axis(ctx))
                    clip_part = convex_hull.project_2d(plane)
                    with BuildPart(part=clip_part, mode=Mode.PRIVATE):
                        direction: Vector = plane.value
                        direction_size = direction * convex_hull.size * 5.0
                        extrude(op=Pos(direction_size), recalc_normals=True)
                        clip_part.loc = Pos(-direction_size * 0.5)
                    add(
                        clip_part,
                        mode=branch.clip_by_parent
                        if isinstance(branch.clip_by_parent, Mode)
                        else Mode.INTERSECT,
                    )

                # Child branches attach at the point where they appeared in items, but run after
                # this branch so parent clipping can use finalized geometry.
                child_parts: list["Part"] = []
                for child_chain, child_prev, child_rot in ctx.deferred_children:
                    child_ctx = ChainBuildContext(
                        branch=child_chain,
                        parent=ctx,
                        prev_item_part=child_prev,
                        rot_transform=child_rot,
                        initial_transforms=ctx.initial_transforms,
                    )
                    if child_chain.clip_by_parent:
                        axis, _ = Axis.from_vector(ctx.branch.resolve_axis(ctx))
                        child_axis, _ = Axis.from_vector(child_chain.axis)
                        if axis in (Axis.X, Axis.Y) and axis != child_axis:
                            norm_axis = (Axis.all() - {axis, child_axis}).pop()
                            cur_size = ctx.branch_part.convex_hull_part.size
                            width = cur_size[axis.index]
                            height = cur_size[norm_axis.index]
                            if axis == Axis.Y:
                                width, height = height, width
                            child_ctx.item_width = width
                            child_ctx.item_height = height

                    child_part = _build(child_ctx)
                    child_parts.append(child_part)

                for child_part in child_parts:
                    add(child_part, mode=Mode.JOIN)

            ctx.branch_part._fix_topology(min_vert_dist)
            ctx.branch_part.add_tags(tag_to_list(branch.tag), domain="FACE")
            if branch.transform:
                ctx.branch_part.transform = branch.transform
            return ctx.branch_part

        final_part = _build(ctx=ChainBuildContext(branch=self))
        add(final_part, mode=mode)
        return final_part

    @staticmethod
    def twist(
        *items: Union[ItemType, Callable[[int], Iterable[ItemType]]],
        angle: float,
        axis: Optional[Axis | Vector] = None,
        segments: int = 1,
        ensure_angle: bool = False,
    ):
        """
        Bends a sequence of PartLike items uniformly into an arc or circle.
        Repeats the input items for the given number of segments and
        injects incremental rotation angles into the chain.
        """

        def is_part(item: chain.ItemType):
            return not isinstance(item, (chain, Rot))

        is_callable = len(items) == 1 and callable(items[0])

        all_segments_items: list[list[chain.ItemType]] = []
        total_parts_count = 0

        for i in range(segments):
            current_source = items[0](i) if is_callable else items
            flat_items = list(_flatten_items(current_source))

            parts_in_segment = sum(1 for item in flat_items if is_part(item))

            total_parts_count += parts_in_segment
            all_segments_items.append(flat_items)

        # Rotations occur between concrete parts. ensure_angle includes the closing transition
        # in the denominator, which makes a repeated pattern complete the requested total angle.
        step_angle = angle / (total_parts_count - (1 if ensure_angle else 0))

        # An explicit axis creates a full rotation; otherwise the scalar is interpreted by
        # the surrounding chain using its configured rotation axis.
        rot_step = Rot(extract_vector(axis) * step_angle) if axis else step_angle

        result_items: list[chain.ItemType] = []
        for current_items in all_segments_items:
            current_parts = [item for item in current_items if is_part(item)]

            for item in current_items:
                if item in current_parts and len(result_items) > 0:
                    result_items.append(rot_step)
                result_items.append(item)

        return result_items
