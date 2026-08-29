# Chain Assembly

`blender_cad.chain` assembles [`Part`](part.md)-like components end to end by aligning [`Joint`](joint.md) objects. It is intended for turning reusable 2D or shallow 3D modules into connected 3D structures: walls into rooms or corridors, panels into a box, or repeated tiles into a circular shell. A chain is an assembly operation, not a general constraint solver. Each concrete item is positioned relative to the preceding concrete item, then the resulting parts are joined into the active `BuildPart`.

Import the public API from the package root:

```python
from blender_cad import *
```

`chain` works particularly well with [`ml`](ml.md). Use ML to make a wall, floor, ceiling, facade, or other layout-driven component, then use `chain` to connect those components in 3D. The relationship also works in reverse: a chain-built `Part` or bounding-box `Joint` can be passed to ML and laid out as reusable component geometry.

## Basic Assembly

Pass parts, `ml` nodes, or other supported `PartLike` values as positional items. `chain.build()` adds the completed assembly to the active `BuildPart`; `.part` builds the same assembly privately as a reusable `Part`.

The default progression axis is `Axis.X`. A new item connects its joint on the negative progression axis (`FROM`) to the preceding item's joint on the positive progression axis (`TO`). If a part has no appropriate named joint, `chain` uses a deformable bounding-box joint on that side.

```python
wall = ml(style(width=3, height=2, background_mat=mat.blue))

with BuildPart() as result:
    chain(
        wall,
        90,
        wall,
        90,
        wall,
        90,
        wall,
        axis=Axis.X,
        rot_axis=Axis.Y,
    ).build()
```

Numbers are rotation steps in degrees around `rot_axis`; `Rot(...)` items are full explicit rotations. Both affect the placement of following concrete parts. The example folds four panels around the Y axis. It is the same composition pattern used by the tested box and corridor assemblies.

Use `side` as a concise planar setup:

| `side` | Progression axis | Default rotation axis |
| --- | --- | --- |
| `"left"` | `-Axis.X` | `Axis.Y` |
| `"right"` | `Axis.X` | `Axis.Y` |
| `"top"` | `-Axis.Y` | `Axis.X` |
| `"bottom"` | `Axis.Y` | `Axis.X` |

An explicit `axis` or `rot_axis` is preferable when the intended local frame is not one of those four cases.

## Items, Joints, And Attachment

The public item stream accepts flattened lists and tuples of:

- A `PartLike` component, including an `ml` component.
- A number or `Rot`, which changes the rotation accumulated for later components.
- `chain.attach(...)`, which changes how the next concrete component is joined.
- `chain.bend(...)`, which deforms the next concrete component before it is joined.
- A nested `chain`, which creates a branch from the most recently built component.
- `chain.clear_rotation`, which resets the accumulated rotation for later items.

`axis` and `rot_axis` are inherited by a nested chain unless that child supplies its own values. `item_width` and `item_height` are passed to components that accept dimensions during extraction, which lets one dimensionless ML wall definition serve multiple assemblies.

`from_joint` and `to_joint` constructor callbacks select custom joints for every applicable item. They receive the current `Part` and should return a `Joint` or `None`; returning `None` falls back to the normal chain-joint lookup. Name a part's joints using the internal chain convention only when maintaining library internals. For public composition, use the callbacks or `chain.attach`.

`chain.attach(to_joint=..., from_joint=..., move_only=False, twist=...)` is a transient override for the next concrete item:

- `to_joint` selects the joint of the preceding component.
- `from_joint` selects the joint of the component being placed.
- Either value may be an `Axis`, a `Vector`, or a callable that receives a `Part` and returns a `Joint`.
- `move_only=True` preserves the incoming component's orientation while moving its selected joint to the target.
- `twist` supplies the optional twist angle used by `Joint.to` during alignment.

This is useful for T-junctions and other branches that must begin from a particular side of the anchor component:

```python
with BuildPart() as result:
    chain(
        corridor,
        chain(
            chain.attach(to_joint=Axis.X),
            corridor,
            corridor,
        ),
        axis=Axis.Y,
        rot_axis=Axis.Z,
    ).build()
```

## Branches And Parent Clipping

A nested `chain` is a subchain. It uses the current branch's previous concrete item as its anchor, so its position in the parent item stream determines where it sprouts. The parent branch is built first and the child branch is evaluated afterward. This ordering lets `clip_by_parent` use completed parent geometry and prevents the branch mesh from being accidentally treated as a later main-path item.

Set `clip_by_parent=True` to intersect a child branch with an extruded projection of its parent's convex hull. Pass a `Mode` such as `Mode.SUBTRACT` instead to use that Boolean mode for the clipping part. For perpendicular parent and child progression axes, a child without its own dimensions also receives dimensions derived from the parent's convex hull. This supports walls or openings that fit the parent panel without duplicating sizes.

```python
wall = ml(style(background_mat=mat.red))

with BuildPart() as result:
    chain(
        wall,
        chain(Rot(X=90), wall, axis=Axis.Y, clip_by_parent=True),
        chain(Rot(X=-90), wall, axis=-Axis.Y, clip_by_parent=Mode.SUBTRACT),
        axis=-Axis.X,
        item_width=2,
        item_height=2,
    ).build()
```

[Tags](tags.md) passed through `tag=` are applied to the completed branch faces. They can then be selected and transformed with the standard [selector API](selectors.md). Connected chain geometry can share topology at joins, so later edits to tagged faces may deform adjacent connected sections as well.

## Repeated Curves And Bends

`chain.twist(*items, angle=..., axis=None, segments=1, ensure_angle=False)` produces an item list for a parent `chain`. It repeats the supplied pattern for `segments` iterations and inserts incremental rotations between concrete parts. This lets repeated panels form arcs, cylindrical shells, or full loops.

```python
with BuildPart() as result:
    chain(
        chain.twist(
            ml(style(width=2, height=1, background_mat=mat.red)),
            ml(style(width=2, height=1, background_mat=mat.blue)),
            axis=Axis.X,
            angle=360,
            segments=5,
        ),
        axis=Axis.Y,
    ).build()
```

With one callable item, `chain.twist` calls it once per segment with the zero-based segment index. The callback can conditionally return nested chains, making radial caps or branches possible. `ensure_angle=True` changes the rotation-step calculation to include the closing transition when distributing the requested angle.

`chain.bend(angle, axis=None, segments=None)` is a one-shot operation item. It deforms the next concrete part with the modifier-level bend operation before connecting it, while registering its chain joints first so subsequent alignment still follows the deformed component. Multiple adjacent bend items apply sequentially to that next component. Use it to create curved roads, corridors, and other assemblies of already-volumetric modules.

## ML Interoperability

ML and chain are composable in both directions:

- **ML inside chain:** define wall panels with ML, including text, windows, borders, extrusion, and cutouts, then connect them into a room or corridor. `test_chain_bend_on_solid_3d_corridor_with_windows` builds a hollow corridor from ML walls, floor/ceiling-like faces, and window inserts, then chains and bends the completed corridor modules.
- **Chain inside ML:** obtain a reusable assembly with `.part` or an anchor with `.bbox_joint(axis)`, then pass it to `ml(...)`. ML measures it as embedded geometry and can place repeated instances in its flow. `test_component_reusability_with_bbox_joints` creates a four-segment chain loop, obtains its X bounding-box joint, and reuses that component three times in one ML layout.

Use `.bbox_joint(axis, selector=None)` when an assembly needs a public connection point. It builds the chain privately and returns the requested bounding-box joint from the resulting `Part`; the optional selector is forwarded to `Part.bbox_joint`.

## Build Behavior And Limits

Chain makes a copy when extracting a reusable source component, so a component definition can appear more than once in an assembly. It records each extracted part's initial transform, reapplies that local transform before resolving its joints, and restores accumulated placement after lookup. This prevents the transform from a prior chain placement from becoming the basis for a later instance.

Building uses an isolated `BuildPart` context per branch. The branch first assembles its main path, applies optional parent clipping, recursively builds deferred child branches, joins those branch results, fixes topology using `min_vert_dist`, applies face tags, then applies the branch `transform`. Keep source components and callback-based joint selection deterministic between builds.

The API does not solve global collisions or force a loop's final seam to close. Select compatible component dimensions, joints, axes, and rotations yourself. Use [`Rule-Based Layout`](rbl.md) when the primary problem is solving spatial constraints rather than joining an ordered assembly.

## Tested References

- [`tests/test_chain.py`](../tests/test_chain.py): sequential assembly, 3D box folding, nested branches, custom attachments, twists, bends, parent clipping, propagated dimensions, tags, and corridor construction.
- [`tests/test_ml.py`](../tests/test_ml.py): `test_component_reusability_with_bbox_joints` verifies a chain-built component and its bounding-box joint reused inside ML.
- [`docs/ml.md`](ml.md): 2D layout, reusable ML components, extrusion, and joints.
- [`docs/part.md`](part.md): `Part`, private building, Boolean modes, transforms, and bounding-box joints.
- [`docs/location.md`](location.md): `Pos`, `Rot`, `Transform`, and axis-oriented placement.
