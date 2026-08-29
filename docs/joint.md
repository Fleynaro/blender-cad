# Joints And Assembly

`Joint` is an oriented attachment point owned by a `Part`. It makes reusable assemblies explicit: define a port on each component, then place one component so its port meets another. A joint wraps a `Location` with ownership and optional deformation tracking; it is not geometry and does not appear in the finished mesh.

Import the public API from the package root:

```python
from blender_cad import *
```

## Core Workflow

Create a joint from a face location, then attach the child joint to the parent joint. The local Z directions face each other after alignment, which is appropriate for two mating surfaces.

```python
with BuildPart() as result:
    Box(5, 5, 1)
    top = result.part.joint(faces().top()[0].at(uv))

    with BuildPart(mode=Mode.PRIVATE) as child:
        Cylinder(0.5, 2)
        bottom = child.part.joint(faces().bottom()[0].at(uv))

    bottom.to(top, mode=Mode.JOIN)
```

`Joint(loc)` creates a joint on the active `BuildPart` result. Prefer `part.joint(loc)` when the owning part should be explicit, especially outside the part's build block. A joint retains its position and orientation relative to its owner, so reading `.loc` always returns its current world-space location after the owner moves.

`Joint.to(target, ...)` moves the source joint's owner and integrates that owner into the active build context. `target` may be another `Joint` or a `Location`. Its main arguments are:

| Argument | Meaning |
| --- | --- |
| `op` | An optional `Transform` applied to the attached object after alignment. |
| `twist` | Optional degrees of rotation around the mating axis. |
| `move_only` | Preserve the source owner's current world rotation instead of rotating it to face the target. |
| `mode` | Integration mode passed to `add`, such as `Mode.JOIN`, `Mode.ADD`, or `Mode.PRIVATE`. |
| `mat`, `tag` | Optional material and tag settings passed to `add`. |

`joint.offset(location)` returns a new joint on the same owner at a location offset from the original. This is useful for a controlled gap between mating faces:

```python
bottom.to(top.offset(Pos(Z=0.1)), mode=Mode.JOIN)
```

The direct connection and offset behavior are exercised by `test_joint_connection` in [`tests/test_locations.py`](../tests/test_locations.py).

## `align`: The Placement Primitive

`align(from_port, to_port, twist=None, rot=None)` is the lower-level operation used by `Joint.to`. It returns a `Location`; it does not create a joint, move a part, or add geometry. Use it directly when two ordinary locations are sufficient:

```python
with BuildPart(mode=Mode.JOIN) as child:
    Box(5, 7, 1)
    source = child.faces().bottom()[0].at(0.5, 0.5)
    target = parent_top_face.at(1.0, 0.5)
    child.loc = align(source, target, twist=45)
```

Without `twist` or `rot`, `align` takes the shortest rotation that makes the source and target local Z axes face each other. `twist=<degrees>` uses the target's complete frame, flips the source to face it, then rotates around the resulting local Z axis. `rot=<Quaternion>` supplies the final world rotation explicitly. In every mode, the source port origin ends at the target port origin.

Use `align` for one-off face, curve, or manually constructed locations. Use `Joint` when the attachment point belongs to a component, needs a stable name, or must follow mesh deformation. The standalone shortest-arc and twist cases are covered by `test_align_joints` and `test_align_joints_with_twist` in [`tests/test_locations.py`](../tests/test_locations.py).

## Named Joints

Parts can expose named connection points in the same way that geometry can carry meaningful labels. Register a joint with `register_joint`, then retrieve it later with `joint_by_name`:

```python
with BuildPart(mode=Mode.PRIVATE) as stair:
    Box(2, 1, 0.4)
    stair.part.register_joint(
        "top",
        stair.part.joint(faces().top()[0].at(uv)),
        propagate=True,
    )

stair.part.joint_by_name("top")
```

`register_joint(name, joint, propagate=False)` requires a joint owned by that same part. Registering an existing name replaces its previous registration and returns the joint. `joint_by_name(name)` returns the first exact match or raises `KeyError`; `has_joint(name)` tests existence. `joints_by_name(*names)` supports exact names and `*` wildcards, preserving the requested-name order. The read-only `.joints` property exposes the registration records.

Set `propagate=True` for an interface joint that should remain available after the part is transferred into another object during assembly. Non-propagating registrations stay local to the source part. `test_joint_registration_and_propagation` verifies named lookup, propagation, and a subsequent connection using the propagated joint.

## Bounding-Box Joints

`part.bbox_joint(axis, selector=None, deformable=False)` creates a joint on an extreme face of the part's axis-aligned bounding-box representation. The signed axis chooses the outer face: `Axis.X` selects the positive X side and `-Axis.X` selects the negative X side. `selector` chooses the UV point on that rectangle and defaults to `uv`.

Bounding-box joints are useful for mostly box-like components where an attachment point should follow the component envelope rather than a specific source face. For example, a stair tread, cabinet, platform, or modular block can expose side anchors without calculating a surface location from its detailed mesh:

```python
left = stair.part.bbox_joint(-Axis.X)
right = stair.part.bbox_joint(Axis.X)
right.to(neighbor.part.bbox_joint(-Axis.X), mode=Mode.JOIN)
```

The bounding box is represented as a 3D part even for flat geometry, so this also works for planar or thin components. `test_flat_object_bbox_joint` and `test_bbox_joint_alignment_across_multiple_axes` in [`tests/test_locations.py`](../tests/test_locations.py) cover these cases.

## Deformable Joints

Set `deformable=True` when a joint must follow mesh-edit operations such as `bend`, `twist`, `simple_deform`, or vertex transforms. A normal joint preserves its stored local frame; it does not infer where an arbitrary deformation moved that frame. A deformable joint is updated after a supported operation:

```python
with BuildPart() as result:
    Box(2, 1, 0.1)
    end = result.part.bbox_joint(Axis.X, deformable=True)
    subdivide(faces=faces().bottom() + faces().top(), cuts=4)
    bend(angle=180, axis=Axis.Y)
    Marker(end.loc, mat=mat.red)
```

Internally, `Part._inject_joint_markers()` adds three temporary loose BMesh vertices for each deformable joint: its origin and tiny local-X and local-Y reference points. Blender transforms these vertices with the surrounding mesh. After the operation, `Part._sync_joint_markers()` reads them back, reconstructs an orthonormal local frame, updates the joint, and removes the temporary vertices. These underscored methods are implementation details, not public API.

The reference frame must remain non-degenerate. If deformation collapses the origin-to-X vector or makes the X and Y vectors collinear, frame reconstruction raises `RuntimeError` rather than leaving a silently invalid joint. `test_bend_operation_deformable_joints` in [`tests/test_locations.py`](../tests/test_locations.py) is the regression test for a bent plate with deformable bounding-box joints.

## Tested References

- [`tests/test_locations.py`](../tests/test_locations.py): direct alignment, joint connection, named registration and propagation, bounding-box joints, and deformable bend tracking.
- [`tests/test_chain.py`](../tests/test_chain.py): chain assembly uses deformable bounding-box joints as default connection points when an item does not provide a custom one.
- [`tests/test_ml.py`](../tests/test_ml.py): reusable ML components connect through named and bounding-box joints.
