# Locations And Transforms

The `location` API is the library's Blender `mathutils`-backed transform layer. It provides readable constructors, matrix composition, deferred object-aware transforms, placement contexts, and locations derived from faces and curves. Import its public API from the package root:

```python
from blender_cad import *
```

`Transform`, `Location`, and their subclasses are wrappers around a Blender world matrix used by [`Part`](part.md), [`Joint`](joint.md), and placement contexts. They use the same fundamental rule as matrix multiplication: the right-hand transform is applied first.

```python
transform = Pos(X=2) * Rot(Z=90)
point = transform * (1, 0, 0)
```

The code above first rotates the point around the origin, then translates it. Keep this ordering in mind when building a chain.

## Transform Values

`Transform` represents translation, Euler rotation, and scale. Its `.matrix` property is the underlying Blender 4x4 matrix. It exposes these decomposed values:

| Property | Meaning |
| --- | --- |
| `.position`, `.x`, `.y`, `.z` | Translation vector and components. |
| `.rotation`, `.rx`, `.ry`, `.rz` | XYZ Euler rotation in degrees. |
| `.euler_rad`, `.quaternion` | Rotation in Blender radians or quaternion form. |
| `.scale`, `.sx`, `.sy`, `.sz` | Per-axis scale. |
| `.forward`, `.left`, `.up` | Normalized local X, Y, and Z directions. |
| `.inverse` | Inverse `Transform`. |
| `.loc`, `.position_loc`, `.rotation_loc` | Location-only views with scale removed. |

`.values` contains nine solver-compatible values in this order: translation (three), rotation in degrees (three), then scale (three). The public `Vector` name is `SVector`, a `mathutils.Vector` subclass that additionally provides `.values` and unbounded `.bounds` for solver integration.

### Core Constructors

| API | Produces |
| --- | --- |
| `Pos(...)` | A `Location` containing translation only. |
| `Rot(...)` | A `Location` containing XYZ Euler rotation only; angles are degrees. |
| `Scale(...)` | A `Transform` containing scale only. |
| `Location(...)` | Translation and rotation, with scale removed. |
| `Transform(matrix)` | A transform from a Blender matrix, or identity when omitted. |

`Location` deliberately has unit scale. Constructing one from a scaled matrix retains its translation and rotation but removes scale. This keeps a placement separate from a part's dimensions. `Pos` and `Rot` are further-specialized `Location` values: they describe only translation or rotation.

Constructors accept either a vector-like positional value, three positional coordinates, or axis keyword arguments. `X`, `Y`, and `Z` apply to one axis. `XY`, `XZ`, `YZ`, and `XYZ` apply to groups. For `Pos` and `Rot`, applicable keyword values are added; for `Scale`, they are multiplied:

```python
offset = Pos(X=1, XY=2)          # (3, 2, 0)
rotation = Rot(X=15, YZ=30)      # (15, 30, 30) degrees
scale = Scale(XY=2, Z=0.5)       # (2, 2, 0.5)
```

`FlipX`, `FlipY`, and `FlipZ` are predefined scales with `-1` on the named axis. Applying `FlipZ` to a cone mirrors it around local Z; this is covered by `test_cone_flip_z_transformation` in [`tests/test_locations.py`](../tests/test_locations.py).

### Composition And Location Utilities

Multiplication is the central operation:

```python
loc = Pos(X=2, Z=1) * Rot(Y=45)
part.loc = loc
part.transform *= Scale(X=2)
```

Multiplying two `Transform` instances returns a `Transform`; multiplying two locations returns a scale-free `Location`; multiplying a transform by a vector returns the transformed vector. The `Part` properties have distinct responsibilities:

| Part property | Behavior |
| --- | --- |
| `.loc` | Read or set translation and rotation while preserving current scale. |
| `.scale` | Read or set relative scale while preserving the current location. |
| `.size` | Read or set scaled dimensions relative to the part's original local size. |
| `.transform` | Read or set the complete transform, including lazy expressions. |

This separation is verified by `test_independent_loc_and_scale` in [`tests/test_locations.py`](../tests/test_locations.py) and `test_location_update_preserves_scale` in [`tests/test_part.py`](../tests/test_part.py).

`Location.look_at(target, flip_z=False)` returns a new location at the same position whose local Z axis points at `target`. Use `flip_z=True` for a camera-style local negative Z direction. `reorder_axes(order)` returns a location with a permuted basis; `x_as_z()` and `y_as_z()` are shortcuts for common axis adaptations. A location can also carry a parent location internally; `.local` returns its position relative to that parent.

## Deferred Expressions

`TransformExpr` is the abstract base for transforms that may need information about their target object. `Transform`, `Location`, `Pos`, `Rot`, and `Scale` resolve immediately because they already own a concrete matrix. `Origin`, `Size`, and the axis-anchored helpers cannot always do so: their final matrix depends on the target part's local bounding box or original size.

Multiplying an expression produces a `TransformChain`, a small symbolic AST. When a part accepts the expression, `.resolve(obj)` walks the chain and returns a concrete `Transform` with a Blender matrix. This is why the following remains a valid expression until it is applied to a part:

```python
operation = Origin(X=0) * Scale(X=2) * Rot(Y=30)
result.transform = operation
```

Resolving eagerly would not work for `Origin`: its fraction is a point in the target object's local bounds, and the bounds are unknown when the expression is first written. It permits anchored rotation or scaling without applying mesh transforms or changing a Blender object pivot.

### Origin And Anchored Operations

`Origin(...)` selects a fractional local-bounds pivot for subsequent concrete transforms in its chain. `0` selects the minimum bound, `0.5` the center, and `1` the maximum bound. It accepts a vector or the same grouped-axis keyword style as other constructors; no arguments means the bounding-box center.

```python
with BuildPart() as result:
    Box(2, 2, 2)
    result.transform = Origin(X=0) * Scale(X=4) * Rot(Y=40) * Origin() * Rot(Y=50)
```

An `Origin` affects transforms that follow it until another `Origin` chooses a new pivot. It is not independently resolvable because it is an instruction, not a matrix. `test_complex_transform_chain_with_origin` in [`tests/test_part.py`](../tests/test_part.py) covers a mixed origin, scale, rotation, and translation chain.

`ScaleAlongAxis(axis, factor, reset=False)` and `SizeAlongAxis(axis, size)` are convenience expressions built from `Origin`, scale/size operations, and a restored center pivot. They change one dimension while keeping one local bounding-box face stationary:

```python
with BuildPart() as result:
    Box(2, 2, 2)
    result.transform *= ScaleAlongAxis(-Axis.X, 2)
    result.transform *= SizeAlongAxis(Axis.Y, 1)
```

For a positive axis, the minimum face of that axis remains fixed. For a negative axis, the maximum face remains fixed. Thus `ScaleAlongAxis(-Axis.X, 2)` keeps the `+X` face in place and extends the object toward `-X`. `reset=True` first resets the selected scale component before applying the new scale. The anchored behavior, direction, reset behavior, sequential absolute sizing, and operation after an existing rotation are verified in `test_set_scale_with_axis_anchor`, `test_set_size_with_axis_anchor`, `test_transform_reset_with_rotation_and_pos`, `test_size_along_axis_sequence`, and `test_complex_transform_with_anchored_scaling` in [`tests/test_part.py`](../tests/test_part.py).

### Relative Scale And Absolute Size

`Scale` expresses a multiplicative factor. `Size` expresses a target absolute dimension, calculated from the object's original local bounding-box dimensions at resolution time:

```python
with BuildPart() as result:
    Box(2, 2, 2)
    result.transform = Size(XY=5, Z=1, X=1)
```

`Size` accepts the same axis and grouped-axis keywords as `Scale`, but unset axes preserve their current scale. In a chain, each `Size` is evaluated against the original size and the scale accumulated so far, so a later explicit size on an axis overrides an earlier target for that axis. `test_size_to_scale_calculation`, `test_uniform_size_assignment`, `test_transform_size_application`, and `test_size_override_chain` in [`tests/test_part.py`](../tests/test_part.py) define these observable behaviors.

## Placement Contexts

`Locations` is a context manager modeled after build123d placement contexts. Geometry made inside the context is created once for every active location:

```python
with BuildPart() as result:
    with Locations(Pos(X=0), Pos(X=2)):
        Box(1, 1, 1)
```

It accepts a `Location`, a 3D vector/tuple/list, or a geometry entity. A face is placed at its UV center; another geometry entity is placed at its center. With no arguments, it contributes identity placement. Contexts nest by Cartesian product: an outer list of two X positions and an inner list of two Y positions yield four placements. The locations are stored on the active `BuildPart`, so nesting builders does not duplicate the child assembly internally. `test_nested_locations_with_child_offset`, `test_multiple_locations_with_nested_assembly`, and `test_nested_locations_multiplication` in [`tests/test_locations.py`](../tests/test_locations.py) cover these rules.

The pattern subclasses generate the locations passed to `Locations`:

| API | Distribution |
| --- | --- |
| `GridLocations(x_spacing, y_spacing, x_count, y_count)` | Centered XY grid. |
| `PolarLocations(radius, count, start_angle=0)` | Evenly spaced circle; each location's local rotation follows its radial angle. |
| `HexLocations(apothem, x_count, y_count)` | Staggered hexagonal pattern. |
| `CurveLocations(curve, count=None, spacing=None, offsets=None, offsets_m=None)` | Locations evaluated along a curve. |

`CurveLocations` chooses its input in this order: `offsets_m` (physical distances, clamped to curve length), `offsets` (normalized curve parameters), `count` (endpoints included when count is at least two), `spacing` (physical interval), then start and end by default. `test_grid_and_rotational_locations`, `test_polar_distribution`, and `test_hexagonal_distribution` cover the planar patterns. Curve placement is also exercised in [`tests/test_curve.py`](../tests/test_curve.py) and [`tests/test_selectors.py`](../tests/test_selectors.py).

## Surface And Curve Locations

`SurfaceLocation` and `CurveLocation` preserve a geometric coordinate system while locations are composed. This is different from an ordinary `Location`, which is only a fixed world matrix; their surface inputs come from [selectors](selectors.md), while curve inputs come from [curves](curve.md).

### SurfaceLocation

Obtain one from a face, normally with `face.location(uv)` or a configured UV selector:

```python
surface = cylinder_face.location(uv.set(v=0.5))
marker_loc = surface * Pos(X=1, Y=2, Z=0.2) * Rot(Y=90)
Marker(loc=marker_loc)
```

For a `SurfaceLocation`, a right-hand `Location` has surface-local semantics:

| Position component | Meaning |
| --- | --- |
| X | Offset in the face U coordinate. |
| Y | Offset in the face V coordinate. |
| Z | Offset along the face normal. |

Rotations are accumulated in the local surface frame. Multiplying it by a parent ordinary `Location` on the left preserves this dynamic surface context while supplying a parent transform. Chained surface-local offsets accumulate, and grid X/Y offsets become U/V offsets. Accessing `.loc` intentionally freezes the current evaluation to a plain static `Location`; later translations then use global matrix semantics rather than U/V/normal semantics. These rules are verified by `test_surface_coordinate_chaining_and_compensation`, `test_surface_to_global_transition_via_loc`, and `test_surface_with_grid_and_cumulative_rotation` in [`tests/test_locations.py`](../tests/test_locations.py). Surface locations under non-uniform part scales are additionally covered in [`tests/test_selectors.py`](../tests/test_selectors.py).

### CurveLocation

`CurveLocation` is the corresponding coordinate system for an `AbstractCurve`. It evaluates `curve.at(t_m=x)` and applies its remaining offsets and local rotation in that curve frame:

| Component | Meaning |
| --- | --- |
| X | Arc-length distance along the curve, in meters. |
| Y, Z | Local offsets in the evaluated curve frame. |
| Rotation | Local rotation composed after the curve frame and offsets. |

As with surface locations, multiplying by a location accumulates these local offsets and rotation while retaining the curve relationship. Use `.loc` when a static world-space matrix is desired instead. For ordinary curve locations created by `curve.at(...)`, the curve API aligns local X with the tangent; see [`docs/curve.md`](curve.md) for curve evaluation details.

## Aligning Ports

`align(from_port, to_port, twist=None, rot=None)` returns a `Location` that moves and rotates `from_port` onto `to_port`, with their Z axes facing each other. It supports three rotation modes:

| Arguments | Rotation behavior |
| --- | --- |
| No `twist` or `rot` | Shortest-arc rotation from the source Z axis to the opposite target Z axis. |
| `twist=<degrees>` | Exact target-frame alignment, a 180-degree facing flip, then the given local-Z twist. |
| `rot=<Quaternion>` | Uses the supplied world rotation. |

This is used by [`Joint`](joint.md) and can also be assigned directly to a child builder's `.loc`. The direct shortest-arc and twist cases are covered by `test_align_joints` and `test_align_joints_with_twist` in [`tests/test_locations.py`](../tests/test_locations.py); joint composition is exercised by `test_joint_connection` and `test_joint_registration_and_propagation`.

## Tested References

- [`tests/test_locations.py`](../tests/test_locations.py): nested contexts, patterns, surface locations, wire locations, alignment, and joints.
- [`tests/test_part.py`](../tests/test_part.py): transform properties, `Size`, `Origin`, axis-anchored scale/size, and transformed bounds.
- [`tests/test_curve.py`](../tests/test_curve.py): `CurveLocations` spacing.
- [`tests/test_selectors.py`](../tests/test_selectors.py): curve and surface-derived placements, including scaled geometry.
