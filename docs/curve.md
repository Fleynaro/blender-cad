# Curve

`Curve` is a Blender curve object and an `Object` subclass. It can contain several spline islands, evaluate positions and orientation along its total length, receive bevel or fill geometry, and convert to a mesh [`Part`](part.md) through `.part`.

## Building Curves

Use `BuildCurve` to collect primitives into a `Curve`:

```python
with BuildCurve() as path:
    Line((0, 0, 0), (5, 0, 0))
    Line((5, 0, 0), (5, 5, 0))

path.bevel(depth=0.1)
```

With the default `merge=True`, consecutive compatible splines merge when the previous endpoint and next start point coincide. In the example, the two `Line` calls form one poly spline. Use `BuildCurve(merge=False)` when separate spline islands are required.

Add a curve to a `BuildPart` through `add(...)`. The addition converts it to a mesh as needed:

```python
with BuildPart() as result:
    with BuildCurve() as pipe:
        Polyline((0, 0, 0), (10, 0, 0), (10, 5, 0))
    add(pipe.bevel(depth=0.15), mode=Mode.JOIN)
```

`JOIN` is usually appropriate for a beveled curve because it preserves its generated mesh instead of performing a boolean union. See [`docs/part.md`](part.md) for mode semantics.

## Primitive Types

Construct these only while a `BuildCurve` context is active:

| API | Result |
| --- | --- |
| `Line(start, end)` | Straight poly-spline segment. If `start` is omitted, it starts at the current endpoint. |
| `Polyline(*points, close=False)` | Connected straight segments; accepts positional points or one sequence of points. |
| `Spline(*points, close=False)` | NURBS spline through the supplied points. |
| `BezierCurve(start, handle1, handle2, end)` | Cubic Bezier segment with explicit handles. |
| `TangentArc(end)` | Bezier approximation that continues from the current curve tangent. |
| `RadiusArc(start, end, radius)` | Bezier approximation of an arc between two points. |
| `CenterArc(center, radius, start_angle, end_angle)` | Polyline arc in the XY plane; angles are degrees. |
| `Jiggle(start, end, noise_factor=1.0, segments=10)` | Randomized polyline between two points. Seed `random` before use when reproducible geometry is required. |

Every primitive accepts `tag=` for curve-domain tags. `close=True` makes `Polyline` and `Spline` cyclic.

`make_curve(rule, limit, resolution=50, close=False, curve_type=Spline)` samples a parametric callback. The callback receives values from `0` through `limit` and returns a 3D point:

```python
import math

arc = make_curve(
    lambda value: (0, math.sin(value) * 3, math.cos(value) * 3),
    math.pi,
)
```

## Evaluation And Geometry

`length()` is the total evaluated length across all spline islands. Evaluate by normalized total length with `t` or by distance with `t_m`:

```python
midpoint = path.position_at(t=0.5)
tangent = path.tangent_at(t_m=2.0)
normal = path.normal_at(t=0.5)
location = path.at(t=0.5)  # its local X axis follows the tangent
```

Use `at()` with `Locations` or directly with components to place geometry along a path. `CurveLocations(curve, spacing=...)` distributes locations along evaluated curve length.

`bevel(depth, resolution=4, fill_caps=True, limits=(0.0, 1.0))` produces a tubular curve. `fill()` switches the curve to 2D filling, and `fill_mode` accepts `FillMode.BOTH`, `FRONT`, `BACK`, or `HALF`. `extrude(amount)` gives filled curve geometry depth. Set `resolution` to control curve evaluation density.

`.part` creates an evaluated mesh `Part`, preserving the curve object's materials. It temporarily links the curve and any tracked dependency needed by modifiers so Blender can evaluate the conversion, then restores the previous linkage.

## Tags And Sub-Curves

Curves support object-level [tags](tags.md) on `CURVE` splines and control-point tags on `POINT` elements. `curve.tagged(*tags)` copies matching splines or control points into a new `Curve`; `untagged(*tags)` is its inverse filter. The builder also exposes system point tags such as `Curve.TAG_POINT_FIRST`, `Curve.TAG_POINT_LAST`, `Curve.TAG_POINT_SMOOTH_FILLET_START`, and `Curve.TAG_POINT_SMOOTH_FILLET_END`.

## Declarative `curve` Builder

The lowercase `curve` class builds a path from instructions. Nested `curve(...)` values create isolated branches while the outer path continues:

```python
outline = curve(
    curve.smooth(radius=2),
    curve.step(10),
    curve.step(10, angle=90),
    curve(
        curve.smooth(False),
        curve.step(5, angle=120),
    ),
    trim_ends=True,
)

path = outline.curve
```

`curve.step(length, angle=..., rot=..., smooth=..., radius=..., tag=...)` advances in the current heading. `curve.smooth(enabled=True, radius=1.0)` configures Bezier fillets, while a step can override it locally. `curve.axis(...)`, `curve.rot(...)`, and `curve.clear_rot()` control heading. `curve.move_to(...)`, `move_to_X()`, and `move_to_Y()` add directed moves. `trim_ends=True` removes the first and last endpoint knots; `close` controls whether the generated Bezier spline is cyclic.

## Tested References

- [`tests/test_curve.py`](../tests/test_curve.py): every documented primitive, merge behavior, evaluation, Bezier paths, parametric curves, tags, BVH use, and the declarative builder.
- [`tests/test_locations.py`](../tests/test_locations.py): curve-derived placement and location contexts.
- [`tests/test_text.py`](../tests/test_text.py): a `make_curve()` path used to deform text.
