# Modifiers And Mesh Operations

`blender_cad` exposes mesh-editing operations from `blender_cad.modifiers` at the package root. They operate on the active `BuildPart` and mutate its mesh immediately. Use [selectors](selectors.md) such as `faces()`, `wires()`, `edges()`, and `vertices()` to control the affected logical geometry. Reacquire selections after every topology-changing operation because previously held selection wrappers can no longer refer to the current mesh.

```python
from blender_cad import *

with BuildPart() as result:
    Box(2, 2, 2)
    extrude(faces().top(), op=Pos(Z=1) * Scale(XY=0.5))
    bevel(edges(), radius=0.1, segments=3)
```

All operations below require an active `BuildPart` unless stated otherwise. See [`docs/part.md`](part.md) for build contexts, `Mode`, and selector lifetime, and [`docs/selectors.md`](selectors.md) for constructing selections.

## Transform

`transform(entities=None, op=Transform(), space=Location(), prop_edit=None)` moves, rotates, or scales vertices with a `TransformExpr`, such as `Pos`, `Rot`, `Scale`, or their product.

- With `entities=None`, it transforms every vertex in the active part.
- A face, edge, wire, vertex, `ShapeList`, `Part`, or `BuildPart` limits the operation to the vertices represented by that selection.
- `space` defines the coordinate system in which the operation is applied. It defaults to the global identity location.
- On the optimized `BoxSetPart`, an unrestricted non-proportional transform is applied through its box representation.

```python
with BuildPart() as result:
    Box(1, 1, 1)
    bottom = vertices() - faces().top().vertices()
    transform(bottom, op=Scale(2))
```

The example expands only the lower vertices, producing a tapered solid. `test_vertex_transform`, `test_global_vertex_transform`, and `test_vertex_transform_excluding_top_face` in [`tests/test_modifiers.py`](../tests/test_modifiers.py) cover face-local, whole-part, and set-subtraction selections.

## Proportional Editing

Pass `prop_edit=` to `transform()` or `extrude()` to interpolate the requested transform separately for every affected vertex. A weight of `0` leaves a vertex unchanged; a weight of `1` applies the full transform. Translation, rotation, and scale are interpolated as one transform, rather than only interpolating position.

### Falloff

`Falloff` controls conversion of a normalized influence into a weight:

| Value | Weight curve |
| --- | --- |
| `Falloff.CONSTANT` | Always `1` inside the edit's domain. |
| `Falloff.LINEAR` | Linear. |
| `Falloff.SHARP` | Squared linear value. |
| `Falloff.SPHERE` | Circular-arc curve. |
| `Falloff.SMOOTH` | Smoothstep curve; the default. |

### Edit Types

| API | Influence |
| --- | --- |
| `RadialPropEdit(origin=Vector((0, 0, 0)), radius=1.0, falloff=Falloff.SMOOTH)` | Full influence at `origin`, falling to zero at `radius`; points beyond the radius have zero weight. `origin` may be a `Vector` or `Location`. A zero radius gives full weight only to points at the origin. |
| `LinearPropEdit(axis=Axis.Z, falloff=Falloff.SMOOTH)` | Maps the selected vertices' bounding range on an axis to a gradient. The positive axis gives greatest influence at the maximum coordinate; a negated axis, such as `-Axis.Z`, reverses it. A zero-length range has full weight. |
| `LambdaPropEdit(func)` | Uses a custom callable that receives `(point, context)` and returns a weight. It is intended for custom weight fields and composed edits. |

`ProportionalEdit` values compose with `+`, `-`, `*`, `/`, and `**`, accepting another edit or a number. Division by zero produces `0`; a power with a negative base also produces `0`. `min(other)`, `max(other)`, `mix(other, factor)`, `clamp(min_val=0.0, max_val=1.0)`, and `invert()` return derived edits. `mix()` performs `self + (other - self) * factor`; its factor can also be an edit.

```python
with BuildPart() as result:
    Box(1, 1, 1)
    taper = LinearPropEdit(Axis.Z)
    transform(op=Scale(XY=0.5), prop_edit=taper)

    top_weight = (
        LinearPropEdit(Axis.X)
        + LinearPropEdit(Axis.Y).clamp(0.5, 0.8)
        - 0.1
    ) / 2
    extrude(faces().top(), op=Pos(Z=1), prop_edit=top_weight)
```

`make_box_sides_edit(neg_x=1.0, pos_x=1.0, neg_y=1.0, pos_y=1.0, multiply=False)` creates a four-sided linear mask for box-like geometry. Each parameter controls the corresponding lateral side. The default combines the side masks with their minimum, so the most restrictive side governs a vertex. With `multiply=True`, it multiplies the masks instead. `test_proportional_linear_transform_and_extrude`, `test_proportional_math_operations_on_extrude`, `test_proportional_box_sides_independent_masking`, `test_proportional_radial_transform`, and `test_proportional_radial_plane_bending` verify these workflows.

## Topology Operations

### `subdivide`

`subdivide(entities=None, cuts=1, faces=None)` splits selected edges with Blender's grid-fill subdivision. Select a face, edge, wire, or another geometry collection through `entities`. `faces=` is a convenience for subdividing each supplied face separately. `cuts` is the number of inserted cuts.

```python
with BuildPart() as result:
    Plane(2)
    subdivide(cuts=6)
    transform(op=Pos(Z=1), prop_edit=RadialPropEdit(radius=1))
```

Subdivide before a deform or proportional edit when additional vertices are needed to represent a smooth shape. The plane deformation and bend tests cover this use.

### `dissolve`

`dissolve(entities=None, angle_limit=5.0)` merges selected coplanar or near-coplanar faces and their related edges when their angle is below `angle_limit`, specified in degrees. It acts on the selected faces; an empty face selection is a no-op. The current modifier test file does not independently exercise `dissolve`, so its exact topology result remains Blender-dependent.

### `extrude`

`extrude(entities=None, op=Transform(), prop_edit=None, delete_source=False, recalc_normals=False, tag=None)` duplicates and connects selected geometry, then transforms the new vertices. The optional `tag` uses the mesh tagging API described in [`tags.md`](tags.md).

| Selected type | Result |
| --- | --- |
| `Face` | Extrudes a face region, adding side faces and edges. |
| `Wire` or `Edge` | Extrudes edges, creating the connected edge-only result. |
| `Vertex` | Individually extrudes vertices into new edges. |

`delete_source=True` removes the original faces or edges after extrusion. `recalc_normals=True` recalculates normals for the complete mesh. New vertices, edges, and faces automatically receive the system extrusion tag and every tag passed through `tag=`.

```python
with BuildPart() as result:
    Box(2, 2, 2)
    extrude(faces().top(), op=Pos(X=1, Z=1) * Rot(Z=20) * Scale(0.8))
```

`test_extrude_all_geometries` covers face, wire, edge, and vertex extrusion. The proportional-edit tests show `prop_edit` applied to the newly generated vertices only.

### `solidify_faces`

`solidify_faces(faces=None, height=0.1, offset=0.0)` returns a new closed `Part` made from the selected faces. It does not add that part automatically; use `add(...)` in the surrounding context.

- `offset` moves the copied source shell along its averaged vertex normals before solidification.
- `height` moves the extruded cap along its averaged vertex normals.
- An empty selection raises `ValueError("No faces selected")`.
- The returned part inherits the source part's world transform.

```python
with BuildPart() as result:
    Cylinder(1, 2)
    shell = solidify_faces(faces().cylinders_only(), height=0.5, offset=0.5)
    add(shell)
```

`test_solidify_faces` covers both curved cylinder sides and planar cap selections under a non-identity source transform.

### `delete`

`delete(entities=None)` removes mesh elements according to their selected type:

| Selected type | Deletion behavior |
| --- | --- |
| `Face` | Removes the faces, retaining shared edges and vertices. |
| `Wire` or `Edge` | Removes the edges and faces using them. |
| `Vertex` | Removes vertices with all connected edges and faces. |

An empty selection is a no-op. `test_delete_all_geometries` demonstrates each domain and repeated deletion after reselection.

## Baked Blender Modifiers

The following operations create a Blender modifier, evaluate it, bake the result back into the active mesh, and remove the temporary modifier. They are destructive mesh edits, not a persistent Blender modifier stack.

### `bevel`

`bevel(entities=None, radius=0.1, segments=10)` bevels the selected logical edges. It clears the bevel weight on all physical edges, assigns weight `1` to the selected edges, and runs Blender's weighted bevel modifier. `radius` is converted to Blender's percentage-width setting; `segments` controls bevel resolution. `test_edge_selection_and_bevel` verifies beveling a box's edges.

### `mirror`

`mirror(axis=Axis.X)` mirrors the whole active part across the selected local axis and bakes the result. It accepts `Axis.X`, `Axis.Y`, or `Axis.Z`; the default is X. It does not expose a selector, clipping, merge, or bisect option. The current modifier test file does not independently exercise this operation.

### `simple_deform`, `bend`, And `twist`

`simple_deform(type=DeformType.BEND, angle=0.0, origin=Location(), axis=Axis.X, limits=(0.0, 1.0))` bakes Blender's Simple Deform modifier.

- `DeformType` provides `TWIST`, `BEND`, `TAPER`, and `STRETCH`.
- `angle` is in degrees.
- `origin` defines the deform origin.
- `axis` selects the deform axis.
- `limits` is the normalized modifier interval passed to Blender.

`bend(angle, axis=Axis.X, segments=None, origin=Location(), limits=(0.0, 1.0))` and `twist(...)` are focused wrappers for the respective deform types. When `segments` is provided, they first call `subdivide(cuts=segments)`. Their axis mapping is internal to the wrappers so that their documented `Axis` matches the library's bend/twist convention.

```python
with BuildPart() as result:
    Box(2, 1, 0.1)
    bend(angle=30, axis=Axis.Y, segments=4)
```

`test_bend_operation` verifies a pre-subdivided bend on an already transformed part. The current modifier test file does not independently exercise `mirror`, raw `simple_deform`, `twist`, `TAPER`, or `STRETCH`.

### `wrap`

`wrap(target, *, loc=None, mode=WrapMode.NEAREST_SURFACEPOINT, offset=0.0, segments=None)` shrinkwraps the active part to a target `Part` or compatible part-like object, then bakes the result.

- `loc`, when supplied, becomes the active part's location before wrapping.
- `segments`, when supplied, calls `subdivide(cuts=segments)` first.
- `WrapMode.NEAREST_SURFACEPOINT` is the default projection method.
- `WrapMode.NEAREST_VERTEX` projects to the target's nearest vertices.
- `offset` is Blender's shrinkwrap offset.

```python
with BuildPart() as result:
    with BuildPart(mode=Mode.PRIVATE) as source:
        Cylinder(radius=1, height=2, segments=64)
    Plane(1)
    wrap(source, segments=4, offset=0.01)
```

`test_wrap_operation` covers wrapping a private cylinder relative to one of its surface locations and then joining that source part.

## Adding Existing Parts

`add(to_add, offset=Location(), transform=None, mode=Mode.ADD, mat=None, tag=None)` integrates an existing `Part`, `BuildPart`, or curve-compatible object into the active `BuildPart`.

- The source is copied by default. Active `Locations` always require copies for each location, except for curve objects handled by the curve workflow.
- `offset` is composed with each active location and the source location. `transform` is additionally multiplied into the added part's transform.
- `mat` replaces the added part's material; `tag` adds face-domain tags to it.
- `Mode.ADD`, `SUBTRACT`, and `INTERSECT` use booleans. Their `_FAST` variants use Blender's floating-point boolean solver. `Mode.JOIN` merges mesh data without boolean cleanup. `Mode.PRIVATE` makes `add()` a no-op.
- Invalid source parts raise `RuntimeError`.

```python
with BuildPart(mode=Mode.PRIVATE) as peg:
    Cylinder(radius=0.25, height=2)

with BuildPart() as result:
    Box(3, 3, 1)
    add(peg, offset=Pos(Z=1), mode=Mode.JOIN, mat=mat.red, tag="peg")
```

The modifier suite uses `add()` to integrate `solidify_faces()` results and to join the wrapped source. Broader boolean and mode behavior is covered by [`tests/test_primitives_booleans.py`](../tests/test_primitives_booleans.py), [`tests/test_context_management.py`](../tests/test_context_management.py), and [`tests/test_part.py`](../tests/test_part.py).

## Tested References

- [`tests/test_modifiers.py`](../tests/test_modifiers.py): transforms, proportional edits, extrusion, solidification, deletion, beveling, bending, and wrapping.
- [`tests/test_primitives_booleans.py`](../tests/test_primitives_booleans.py): boolean mode behavior.
- [`tests/test_context_management.py`](../tests/test_context_management.py): active `BuildPart` and nested integration behavior.
