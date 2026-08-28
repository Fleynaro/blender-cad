# Part And BuildPart

`Part` is the library's polygon-mesh object. It owns a Blender mesh object and inherits common scene-object behavior from `Object`, including transforms, bounds, joints, tagging, `show()`, and `remove()`. Primitives such as `Box`, `Sphere`, and `Cylinder` build `Part` geometry; they do not create CAD BREP solids.

Import the public API from the package root:

```python
from blender_cad import *
```

## Build Context

Build mesh geometry inside `BuildPart`. The context exposes its completed mesh as `result.part` after the block:

```python
with BuildPart(mat=mat.blue) as result:
    Box(10, 10, 1)
    Sphere(3, mode=Mode.SUBTRACT)

result.part.show("cut_box")
```

Nested `BuildPart` contexts compose into the parent when the inner block exits. To reopen a completed context and keep editing the same part, create it with `mode=Mode.PRIVATE`:

```python
with BuildPart(mode=Mode.PRIVATE) as result:
    Box(10, 10, 1)

with result:
    Sphere(3, mode=Mode.SUBTRACT)
```

`BuildPart` and the active-context helpers `faces()`, `wires()`, `edges()`, `vertices()`, `set_mat()`, and tag helpers require an active `BuildPart` context. These patterns are verified by [`tests/test_context_management.py`](../tests/test_context_management.py).

### Modes

`Mode` determines how newly created or explicitly added geometry is integrated:

| Mode | Behavior |
| --- | --- |
| `ADD` | Boolean union. It can alter topology and remove interior geometry. |
| `SUBTRACT` | Boolean difference. |
| `INTERSECT` | Boolean intersection. |
| `JOIN` | Blender mesh join. It copies mesh data into the target without boolean processing, preserving overlapping/internal geometry and material slots. |
| `PRIVATE` | Keeps the built object out of automatic parent integration. Use it when a temporary or reusable part must be added later with `add(...)`. |
| `ADD_FAST`, `SUBTRACT_FAST`, `INTERSECT_FAST` | Fast variants of the respective boolean modes. |

`JOIN` is intentionally distinct from `ADD`: use it for assemblies, converted curves, or text where an expensive or topology-changing boolean is not wanted. `test_nested_builders`, `test_explicit_add_operation`, and the `JOIN` cases in [`tests/test_part.py`](../tests/test_part.py) demonstrate the distinction in context usage.

## Selecting And Editing Geometry

`Part.faces()`, `wires()`, `edges()`, and `vertices()` return `ShapeList` collections backed by the current topology. The corresponding `BuildPart` helpers select from the active result. Selectors are intended for material assignment, locations, topology queries, and modifier operations.

```python
with BuildPart() as result:
    Box(2, 2, 2)
    top = faces().top()[0]
    top.mat = mat.red
```

After an operation that changes topology, reacquire selectors instead of retaining old elements. The selected geometry wrappers map to the current BMesh and can become invalid after a mesh edit.

`mat` assigns material to selected faces, or to all faces when no face selection is supplied. `default_mat` sets material slot zero and leaves explicit non-default face materials intact. `BuildPart(mat=...)` applies its material as the completed part's default material.

`get_tags()`, `set_tags()`, `add_tags()`, and `remove_tags()` work over face, edge, and point domains. Tags are stored on geometry, not merely on the Python wrapper.

## Transform, Bounds, And Conversion

All `Object` properties apply to a `Part`:

```python
with BuildPart() as result:
    Box(2, 2, 2)
    result.scale = 2
    result.loc = Pos(2, 3, 4) * Rot(X=30)
    result.transform *= ScaleAlongAxis(-Axis.X, 2)
```

`transform`, `loc`, `scale`, and `size` update the Blender object transform. Assigning `loc` preserves the current scale. `bbox` and `local_bbox` expose bounds, while `bbox_part` and `convex_hull_part` create cached `Part` representations useful for visualization or later joining. `project_2d()` produces a flattened mesh silhouette on an axis-aligned plane.

The transform and derived-mesh behavior is verified throughout [`tests/test_part.py`](../tests/test_part.py), including anchored scaling, `bbox_part`, `convex_hull_part`, and 2D projection.

`Part.hash(precision=4, use_materials=False)` produces a deterministic hash from rounded vertex coordinates, transform, and optionally face materials. Tests use this to assert generated geometry; it is not a general-purpose content hash for every Blender data property.

### BoxSetPart

`Part.box_set_empty()` creates `BoxSetPart`, an optimized `Part` subtype that stores box descriptors and creates a Blender mesh only when required. It is not merely a faster `Box`: its primary purpose is to support the solver and rule-based layout (RBL) in [`ml.py`](../blender_cad/ml.py). RBL uses rules as a declarative description of the desired arrangement, and the solver needs a fast representation for trying candidate layouts.

Use `BoxSetPart` to quickly sketch an approximate scene before drawing the actual geometry: it represents real objects as a set of bounding boxes, so the solver can reason about their placement, including collisions and other box-level constraints. This makes `BoxSetPart` primarily an optimization for the solver: it is about speed and "trying out" a layout, not about replacing the final modeled parts. Adding a normal `Part` to it reduces that object to its local bounding box, so use it only when box-level approximation is acceptable. This behavior is covered by `test_box_set_part_primitive_storage_and_optimization` in [`tests/test_part.py`](../tests/test_part.py).

## Object Lifecycle

`Part` starts unlinked to the scene. Call `part.show(name="...")` to link it to the current collection and retain it in the scene. Temporary wrappers clean up their Blender data when removed or collected. Use `copy()` for an independent Blender object and mesh copy, and `remove()` when the object is no longer needed.

For common shared object behavior, see [`blender_cad/object.py`](../blender_cad/object.py). For primitive constructors, see [`blender_cad/primitives.py`](../blender_cad/primitives.py).

## Tested References

- [`tests/test_context_management.py`](../tests/test_context_management.py): nested contexts, explicit integration, and reopening a result context.
- [`tests/test_part.py`](../tests/test_part.py): transforms, size, bounds, projection, and `BoxSetPart`.
- [`tests/test_primitives_booleans.py`](../tests/test_primitives_booleans.py): primitive creation and boolean modes.
- [`tests/test_selectors.py`](../tests/test_selectors.py): face, wire, edge, and vertex selection.
