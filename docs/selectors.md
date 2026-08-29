# Selectors And Mesh Topology

`blender_cad` exposes a CAD-like selection API over Blender polygon meshes. Import it from the package root:

```python
from blender_cad import *
```

The API resembles `build123d` selectors, but it does not operate on BREP or NURBS topology. A Blender mesh has physical polygons, edges, and vertices. Before `faces()`, `wires()`, `edges()`, or `vertices()` returns a collection, `Topology` analyzes that mesh and reconstructs logical entities suitable for CAD-style work.

This distinction is important: a cylinder side may be dozens of Blender polygons at one resolution and a different number at another, yet normally appears as one logical `Face`. The logical face supports one surface-selection and placement workflow regardless of its tessellation.

## Selecting From A Part

Inside an active `BuildPart`, use the global helpers. Outside it, select from a completed `Part` through the corresponding instance methods.

```python
with BuildPart() as result:
    Box(10, 12, 2)

    top_face = faces().top()[0]
    long_edges = top_face.edges().sort_by(SortBy.LENGTH)[-2:]
    top_face.mat = mat.red

part_faces = result.part.faces()
```

`faces()`, `wires()`, `edges()`, and `vertices()` return `ShapeList` collections. They describe the topology of the current BMesh, so reacquire a collection after a boolean, modifier, extrusion, or any other mesh edit. A wrapper retained across an edit can refer to no-longer-current BMesh elements.

## Physical And Logical Entities

`Topology` builds these logical entities from physical mesh data:

| Logical entity | Physical representation             | Purpose                                                                                                                     |
| -------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `Face`         | One or more adjacent BMesh polygons | A continuous, smooth logical surface. It provides surface classification, UV evaluation, boundaries, and surface placement. |
| `Edge`         | One or more connected BMesh edges   | A logical curve segment. Smoothly connected physical edges can form one logical edge.                                       |
| `Vertex`       | One BMesh vertex                    | A discrete topological point.                                                                                               |
| `Wire`         | An ordered set of logical edges     | A continuous boundary/path. Face-boundary wires are closed loops; loose mesh-edge topology may also produce an open wire.   |

Use `.split()` when the physical mesh resolution matters. It ungroups each selected logical entity into single physical entities:

```python
with BuildPart() as result:
    Cylinder(radius=2, height=5, segments=64)

    side = faces().filter_by(GeomType.CYLINDER)
    logical_count = len(side)             # normally one cylindrical logical face
    polygon_count = len(side.split())     # individual BMesh polygons
```

The `all_single_faces()`, `all_single_edges()`, and `all_single_vertices()` helpers also flatten selected entities into physical single-element wrappers. `.split()` is the usual uniform operation when the current collection type is already known.

## Topology Reconstruction

Topology is rebuilt from the mesh using `Topology` in `geometry.py`. For every unvisited physical polygon, the analyzer grows a logical face across adjacent polygons whose normals differ by no more than the configured smooth-angle threshold. It then extracts that group's boundary loops as wires and segments each loop into logical edges at sharp corners. [`docs/topology.md`](topology.md) explains the complete reconstruction pipeline, traversal rules, thresholds, and implementation limits.

```python
with BuildPart(
    topology=TopologyConfig(smooth_angle=10, edge_break_angle=20)
) as result:
    Cylinder(radius=2, height=5)
```

`TopologyConfig` has two degree-valued thresholds:

| Option             | Default | Effect                                                                                                                                       |
| ------------------ | ------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `smooth_angle`     | `15.0`  | Adjacent polygon faces at or below this normal-angle difference are grouped into one logical `Face`.                                         |
| `edge_break_angle` | `15.0`  | Connected boundary edges are split into separate logical `Edge` values when their deviation from a straight continuation exceeds this value. |

Use a lower `smooth_angle` when a shallow crease must remain separate. Use a lower `edge_break_angle` when more bends must appear as distinct logical edges. These values alter logical selection topology, not the underlying Blender mesh.

You can provide the configuration when opening `BuildPart`, or update the active part with `set_topology(TopologyConfig(...))`. The type-identification test includes a thin cone built with `smooth_angle=1`; see `test_geometry_type_identification` in [`tests/test_selectors.py`](../tests/test_selectors.py).

## ShapeList Operations

`ShapeList` is an ordered, duplicate-free collection with selector operations. Filtering, sorting, slicing, and set-style operations return another `ShapeList`, preserving the original collection as the universe for inversion.

```python
with BuildPart() as result:
    Box(10, 10, 2)

    top = faces().top()
    side = faces().side()
    selected = top + side       # union
    only_top = selected & top   # intersection
    without_sides = selected - side
    not_top = ~top              # all faces in the original faces() result except top
```

| Operation           | Meaning                                                                                            |
| ------------------- | -------------------------------------------------------------------------------------------------- |
| `a + b`             | Union, with duplicate entities removed.                                                            |
| `a - b`             | Difference.                                                                                        |
| `a & b`             | Intersection.                                                                                      |
| `~a`                | Complement relative to the `ShapeList` from which `a` was derived. It is not a global scene query. |
| `items[n]`          | One entity.                                                                                        |
| `items[start:stop]` | A `ShapeList`, not a plain Python list.                                                            |

### Filter, Sort, And Group

`filter_by(...)` accepts either a predicate or a `GeomType`:

```python
cylindrical = faces().filter_by(GeomType.CYLINDER)
new_faces = faces().split().filter_by(lambda face: face.is_new())
```

Face classification is an analysis result. The supported types are `GeomType.PLANE`, `CYLINDER`, `CONE`, `SPHERE`, and `UNKNOWN`. Classification is based on the logical face's polygon normals and geometry, not on a native CAD surface. `_detect_geom_type()` first recognizes co-directional normals as planes, then uses normal differences to estimate a revolve axis for cylinders and cones, and finally checks whether normals point outward from the face's average polygon center for spheres. It is a tessellation-dependent heuristic, not analytic surface fitting. Cylinder, cone, sphere, and plane selection is exercised by `test_geometry_type_identification` in [`tests/test_selectors.py`](../tests/test_selectors.py); see [`docs/geometry.md`](geometry.md#geometry-type-detection) for thresholds, formulas, and limits.

`sort_by(...)` accepts an `Axis`, a vector direction, a `SortBy` value, or a key function. `group_by(...)` accepts the analogous `Axis`, vector, `GroupBy`, or key-function form and returns a list of `ShapeList` groups. Numeric group keys are compared using `tolerance`, which defaults to `1e-4`.

```python
top = faces().sort_by(Axis.Z)[-1]
largest = faces().sort_by(SortBy.AREA)[-1]
z_levels = faces().group_by(Axis.Z)
```

Directional shortcuts select the extreme coordinate group: `min_x()`, `max_x()`, `min_y()`, `max_y()`, `min_z()`, `max_z()`, `top()`, and `bottom()`. `side()` means every item except `top()` and `bottom()` in the same collection. Grouping and set combinations are covered by `test_grouping_and_filtering` and `test_complex_selection_logic` in [`tests/test_selectors.py`](../tests/test_selectors.py).

### Geometry Checkpoints

`GeometryCheckpoint` records the current spatial state of a part so selectors can distinguish geometry created after that state. Every modeling operation creates an operation checkpoint; therefore `entity.is_new()` compares the entity with the checkpoint from the most recent operation. Use it as a predicate with `filter_by(...)`:

```python
with BuildPart() as result:
    Box(3, 3, 3)
    with Locations(Pos(Z=3)):
        Box(2, 2, 2)

    # Select only faces introduced by the second operation.
    faces().filter_by(lambda face: face.is_new()).mat = mat.red
```

The comparison is spatial rather than based only on mesh element identity. Existing exterior faces remain old when a boolean modifies them, while genuinely generated faces, such as cutout walls or bevel transitions, are new. If physical mesh faces must be inspected individually, call `split()` before `is_new()`.

Create an explicit checkpoint with `make_checkpoint()` when several later operations should be treated as one interval. Pass that checkpoint to `is_new(checkpoint)` to select everything created since it:

```python
with BuildPart() as result:
    Box(2, 2, 2)
    checkpoint = make_checkpoint()
    extrude(edges().top(), op=Pos(Z=1))
    extrude(edges().top(), op=Pos(Z=1))

    faces().split().filter_by(
        lambda face: face.is_new(checkpoint)
    ).mat = mat.red
```

`is_new()` is available on `Face`, `Edge`, and `Vertex` entities. Checkpoint behavior is covered by `TestGeometryCheckpoints` in [`tests/test_selectors.py`](../tests/test_selectors.py).

### Tags, Materials, And Conversion

`tagged(*tags)` selects entities carrying requested tags; wildcard patterns are supported. `untagged(*tags)` removes matching tagged entities. `add_tags(...)`, `remove_tags(...)`, and the `tags` property update all selected entities. Tags are stored on mesh geometry, rather than on a temporary selector wrapper.

Assigning `ShapeList.mat` assigns the material to every selected entity that supports materials, such as faces:

```python
faces().tagged("panel_*").mat = mat.blue
faces().filter_by(GeomType.PLANE).mat = mat.green
```

`ShapeList.part()` copies selected entities into a new `Part`. For an edge collection, `.to_wire()` constructs a `Wire` from the selected edges; the selected edges must be continuous for the resulting path to be meaningful. `test_wire_creation_from_filtered_edges_and_curve_distribution` and `test_part_extraction` in [`tests/test_selectors.py`](../tests/test_selectors.py) cover these workflows.

## Faces, UV, And Surface Placement

A logical `Face` may be planar, cylindrical, conical, or spherical even though it is physically tessellated. `Face.at(u, v)` evaluates a location on that logical surface, with normalized coordinates in the `[0, 1]` domain:

```python
with BuildPart() as result:
    Cylinder(radius=5, height=1)
    side = faces().filter_by(GeomType.CYLINDER)[0]

    Marker(side.at(0.25, 0.5))
```

For known logical geometry, the implementation chooses a matching projection: planar for planes, spherical for spheres, and cylindrical for cylinders and cones. Otherwise it selects the available projection with the least measured distortion. The returned `Location` has its local Z aligned to the sampled surface normal, so it can be supplied directly to `Locations(...)` to attach new geometry.

The global `uv` value is an immutable fluent `UVSelector` entry point. Pass it to `Face.at(...)` or `Face.location(...)` when selection should be described rather than specified by fixed coordinates:

```python
with BuildPart() as result:
    Plane(2)
    face = faces()[0]

    Marker(face.at(uv.local().max_x().min_y()))
    Marker(face.at(uv.bottom().offset(u=0.2, v=-0.1)))
```

`min_u()`, `max_u()`, `min_v()`, and `max_v()` choose UV preferences. `min_x()` through `max_z()` choose among sampled locations in world coordinates by default; add `.local()` to compare in the face's local frame. `top()` and `bottom()` are aliases for `max_z()` and `min_z()`.

`set(u=..., v=...)` fixes normalized coordinates. `offset(u=..., v=...)` shifts normalized coordinates and wraps at the UV boundary. `set_m(...)` starts at UV origin and applies absolute metric offsets, while `offset_m(...)` walks a physical distance across the tessellated surface. `with_projection(...)` selects a specific `UVProjection`; `offset_final(...)` composes a final location offset after surface evaluation.

Metric offsets account for the part's current scale while crossing polygon triangles. This behavior, UV wrapping, transformed surfaces, and scaled cylinders are covered by `test_uv_selector_offset_with_wrap_around`, `test_surface_at_mapping_with_complex_deformation`, and `test_uv_metric_offsets_on_scaled_cylinder` in [`tests/test_selectors.py`](../tests/test_selectors.py). See [`docs/geometry.md`](geometry.md) for projection, triangle interpolation, metric walking, and the full `UVSelector` ranking algorithm, and [`docs/location.md`](location.md) for `SurfaceLocation` composition and grid placement.

## Wires And Edges

Use `face.wires()` to obtain its boundary loops. A face with a hole has an outer wire and one or more inner wires; use `outer_wires()` and `inner_wires()` when that distinction matters.

```python
with BuildPart() as result:
    Cylinder(radius=5, height=1)
    Cylinder(radius=3, height=1, mode=Mode.SUBTRACT)

    cap = faces().top()[0]
    outer = cap.outer_wires()[0]
    holes = cap.inner_wires()

    with CurveLocations(outer, count=5):
        Marker()
```

`Wire` inherits curve evaluation from `Edge`, so both support `.at(t)` for normalized path positions and can be passed to `CurveLocations`. A wire preserves traversal order and may contain multiple logical edges. `Wire.edges()` returns those logical edges; `Wire.vertices()` returns traversal-order vertices without duplicating the closing vertex of a closed wire. Rotated cylinder and annulus wire placement are verified by `test_rotated_cylinder_wire` and `test_rotated_annulus_wire` in [`tests/test_selectors.py`](../tests/test_selectors.py).

## Tested References

- [`tests/test_selectors.py`](../tests/test_selectors.py): topology reconstruction, logical-entity selection, geometry classification, UV placement, wire evaluation, grouping, tags, and checkpoints.
- [`tests/test_locations.py`](../tests/test_locations.py): `SurfaceLocation`, surface-local transforms, and grid placement.
