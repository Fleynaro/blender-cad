# Geometry And Surface Mapping

`blender_cad` models with Blender meshes, then exposes a CAD-like geometry layer over that mesh. This page explains the mathematical implementation behind logical faces, `Face.at(...)`, `UVSelector`, and `GeomType` classification. It describes the current implementation, not an analytic BREP or NURBS surface API.

For how the `Topology` analyzer first groups physical polygons into logical entities, see [`docs/topology.md`](topology.md). For the public collection and selector API, see [`docs/selectors.md`](selectors.md).

## Basic Entities

Topology-derived entities are wrappers around the active BMesh:

| Entity | Backing mesh data | Meaning |
| --- | --- | --- |
| `Vertex` | One BMesh vertex | A discrete point. `vertex.center()` returns its position as a `Location`. |
| `Edge` | One or more ordered BMesh edges | A continuous logical path between its endpoints. It supports length evaluation through `.at(t=...)` or `.at(t_m=...)`. |
| `Wire` | An ordered chain of logical edges | A continuous boundary or loose path. A face can have one outer boundary and zero or more inner boundaries. |
| `Face` | One or more smoothly connected BMesh polygons | A logical surface patch. It owns boundary wires, exposes `geom_type`, and maps normalized UV coordinates to an oriented `Location`. |

The topology layer is deliberately separate from the underlying tessellation. For example, the many polygons forming a cylinder side normally become one logical `Face`. The geometry methods below still evaluate that logical face through its physical polygons and triangles, so a result follows the actual mesh after scaling or other deformation.

## Face Coordinate Contract

`Face.at(...)` has two forms:

```python
point = face.at(0.25, 0.5)
point = face.at(uv.local().max_x().min_y())
```

The direct form accepts normalized `u` and `v` values. Before evaluation, each is clamped just inside the nominal range, `[1e-6, 1 - 1e-6]`. This avoids ambiguous triangle-boundary and periodic-seam hits. It means direct evaluation at `0.0` or `1.0` produces a point immediately inside the boundary, rather than an exact boundary vertex.

The result is a `Location`, not only a position:

- Its origin is the sampled point on the surface.
- Its local Z axis follows the interpolated surface normal.
- The owning part's scale and location are composed before it is returned.

Consequently, it can be used to attach geometry directly:

```python
with BuildPart() as result:
    Sphere(radius=2)
    surface = faces().filter_by(GeomType.SPHERE)[0]

    with Locations(surface.at(0.3, 0.3)):
        Cone(radius_bottom=1, radius_top=0.2, height=2)
```

`test_sphere_cone_attachment_and_selection` in [`tests/test_selectors.py`](../tests/test_selectors.py) exercises this composition.

## From 3D Mesh Coordinates To UV

`Face._at_uv(...)` needs a UV coordinate for every unique vertex in the logical face. It first chooses a projection, projects the face vertices, then normalizes the projected coordinates into a face-local `[0, 1] x [0, 1]` domain.

### Projection Choice

For a recognized `geom_type`, the mapping is fixed:

| Geometry type | Projection |
| --- | --- |
| `PLANE` | `PLANAR` |
| `CYLINDER` | `CYLINDRICAL` |
| `CONE` | `CYLINDRICAL` |
| `SPHERE` | `SPHERICAL` |

For `UNKNOWN` geometry, `_detect_best_uv_projection()` evaluates every `UVProjection` and keeps the lowest distortion score. The score combines variance in projected-to-3D edge-length ratios with a penalty for planar projections that collapse polygons whose normals are far from the projection axis. The selected normalized UVs are cached on the face. Supplying `projection=...` or `uv.with_projection(...)` bypasses automatic selection; the most recently used explicit projection is cached separately.

### Establishing A Projection Frame

`project_coords(...)` builds a rotation that aligns the face's `main_axis` with local Z. It rotates every vertex into that frame and estimates the projection center as the center of the local axis-aligned bounding box. Coordinates are measured relative to this center.

The implementation then uses the following formulas:

| Projection | U | V |
| --- | --- | --- |
| Planar | local `x` | local `y` |
| Cylindrical | `0.5 + atan2(y, x) / (2*pi)` | local `z` |
| Spherical | `0.5 + atan2(y, x) / (2*pi)` | `0.5 + asin(z / r) / pi` |

For a sphere, `r` is the length of the centered local coordinate. A zero-length coordinate falls back to `(0.5, 0.5)` to avoid division by zero.

### Normalization And Circular Seams

`normalize_uvs(...)` maps both projected coordinate spans to `[0, 1]`. Cylindrical and spherical U are circular, so the function first sorts U coordinates and compares all internal gaps with the gap across `1 -> 0`. When an internal gap is largest, coordinates below its upper endpoint are shifted by `+1` before normalization. This moves a seam that cuts through the sampled mesh out of the main coordinate interval where possible.

Later, per-triangle processing performs a second seam adjustment. For spherical and cylindrical triangles, a vertex whose U differs from the triangle average by more than `0.5` is moved across the periodic boundary before containment tests. This lets a triangle near the seam remain locally continuous in UV space.

These projections are mesh-derived parameterizations. They are not persisted Blender UV layers and do not promise a stable global orientation across arbitrary topology, transformations, or changes in tessellation.

## Evaluating A UV Point

After projection, `_calc_uv_location(...)` converts the requested UV point back into a 3D location.

1. Each BMesh polygon is represented as a triangle fan: `(vertex[0], vertex[i], vertex[i + 1])`.
2. `_find_face_and_uv_tri(...)` searches those UV triangles for one containing the requested point.
3. If no triangle contains it, evaluation chooses the closest triangle in UV space. This makes points in gaps or on numerically ambiguous boundaries evaluate predictably rather than fail immediately.
4. `get_barycentric_2d(...)` computes barycentric weights `(a, b, c)` for the point in the selected UV triangle.
5. The same weights interpolate the triangle's 3D vertex positions and normals:

```text
position = a * P0 + b * P1 + c * P2
normal   = normalize(a * N0 + b * N1 + c * N2)
```

6. The interpolated normal is converted into an orientation with local Z tracking that normal, then the result is composed through the part's scale and parent location.

Smooth polygons interpolate vertex normals; non-smooth polygons use the polygon normal at all three corners. This is why the returned frame follows smooth curved surfaces while retaining a stable normal on a faceted face.

## Metric Offsets: Walking Across Triangles

Normalized UV offsets and metric offsets solve different problems. `offset(u=..., v=...)` adds normalized coordinates, while `offset_m(u=..., v=...)` asks to travel an actual distance along the tessellated surface.

```python
with BuildPart() as result:
    Cylinder(radius=2, height=5, segments=64)
    side = faces().filter_by(GeomType.CYLINDER)[0]

    # Start at the center, then travel in the face-local U direction.
    Marker(side.at(uv.set(u=0.5, v=0.5).offset_m(u=1.0)))
```

`Face._walk_metric(...)` performs that travel one triangle at a time:

1. Find the triangle containing the current UV point.
2. Scale the triangle's 3D vertices by the part's current scale. This is essential: a non-uniformly scaled cylinder has a different physical distance per UV unit at different points.
3. For triangle edges `E1_3d`, `E2_3d` and their UV equivalents `E1_uv`, `E2_uv`, solve the local Jacobian:

```text
[E1_3d E2_3d] = [dP/du dP/dv] * [E1_uv E2_uv]
```

The inverse 2D UV determinant yields the two 3D tangents `dP/du` and `dP/dv`. Their lengths are the current meters per one normalized U or V unit.

4. Cast a UV-space ray along positive or negative U/V and find the nearest triangle edge ahead of the current point.
5. If the remaining distance fits before that edge, convert meters to UV distance with the current tangent length and stop.
6. Otherwise, subtract the distance to the edge, nudge the UV point by `1e-6` across it, and locate the next triangle.

The next-triangle search first checks another fan triangle in the same n-gon, then polygons sharing the crossed edge, and finally polygons sharing either crossed vertex. The final fallback tolerates imperfect mesh connectivity, but it is an implementation recovery path rather than a topological guarantee.

The loop has a 500-step safety limit. A degenerate UV triangle, zero-length tangent, or missing next triangle ends the walk at its last valid point. U is returned modulo `1.0`; V is not wrapped. The tests deliberately cover seam crossings, quad diagonals, negative walks, and non-uniform scale in `test_uv_metric_offsets_on_scaled_cylinder` in [`tests/test_selectors.py`](../tests/test_selectors.py).

## UVSelector

`uv` is the global `UVSelector` entry point. Each fluent method returns a new selector, so an expression can be reused safely:

```python
center_bottom = uv.bottom()
Marker(face.at(center_bottom))
Marker(another_face.at(center_bottom))
```

`Face.at(selector)` calls `selector.select(face._at_uv)`. Selection is a small, deterministic sample-and-rank operation, not a continuous optimization:

1. If U or V is fixed with `set(...)`, that axis has one sample. Otherwise it samples `0.0`, `0.5`, and `1.0`.
2. It evaluates every U/V combination on the face.
3. Each world- or local-coordinate criterion narrows the candidates to the best value within `tolerance()` (default `1e-4`). Criteria run in chain order, so later criteria break ties left by earlier ones.
4. Remaining candidates are ranked by squared UV distance from the selector's preferred `(pref_u, pref_v)`, which defaults to `(0.5, 0.5)`.
5. Normalized offsets wrap with modulo one. Metric offsets are then evaluated by the triangle walk described above. `offset_final(...)` composes a final `Location` transform after surface evaluation.

### Common Selector Patterns

```python
# Default preference: the sample nearest the UV center.
Marker(face.at(uv))

# Pick the sample highest in the face's local coordinate system.
Marker(face.at(uv.local().max_y()))

# Lexicographic selection: maximum local X, then minimum local Y.
Marker(face.at(uv.local().max_x().min_y()))

# Fix coordinates before applying a metric walk.
Marker(side.at(uv.set(u=0.5, v=0.0).offset_m(u=1.0)))

# Start from UV origin, then travel absolute metric distances.
Marker(side.at(uv.set_m(u=2.0, v=0.5)))

# Move a selected sample in normalized UV, wrapping at the boundary.
Marker(face.at(uv.bottom().offset(u=0.2, v=-0.1)))
```

`min_u()`, `max_u()`, `min_v()`, and `max_v()` set only the final tie-break preference. They do not constrain an axis; use `set(...)` when an exact normalized coordinate is required. `min_x()` through `max_z()` compare world positions by default. `.local()` makes those criteria compare each candidate's location in the face-local frame instead.

The examples above correspond to `test_plane_at_with_complex_transform`, `test_uv_selector_offset_with_wrap_around`, and `test_uv_metric_offsets_on_scaled_cylinder` in [`tests/test_selectors.py`](../tests/test_selectors.py).

## Geometry Type Detection

`Face.geom_type` and `Face.main_axis` are lazily calculated by `_detect_geom_type()` and cached. The detector is a mesh-normal heuristic used by `faces().filter_by(GeomType...)`; it does not fit exact analytic surfaces.

| Step | Decision | Result |
| --- | --- | --- |
| 1 | Every polygon normal has dot product greater than `0.999` with the first normal. | `PLANE`; the axis is the normalized sum of normals. |
| 2 | Build normalized differences between consecutive polygon normals. Cross the first and middle usable differences to propose an axis. | Continue only if at least two differences exist; otherwise `UNKNOWN`. |
| 3 | Angles from all normals to that axis have mean absolute deviation below `0.02` radians. | `CYLINDER` if their mean is within `0.05` radians of 90 degrees; otherwise `CONE`. |
| 4 | If revolution checks do not classify the face, normals point from the average polygon center with average dot product above `0.98`. | `SPHERE`; the axis is estimated from the two highest-valence vertices when available. |
| 5 | No test matches. | `UNKNOWN` with Z as the default axis. |

The proposed revolution axis is flipped when necessary to agree with the average normal direction. For a sphere, the `mesh_center` is the average of polygon centers, not a least-squares sphere center. These choices make detection inexpensive and adequate for the primitives covered by the selector tests, but classification can change with irregular tessellation, disconnected geometry, degenerate polygons, or highly deformed surfaces.

```python
with BuildPart() as result:
    Cylinder(radius=1, height=2)
    side = faces().filter_by(GeomType.CYLINDER)[0]
    Marker(side.at(0.25, 0.5))
```

`test_geometry_type_identification` in [`tests/test_selectors.py`](../tests/test_selectors.py) verifies plane, sphere, cylinder, and cone detection, including a thin cone that requires a lower topology smooth-angle threshold to remain a distinct logical face.

## Tested References

- [`tests/test_selectors.py`](../tests/test_selectors.py): geometry classification, direct UV placement, selector ranking, UV offsets, metric walks, surface transformations, and surface-attached locations.
- [`tests/test_locations.py`](../tests/test_locations.py): `Face.at(...)` locations used for joints and parent/child placement.
- [`docs/topology.md`](topology.md): the separate logical-face, wire, edge, and vertex reconstruction algorithm.
