# Topology Reconstruction

`blender_cad` stores geometry as a Blender mesh, not as a BREP model. Blender therefore exposes physical polygons, mesh edges, and mesh vertices. Those elements are useful for rendering, but they are too fine-grained for most CAD-like operations: for example, the curved side of a cylinder is normally tessellated into many polygons even though a user expects to select one surface.

`Topology` in [`blender_cad/geometry.py`](../blender_cad/geometry.py) reconstructs a temporary, CAD-like view of the current BMesh. It is the foundation for `faces()`, `wires()`, `edges()`, and `vertices()`, and consequently for [selector filtering and ordering](selectors.md), [tagging](tags.md), [surface placement](geometry.md), and [curve placement](curve.md).

For public selector usage, see [`docs/selectors.md`](selectors.md). This page explains the current reconstruction algorithm and its implementation-level limitations; it is not a promise of a BREP kernel.

## Why Logical Topology Exists

The analyzer creates a layer of *logical* entities over physical mesh elements:

| Logical entity | Physical backing | Meaning |
| --- | --- | --- |
| `Face` | One or more adjacent BMesh polygons | A smooth connected surface patch, such as a cylinder side or a planar cap. |
| `Wire` | An ordered boundary chain of logical edges | A closed face boundary, a hole boundary, or a loose open/closed mesh path. |
| `Edge` | One or more consecutive BMesh edges | A path segment between sharp turns. |
| `Vertex` | One BMesh vertex | A discrete mesh point. |

Grouping decouples selector intent from tessellation. The side of a 16-segment cylinder and the side of a 64-segment cylinder can each be one logical `Face`, so `faces().filter_by(GeomType.CYLINDER)` expresses the same operation for both. Call `.split()` or an `all_single_*()` helper only when an operation deliberately needs physical mesh resolution.

## Lifecycle And Inputs

Each `Part` owns a `TopologyConfig` and creates a `Topology` lazily when a selector is first requested. `Part` caches that topology until the mesh changes or the configuration differs. Selector wrappers refer to the BMesh used for that reconstruction, so reacquire selections after an operation that edits the mesh.

Configure the thresholds when creating a part or update the active build context:

```python
with BuildPart(
    topology=TopologyConfig(smooth_angle=10, edge_break_angle=20),
) as result:
    Cylinder(radius=2, height=5)

set_topology(TopologyConfig(smooth_angle=8, edge_break_angle=12))
```

Both values are expressed in degrees.

| Setting | Default | Exact decision |
| --- | --- | --- |
| `smooth_angle` | `15.0` | Adjacent polygons join one logical face when their adjusted normal angle is less than or equal to this threshold. |
| `edge_break_angle` | `15.0` | Consecutive boundary edges start a new logical edge when their deviation from a straight continuation is strictly greater than this threshold. |

These settings change only the logical view. They do not weld vertices, retessellate polygons, alter normals, or otherwise change the source mesh.

## Reconstruction Pipeline

`Topology._build_topology()` performs the following stages for one BMesh.

1. Visit every unassigned physical polygon and grow one logical face from it.
2. Find the physical edges on that face group's boundary.
3. Order each boundary into a wire and split the wire at sharp turns into logical edges.
4. Independently collect mesh edges with no linked faces as loose wires.
5. Independently collect mesh vertices with no linked edges as loose vertices.
6. Flatten the face and loose-wire results into the top-level `faces`, `wires`, `edges`, and `vertices` collections.

The object graph is therefore directional: a `Face` owns its boundary `Wire` objects, a `Wire` owns its logical `Edge` objects, and an `Edge` knows the physical edges that form its continuous chain. The top-level collections are convenience views over that graph plus loose topology.

## Stage 1: Growing A Logical Face

`_grow_face_group()` uses breadth-first search. The seed polygon is immediately marked visited, then the algorithm examines every physical edge of every queued polygon and considers the edge's linked polygons as candidates.

For each unvisited neighbor with a usable normal, it calculates the angle between the current polygon normal and the neighbor normal. When that angle is greater than 90 degrees, the analyzer replaces it with `abs(pi - angle)` before comparing it with `smooth_angle`. This deliberately treats nearly opposed normals as nearly aligned. It makes reconstruction more tolerant of the flipped-normal geometry exercised by `test_topology_reconstruction_complex`, but it also means normal direction alone is not a reliable separator.

The threshold is evaluated at each adjacency, not against the initial seed normal. Consequently, grouping is transitive: `A` can join `B`, and `B` can join `C`, even if `A` and `C` differ by more than `smooth_angle`.

```text
seed polygon
    |
    +-- adjacent polygon within smooth_angle -> same logical Face
    |       |
    |       +-- its eligible neighbor -> same logical Face
    |
    +-- sharper adjacent polygon -> boundary between logical Faces
```

Degenerate polygons require special care. A queued polygon whose normal length is effectively zero does not expand the search. A zero-normal neighbor of a valid polygon is marked visited rather than queued. This is defensive handling of invalid mesh data, not a geometric classification guarantee; avoid relying on logical faces produced from degenerate polygons.

## Stage 2: Extracting Face Boundaries

Once a group is complete, `_build_topology()` examines every physical edge used by the group's polygons. An edge is a boundary edge when **exactly one** of its linked faces belongs to the group:

```python
sum(1 for linked_face in edge.link_faces if linked_face in face_group) == 1
```

An edge shared by two group polygons is internal tessellation and disappears from the logical boundary. An edge shared by a group polygon and a polygon outside the group remains a boundary, even if the physical edge also has unusual non-manifold linkage. This simple membership rule is why logical-face selection can ignore the triangles or quads used to render a smooth surface.

## Stage 3: Building Wires And Logical Edges

`_build_wires()` repeatedly removes one unconsumed boundary edge, follows any remaining boundary edge incident to the current vertex, and stops when it returns to the start vertex or cannot continue. The ordered physical chain becomes one `Wire` after segmentation.

`_segment_loop()` walks consecutive physical edges in that chain. At their shared vertex it builds two vectors pointing away from the vertex, then measures their angle. A straight continuation has an angle near 180 degrees. The chain is split whenever:

```text
abs(180 degrees - angle) > edge_break_angle
```

Each resulting `Edge` stores every physical edge from its segment and a traversal start vertex. This ordering is important: `Edge.at(t)`, `Edge.curve()`, and `Wire.at(t)` traverse the stored chain, rather than treating the backing edges as an unordered set.

The current outer/inner classification is intentionally simple: the first boundary wire found for a face is marked `is_outer=True`; later wires are marked as inner. It is not based on projected area, winding, containment, or surface orientation. For ordinary primitive caps and holes this supplies the expected `outer_wires()` and `inner_wires()` interface, but complex, disconnected, or ambiguous boundary arrangements should not rely on that classification as a formal topological proof.

## Loose Edges And Loose Vertices

Mesh edges with no linked faces do not belong to a face boundary. `_process_loose_wires()` still exposes them to selectors by tracing continuous chains of such edges and creating `Wire(is_outer=False)` objects.

The loose-wire start policy makes the resulting traversal more useful:

1. Prefer a vertex connected to exactly one loose edge, which gives an open path a natural endpoint.
2. For a closed chain, prefer a sharp corner so the unavoidable seam falls at an existing visible break.
3. If the loop is smooth everywhere, use the first available edge and vertex as the seam.

The traced chain is then segmented with the same `edge_break_angle` rule as face boundaries. Vertices with no linked edges at all are emitted as separate logical `Vertex` values.

## What The Analyzer Does Not Infer

This topology is derived from connectivity and local angles. It does not infer analytic BREP surfaces, repair invalid meshes, merge disconnected islands, or provide a general non-manifold solver. In particular:

- Face grouping depends on shared physical edges. Touching geometry that does not share an edge remains separate.
- A smooth-angle threshold groups gradual tessellation changes; it is not curvature fitting or a feature-recognition algorithm.
- Boundary traversal selects the first available continuation. Branching and otherwise ambiguous non-manifold boundaries are not canonicalized.
- Wire outer/inner status is a discovery-order heuristic, not containment analysis.
- Logical wrappers are snapshots of the active BMesh topology. Retain only their semantic result, not the wrapper itself, across mesh edits.

## Selector Consequences

The analyzer is what makes selector expressions operate on modeling intent rather than raw tessellation:

```python
with BuildPart() as result:
    Cylinder(radius=5, height=2, segments=64)

    side = faces().filter_by(GeomType.CYLINDER)[0]
    side.mat = mat.blue
    top_boundary = faces().top()[0].outer_wires()[0]
```

`Face.geom_type`, `Face.at(...)`, `Wire.at(...)`, `CurveLocations`, material assignment, and selector set operations all receive the logical wrappers produced here. A topology configuration should therefore be chosen for the feature scale a model intends to select: a low `smooth_angle` preserves shallow creases as separate faces, while a low `edge_break_angle` exposes more path corners as separate edges.

## Tested References

- [`tests/test_selectors.py`](../tests/test_selectors.py) covers primitive grouping and classification, wire traversal, topology reconstruction with flipped normals and a T-junction, and face/edge selector workflows.
- `TestSelectorsAndLocations.test_geometry_type_identification` verifies that cylinder, cone, sphere, and planar physical polygons are presented as logical surfaces.
- `TestSelectorsAndLocations.test_topology_reconstruction_complex` exercises reconstruction after beveling, an edge extrusion, and potentially flipped normals.
- `TestSelectorsAndLocations.test_polygon_free_curve_mesh` verifies loose-wire and sharp-corner segmentation behavior for a mesh with no polygons.
