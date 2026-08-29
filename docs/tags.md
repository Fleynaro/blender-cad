# Tags

Tags are lightweight string labels attached to geometry or layout nodes. They let a model mark meaningful regions while it is being built, then find those regions later with selectors instead of preserving temporary Python variables or reconstructing the selection from geometric predicates.

Typical uses include assigning materials to selected faces, modifying geometry produced by an operation, isolating a branch of a curve, and targeting nodes in a rule-based layout (RBL) tree.

```python
from blender_cad import *

with BuildPart() as result:
    Box(4, 4, 1)
    faces().top().add_tags("panel:top")
    faces().tagged("panel:top").mat = mat.blue
```

Tags are not a type system or a replacement for geometric selection. Use them when the important property is semantic, such as "this is the panel background" or "this geometry came from this extrusion," rather than a property that should be recomputed from geometry, such as the largest face.

## Matching Tags

The same matching convention is used by mesh selectors, curves, and RBL queries:

- `tagged(*tags)` keeps items with at least one matching requested tag.
- `untagged(*tags)` keeps items with no matching requested tag.
- Passing several tags is an OR query: an item matches if it has any requested tag.
- A `*` in a requested tag is a wildcard. For example, `tagged("panel:*")` matches `panel:top` and `panel:side`.

Use namespaced tags to make broad wildcard queries useful and avoid accidental collisions:

```python
paintable = faces().tagged("finish:*")
paintable.mat = mat.red

non_background = faces().untagged("ml:background")
```

`tagged(...)` and `untagged(...)` return filtered results. They do not remove labels from the source object. To edit labels, use `add_tags(...)`, `remove_tags(...)`, or assign the `tags` property on individual geometry entities or a `ShapeList`.

## Mesh Part Tags

`Part` inherits the tag API from `Object`. A mesh can store tags independently on its `FACE`, `EDGE`, and `POINT` domains. `POINT` is the mesh-vertex domain; it is also the name used by Blender's geometry-attribute API.

Inside an active `BuildPart`, the public helpers operate on the current part:

```python
with BuildPart() as result:
    Box(2, 2, 2)

    # Tag only the selected faces.
    faces().top().add_tags("finish:highlight")

    # Query all tags currently present on the part.
    print(get_tags())

    # Replace or remove tags across an explicit domain when needed.
    set_tags(["source:example"], domain="FACE")
    remove_tags(["source:example"], domain="FACE")
```

The equivalent object-level methods are `part.get_tags(...)`, `part.set_tags(...)`, `part.add_tags(...)`, and `part.remove_tags(...)`. With no domain, a mesh operation covers faces, edges, and points. A selector is usually more precise because it updates only the selected entities:

```python
faces().tagged("finish:highlight").mat = mat.yellow
edges().tagged("panel:*").add_tags("export:visible")
vertices().untagged("anchor").add_tags("anchor")
```

### Blender Storage And 3D Operations

For mesh data, tags are stored in Blender string geometry attributes named `_tag_face`, `_tag_edge`, and `_tag_vert`. They belong to the mesh data block, not to a temporary `ShapeList` or selector wrapper. Consequently, selecting an element, assigning its tags, and later creating a new selector still addresses the same stored metadata.

This storage model also makes tags suitable for multi-step modeling. The mesh attribute data remains associated with the mesh while Blender CAD operations read and write that mesh. Operations that preserve or copy source mesh elements can preserve their attributes, while operations that create new topology must assign or propagate tags deliberately. Do not treat arbitrary topology changes as a promise that every semantic label will be inferred for every newly created element.

The library uses this mechanism during its own operations. `tests/test_selectors.py` covers tag propagation through nested building and transforms, then uses tagged face selections to assign different materials.

## Curves

Curves use the same public concepts but have different storage constraints. Curve tags may belong to a spline (`CURVE` domain) or to a control point (`POINT` domain):

```python
path = curve(
    curve.tag("route:main"),
    curve.step(10, tag="route:first-segment"),
    curve.step(10, angle=90, tag="route:turn"),
).build()

turn = path.tagged("route:turn")
without_main = path.untagged("route:main")
```

`Curve.tagged(...)` creates a new sub-curve containing matching splines or control points. `Curve.untagged(...)` is its inverse filter. Tags copied into a sub-curve remain available for later selection.

### Curve Storage And Filter Tags

Blender's native geometry-attribute support used for meshes is not available for the required curve domains. `Object._ensure_tag_attribute(...)` therefore falls back to library-managed custom storage for `bpy.types.Curve`. This storage is associated with the `blender_cad` object layer, rather than being a native Blender curve-data attribute. Code that needs Blender-native, mesh-persistent tag attributes should convert or work with a `Part`; code working with `Curve` should use the curve tagging API directly.

Some curve tags are generated only while matching and are not stored in that custom storage. In particular, `Curve.TAG_POINT_INDEX` is a filter namespace that exposes a control point's global index. The constants `Curve.TAG_POINT_FIRST` and `Curve.TAG_POINT_LAST` are convenient index filters for the first and last control points. They can be used to refine a selection without adding permanent metadata:

```python
body = path.tagged("route:turn").untagged(
    Curve.TAG_POINT_FIRST,
    Curve.TAG_POINT_LAST,
)
```

Curve smoothing can also generate `Curve.TAG_POINT_SMOOTH_FILLET_START` and `Curve.TAG_POINT_SMOOTH_FILLET_END` for its generated boundary points. See [`curve.md`](curve.md#tags-and-sub-curves) and `tests/test_curve.py` for the tested curve-tag workflows.

## System Tags

The library adds several tags automatically when it creates geometry. These labels are intended to make the output of a higher-level operation immediately selectable.

| Tag | Added by | Purpose |
| --- | --- | --- |
| `Object.TAG_OP_EXTRUDE` (`"op:extrude"`) | `extrude(...)` | Marks newly generated vertices, edges, and faces. |
| `Object.TAG_ML_BACKGROUND` (`"ml:background"`) | ML background generation | Marks mesh geometry generated for an ML node background. |
| `Object.TAG_ML_BORDER` (`"ml:border"`) | ML border generation | Marks mesh geometry generated for an ML node border. |

For example, after extrusion, selecting `Part.TAG_OP_EXTRUDE` directly identifies the generated geometry. This is often clearer and more robust than creating a checkpoint and using `filter_by(lambda entity: entity.is_new())`:

```python
with BuildPart() as result:
    Box(2, 2, 1)
    extrude(faces().top(), op=Pos(Z=1))

    faces().tagged(Part.TAG_OP_EXTRUDE).mat = mat.red
```

ML uses the same idea to expose meaningful parts of generated layout geometry:

```python
with BuildPart() as result:
    ml(
        style(
            width=4,
            height=4,
            background_mat=mat.red,
            border_width=0.2,
            border_style="solid",
        ),
    ).build()

    faces().tagged(Part.TAG_ML_BACKGROUND).mat = mat.blue
    extrude(edges().tagged(Part.TAG_ML_BORDER), op=Pos(Z=-1))
```

This lets an ML component describe its generated regions declaratively, then lets downstream modeling or material work select them without depending on their final topology. See [`ml.md`](ml.md) for the layout system and `tests/test_selectors.py` for tested background, border, and extrusion examples.

## RBL Node Tags

Tags also identify logical nodes in an RBL tree. These are node tags, not mesh geometry attributes, so they are useful before or independently of final physical geometry creation.

```python
red = rl.object(ml(style(width=1, height=1)), tag="accent:red")
blue = rl.object(ml(style(width=1, height=1)), tag=["accent:blue", "shared"])

layout = rl.group(
    red,
    blue,
    rl.tagged("accent:*") | rl.transform(x=1),
)
layout.resolve()
```

`rl.tagged(*tags, root=None)` and `rl.untagged(*tags, root=None)` return an RBL query filter chain. The query can be scoped with `rl.root(node)` and composed with rules using `|`. `rl.object(...)` automatically adds `rl.TAG_OBJECT`, and `rl.group(...)` automatically adds `rl.TAG_GROUP`.

See [`rbl.md`](rbl.md#groups-tags-and-queries) for RBL query scoping and rule composition.

## Related Coverage

- `tests/test_selectors.py` verifies mesh tag propagation, wildcard selection, material assignment, extrusion tags, and ML background/border tags.
- `tests/test_curve.py` verifies curve builder tags, sub-curve isolation, and first/last point filtering.
- `docs/selectors.md` covers the selector API alongside other mesh-selection tools.
- `docs/curve.md` covers curves and sub-curve selection.
- `docs/rbl.md` and `docs/ml.md` describe the two systems that use tags for declarative node and generated-geometry selection.
