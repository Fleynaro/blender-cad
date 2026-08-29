# blender_cad

**Code-first polygonal modeling for Blender.**

`blender_cad` lets you describe structured 3D models with Python and build them directly as Blender meshes. It combines a build123d-inspired modeling workflow with Blender-native materials, modifiers, curves, and mesh data.

Use it for buildings, furniture, modular environments, hard-surface props, panels, facades, dashboards, and other assets that benefit from explicit dimensions, reusable components, and repeatable rules.

## Why blender_cad?

- **Keep models editable.** Change a dimension, a layout rule, or a reusable component, then rebuild the result.
- **Work with Blender meshes.** Create render-ready polygonal geometry without an export/import loop.
- **Express intent, not only coordinates.** Use contextual placement, semantic selection, declarative layout, and assembly relationships.
- **Version and review models as code.** Model sources are compact, diffable, and easy to regenerate.
- **Stay close to Blender.** Use Blender-native materials, curves, mesh operations, and scene objects.

## Quick Start

Run modeling scripts in Blender's Python environment. The recommended workflow is VS Code with the [Blender Development extension](https://github.com/JacquesLucke/blender_vscode): open this repository as a folder, start Blender with **Blender: Start**, then execute a script with **Blender: Run Script**.

Alternatively, use Blender's Text Editor after making the `blender_cad` package available to Blender's Python environment.

Create a file such as `scratch.py` with:

```python
from blender_cad import *

with BuildPart(mat=mat.blue) as result:
    Box(2, 2, 2)
    faces().top().mat = mat.gold

result.part.show(name="gift_box")
```

Run it to add a blue box with a gold top to the current Blender scene. Dimensions are Blender units; use one consistent unit convention for a project.

## Build With Code

Model geometry in a `BuildPart` context, then show the resulting `Part` in Blender:

```python
from blender_cad import *

with BuildPart(mat=mat.iron) as result:
    Box(4, 4, 1)
    Cylinder(radius=0.75, height=2, mode=Mode.SUBTRACT)

result.part.show(name="plate_with_hole")
```

Use `Mode.ADD`, `Mode.SUBTRACT`, `Mode.INTERSECT`, and `Mode.JOIN` to compose geometry. `SUBTRACT` creates real mesh cuts; `JOIN` combines mesh data without a boolean operation.

Placement contexts make repeated geometry readable:

```python
from blender_cad import *

with BuildPart(mat=mat.green) as result:
    with GridLocations(2, 2, 3, 2):
        Cylinder(radius=0.4, height=1)

result.part.show(name="cylinder_grid")
```

For finished geometry, select and edit the active mesh:

```python
from blender_cad import *

with BuildPart(mat=mat.plastic_glossy_red) as result:
    Box(2, 2, 1)
    extrude(faces().top(), op=Pos(Z=1) * Scale(XY=0.6))
    bevel(edges(), radius=0.1, segments=3)

result.part.show(name="stepped_box")
```

Reacquire `faces()`, `edges()`, `wires()`, or `vertices()` after operations that change the mesh: earlier selections can refer to outdated mesh elements.

## What You Can Build

| Area | Capabilities |
| --- | --- |
| Core modeling | `BuildPart`, primitives, booleans, mesh joins, reusable parts, and scene display |
| Placement | `Pos`, `Rot`, `Scale`, `Transform`, surface and curve locations, plus grid, polar, hexagonal, and curve distributions |
| Selection | Logical faces, wires, edges, and vertices with filtering, sorting, grouping, checkpoints, tags, and face-level materials |
| Mesh editing | Extrude, bevel, subdivision, proportional editing, solidify, deletion, mirror, bend, twist, wrap, and other Blender mesh operations |
| Materials | Layered PBR materials, procedural surfaces, textures, shader variables, and per-face overrides |
| Curves and text | Paths and splines, bevel/fill/extrude, curve placement, font geometry, styled text, and mesh conversion |
| Declarative layout | HTML/CSS-inspired markup layout (`ml`) and rule-based layout (`rl`) backed by an optimizer |
| Assembly | Named joints, ports, and ordered chain assemblies for modular components |

## Choose A Workflow

- Start with [Getting Started](docs/getting-started.md) for a guided first model.
- Use [Part and BuildPart](docs/part.md) for primitives, modes, composition, and object lifecycle.
- Use [Locations and Transforms](docs/location.md) for placement, orientation, repetition, and alignment.
- Use [Selectors and Mesh Topology](docs/selectors.md) to target precise geometry after it is built.
- Use [Modifiers and Mesh Operations](docs/modifiers.md) to edit an active mesh.
- Use [Materials](docs/materials.md) for procedural and textured surfaces.
- Use [Curve](docs/curve.md) and [Text](docs/text.md) for paths, typography, and curve-driven geometry.
- Use [Markup Layout](docs/ml.md) for structured panels, facades, labels, and other mostly 2D arrangements.
- Use [Rule-Based Layout](docs/rbl.md) and the [Solver](docs/solver.md) when relationships should determine positions and sizes instead of fixed coordinates. Direct solver use requires SciPy in Blender's embedded Python.
- Use [Joints and Assembly](docs/joint.md) and [Chain Assembly](docs/chain.md) to connect reusable components.

## Documentation

| Guide | Description |
| --- | --- |
| [Overview](docs/overview.md) | Architecture, scope, and subsystem guide |
| [Getting Started](docs/getting-started.md) | First parts, materials, placement, booleans, and edits |
| [Part and BuildPart](docs/part.md) | Core part modeling and composition modes |
| [Locations and Transforms](docs/location.md) | Transforms, placement contexts, and alignment |
| [Selectors](docs/selectors.md) | Filtering and ordering logical mesh selections |
| [Topology](docs/topology.md) | Logical topology reconstructed from Blender meshes |
| [Geometry](docs/geometry.md) | Surface classification, UV mapping, and `Face.at(...)` |
| [Modifiers](docs/modifiers.md) | Mesh editing and baked Blender modifiers |
| [Materials](docs/materials.md) | PBR layers, textures, and shader customization |
| [Curves](docs/curve.md) | Curve primitives, building, placement, and conversion |
| [Text](docs/text.md) | Font geometry, styling, wrapping, and curve text |
| [Tags](docs/tags.md) | Semantic labels and wildcard queries |
| [Markup Layout](docs/ml.md) | CSS-like layout and generated geometry |
| [Rule-Based Layout](docs/rbl.md) | Relationship-driven object layout and constraints |
| [Solver](docs/solver.md) | Parameter optimization and solver strategies |
| [Joints](docs/joint.md) | Named attachment points and component alignment |
| [Chain Assembly](docs/chain.md) | Sequential, branched, and modular assemblies |

## Mesh-First By Design

`blender_cad` works with Blender polygon meshes, not analytic BREP or NURBS solids. Its topology layer groups compatible physical polygons into logical faces, wires, edges, and vertices so you can write surface-level modeling code without losing Blender's native workflow.

Logical topology and `GeomType` classification are tessellation-dependent approximations, not analytic CAD guarantees. The library is designed for structured hard-surface and modular assets rather than sculpting, highly irregular organic forms, or freeform art-directed detail.

## Further Reading

See the [project overview](docs/overview.md) for the full architecture and guidance on selecting the right modeling system for a task.
