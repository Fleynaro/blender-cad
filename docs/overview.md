# blender_cad Overview

`blender_cad` is a convenient way to describe 3D models with code: **3D modelling as code**. This code-first approach is increasingly important in the industry because a model can be read, changed, reviewed, generated, and rebuilt like any other software artifact.

The library is designed primarily for structured polygonal geometry: buildings, walls, furniture, modular environments, hard-surface props, panels, and other assets whose shape follows clear spatial rules. It is not intended to replace Blender's sculpting workflow. Organic sculpture, highly irregular biological forms, and art-directed freeform detail are usually better made with dedicated artistic tools.

The public interface is strongly inspired by [build123d](https://build123d.readthedocs.io/), but is substantially extended and adapted for Blender meshes rather than CAD BREP solids. It provides CAD-like contexts, selection, and topology over render-ready polygonal geometry while retaining Blender-native materials, modifiers, and mesh workflows.

## Why Model With Code

- **Editable models.** Change a dimension, a rule, or a reusable component and rebuild the result instead of manually repairing a binary scene asset.
- **Portable models.** Source code is compact, version-controllable, reviewable, and easy to share or reuse.
- **Modular composition.** A wall, panel, furniture unit, or other component can be replaced without reconstructing the surrounding model.
- **Parametric rebuilds.** Models express inputs and relationships. For example, changing a four-column layout to six columns can rebuild the entire structure consistently.
- **Agent-friendly authoring.** The library is designed first of all for agentic development. Agents can inspect and modify explicit code much more reliably than opaque binary models.
- **Lower spatial cognitive load.** Raw 3D coordinates are difficult for both people and language models to reason about. `blender_cad` encourages declarative construction, semantic selectors, parent-child structure, and relationships instead of scattered magic numbers.
- **Solvable layouts.** Rules can describe the intended relationship, while the built-in solver finds concrete positions and sizes that satisfy it. See [Rule-Based Layout](rbl.md) and [Solver](solver.md).

## Architecture At A Glance

```text
Declarative layer:  ml, rbl, chain
                     |
                     v
Core modelling:    BuildPart / BuildCurve, operations, Part / Curve
                     |
                     v
Placement:         Transform / Location / Locations
                     |
                     v
Blender mesh layer: polygon meshes, curves, materials, modifiers
                     |
                     v
CAD-like view:     logical topology, selectors, surface placement
```

### Core Modelling Layer

The core follows the familiar `build123d` context pattern, but every solid result is a Blender mesh:

- `BuildPart` collects mesh geometry into a `Part`; primitives, `add(...)`, boolean modes, and mesh joins operate in this context. See [Part And BuildPart](part.md).
- `BuildCurve` collects spline primitives into a `Curve`, which can be beveled, filled, extruded, or converted to a mesh `Part`. See [Curve](curve.md).
- Mesh operations such as `extrude`, `bevel`, and selected-vertex transforms edit the active part. See [Modifiers And Mesh Operations](modifiers.md).
- `Transform`, `Location`, `Pos`, `Rot`, and `with Locations(...)` provide readable transform composition and contextual placement. See [Locations And Transforms](location.md).

### Meshes With A CAD-Like Topology View

`blender_cad` is mesh-first, not a NURBS or BREP kernel. Blender stores a cylinder side, for example, as many physical polygons. The topology layer temporarily groups compatible polygons into logical `Face`, `Wire`, `Edge`, and `Vertex` entities so code can work with a surface-level concept instead of tessellation details.

- [Topology Reconstruction](topology.md) explains how physical mesh elements are grouped into logical topology and where this approximation differs from BREP/NURBS CAD.
- [Selectors And Mesh Topology](selectors.md) explains `faces()`, `wires()`, `edges()`, `vertices()`, filtering, ordering, and logical-versus-physical selections.
- [Geometry And Surface Mapping](geometry.md) explains geometry classification and `Face.at(...)`, which produces oriented locations on mesh surfaces for attachment and placement.

This layer offers CAD-like selection and surface workflows without claiming analytic CAD surfaces or turning the underlying mesh into NURBS.

### Declarative Composition Layer

The layers above the core reduce procedural coordinate work further:

- [Markup Layout (ML)](ml.md) builds structured, mostly 2D layouts with an HTML/CSS-inspired parent-child tree, sizing, flow, flex layout, text, and mesh-producing styles. It is useful for facades, walls, panels, consoles, dashboards, and labels.
- [Rule-Based Layout (RBL)](rbl.md) declares relationships between objects and groups: position, size, orientation, containment, collision avoidance, curve following, and stacks. Calling `resolve()` turns those rules into a concrete layout.
- [Solver](solver.md) is the optimization backend for RBL and can also be used directly. It searches adjustable positions, rotations, scales, and other parameters to minimize declared layout errors.
- [Chain Assembly](chain.md) connects reusable components end to end through joints. It is well suited to rooms, corridors, folded wall runs, and repeated modular structures.
- [Joints And Assembly](joint.md) covers explicit oriented attachment points and one-off component alignment.

These systems preserve intent. Instead of writing a final coordinate such as `Pos(X=12.37)`, describe that an object belongs to a group, stacks after another object, fits a boundary, follows a curve, or must face a target. When inputs change, rebuild or resolve the model to obtain a new consistent result.

## Supporting Capabilities

| Capability | Use it for | Documentation |
| --- | --- | --- |
| Materials | Layered PBR materials, procedural surfaces, textures, and face-level assignment | [Materials](materials.md) |
| Tags | Semantic labels for geometry, curves, and layout nodes | [Tags](tags.md) |
| Text | Font geometry, styled fragments, curve text, wrapping, and mesh conversion | [Text](text.md) |
| Curves | Paths, splines, bevels, fills, curve placement, and conversion to mesh parts | [Curve](curve.md) |
| Surface placement | Attach components to an oriented point sampled from a logical mesh face | [Geometry And Surface Mapping](geometry.md) |
| Mesh editing | Extrusion, bevels, local transforms, and proportional editing | [Modifiers And Mesh Operations](modifiers.md) |
| Reusable assembly | Named joints and chain-based connections between components | [Joints And Assembly](joint.md) and [Chain Assembly](chain.md) |

## Choosing A Starting Point

- Start with [Getting Started With blender_cad](getting-started.md) for a beginner-friendly walkthrough from one box to a small model.
- Start with [Part And BuildPart](part.md) for primitives, booleans, mesh joins, and the main modelling context.
- Use [Locations And Transforms](location.md) when placement or coordinate-frame composition is central to the model.
- Use [Selectors And Mesh Topology](selectors.md) to target faces, edges, wires, or vertices after geometry has been built.
- Use [Markup Layout (ML)](ml.md) for structured panels, facades, text-heavy surfaces, and CSS-like 2D arrangement.
- Use [Rule-Based Layout (RBL)](rbl.md) and [Solver](solver.md) when objects should maintain relationships rather than fixed coordinates.
- Use [Chain Assembly](chain.md) and [Joints And Assembly](joint.md) when repeated components need to connect in 3D.

The [README](../README.md) contains a short introduction and quick-start examples.
