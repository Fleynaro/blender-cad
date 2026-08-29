# Getting Started With blender_cad

`blender_cad` lets you create Blender models with Python code. You describe a shape, run the script in Blender, and the model appears in the scene.

This tutorial starts with one box and gradually adds materials, repeated objects, holes, and mesh editing. Each example is a complete script. The recommended way to run the examples is from VS Code; you can also use Blender's built-in Text Editor.

## Before You Start

### Recommended: Run Scripts From VS Code

The recommended workflow is to edit and run scripts from VS Code with the [Blender VS Code extension](https://github.com/JacquesLucke/blender_vscode):

1. Install the extension from the VS Code Extensions view. Search for **Blender Development** (`JacquesLucke.blender-development`) and verify that it is published by Jacques Lucke.
2. Install and open Blender, then open this `blender_cad` repository as a folder in VS Code.
3. Create an empty Python file in the repository, for example `scratch.py`.
4. Add an import and one of the examples from this tutorial to the file:

   ```python
   from blender_cad import *
   ```

5. In VS Code open the Command Palette with `Ctrl+Shift+P`, choose **Blender: Start**, select your Blender executable, and wait for Blender to start. Then choose **Blender: Run Script** and select the file.

The script runs in Blender's Python environment, so the repository's `blender_cad` package is imported directly while you edit the source in VS Code. Keep the script open in the repository so it can be rerun after each change.

### Alternative: Blender's Text Editor

Open Blender and make sure the `blender_cad` package is available in its Python environment. Create a new text block in Blender's **Text Editor**, paste an example, and choose **Run Script**.

Every example starts with this import:

```python
from blender_cad import *
```

The `*` import is intentional here. It makes the library's public modelling tools, such as `BuildPart`, `Box`, `Pos`, and `mat`, available without long module names.

## The Smallest Useful Model

Make a box:

```python
from blender_cad import *

with BuildPart() as result:
    Box(2, 2, 2)

result.part.show(name="my_box")
```

Run the script. A box named `my_box` appears in the Blender scene.

Here is what each line does:

- `BuildPart()` starts a modelling area. Shapes created inside it become one result.
- `Box(2, 2, 2)` creates a box that is 2 units wide, 2 units deep, and 2 units high.
- `as result` gives us access to the finished model after the indented block.
- `result.part` is the finished `Part`, the library's mesh object.
- `show(...)` adds that part to the current Blender collection so you can see it.

Most dimensions in `blender_cad` are Blender units. Use a consistent unit convention in your project. For many Blender scenes, one unit is treated as one meter.

For a deeper explanation of `Part` and `BuildPart`, see [Part And BuildPart](part.md).

## Add A Material

Pass a material to `BuildPart` to give the whole model a default material:

```python
from blender_cad import *

with BuildPart(mat=mat.blue) as result:
    Box(2, 2, 2)

result.part.show(name="blue_box")
```

`mat.blue` is one of the ready-to-use materials. Try `mat.red`, `mat.green`, `mat.gold`, or `mat.wood_oak` too.

You can also use a different material on selected faces. This example makes only the top face gold:

```python
from blender_cad import *

with BuildPart(mat=mat.blue) as result:
    Box(2, 2, 2)
    faces().top().mat = mat.gold

result.part.show(name="gift_box")
```

`faces()` gets the faces of the part currently being built. `.top()` keeps the top face, and `.mat = ...` assigns its material.

Read [Materials](materials.md) for procedural materials, textures, and more material options. Read [Selectors And Mesh Topology](selectors.md) to learn how to select other faces, edges, and vertices.

## Put Shapes In Different Places

By default, shapes are made around the origin, the point `(0, 0, 0)`. Use `Locations` and `Pos` to make the same shape in several positions:

```python
from blender_cad import *

with BuildPart(mat=mat.green) as result:
    with Locations(Pos(X=-2), Pos(X=2)):
        Box(1, 1, 1)

result.part.show(name="two_boxes")
```

This creates two 1 by 1 by 1 boxes:

- The first box is moved 2 units along negative X.
- The second box is moved 2 units along positive X.

The shapes are collected into one `Part` because both are inside the same `BuildPart` block.

For a simple grid of objects, use `GridLocations`:

```python
from blender_cad import *

with BuildPart(mat=mat.red) as result:
    with GridLocations(2, 2, 3, 2):
        Cylinder(radius=0.4, height=1)

result.part.show(name="cylinder_grid")
```

This makes a centered 3 by 2 grid of cylinders, with 2 units between neighbors in X and Y.

See [Locations And Transforms](location.md) for rotations, scale, circular patterns, and combining transforms.

## Cut A Hole With A Boolean

Shapes normally use `Mode.ADD`, which combines them into the current model. Use `Mode.SUBTRACT` to cut one shape from another.

```python
from blender_cad import *

with BuildPart(mat=mat.iron) as result:
    Box(4, 4, 1)
    Cylinder(radius=0.75, height=2, mode=Mode.SUBTRACT)

result.part.show(name="plate_with_hole")
```

The box is the starting solid. The taller cylinder passes through it and is subtracted, leaving a circular hole.

The most common modes are:

| Mode | What it does |
| --- | --- |
| `Mode.ADD` | Combines a new shape with the current model. This is the default. |
| `Mode.SUBTRACT` | Cuts the new shape out of the current model. |
| `Mode.INTERSECT` | Keeps only the volume shared by the current model and new shape. |
| `Mode.JOIN` | Joins mesh data without a boolean operation. Useful for assemblies. |

Boolean operations are more expensive than simply adding a primitive, so use them when you need a real cut, overlap, or intersection. The full mode reference and composition behavior are in [Part And BuildPart](part.md).

## Build A Small Table

This example combines everything so far. It builds a tabletop and four legs, assigns materials, and uses locations to place the legs.

```python
from blender_cad import *

with BuildPart() as result:
    # The tabletop is centered at Z = 2.
    with Locations(Pos(Z=2)):
        with BuildPart(mat=mat.wood_oak):
            Box(4, 3, 0.3)

    # Make one leg at every corner below the tabletop.
    with Locations(
        Pos(X=-1.6, Y=-1.1, Z=0.85),
        Pos(X=-1.6, Y=1.1, Z=0.85),
        Pos(X=1.6, Y=-1.1, Z=0.85),
        Pos(X=1.6, Y=1.1, Z=0.85),
    ):
        with BuildPart(mat=mat.iron):
            Box(0.3, 0.3, 1.7)

result.part.show(name="simple_table")
```

Notice that the shapes use coordinates only to describe their relationship to the center. Each `Box` is centered on its location. The legs have a height of `1.7`, so placing their centers at `Z=0.85` puts their bottoms on the ground at `Z=0`.

When a model gets more complex, avoid scattering unexplained numbers through the code. Give dimensions meaningful variable names, then use those variables in your locations and primitives.

## Edit A Finished Shape Inside BuildPart

You can select geometry and edit the active mesh. This example extrudes the top face of a box upward, then rounds all edges:

```python
from blender_cad import *

with BuildPart(mat=mat.plastic_glossy_red) as result:
    Box(2, 2, 1)
    extrude(faces().top(), op=Pos(Z=1) * Scale(XY=0.6))
    bevel(edges(), radius=0.1, segments=3)

result.part.show(name="stepped_box")
```

The model is made in three steps:

1. Create a short box.
2. Select its top face and extrude it upward while making the new top smaller.
3. Select all edges and bevel them.

After an operation that changes the mesh, such as `extrude` or `bevel`, get a fresh selection with `faces()`, `edges()`, or `vertices()`. Old selections may no longer point to valid mesh elements.

Read [Modifiers And Mesh Operations](modifiers.md) for extrusion, bevels, transforms, subdivision, deformation, and other mesh edits.

## Where To Go Next

- [Overview](overview.md): choose the right high-level feature for your model.
- [Part And BuildPart](part.md): primitives, boolean modes, existing parts, and object lifecycle.
- [Locations And Transforms](location.md): move, rotate, scale, and repeat geometry.
- [Materials](materials.md): build layered PBR materials and apply them to faces.
- [Selectors And Mesh Topology](selectors.md): select logical faces, wires, edges, and vertices.
- [Curve](curve.md): create paths and turn them into geometry.
- [Markup Layout](ml.md): make structured panels, facades, and text-heavy layouts.
- [Rule-Based Layout](rbl.md): describe relationships between objects instead of manually calculating every coordinate.
- [Joints And Assembly](joint.md) and [Chain Assembly](chain.md): connect reusable components together.

Start by making a small object with a few dimensions, then rebuild it after every change. That quick loop is the main advantage of modelling with code: a changed value can update the whole model consistently.
