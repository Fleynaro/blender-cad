# 🧊 blender_cad

**The power of declarative CAD, the flexibility of polygons.**

`blender_cad` is a Python framework for procedural polygonal modeling inside Blender. Inspired by the philosophy of `build123d`, it brings a code-first, non-destructive workflow to the world of meshes.

Whether you are generating complex game levels, modular environment assets, or procedural hard-surface props, `blender_cad` allows you to describe your geometry with code while leveraging Blender's powerful rendering and material system.

---

## 🚀 Why blender_cad?

While traditional CAD (like `build123d` or `CadQuery`) excels at BREP geometry for engineering, it often struggles with the specific needs of game development: UV maps, vertex colors, and efficient mesh topologies.

**blender_cad bridges this gap:**

- **Context-Driven Modeling:** Use `with BuildPart():` blocks to manage state and geometry cleanly.
- **Polygonal Logic:** Native support for meshes, meaning you get lightweight geometry ready for game engines.
- **Advanced Selectors:** Filter and sort faces, edges, and vertices by geometry type (`GeomType.SPHERE`, `GeomType.CONE`, etc.), location, or custom logic.
- **First-Class Materials:** Unlike standard CAD, materials are integrated into the core workflow. Assign textures and colors dynamically as you build.
- **Blender Native:** It lives inside Blender. No export/import loops. Immediate visual feedback.

---

## 🛠 Features

- **Modular Locations:** Complex transformations using `Pos`, `Rot`, and distribution patterns like `GridLocations` or `PolarLocations`.
- **Surface Attachment:** Attach objects directly to the surface of other objects using UV-coordinates or normals.
- **Boolean & Modifiers:** Native `JOIN`, `CUT`, and `INTERSECT` modes, plus procedural `bevel` and other mesh-specific operations.
- **Smart Selection:** Use the `~` operator for inverse selection or chain filters to find exactly the face you need.
- **Hot Reload:** Built-in support for instant code updates without restarting Blender.

---

## 📖 Quick Start

Here is a glimpse of how you can build complex, material-aware geometry in just a few lines of code:

```python
from blender_cad import *

# Create a part with a default material
with BuildPart(mat=Material(color=(0.1, 0.1, 0.1))) as example:
    # 1. Base Geometry
    Box(10, 10, 1)

    # 2. Advanced Selection & Material Override
    # Group faces by Z-axis: [Bottom, Sides, Top]
    z_groups = faces().group_by(Axis.Z)
    z_groups[2].mat = Material(color=(1, 0, 0)) # Set Top face to Red

    # 3. Procedural Distribution
    with Locations(Pos(Z=1)):
        with GridLocations(x_spacing=3, y_spacing=3, x_count=3, y_count=3):
            # Each instance can have its own local transform
            with Locations(Rot(Y=45)):
                Cylinder(radius=0.5, height=2)

    # 4. Inverse Selection
    # Find all faces that are NOT part of the cylinders (the base box)
    base_faces = ~faces().filter_by(GeomType.CYLINDER)
```

---

## 🕹 Game Level Generation Example

`blender_cad` is particularly powerful for generating modular environments. You can define "Rules" for your rooms and let the script populate the scene:

```python
with BuildPart() as level:
    # Generate a floor
    Box(20, 20, 0.5)

    # Attach a cone to a sphere's surface location
    Sphere(radius=2)
    surface_loc = faces().filter_by(GeomType.SPHERE)[0].at(0.3, 0.3)

    with Locations(surface_loc):
        Cone(radius_bottom=1, radius_top=0, height=3)
```

---

## 🧪 Testing & Reliability

The library is built with stability in mind. We use a comprehensive unit testing suite that verifies:

- **Geometry Integrity:** Hashing mesh data to ensure consistent generation.
- **Material Slots:** Verifying correct material indices and inheritance.
- **Topology Graphs:** Ensuring the relationships between vertices, edges, and faces remain valid after complex operations.

---

## 📝 License

`blender_cad` is available under the MIT License. Feel free to use it in your commercial or personal projects.

_Built for artists who code and coders who create._
