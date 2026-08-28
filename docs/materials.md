# Materials

`blender_cad` materials are a code-first alternative to manually wiring Blender shader nodes. A material is an immutable description made of ordered `MaterialLayer` objects. The description is compiled to one Principled BSDF node tree only when it is assigned to geometry or passed to `build_material()`.

Import the public material namespace and blend modes from the package root:

```python
from blender_cad import *
```

Use `mat` as a small material library. It exposes layers, textures, variables, predefined materials, and simple named colors.

## Layered PBR Model

Each layer reads and writes PBR channels such as base color, metallic, roughness, normal, transmission, emission, and alpha. Adding a layer creates an ordered composition: layers on the right build after layers on the left, so they see the PBR state established by earlier layers.

```python
gold = mat.gold
dirty_gold = gold + mat.Dirt(scale=12.0) * 0.5
```

`* factor` wraps a layer in a scoped weight. A layer can use that weight through `ctx.factor`; `ctx.blend()` uses it when no explicit `factor=` is supplied. Built-in procedural overlays such as `Dirt`, `Rust`, `Dust`, and `Cracks` use the scoped factor to control their masks. A weighted layer is not a universal opacity switch: custom layers must deliberately use `ctx.factor` or pass a factor to `ctx.blend()`.

Parentheses create reusable groups of layers. The group factor composes with every nested weight:

```python
weathered_gold = mat.gold + (mat.Rust() + mat.Dirt(scale=8.0)) * 0.35
```

`PBR` supplies explicit channel values. Without `mode`, each supplied channel replaces the current channel value. Give it a `BlendMode` to blend supplied values with the current values instead:

```python
base = mat.PBR(base_color=(0.1, 0.1, 0.8, 1.0), metallic=1.0)
less_metal = base + mat.PBR(metallic=0.7, mode=BlendMode.SUBTRACT)
```

`BlendMode` includes `MIX`, `MULTIPLY`, `ADD`, `SUBTRACT`, contrast modes, and other Blender Mix-node blend modes. Plain Python values are folded before graph creation only for the supported constant-folding modes; expressions and textures compile to Blender nodes.

## Start With Built-Ins

`mat.PBR(...)` is the general base layer. Its optional channels are `base_color`, `metallic`, `roughness`, `specular`, `ior`, `transmission`, `normal`, `emission_color`, `emission_strength`, `alpha`, and `ao`.

The library also provides procedural bases and overlays:

| Layer | Purpose |
| --- | --- |
| `mat.Glass` | Transmission, IOR, color, and roughness for glass. |
| `mat.Metal` | Metallic base with noise-driven roughness. |
| `mat.Wood`, `mat.Concrete`, `mat.Brick`, `mat.Sand` | Procedural surface materials. |
| `mat.Fabric`, `mat.Leather`, `mat.Plastic` | Procedural fabric, leather, and plastic surfaces. |
| `mat.Glow` | Blends emission color and strength. |
| `mat.Dirt`, `mat.Rust`, `mat.Dust`, `mat.Cracks` | Masked procedural overlays that alter several PBR channels. |
| `mat.LaserGrid` | Object-space emissive grid. |
| `mat.DirtImg`, `mat.RustImg` | Image-based overlay layers using their configured textures. |

Ready-to-use definitions include `mat.red`, `mat.green`, `mat.blue`, `mat.yellow`, `mat.iron`, `mat.gold`, `mat.copper`, `mat.wood_oak`, `mat.wood_pine`, `mat.concrete`, `mat.brick_red`, `mat.sand_desert`, `mat.denim`, `mat.leather_brown`, `mat.plastic_glossy_red`, and `mat.plastic_matte_grey`.

Use `mat.Meta(name="...")`, or its alias `mat.Name`, to give a compiled Blender material a stable display name:

```python
brushed_gold = mat.gold + mat.Meta(name="BrushedGold")
```

Material descriptions have stable hashes. `build_material(layer)` reuses an existing Blender material with the same layer hash unless `rebuild=True` is requested. `bpy_material_hash(material)` hashes the generated node graph for tests and diagnostics; it is not a general asset-versioning API.

## Images, Mapping, And Camera Textures

Use `mat.Tex(image_path=...)` for an image descriptor. Pass the texture to a PBR channel or use it in a custom layer; the builder creates and caches the corresponding Image Texture node.

```python
albedo = mat.Tex(image_path="assets/albedo.png")
painted = mat.PBR(base_color=albedo, roughness=0.45)
```

`mat.Mapping` configures texture coordinates and transforms. It accepts a `Transform`, coordinate type, and triplanar-related settings. Mapping is resolved lazily when a texture or procedural expression uses it.

```python
from blender_cad.material import CoordType

grain_mapping = mat.Mapping(coord_type=CoordType.OBJECT, transform=Rot(Z=25))
wood = mat.PBR(base_color=mat.Tex(image_path="assets/wood.png", mapping=grain_mapping))
```

`mat.CameraTex` renders the current scene from a temporary camera defined by a `Location`. It packs the captured image into the Blender file and can be used as an ordinary texture.

`test_material_shoot_projection` in [`tests/test_materials.py`](../tests/test_materials.py) demonstrates the higher-level workflow: build a child part in a private context, call `child.shoot(camera_location)`, then apply the captured texture to a selected face. That test hashes image pixels because the render output is part of the expected material result.

## Variables And Shader Expressions

`mat.Var(name, default)` creates a node-backed value that can be reused in arithmetic and blend factors. It is useful when a material's node tree should expose a named tunable value instead of hard-coding a constant.

```python
glow = mat.Glow(emission_color=(1.0, 0.0, 0.0, 1.0))
adjustable = mat.gold + glow * mat.Var("intensity", 0.5) * 0.2
```

Expression objects form an AST before compilation. Arithmetic such as `value * 0.5`, `value + 0.1`, comparisons, and `value.min(...)` / `value.max(...)` become shader math when an input is node-backed. The build context provides the expression constructors used by built-in layers:

| Context helper | Produces |
| --- | --- |
| `ctx.var(...)` | A float, vector, or color variable node. |
| `ctx.mapping(...)` | Texture-coordinate mapping settings. |
| `ctx.noise(...)`, `ctx.voronoi(...)`, `ctx.wave(...)`, `ctx.brick(...)` | Procedural texture expressions. |
| `ctx.color_ramp(...)`, `ctx.map_range(...)` | Value and color remapping expressions. |
| `ctx.bump(...)` | A normal-vector expression from a height input. |
| `ctx.blend(a, b, factor=..., mode=...)` | A constant result or a Blender Mix-node expression. |

The builder caches expressions by stable key within one material build. Reusing the same expression object or equivalent description avoids duplicate shader subgraphs.

## Create A Custom Layer

Subclass `mat.Layer` and implement `build(self, ctx)`. Read the current channel, construct the procedural mask or expression, and assign the blended result back to that channel. This approach provides precise per-channel control while keeping the material description composable.

```python
from dataclasses import dataclass
from blender_cad.material import CoordType

@dataclass(frozen=True, slots=True)
class Speckles(mat.Layer):
    color: object = (0.08, 0.04, 0.02, 1.0)
    scale: float = 30.0

    def build(self, ctx):
        mapping = ctx.mapping(coord_type=CoordType.OBJECT)
        scale = ctx.var("speckle_scale", self.scale)
        noise = ctx.noise(scale=scale, detail=8.0, mapping=mapping)
        mask = ctx.map_range(noise, from_min=0.65, from_max=0.8)

        # Respect `Speckles() * factor` and preserve the existing PBR channels.
        coverage = ctx.blend(0.0, mask, factor=ctx.factor)
        ctx.channels.base_color = ctx.blend(
            ctx.channels.base_color, self.color, factor=coverage
        )
        ctx.channels.roughness = ctx.blend(
            ctx.channels.roughness, 0.9, factor=coverage
        )

speckled_gold = mat.gold + Speckles() * 0.4
```

The custom-layer and channel-blending patterns are exercised by `test_variable_internal` and `test_base_color_blending` in [`tests/test_materials.py`](../tests/test_materials.py). Follow those patterns rather than mutating Blender nodes directly: the AST is resolved into the final PBR node graph when the material is built.

## Apply Materials To Parts And Faces

Pass `mat=` to `BuildPart` to set the completed part's default material, or assign `part.default_mat` directly. The default is used for faces with no explicit material assignment.

```python
with BuildPart(mat=mat.blue) as result:
    Box(10, 10, 1)
    faces().top().mat = mat.gold + mat.Dirt() * 0.25

result.part.default_mat = mat.green
```

Assign `.mat` to a selected `Face` or `ShapeList` to override only those faces. Changing `part.default_mat` does not overwrite explicit face materials. `test_material_assignment_by_z_groups` and `test_nested_material_logic_and_overrides` in [`tests/test_materials.py`](../tests/test_materials.py) verify default slots, face overrides, nested `BuildPart` behavior, and `set_default_mat(...)`.

See [`docs/part.md`](part.md) for the surrounding `BuildPart` and face-selection APIs.

## Tested References

- [`tests/test_materials.py`](../tests/test_materials.py): layer composition, variables, custom layers, PBR channel modes, default and face-specific assignment, procedural materials, predefined materials, and camera projection.
- [`tests/test_part.py`](../tests/test_part.py): material-inclusive part hashing and broader part behavior.
