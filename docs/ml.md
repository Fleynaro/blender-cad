# Markup Layout (ML)

`blender_cad.ml` provides the public `ml` node and `style` alias (`MLStyle`). It is a declarative, HTML/CSS-inspired way to construct primarily 2D layouts and then turn them into Blender mesh geometry. A node is analogous to a markup element: it accepts child nodes, text, existing parts, layout rules, and one or more style objects.

ML is most useful for structured panels, architectural facades, sci-fi walls, control consoles, labels, dashboards, and other designs whose main complexity is spatial arrangement. It is not a general scene graph or a replacement for arbitrary 3D modeling: layout happens in the local XY plane first, then styles such as `extrude`, `bevel`, `bend`, and transforms produce or modify the final mesh.

Import the public API from the package root:

```python
from blender_cad import *
```

## Basic Model

`ml(*items, **attrs)` accepts a flattened mix of children and `style(...)` values. Styles merge from left to right, so a later value overrides an earlier one. Strings become text nodes. A `Part`, `BuildPart`, `BaseCurve`, `BuildCurve`, or `Joint` becomes a layout-aware part node. Child nodes have exactly one parent.

```python
with BuildPart() as result:
    panel = ml(
        style(width=8, height=4, padding=0.5, background_mat=mat.blue),
        ml(style(width=2, height=1, background_mat=mat.red)),
        "Status",
    )
    panel.build()
```

Use `node.build()` to add the resulting geometry to the active `BuildPart`. `node.part` and `node.to_part(width=..., height=...)` build a private reusable `Part` instead. The geometry follows the active `BuildPart` mode unless a node style changes it. See [`Part And BuildPart`](part.md) for context and boolean modes.

ML supports component-style composition: ordinary Python functions can return an `ml` tree and callers can nest or repeat the component. The tree is declarative, so changing a parent size causes percentage sizes, standard flow, and flex layout to be recalculated for its children on the next build.

```python
def indicator(color):
    return ml(style.circle(0.25, mat=color), style(extrude=0.1))

with BuildPart() as result:
    ml(
        style(width=6, height=2, background_mat=mat.blue),
        style.flex_center(gap=0.5),
        indicator(mat.red),
        indicator(mat.yellow),
        indicator(mat.green),
    ).build()
```

## Layout And Sizing

Lengths are local units multiplied by `unit_scale`. Numbers and numeric strings are local units; percentage strings resolve against the relevant parent dimension. `width`, `height`, min/max dimensions, `aspect_ratio`, padding, margin, border width, gap, positions, and many other fields accept the same length form where applicable.

The box model supports CSS-like shorthand:

| Value | Meaning for `padding` and `margin` |
| --- | --- |
| one value | all sides |
| two values | top/bottom, left/right |
| four values | top, right, bottom, left |

`padding_tb`, `padding_lr`, `margin_tb`, and `margin_lr` override the shorthand. The per-side fields, such as `padding_left` and `margin_bottom`, override those values. `border_in_measure` controls whether the border contributes to the measured size.

Nodes normally participate in standard inline flow. `align` controls horizontal content alignment (`left`, `center`, `right`, or `justify`) and `align_y` controls vertical placement (`start`, `center`, or `end`). `position="absolute"` removes a child from normal flow; place it using `left`, `right`, `top`, `bottom`, and `anchor_x`/`anchor_y`. `position="relative"` keeps the flow position while applying those directional offsets. `display="none"` omits the node from layout and rendering.

`ml.new_line()` inserts an explicit line break in standard flow. `style.square`, `style.circle`, `style.absolute_center`, `style.flex_center`, `style.row`, and `style.column` are concise presets.

The sizing, percentage, alignment, positioning, and preset behavior is covered by `test_basic_block_sizing_and_padding`, `test_min_max_sizes_and_aspect_ratio`, `test_relative_and_absolute_positioning`, `test_relative_position_offsets`, `test_percentage_overflow_and_alignment`, and `test_preset_shapes_with_absolute_centering` in [`tests/test_ml.py`](../tests/test_ml.py).

### Flex Layout

Set `display="flex"` to use row or column layout. The supported controls are `flex_direction`, `flex_wrap`, `gap`, `justify_content`, `align_items`, `align_content`, `flex_grow`, and `flex_shrink`. The engine lays out flex lines, shrinks overflowing items down to their minimum size where possible, distributes free main-axis space to growable items, then applies the requested alignment.

```python
ml(
    style(
        width=12,
        height=6,
        padding=1,
        display="flex",
        flex_direction="row",
        flex_wrap="wrap",
        gap=0.5,
        justify_content="space-between",
        align_items="center",
        align_content="space-around",
        background_mat=mat.blue,
    ),
    *[ml(style(width=1, height=1, background_mat=mat.red)) for _ in range(6)],
)
```

See `TestMLFlexFlow` in [`tests/test_ml.py`](../tests/test_ml.py) for row and column flow, wrapping, grow/shrink, gap, and cross-axis alignment.

## Text And Inline Content

Strings are automatically laid out as text. Standard flow can mix text with atomic inline boxes. `ml.b(...)` and `ml.i(...)` create bold and italic inline wrappers. A wrapper with no box-affecting style is transparent to the surrounding text flow; a wrapper with a box style, such as a background or padding, is measured as one inline item.

Text styles include `font_size`, `font_weight`, `font_style`, `line_height`, `letter_spacing`, `word_spacing`, `text_align`, `white_space`, and `wrap_mode`. Supported wrapping modes are `word`, `character`, `anywhere`, and `none`; `white_space="nowrap"` and `white_space="pre"` disable width-based wrapping. Explicit newline characters remain line breaks. Text measurement uses Blender text geometry, so wrapping adapts to the available layout width without manually placing glyphs.

`mat` colors text; `text_extrude` gives text depth. `text_stroke_width`, `text_stroke_mat`, `text_stroke_opacity`, `text_stroke_extrude`, and `text_stroke_samples` create an outlined text effect.

The tested behavior is documented by `TestMLStandardFlowText`, especially `test_text_alignment_center_and_wrap_character`, `test_inline_bold_italic_and_nested_background`, `test_text_stroke_rendering`, and `test_overflow_clipping_behavior` in [`tests/test_ml.py`](../tests/test_ml.py).

## Paint, Boundaries, And Clipping

`background_mat` fills a node's 2D outline; `border_mat`, `border_width`, `border_style`, `border_offset`, and `border_extrude` create a border. Border styles are `solid`, `dashed`, `dotted`, `double`, and `none`. `overflow="hidden"` clips children to the node outline; `overflow="hidden-border"` includes the border extent in that clipping boundary.

Rounded corners support `border_radius`, side-specific radii, and corner-specific radii. The radius normalization prevents adjacent radii from exceeding the box size. Unlike CSS, negative radii are supported: they produce concave, inward corner arcs. `top_scale`, `right_scale`, `bottom_scale`, and `left_scale` visually warp the outline while retaining the original rectangular layout measurement.

`background_from_curve` uses a closed curve as the background outline and derives unspecified width and height from it. This permits non-rectangular panels while retaining the same child layout model.

```python
with BuildCurve() as boundary:
    Spline((0, 0, 0), (6, 0, 0), (6, 4, 0), (0, 4, 0), close=True)

ml(
    style(
        background_from_curve=boundary,
        background_mat=mat.blue,
        border_width=0.1,
        border_mat=mat.yellow,
        border_radius="10%",
    ),
).build()
```

The border, curve background, clipping, concave radii, and visual outline behavior is verified by `TestMLBorder`, `test_circular_primitive_overflow_clipping`, `test_background_generation_from_curve_boundary`, and `test_top_scale_with_border_warp` in [`tests/test_ml.py`](../tests/test_ml.py).

### Border Layouts

`style.border_ml(...)` builds a second ML layout along a border curve. Use `side="left"`, `"right"`, `"top"`, or `"bottom"` for a rectangular outline, or provide `selector=lambda: curve_selection` when the background comes from a tagged curve. The nested layout receives the selected curve's length as its width and is positioned using the curve's location. It can use a separate solver and objective through `layout_solver` and `layout_objective`.

```python
style.border_ml(
    style(display="flex", justify_content="space-around"),
    *[ml(style(width=0.25, height=0.25, background_mat=mat.red)) for _ in range(4)],
    side="top",
)
```

`test_background_curve_tag_selection_and_border_mapping` demonstrates tagged curve selections for individual border segments. `test_border_element_layout_solver_isolated_execution` verifies that this child layout is solved separately after the parent curve layout is available.

## 3D Operations And Reusable Geometry

Layout is 2D, but emitted geometry can become 3D:

| Style or helper | Effect |
| --- | --- |
| `extrude` / `text_extrude` | Give the background/border or text depth. |
| negative extrusion | Queues a cutter that is subtracted from its parent after child emission. |
| `extrude_transform` / `border_extrude_transform` | Transform the extrusion profile. |
| `bevel` and `style.bevel_box_top(...)` | Apply bevel operations to emitted faces. |
| `bend_angle`, `bend_direction`, `bend_segments` | Bend the emitted node geometry. |
| `style.prop_box_extrude(...)` | Apply directional proportional editing to background extrusion. |
| `ml.hole(...)` | Convenience node for a negative-extrusion cutout. |
| `mode="add"` | Use boolean union instead of joining the node geometry. |

Negative extrusion is not merely a negative Z offset. ML creates oversized cutter geometry, carries it up the node tree, and applies it as a fast Boolean subtraction after the parent content has been emitted. This is why nested cutouts and engraved text can pierce an already-built parent surface.

Pass a completed `Part` or use `ml.from_part(part, style(...))` to place existing mesh geometry in a layout. It is measured from its bounding box, so it occupies flow space rather than overlapping following nodes by default. `adapt_transform=True` adapts an embedded part to ML's coordinate and unit conventions.

`ml.mirror(factory, side=..., flip=True, offset=...)` produces two component instances and flips the second across the relevant local axis with negative scale. Equivalent render geometry can be reused by ML's cache.

`ml.joint(...)` registers a named joint at an absolute layout position. The result also receives standard X/Y bounding-box joints when `build(register_chain_joints=True)` is left enabled.

See `TestMLThreeDOperations` in [`tests/test_ml.py`](../tests/test_ml.py), including `test_negative_extrusion_boolean_subtraction`, `test_text_negative_extrusion_engraving`, `test_part_3d_integration_and_transformation`, `test_nested_mirroring_and_reflection_scales`, `test_bend_cylindrical_deformation`, and `test_joint_and_attachment`.

## Generation And Caching

`ml.generate(factory)` expands children dynamically during layout. `ml.generate_array(factory, fill_mode="box" | "line")` repeatedly calls a factory until the parent overflows:

- `fill_mode="box"` fills the available layout area.
- `fill_mode="line"` stops when generation would add a second flow line.

The implementation first grows the candidate count and then uses binary search to find the largest fitting count. Factories may accept no argument, an index, or `(index, generator_node)`. Generated nodes are retained by index while the count converges, so a factory is not needlessly recreated for every pass.

Rendering also has a per-build `part_cache`. ML caches equivalent final node geometry, including material-sensitive rendering styles and callbacks, after layout is resolved. Layout-only differences such as a node's flow position are applied outside that cached mesh, allowing repeated components and mirrored copies to avoid duplicate mesh construction. The cache is scoped to one `MLBuildContext`; it is not a persisted asset cache.

`test_generate_array_fill_mode_box`, `test_generate_array_fill_mode_line`, `test_generate_array_nested_matrix`, `test_generate_array_dynamic_incremental_sizing`, and `test_layout_engine_geometry_caching` cover these behaviors.

## Rules, Evaluation, And Tags

ML accepts `rl.Rule` values alongside children and styles. During layout it builds an RBL tree and evaluates the rules against lightweight `BoxSetPart` evaluation geometry. This lets rules optimize position, size, or containment before expensive mesh generation. With `rl.inside()` targeting an ML parent, the parent uses its evaluated outline curve as the collision boundary, including curve-backed and concave shapes.

Use `ml.dof(...)` for a solver-controlled numeric style value, `ml.dof_p(...)` for a percentage value, or `ml.dof_get(...)` inside a style callback. The layout must retain stable property evaluation order between solver passes.

`show_eval_box=True` or a material layer displays the evaluation boxes and boundary curves for inspection. `build(evaluate=True)` emits only the lightweight evaluation representation.

`tag` is inherited by descendants. `root_tag` applies only to the current node. After emission, use the selector API to find tagged mesh geometry; see [`Selectors`](selectors.md). RBL tags and ML tags cooperate when rules are compiled from the ML tree.

The RBL integration is covered by `TestRuleBasedLayout` in [`tests/test_ml.py`](../tests/test_ml.py), especially `test_inside_rule_with_curve_boundary_and_evaluation`, `test_tag_propagation_and_chained_rule_selectors`, `test_size_rule_expansion_in_concave_geometry`, and `test_dof_in_transform_callback`. See [`Rule-Based Layout`](rbl.md) and [`Solver`](solver.md) for the standalone APIs.

## Build Phases And Limits

`build()` deliberately separates layout from mesh construction:

1. The layout phase resolves inherited styles, expands generators, measures standard or flex flow, stabilizes dynamic values, creates evaluation boxes, and runs the configured RBL solver.
2. The emit phase creates final backgrounds, text, borders, embedded parts, clipping, Boolean cutters, materials, transforms, and cache entries.

Styles may be callables accepting no argument or the current `ml` node. Because a callback can depend on another node's dimensions, ML repeats layout passes until its style-and-box snapshot no longer changes. A dependency cycle raises `RuntimeError` after the pass limit instead of producing unstable geometry.

Do not treat ML as a continuously reactive scene system. Rebuild the root tree after changing an input. Keep dynamic style callbacks and generator factories deterministic, avoid cyclic dimension dependencies, and use a standalone RBL tree when the task is arranging arbitrary parts rather than an ML layout.

## Tested References

- [`tests/test_ml.py`](../tests/test_ml.py): box model, text, flex, borders, generators, 3D operations, parts, tags, and RBL integration.
- [`docs/rbl.md`](rbl.md): rule semantics and ML evaluation-box integration.
- [`docs/part.md`](part.md): `Part`, `BuildPart`, Boolean modes, and `BoxSetPart`.
- [`docs/selectors.md`](selectors.md): selecting the tagged geometry produced by ML.
