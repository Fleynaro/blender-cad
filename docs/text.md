# Text

`Text` is a Blender FONT curve and an `Object` subclass. It supports text geometry, per-character material and typography through `t`, curve deformation, shrinkwrapping, and conversion to a mesh `Part` using `.part`.

## Basic Text

Create `Text` directly, configure its curve geometry, then add it to a `BuildPart`:

```python
with BuildPart() as result:
    label = Text("Hello", size=1, loc=Pos(Y=0.5, Z=0.25))
    label.align("CENTER", "CENTER")
    label.extrude(0.1)
    add(label, mode=Mode.JOIN)
```

`Text(text="", size=1.0, max_width=0.0, loc=Location())` defaults to left and bottom-baseline alignment. It has the shared `Object` transform, bounds, scene, and lifecycle API. As a curve-derived object, `fill_mode`, `bevel(...)`, `extrude(...)`, `resolution`, and `.part` are also available.

Use `JOIN` when integrating text with an existing `Part` unless a boolean result is specifically required. `add()` converts the text through its evaluated `.part` representation.

## Styled Fragments With `t`

`t` is a tree of text fragments. Build a string with nested fragments; child styles override parent styles during flattening:

```python
label = Text(
    "Hello " + t("green", mat=mat.green) + " ",
    size=1,
)
label.text += t.b("bold " + t.i("italic", mat=mat.red))
label.text += "!"
```

Available fragment styling:

| API | Effect |
| --- | --- |
| `t(value, mat=..., bold=..., italic=...)` | Creates a fragment from a string, another fragment, or a fragment list. |
| `t.b(value, **kwargs)` | Creates a bold fragment. |
| `t.i(value, **kwargs)` | Creates an italic fragment. |
| `+`, reversed `+`, `+=` | Concatenate strings and fragments into a new or existing fragment tree. |
| `.plain` | Returns the unformatted concatenated string. |
| `.flatten()` | Returns character/style pairs used to rebuild Blender's font formatting. |

Assign `label.text` to a string or `t` tree to rebuild the body and Blender character formatting. Each styled character receives its material index, bold flag, and italic flag from the flattened fragment tree.

## Typography And Fonts

`font_size` changes the Blender font size. `max_width` maps to the first Blender text box width. `spacing_character`, `spacing_word`, `spacing_line`, and `offset` expose Blender's corresponding font-curve settings.

`align(x, y)` forwards alignment values to Blender's font object. The tested direct API uses `LEFT`, `CENTER`, and `RIGHT` horizontally and Blender's vertical values such as `TOP`, `CENTER`, and `BOTTOM_BASELINE`.

`load_fonts(regular, bold=None, italic=None, bold_italic=None)` loads font files and assigns the normal and optional style slots. Font paths must be readable by Blender.

## Paths And Surfaces

`put_on_curve(curve)` clears existing modifiers and adds a Curve modifier. It also records the path as a conversion dependency, so `.part` can evaluate the deformation even when the source curve is not linked into the scene.

```python
path = make_curve(lambda value: (0, value, 0), limit=5)
label.put_on_curve(path)
label.loc = Pos(X=path.length() / 2) * Rot(X=30)
```

`wrap(target, loc=None, mode=WrapMode.NEAREST_SURFACEPOINT, offset=0.0)` clears existing modifiers and adds a Shrinkwrap modifier against a `Part`-like target. The target is likewise tracked for evaluated mesh conversion. These two helpers replace any previous modifiers on the text object.

## ML Integration

The `ml` layout system creates `Text` objects internally for textual leaves. Its tested style fields include font size, bold and italic font styling, character and word spacing, text alignment, wrapping, line height, extrusion, and text stroke geometry. This is layout-level behavior rather than extra `Text` public methods.

The implementation measures text using temporary `Text` objects and a cached differential measurement so normal font side bearings do not inflate content width. See [`blender_cad/ml.py`](../blender_cad/ml.py) and text-focused coverage in [`tests/test_ml.py`](../tests/test_ml.py).

## Tested References

- [`tests/test_text.py`](../tests/test_text.py): nested `t` fragments, per-character materials, bold/italic formatting, path deformation, extrusion, and mesh joining.
- [`tests/test_ml.py`](../tests/test_ml.py): text block layout, horizontal alignment, wrapping, font weight/style, line height, inline layout, strokes, and text extrusion.
- [`tests/test_curve.py`](../tests/test_curve.py): curve construction and conversion used by path-based text.
