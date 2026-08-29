# Rule-Based Layout (RBL)

`blender_cad.rbl` provides `rl`, a rule-based layout API for arranging parts and markup-layout (`ml`) nodes. Rather than encoding every final coordinate as an absolute value, build a hierarchy and declare the relationships that should hold: move toward a point, stay inside a boundary, remain outside another object, follow a curve, face a target, or use a requested size. Call `resolve()` to evaluate those relationships and add the positioned physical parts to the active build.

Import the public alias from the package root:

```python
from blender_cad import *
```

RBL is especially useful when a layout contains dependent objects. A rule can read another node's current position or size through a lambda, so the relationship remains meaningful if the source object changes. This is analogous to web layout: a CSS relationship such as a flex or percentage-based layout is re-evaluated when its container changes instead of preserving fixed screen coordinates. RBL is not a continuously reactive scene system, however: edit the source geometry, initial transform, size, or rule inputs, then call `resolve()` again to construct a fresh solved layout.

## Key Concepts

An RBL tree has two kinds of node:

- A **physical node** has a `part` and can be instantiated into the result.
- A **structural group** contains nodes and can carry rules, transforms, and a coordinate space. A group may also have a physical `part`.

Use `rl.object(part, tag=...)` to wrap one part and `rl.group(*items, part=None, tag=...)` to form a hierarchy. Raw parts passed as children are wrapped automatically. The `|` operator is the declarative composition operator:

```python
node = rl.object(part) | rl.gravity(Pos(X=10, Y=5))
layout = rl.group(node) | rl.transform(rz=45).on_self()
layout.resolve()
```

The first expression attaches a rule to one node. The second attaches a rule to the group itself. `resolve()` compiles rules from the complete hierarchy, initializes their degrees of freedom (DOFs), runs the configured solver, and then adds each physical part once with its computed global transform.

Rules are applied to a scope. The default is `DEEP_PHYSICAL_WITH_SELF`: all descendant physical nodes, plus the node itself when it is physical. The fluent scope methods return a modified rule:

| Method | Targets |
| --- | --- |
| `rule.on_self()` | The node that owns the rule. |
| `rule.on_each(include_self=False)` | Immediate children; include the owner when requested. |
| `rule.on_deep_physical(include_self=False)` | Descendant physical nodes; optionally a physical owner. |
| `rule.on_deep_all(include_self=False)` | All descendants, including structural nodes; optionally the owner. |

Combine rules with `rule_a | rule_b`. The result is a `RuleGroup` and can be applied just like a single rule. Use `rule.with_priority(weight)` when competing soft objectives need a different relative weight.

## Solver Relationship

RBL uses the [`Solver`](solver.md) module as its optimization backend. At initialization it creates DOF parameters for layout nodes. During each solver pass, enabled parameters are registered through `s.param(...)`, transforms are rebuilt, and rules are evaluated. Hard rules write transforms directly; gravity, soft look-at, soft size, and collision rules add objective penalties. The final resolved transforms are then applied to the actual parts.

This gives RBL its declarative behavior. The following layout does not calculate the blue position manually. Its target is evaluated while resolving, from the red node's current resolved position:

```python
with BuildPart() as result:
    red = rl.object(ml(style(width=1, height=1, background_mat=mat.red)))
    blue = rl.object(ml(style(width=1, height=1, background_mat=mat.blue)))

    rl.group(
        red | rl.gravity(Pos(X=10, Y=5)),
        blue | rl.gravity(lambda: Pos(rl.get_position(red)) * Pos(Y=1)),
    ).resolve()
```

If the red target, initial transform, or size-dependent rule changes, a later `resolve()` re-evaluates the lambda and the solver objectives. The outcome can therefore adapt without replacing the relationship with a new absolute blue coordinate. Solver candidates replay the rule graph, so rule traversal and enabled DOFs must remain stable across passes; see [Solver: Parameter Order Is a Contract](solver.md#parameter-order-is-a-contract).

`resolve(solver=Solver(), mode=Mode.ADD, instantiation_delay_sec=0.1)` accepts either a `Solver` or a solver strategy. `mode` is used when final parts are added. The default `Mode.ADD` performs boolean addition; use `Mode.JOIN` for an assembly-style result when appropriate.

## Rules And Transforms

### Initial State And Fixed Transforms

`rl.transform(x=..., y=..., z=..., rx=..., ry=..., rz=..., sx=..., sy=..., sz=..., local=True, init=False)` sets supplied transform fields. Values may be numbers or zero-argument callables. By default it is a hard rule and disables the corresponding DOFs. With `init=True`, it applies only during initialization, making it useful as a solver starting state rather than a continually enforced transform.

```python
with BuildPart() as result:
    (
        ml(style(width=1, height=1, background_mat=mat.red))
        | rl.transform(x=-5, y=-5, rz=45, init=True)
        | rl.gravity(Pos(X=5, Y=5))
    ).resolve()
```

`local=True` updates the target's local transform; `local=False` requests a global transform. Group transforms are inherited by their descendants, so a group rule on `on_self()` can move or rotate an assembled sublayout together.

### Gravity, Orientation, Curves, And Stacks

`rl.gravity(target, pull=None, push=None)` minimizes distance to a position, a node, or a zero-argument position callable. It enables translational DOFs. It pulls by default; `push=True` produces a reciprocal-distance objective that favors separation.

`rl.look_at(target, soft=False)` orients an element toward a position or node. The hard form directly sets rotation. `soft=True` leaves rotational DOFs enabled and makes orientation an objective. `rl.look_along(direction, soft=False)` is the corresponding constant-direction form.

`rl.at_curve(curve)` removes x/y/z translation DOFs and adds a bounded `t` DOF from `0.0` to `1.0`; the element is placed at `curve.at(t)`.

`rl.stack(direction, gap=1.0)` operates on a group itself and positions its deep physical descendants cumulatively along a normalized direction. `gap` can be a number, an `(min, max)` interval optimized as a group DOF, or a callable accepting the zero-based child index. The stack rule takes control of every descendant DOF.

```python
with BuildPart() as result:
    (
        rl.group([
            ml(style(width=1, height=1, background_mat=mat.red))
        ] * 5)
        | rl.stack(-Axis.X, gap=lambda index: index + 1)
        | rl.gravity(Pos(X=5, Y=5)).on_self()
    ).resolve()
```

### Size And Dynamic Dependencies

`rl.size(x=None, y=None, z=None, soft=False, local=True, init=False)` targets positive dimensions by scaling the underlying part. A size can be numeric or a zero-argument callable. A hard size writes scale directly; a soft size adds an equality objective. `init=True` applies size during initialization only.

```python
with BuildPart() as result:
    red = (
        ml(style(width=1, height=1, background_mat=mat.red))
        | rl.size(x=2, init=True)
    )
    blue = (
        ml(style(width=1, height=1, background_mat=mat.blue))
        | rl.transform(y=2)
        | rl.size(x=lambda: rl.get_size(red).x * 2)
    )
    rl.group(red, blue).resolve()
```

`rl.grow(size=1e4)` is a soft request for all dimensions to reach `size`, with a scaled priority. `rl.shrink()` requests soft size `(0, 0, 0)`; since `rl.size` asserts that requested dimensions are positive, this helper cannot complete successfully with the current implementation.

## Degrees Of Freedom

Each node starts with positional DOFs `x`, `y`, and `z` enabled only when it is physical. Rotational (`rx`, `ry`, `rz`) and scaling (`sx`, `sy`, `sz`) DOFs start disabled. Rules may enable, disable, or add DOFs. Configure them explicitly with:

```python
rl.configure_dofs(
    x=True, y=False, z=True,
    rot=True,
    scale=False,
)
```

Individual axis settings are `x`, `y`, `z`, `rx`, `ry`, `rz`, `sx`, `sy`, and `sz`. Shorthand settings are `pos`, `rot`, and `scale`; a supplied shorthand sets all three axes in that category. For example, `rl.configure_dofs(y=False)` locks vertical translation while `rl.gravity(Pos(X=5, Y=5))` can still optimize x.

Rule initialization is ordered by `init_priority`. `configure_dofs` has the highest initialization priority, transform rules run next, and most other rules initialize afterward. A later rule can therefore intentionally adjust a DOF selected earlier.

## Groups, Tags, And Queries

Groups make coordinate spaces and rule application explicit. They can be nested, and the same node can be present through multiple group paths. Global transforms accumulate through parent groups.

Tags select nodes declaratively rather than retaining every Python variable. `rl.object` adds `rl.TAG_OBJECT`; `rl.group` adds `rl.TAG_GROUP`. The available query helpers are:

- `rl.tagged(*tags, root=None)` selects nodes matching the tags.
- `rl.untagged(*tags, root=None)` selects nodes not matching the tags.
- `rl.root(node=None)` starts a query scoped to a root node or query.
- `query.filter(callback)` appends a custom node predicate.

Queries can receive rules with `|`. The following applies different transforms to tagged nodes and limits one search to a nested group:

```python
red_blue = rl.group(
    rl.object(ml(style(background_mat=mat.red)), tag="red"),
    rl.object(ml(style(background_mat=mat.blue)), tag=["blue", "shared"]),
    tag="red_blue",
)

layout = rl.group(
    red_blue,
    rl.object(ml(style(background_mat=mat.green)), tag=["green", "shared"]),
    rl.tagged("red") | rl.transform(x=1),
    rl.root(red_blue).tagged("shared") | rl.transform(x=-1),
)
layout.resolve()
```

## Collision Rules

Collision rules are high-priority solver penalties. They work by transforming vertices of the directed source object into the target's local space and finding each vertex's nearest point on the target BVH. The sign of the nearest-surface normal determines whether a source vertex is outside or inside. They are not Boolean operations and do not alter mesh topology.

Use these constructors:

| Rule | Meaning |
| --- | --- |
| `rl.outside(target=None, margin=0, full_check=False, shell_override=None)` | Require source vertices to remain outside the target, at least `margin` away. Without a target, compare pairs of physical objects in the rule scope. |
| `rl.inside(target=None, margin=0, full_check=False, shell_override=None)` | Require source vertices to remain inside the target, at least `margin` inward. With no target, the current rule-owning element is used. |
| `rl.intersect(target=None, margin=0, full_check=False, shell_override=None)` | Uses the same inward-side penalty as the current `inside` implementation. It does not separately test that two surfaces overlap. |

The collision check is directed by default: it tests vertices of A against the surface of B. This matters when one mesh contains the other, meshes have different vertex density, or only one side's vertices reveal an invalid condition. Set `full_check=True` to also test B against A. When no target is supplied, default checking performs one directed check for each unordered physical pair; `full_check=True` adds the reverse direction.

### Keep Two Parts Apart

Give each part a starting state, attract them, and attach `outside` to one source. The collision objective prevents the red source's vertices from entering the blue target while gravity brings the parts together:

```python
with BuildPart() as result:
    red = rl.object(ml(style(width=1, height=1, background_mat=mat.red)))
    blue = rl.object(ml(style(width=1, height=1, background_mat=mat.blue)))

    rl.group(
        red | rl.transform(x=-5, rz=45, init=True) | rl.gravity(blue) | rl.outside(blue),
        blue | rl.transform(x=5, init=True) | rl.gravity(red),
    ).resolve(mode=Mode.JOIN)
```

### Keep Parts In A Boundary

For `inside`, the target may be a part or a curve. During initialization, the target bounding-box minimum and maximum become the source x/y/z DOF bounds and the source is initialized at that box center. The later vertex/BVH penalty enforces the boundary relationship.

```python
with BuildPart() as result:
    with BuildCurve() as boundary:
        Spline((0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), close=True)

    (
        rl.group(
            ml(style(width=1, height=1, background_mat=mat.red))
            | rl.gravity(Pos(X=10, Y=10))
        )
        | rl.inside(boundary)
    ).resolve(mode=Mode.JOIN)
```

`margin` is a signed-distance buffer: `outside` penalizes a source vertex closer than `margin` to the outside of the target, while `inside` penalizes a source vertex that is not at least `margin` inside. `shell_override` replaces the source collision shell for checks where that approximation is useful. `rl.point()` creates a private point part and is used in the tests as a point-like shell when several objects are constrained inside a curve.

`target_shell_selector` is accepted by the public constructors, but the current `RuntimeContext.get_part_data()` caches only the default part geometry. Do not rely on it to select a target shell in standalone RBL.

## ML Integration

`ml` accepts `rl.Rule` instances and RBL elements directly among its constructor items. Rules become `ml` node rules; RBL elements are collected as global layout elements. While an ML tree is built, it is converted to an RBL tree and evaluated in the ML layout solver. This lets markup nodes declare RBL relationships next to their style and children:

```python
with BuildPart() as result:
    ml(
        style(width=10, height=10, background_mat=mat.blue),
        rl.inside().on_each(),
        ml(
            style(width=1, height=1, background_mat=mat.red),
            rl.gravity(Pos(X=100, Y=0)),
        ),
    ).build()
```

In the ML runtime, RBL rules are forced soft so they participate in ML's layout objective. ML nodes use their evaluation boxes as collision shells: the default is a box, while an `inside` rule targeting the current ML node uses its generated boundary curve. This supports containment, size growth, and gravity-driven placement without emitting final geometry for every candidate pass.

The tested ML pattern is to supply `rl.Rule` instances as constructor items. The constructor also accepts an RBL node as an item and records it as a global layout element. Use a standalone RBL tree and `.resolve()` when the final operation is arranging ordinary parts.

## Limitations And Common Errors

- Call `resolve()` for standalone RBL. Declaring a hierarchy does not add or update scene geometry by itself.
- Treat the result as an optimization. Competing soft objectives use weighted penalties and may settle at a compromise rather than satisfy every target exactly.
- Keep solver structure stable across passes: do not make rule targets, enabled DOFs, or rule traversal conditional on candidate-only state.
- A callable position or size is invoked during evaluation. It must return a supported position/vector or scalar respectively, and it must not create an unstable dependency cycle.
- `rl.get_position`, `rl.get_size`, `rl.get_transform`, `rl.current_rule`, `rl.current_rule_element`, and `rl.current_target_element` require an active RBL runtime context. Calling them outside rule evaluation raises `RuntimeError`.
- `rl.get_size` requires a physical element; a structural group without a part has no part data to size.
- Collision checks use source vertices against a target BVH. Use `full_check=True` when the directed default is not sufficient for the shapes being constrained.
- `rl.inside()` without a target is meaningful only when attached to an element that can serve as the current rule element. It defaults its target to that owning element.
- The standalone resolver includes a default `0.1` second instantiation delay before adding final parts.

## Tested References

- [`tests/test_rbl.py`](../tests/test_rbl.py): gravity, transforms, rule groups, priorities, scopes, dynamic lambdas, size, collisions, curve constraints, orientation, stacks, DOFs, and tags.
- [`tests/test_ml.py`](../tests/test_ml.py): RBL rules embedded in ML nodes, containment, growth, and query-based ML layout rules.
- [`docs/solver.md`](solver.md): optimization passes, parameter ordering, objectives, solver strategies, and RBL's solver integration.
