# Solver

`blender_cad.solver` turns a declarative model into an optimization problem. Instead of calculating coordinates or dimensions procedurally, declare adjustable values with `s.param(...)` and declare the desired relationships with `s.aim_*()` calls. The solver searches for parameter values that minimize the accumulated error, then executes the block once more with the best values.

The public exports are `Solver`, `sm`, and `solver`:

```python
from blender_cad import Solver, sm, solver
```

The module requires `scipy` in Blender's embedded Python runtime. It imports `scipy.optimize` directly, so installing SciPy only in the system Python used to launch Blender is insufficient.

## Basic Session

Use a solver as the iterator in a `for` loop. The loop body is evaluated several times by design. Only act on the solved result during the final pass.

```python
answer = None

for s in Solver(sm.nelder_mead()):
    x = s.param(0.0)
    s.aim_equal(x * x, 25.0)

    if s.is_final:
        answer = x
```

`answer` is approximately `5.0` or `-5.0`. `test_single_parameter_optimization` in [`tests/test_solver.py`](../tests/test_solver.py) verifies this session shape.

`sm` is a strategy factory. Passing a strategy is optional: `Solver()` uses `sm.auto()`, which tries Nelder-Mead, SLSQP, and L-BFGS-B in order. A strategy can also be iterated directly because `sm.nelder_mead()` creates a default `Solver` when used in a `for` loop.

## Execution Model

One loop has three phases:

1. **Registration pass.** The first yielded session has `s.is_init == True`. Each `s.param()` records its initial flattened values, bounds, reconstruction function, and offset in the optimizer vector. Aim and mode calls contribute structural information for the cache key, but do not calculate an objective error.
2. **Candidate passes.** SciPy runs in a background thread. Each time its objective callback needs an error for a candidate vector, it sends that vector to the main thread. The `for` loop receives it as a regular session, rebuilds parameters, and returns the sum of `s.aim*()` errors through a queue. This thread bridge preserves simple declarative syntax while allowing the SciPy callback to wait for the loop body.
3. **Final pass.** After the strategy returns, the loop is yielded once with `s.is_final == True` and the best parameter vector. Build final geometry, capture outputs, or perform other final-only effects here.

The loop body must be safe to execute many times. Do not append persistent results, perform irreversible operations, or rely on the candidate evaluation count outside an `if s.is_final:` guard.

### Parameter Order Is a Contract

Parameters are positional, like hook calls in React. Their definition order and structure during every candidate and final pass must exactly match the registration pass. Do not conditionally add, remove, reorder, or change the shape of `s.param()` calls between passes. The solver reconstructs values from the offsets recorded in the first pass; changing the order associates a slice of the optimizer vector with the wrong parameter.

Loops are supported when their iteration count and order are stable. `test_dynamic_param_calls_in_loop` in [`tests/test_solver.py`](../tests/test_solver.py) registers one parameter per element in a fixed list.

Nested solvers are supported. `solver()` always returns the most deeply active session, so an inner `Solver` can optimize a value derived from an outer parameter. See `test_nested_solver_optimization` in [`tests/test_solver.py`](../tests/test_solver.py).

## Parameters And Bounds

`s.param(init_value, min=None, max=None, steps=10, step=None)` returns `init_value` during registration and a reconstructed candidate value thereafter.

Supported values are:

- `int` and `float`
- `Vector` and the library's value objects that provide `values`, `bounds`, and `copy`, including positions, rotations, and transforms
- nested `list` and `tuple` structures containing supported values

Bounds may be scalars, matching supported structures, or `None`. Scalar bounds broadcast over each degree of freedom in a complex value. A nested `(value, (min, max))` tuple lets individual list elements have distinct bounds.

```python
for s in Solver():
    location = s.param(
        Pos(1, 1, 1),
        min=Pos(0, 0, -100),
        max=Pos(5, 5, 100),
    )
    s.aim_equal(location, Pos(10, 10, 10))
```

The solver keeps `x` and `y` bounded at `5` while the free `z` coordinate reaches `10`; this is covered by `test_pos_with_bounds`.

`steps` controls the number of grid segments used by `sm.brute()`. `step` is a convenience form that computes that count from finite scalar `min` and `max` bounds. It does not make a parameter intrinsically integer: round or otherwise transform the returned value in the declarative model when discrete behavior is required. `test_single_integer_parameter_optimization` and `test_brute_mixed_int_float` demonstrate this pattern.

## Objectives And Constraints

Every candidate starts with zero error. Calls to `s.aim(error)` add their errors, so several aims form one summed objective.

- `s.aim(error)` adds a numeric error directly.
- `s.aim_equal(a, b, k=1.0)` minimizes squared L2 distance to equality.
- `s.aim_dist(a, b, dist, k=1.0)` minimizes squared error from the requested distance. It supports scalar values, nested sequences, mathutils vectors/eulers/quaternions, and `Transform` values. Transform rotation differences use the shortest angular distance.
- `s.constraint(condition)` adds a large penalty when the Boolean condition is false. It is a penalty, not a SciPy constrained-optimization object.

Use weights (`k`) or explicit errors to balance goals with different units. Tests for multi-objective solving, bounds, constraints, vectors, nested lists, locations, and rotations are in [`tests/test_solver.py`](../tests/test_solver.py).

## Geometry During Optimization

Optimizing a geometry-producing component can be expensive because every candidate replays the model. `s.mode()` returns `Mode.PRIVATE` during registration and candidate passes, then returns its requested mode (default `Mode.ADD`) only on the final pass. Pass it to operations that accept a boolean mode so intermediate candidate geometry stays private:

```python
for s in Solver(sm.brute().with_polish(), max_steps=20):
    count = s.param(1, min=5, max=12, step=1.0)
    turn_angle = s.param(0, min=0, max=40, steps=5)
    staircase = StaircaseBuilder(step, int(count), turn_angle)
    staircase.j_bottom().to(start, mode=s.mode())
    s.aim_equal(staircase.j_top().loc.position, end.position)
```

The complete staircase example is [`test_staircase_solver_alignment`](../tests/test_components.py). It uses parameters rather than manually chosen endpoint coordinates and only commits `ADD` operations for the final solution.

`solver()` is useful inside helpers or components that should participate in the active session without accepting `s` explicitly. Outside an active solver it returns a dummy registration/final session; this is intended for declarative helper code that needs a session-shaped object, not as a substitute for an active optimization pass.

## Cache

`Solver` has a process-local class cache. After a successful final pass, it stores the best flattened vector under a key assembled during registration from parameter definitions and declarative objective structure, plus the initial objective error. Re-running the same solver structure first evaluates that cached vector. If it is better than the new initial vector and already satisfies `tol`, the solver returns it without another strategy search.

The cache is an optimization, not persistent storage or a guarantee that no candidate passes occur. The cached vector is still evaluated before it can be accepted, and cache contents are lost when the Blender Python process restarts. `test_solver_caching` verifies that the second identical run uses fewer passes.

Keep all problem-defining inputs represented by the registered parameters or the same declarative calls. External mutable state that changes an objective without changing its registration data can make a cached starting point less useful.

## Layout Integrations

The solver is also the optimization backend for rule-based and markup layouts. These integrations retain the same positional parameter contract, so changing traversal order, enabled degrees of freedom, or style-resolution order between passes is unsafe.

`rl.resolve(...)` accepts either a `Solver` or a strategy. It initializes and orders rule bindings, registers the enabled degrees of freedom for layout nodes, evaluates soft rules through the session objectives, and instantiates physical parts after the solve. Hard rules apply transforms directly; soft collision, gravity, look-at, and size rules contribute optimization error. See [`tests/test_rbl.py`](../tests/test_rbl.py) for scopes, priorities, degree-of-freedom locking, collisions, and solver interactions.

`ml` uses a solver loop for layout degrees of freedom and defaults to Nelder-Mead. Its degree-of-freedom lookup includes a call index in its identity, so a stable style-field evaluation order is required for the same reason as stable `s.param()` order. Markup border sub-builders can use separate solver and objective settings. The relevant layout integration coverage is in [`tests/test_ml.py`](../tests/test_ml.py).

## Strategies

`sm` provides these strategy builders:

- `sm.nelder_mead(...)`: derivative-free local simplex search; useful for discontinuous or geometry-heavy objectives.
- `sm.slsqp(...)`: bounded local optimization.
- `sm.l_bfgs_b(...)`: bounded local optimization suited to smooth, larger parameter spaces.
- `sm.brute()`: finite bounded grid search; use `steps` or `step` to control the grid.
- `sm.stochastic(...)`: differential-evolution global search.
- `sm.shgo(...)`: SHGO global search with a local SLSQP refiner by default.
- `sm.auto()`: ordered selection of the three local strategies listed above.
- `strategy.with_polish(...)`: pipeline a global or grid strategy into a local refiner; default refiner is SLSQP.
- `sm.pipeline(...)`: feed each strategy the preceding strategy's result.
- `sm.selector(...)`: try each candidate from the same original starting vector and retain the best result.
- `strategy.repeat(n)`: run a continuation pipeline `n` times. Within it, `s.homotopy_parameter` progresses from `0.0` to `1.0` and can progressively tighten a penalty or constraint.

`test_homotopy_constraint_relaxation_with_tracking` demonstrates `repeat()` and `s.homotopy_parameter`.

## Tested References

- [`tests/test_solver.py`](../tests/test_solver.py): strategies, parameter types, bounds, cache, nested sessions, constraints, and homotopy.
- [`tests/test_components.py`](../tests/test_components.py): declarative staircase construction and `s.mode()` for geometry operations.
- [`tests/test_rbl.py`](../tests/test_rbl.py): rule-based-layout interactions that rely on solver-backed degrees of freedom.
