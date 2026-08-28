from abc import ABC, abstractmethod
import builtins
from dataclasses import dataclass, replace
import math
import threading
from contextvars import ContextVar
from queue import Queue
from typing import Any, Callable, Dict, List, Literal, Tuple, Optional, TypeVar, Union
import numpy as np
from scipy.optimize import minimize, differential_evolution, brute, shgo
from mathutils import Vector, Euler, Quaternion

from .location import SVector, Transform
from .build_part import Mode

SolverLike = Union['OptimizationStrategy', 'Solver']

# The optimizer runs outside the loop's thread; this context variable lets strategy
# wrappers report their active stack back to the corresponding Solver instance.
_current_running_solver: ContextVar[Optional['Solver']] = ContextVar('_current_running_solver', default=None)

@dataclass
class SolverInput:
    """Encapsulates all parameters required for an optimization execution."""
    objective: Callable[[np.ndarray], float]
    x0: np.ndarray
    bounds: List[Tuple[float, float]]
    steps: List[int]
    tol: float
    max_steps: Optional[int]

@dataclass
class SolverResult:
    """Standardized result container for all solver strategies."""
    x: np.ndarray
    fun: float
    success: bool
    message: str

class OptimizationStrategy(ABC):
    """
    Base class for all optimization building blocks.
    """

    class _Context:
        """Context manager to notify the main thread when a strategy starts and finishes execution."""
        def __init__(self, strategy: 'OptimizationStrategy'):
            self.strategy = strategy

        def __enter__(self):
            srv = _current_running_solver.get()
            if srv is not None:
                srv._from_solver.put(("push", self.strategy))
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            srv = _current_running_solver.get()
            if srv is not None:
                srv._from_solver.put(("pop",))

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def _solve(self, input: SolverInput) -> SolverResult:
        """Execute the optimization logic using provided inputs."""
        pass

    def to_solver(self, max_steps: int | None = 200, tol: float = 1e-4, debug: bool = False) -> 'Solver':
        """Convenience method to create a Solver instance with this strategy."""
        return Solver(self, max_steps, tol, debug)
    
    def __iter__(self):
        return self.to_solver().__iter__()

    def with_polish(self, method: Optional['OptimizationStrategy'] = None) -> 'SolverPipeline':
        """
        Syntactic sugar to create a pipeline that refines the current strategy 
        with a local minimizer.
        """
        # Default to SLSQP if no polishing method provided
        refiner = method if method is not None else ScipyMinimize(method="SLSQP")
        return sm.pipeline(self, refiner)
    
    def repeat(self, n: int, name: Optional[str] = None) -> 'SolverPipeline':
        """Repeats the specified method N times using a pipeline structure for continuation methods."""
        return sm.repeat(self, n, name)

class ScipyMinimize(OptimizationStrategy):
    """Wrapper for scipy.optimize.minimize (Gradient and Simplex methods)."""
    def __init__(self, method: str, tol=None, max_steps=None, xatol=None, fatol=None, eps=1e-6, name: Optional[str] = None):
        super().__init__(name or method)
        self.method = method
        self.override_tol = tol
        self.override_max_steps = max_steps
        self.override_xatol = xatol
        self.override_fatol = fatol
        self.eps = eps

    def _get_minimize_params(self, input_context: SolverInput) -> Dict[str, Any]:
        """
        Returns a dictionary of arguments compatible with scipy.optimize.minimize
        or shgo's minimize_kwargs.
        """
        tol = self.override_tol or input_context.tol
        max_steps = self.override_max_steps or input_context.max_steps
        xatol = self.override_xatol or tol
        fatol = self.override_fatol or tol
        
        options = {'maxiter': max_steps}
        if self.method == "Nelder-Mead":
            options.update({'xatol': xatol, 'fatol': fatol})
        if self.eps is not None:
            options.update({'eps': self.eps})
            
        return {
            'method': self.method,
            'tol': tol,
            'options': options
        }

    def _solve(self, input: SolverInput) -> SolverResult:
        with self._Context(self):
            params = self._get_minimize_params(input)
            res = minimize(
                input.objective, 
                x0=input.x0, 
                bounds=input.bounds,
                **params
            )
            return SolverResult(x=res.x, fun=res.fun, success=res.success, message=res.message)

class DiffEvoSolver(OptimizationStrategy):
    """Wrapper for scipy.optimize.differential_evolution (Global optimization)."""
    def __init__(self, popsize: int, mutation: Tuple[float, float], recombination: float, name: Optional[str] = None):
        super().__init__(name or "DifferentialEvolution")
        self.popsize = popsize
        self.mutation = mutation
        self.recombination = recombination

    def _solve(self, input: SolverInput) -> SolverResult:
        with self._Context(self):
            res = differential_evolution(
                input.objective,
                bounds=input.bounds,
                strategy='best1bin',
                maxiter=input.max_steps,
                popsize=self.popsize,
                tol=input.tol,
                mutation=self.mutation,
                recombination=self.recombination,
                init='latinhypercube',
                polish=False # Disable polishing as it is handled by the pipeline
            )
            return SolverResult(x=res.x, fun=res.fun, success=res.success, message=res.message)

class BruteSolver(OptimizationStrategy):
    """Wrapper for scipy.optimize.brute (Grid search)."""
    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "BruteForce")

    def _solve(self, input: SolverInput) -> SolverResult:
        with self._Context(self):
            # Create slices representing the grid ranges: (start, stop, step)
            # We divide the interval into 'grid_points' segments
            ranges = []
            for i, (b, s) in enumerate(zip(input.bounds, input.steps)):
                if b[0] is None or b[1] is None:
                    raise ValueError(f"BruteSolver requires finite bounds. Index {i} is {b}")
                # Create slices using the specific 'steps' for this parameter
                ranges.append(slice(b[0], b[1], (b[1] - b[0]) / max(1, s)))

            # full_output=True returns (x0, fval, grid, Jout)
            # finish=None prevents brute from running a local minimizer (use .polish() for that)
            best_x, best_fun, _, _ = brute(
                input.objective,
                ranges,
                full_output=True,
                finish=None
            )
            if np.isscalar(best_x):
                best_x = np.array([best_x])
            return SolverResult(x=best_x, fun=best_fun, success=True, message="Brute force completed")
    
class SHGOSolver(OptimizationStrategy):
    """
    Wrapper for scipy.optimize.shgo (Simplicial Homology Global Optimization).
    Uses a ScipyMinimize strategy for local refinement of discovered minima.
    """
    def __init__(
        self, 
        sampling_method: Literal['sobol', 'simplicial', 'halton'] = 'sobol',
        n_points: int = 64,
        iters: int = 1,
        local_strategy: Optional[ScipyMinimize] = None,
        name: Optional[str] = None
    ):
        super().__init__(name or "SHGO")
        self.sampling_method = sampling_method
        self.n_points = n_points
        self.iters = iters
        # Default to SLSQP if no specific local strategy is provided
        self.local_strategy = local_strategy or ScipyMinimize(method="SLSQP")

    def _solve(self, input: SolverInput) -> SolverResult:
        with self._Context(self):
            # Extract local minimizer settings from the provided strategy
            minimizer_kwargs = self.local_strategy._get_minimize_params(input)
            
            res = shgo(
                input.objective,
                bounds=input.bounds,
                n=self.n_points,
                iters=self.iters,
                sampling_method=self.sampling_method,
                minimizer_kwargs=minimizer_kwargs,
                options={'maxiter': input.max_steps},
            )
            
            return SolverResult(
                x=res.x, 
                fun=res.fun, 
                success=res.success, 
                message=res.message
            )

class SolverPipeline(OptimizationStrategy):
    """
    Sequential optimization: Each step starts from the result of the previous step.
    (Polishing mode).
    """
    def __init__(self, steps: List[OptimizationStrategy], stop_on_best_x: bool = True, name: Optional[str] = None):
        super().__init__(name or "Pipeline")
        self.steps = steps
        self.stop_on_best_x = stop_on_best_x

    def _solve(self, input: SolverInput) -> SolverResult:
        with self._Context(self):
            current_x = input.x0
            last_res = None

            for strategy in self.steps:
                # Create a new input context with the updated starting point
                last_res = strategy._solve(replace(input, x0=current_x))
                current_x = last_res.x
                
                if self.stop_on_best_x and last_res.fun <= input.tol:
                    break
                    
            return last_res

class SolverSelector(OptimizationStrategy):
    """
    Parallel-style optimization: Each step starts from the ORIGINAL x0.
    The best overall result is selected.
    """
    def __init__(self, candidates: List[OptimizationStrategy], name: Optional[str] = None):
        super().__init__(name or "Selector")
        self.candidates = candidates

    def _solve(self, input: SolverInput) -> SolverResult:
        with self._Context(self):
            best_res: Optional[SolverResult] = None

            for strategy in self.candidates:
                res = strategy._solve(input)
                
                if best_res is None or res.fun < best_res.fun:
                    best_res = res
                
                if best_res.fun <= input.tol:
                    break
                    
            return best_res
        
class RepeatStrategyWrapper(OptimizationStrategy):
    """Wrapper strategy representing a specific iteration within a continuation loop."""
    def __init__(self, strategy: OptimizationStrategy, current_iteration: int, total_iterations: int):
        super().__init__(f"Repeat_{current_iteration}")
        self.strategy = strategy
        self.current_iteration = current_iteration
        self.total_iterations = total_iterations

    @property
    def progress(self) -> float:
        """Calculates continuation progress mapped from 0.0 to 1.0."""
        if self.total_iterations <= 1:
            return 1.0
        return self.current_iteration / (self.total_iterations - 1)

    def _solve(self, input: SolverInput) -> SolverResult:
        with self._Context(self):
            return self.strategy._solve(input)

class SolverMethod():
    """Factory for creating optimization strategies."""
    def pipeline(self, *steps: 'OptimizationStrategy', stop_on_best_x: bool = True, name: Optional[str] = None) -> 'SolverPipeline':
        return SolverPipeline(steps=[*steps], stop_on_best_x=stop_on_best_x, name=name)
    
    def selector(self, *selector: 'OptimizationStrategy', name: Optional[str] = None) -> 'SolverSelector':
        return SolverSelector([*selector], name=name)
    
    def repeat(self, strategy: OptimizationStrategy, n: int, name: Optional[str] = None) -> 'SolverPipeline':
        """Repeats the specified method N times using a pipeline structure for continuation methods."""
        steps = [RepeatStrategyWrapper(strategy, i, n) for i in range(n)]
        return self.pipeline(*steps, stop_on_best_x=False, name=name)
    
    def slsqp(self, tol: Optional[float] = None, max_steps: Optional[int] = None, name: Optional[str] = None) -> 'ScipyMinimize':
        """
        Sequential Least Squares Programming.
        Best overall for kinematics. Handles bounds and constraints. 
        Uses gradients to find the 'steepest descent' efficiently.
        """
        return ScipyMinimize(method="SLSQP", tol=tol, max_steps=max_steps, name=name)

    def l_bfgs_b(self, tol: Optional[float] = None, max_steps: Optional[int] = None, name: Optional[str] = None) -> 'ScipyMinimize':
        """
        Limited-memory Broyden–Fletcher–Goldfarb–Shanno (Extended).
        Very memory-efficient. Excellent for smooth functions with many parameters.
        Handles bounds but is slightly more robust than SLSQP on noisy surfaces.
        """
        return ScipyMinimize(method="L-BFGS-B", tol=tol, max_steps=max_steps, name=name)

    def nelder_mead(self, tol: Optional[float] = None, max_steps: Optional[int] = None, name: Optional[str] = None) -> 'ScipyMinimize':
        """
        Simplex algorithm. Does not use gradients. 
        Robust for 'jumpy' or non-differentiable logic, but prone to 
        getting stuck on flat plateaus (as you observed).
        """
        return ScipyMinimize(method="Nelder-Mead", tol=tol, max_steps=max_steps, name=name)

    def brute(self, name: Optional[str] = None) -> 'BruteSolver':
        """
        Grid search. Slow, but robust.
        """
        return BruteSolver(name=name)
    
    def shgo(
        self, 
        sampling: Literal['sobol', 'simplicial', 'halton'] = 'sobol',
        n: int = 64,
        iters: int = 1,
        local_strategy: Optional[ScipyMinimize] = None,
        name: Optional[str] = None
    ) -> 'SHGOSolver':
        """
        Simplicial Homology Global Optimization.
        """
        return SHGOSolver(sampling_method=sampling, n_points=n, iters=iters, local_strategy=local_strategy, name=name)

    def stochastic(self, popsize: int = 15, mutation=(0.5, 1.0), recombination=0.7, name: Optional[str] = None) -> 'DiffEvoSolver':
        """
        Slow global search
        """
        return DiffEvoSolver(popsize=popsize, mutation=mutation, recombination=recombination, name=name)

    def auto(self, name: Optional[str] = None) -> 'SolverSelector':
        """
        Sequentially tries Nelder-Mead, then SLSQP, then L-BFGS-B.
        Stops at the first method that satisfies the tolerance.
        """
        return self.selector(self.nelder_mead(), self.slsqp(), self.l_bfgs_b(), name=name)

sm = SolverMethod()

T = TypeVar('T')

class Solver:
    class Session:
        """Proxy object to hold values and manage parameter registration/retrieval."""
        def __init__(self, solver: 'Solver', values: Optional[np.ndarray], is_init: bool = False, is_final: bool = False):
            self._solver = solver
            self._values = values
            self.is_init = is_init
            self.is_final = is_final
            self._param_counter = 0

        @property
        def solver(self) -> Optional['OptimizationStrategy']:
            """Returns the current active optimization strategy from the end of the stack."""
            if self._solver.methods_stack:
                return self._solver.methods_stack[-1]
            return None
        
        @property
        def solver_name(self) -> str:
            """Returns the name of the current active optimization strategy."""
            return self.solver.name if self.solver else ""

        @property
        def homotopy_parameter(self) -> float:
            """
            Calculates the continuation/homotopy parameter (t) mapped from 0.0 to 1.0.
            Used to dynamically relax or tighten constraints during optimization.
            """
            for strategy in reversed(self._solver.methods_stack):
                if isinstance(strategy, RepeatStrategyWrapper):
                    return strategy.progress
            return 1.0

        def __enter__(self):
            new_stack = _solver_session_stack.get() + [self]
            self._token = _solver_session_stack.set(new_stack)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            _solver_session_stack.reset(self._token)

        def param(self, init_value: T, min: Optional[T] = None, max: Optional[T] = None, steps: int = 10, step: Optional[float] = None) -> T:
            """
            Registers or retrieves a parameter. 
            In the first iteration, it collects bounds and initial values.
            In subsequent iterations, it returns reconstructed objects from the solver state.
            """
            if self.is_init:
                # Calculate steps if step size is explicitly provided
                final_steps = steps
                if step is not None:
                    if min is None or max is None:
                        raise ValueError("If step is provided, min and max must be provided as well.")
                    # We assume min/max/step are for the scalar range
                    # For complex types, we use the same step logic across all components
                    range_val = float(max) - float(min)
                    final_steps = int(builtins.max(1, math.ceil(abs(range_val / step))))

                # Registration defines the positional layout of the optimizer vector.
                # Candidate and final passes must call param() in this exact order.
                param_def = (init_value, (min, max))
                v, b, builder, length = _parse_param(param_def)
                
                self._solver.init_values.extend(v)
                self._solver.bounds.extend(b)
                self._solver.steps_list.extend([final_steps] * length)
                self._solver._builders.append(builder)
                
                start = self._solver._current_total_length
                end = start + length
                self._solver._offsets.append((start, end))
                self._solver._current_total_length += length

                self.add_cache_key_part(("param", tuple(v), tuple(b), length))
                
                return init_value
            
            # Candidate/final phases reconstruct this parameter from its registered slice.
            start, end = self._solver._offsets[self._param_counter]
            builder = self._solver._builders[self._param_counter]
            flat_slice = self._values[start:end]
            
            self._param_counter += 1
            return builder(flat_slice)

        def aim(self, error: float) -> None:
            """Called by the user to notify the solver of an error."""
            if self.is_final:
                return
            if self.is_init:
                self.add_cache_key_part(("aim", round(error, 6)))
                return
            # Sum up errors (usually using MSE or absolute sum)
            self._solver._current_error += float(error)

        def aim_dist(self, value1: Any, value2: Any, dist: float, k = 1.0) -> None:
            """
            Notifies the solver that the L2 distance between value1 and value2 
            should approach 'dist'. 
            
            Supports floats, mathutils types, Transform objects, and nested sequences.
            The error is calculated as the square of the difference from the target distance:
            error = (current_dist - target_dist)^2
            """
            if self.is_final:
                return
            if self.is_init:
                self.add_cache_key_part(("aim_dist", type(value1), type(value2), round(dist, 6), round(k, 6)))
                return

            def flatten(v: Any) -> List[float]:
                if hasattr(v, 'values'):
                    return list(v.values)
                
                if isinstance(v, (int, float)):
                    return [float(v)]
                
                if isinstance(v, (list, tuple, Vector, Euler, Quaternion)):
                    # Recursively flatten in case of nested lists/tuples
                    res = []
                    for x in v:
                        res.extend(flatten(x))
                    return res
                
                raise TypeError(f"Solver cannot flatten type: {type(v)}")

            if isinstance(value1, Transform) and isinstance(value2, Transform):
                v1 = Transform.values.fget(value1)
                v2 = Transform.values.fget(value2)

                current_dist_sq = 0.0
                for i in range(len(v1)):
                    diff = v1[i] - v2[i]
                    # Indexes 3, 4, 5 — (rx, ry, rz) in degrees
                    if 3 <= i <= 5:
                        # Find the shortest angle between two rotations
                        diff = (diff + 180) % 360 - 180
                    current_dist_sq += diff ** 2
            else:
                v1 = flatten(value1)
                v2 = flatten(value2)
                if len(v1) != len(v2):
                    raise ValueError(f"Solver dimension mismatch: {len(v1)} vs {len(v2)}")
                current_dist_sq = sum((a - b) ** 2 for a, b in zip(v1, v2))
            
            if dist == 0:
                # Optimized path for equality: error is simply the squared distance
                self.aim(current_dist_sq * k)
            else:
                # Error is the square of the difference between current and target distance
                current_dist = math.sqrt(current_dist_sq)
                self.aim((current_dist - dist) ** 2 * k)

        def aim_equal(self, value1: Any, value2: Any, k = 1.0) -> None:
            """
            Specialized case of aim_dist where target distance is 0.
            Notifies the solver that two structures should be identical.
            """
            self.aim_dist(value1, value2, dist=0.0, k=k)

        def constraint(self, condition: bool) -> None:
            """
            Applies a large penalty if the condition is False.
            Useful for hard boundaries or logic-based restrictions.
            """
            if self.is_final:
                return
            if self.is_init:
                self.add_cache_key_part(("constraint", condition))
                return
            if not condition:
                # Apply a massive penalty to discourage this path
                self.aim(1e9)

        def mode(self, value=Mode.ADD):
            """Keeps candidate geometry private and applies ``value`` only in the final pass."""
            if self.is_init:
                self.add_cache_key_part(("mode", value.name))
            return value if self.is_final else Mode.PRIVATE
        
        def add_cache_key_part(self, part: Tuple) -> None:
            """Adds a part to the cache key."""
            self._solver._cache_key_parts.append(part)

        @staticmethod
        def dummy():
            solver = Solver()
            return solver.Session(solver, None, is_init=True, is_final=True)

    _CACHE: Dict[Tuple, np.ndarray] = {}

    @property
    def cache_key(self) -> Tuple:
        return tuple(self._cache_key_parts)

    def __init__(self, method: OptimizationStrategy = sm.auto(), max_steps: int | None = 200, tol: float = 1e-4, debug: bool = False):
        self.init_values: List[float] = []
        self.bounds: List[Tuple[Optional[float], Optional[float]]] = []
        self.steps_list = []

        self._builders = []
        self._offsets = []
        self._current_total_length = 0
        self._cache_key_parts = []

        self.iterations = 0

        self.max_steps = max_steps
        self.tol = tol
        self.method = method
        self.debug = debug

        self.methods_stack: List[OptimizationStrategy] = []
        
        # The worker's SciPy callback sends a candidate through _from_solver, then
        # blocks until the main-thread loop supplies its accumulated error here.
        self._to_solver: Queue[float] = Queue()
        # _from_solver also carries strategy-stack and termination control messages.
        self._from_solver: Queue[Tuple[str, Any]] = Queue()
        
        self._best_x: Optional[np.ndarray] = None
        self._current_error: float = 0.0

    def __iter__(self):
        """ Handles registration pass, optimization thread, and yielding results. """
        # Registration fixes parameter offsets and the cache-key structure before
        # optimization. The loop body must preserve this param() call order later.
        self.init_values = []
        self.bounds = []
        self._builders = []
        self._offsets = []
        self._current_total_length = 0
        self._cache_key_parts = []
        self.methods_stack = []
        init_session = self.Session(self, None, is_init=True)
        with init_session:
            yield init_session

        if self._current_total_length == 0:
            return
        
        # After the first yield, we have all init_values and bounds
        self._best_x = None
        self.iterations = 0

        # SciPy's callback blocks for each objective evaluation. Keeping it on a
        # worker thread allows the main thread to expose those evaluations as a for-loop.
        thread = threading.Thread(target=self._run_minimize, daemon=True)
        thread.start()

        while True:
            msg = self._from_solver.get()
            assert isinstance(msg, tuple)
            
            # Process stream messages from the optimizer thread
            if msg[0] == "push":
                self.methods_stack.append(msg[1])
                continue
            elif msg[0] == "pop":
                if self.methods_stack:
                    self.methods_stack.pop()
                continue
            elif msg[0] == "value":
                x = msg[1]
            elif msg[0] == "break":
                break
            else:
                raise ValueError(f"Unknown message: {msg}")
            
            self._current_error = 0.0
            session = self.Session(self, x)
            
            with session:
                yield session

            # After the user's code block finishes, we send the accumulated error
            self._to_solver.put(self._current_error)

            if self.debug:
                print(f"[Solver Debug] {session.solver_name} | Iter: {self.iterations} | X: {', '.join(f'{v:.4f}' for v in x)} | Error: {self._current_error:.6f} | H: {session.homotopy_parameter:.1f}")
            self.iterations += 1

        # Persist the best vector before replaying the model one final time with
        # normal geometry modes and final-only side effects enabled.
        if self._best_x is not None:
            Solver._CACHE[self.cache_key] = self._best_x
            final_session = self.Session(self, self._best_x, is_final=True)
            with final_session:
                yield final_session

    def _run_minimize(self) -> None:
        token = _current_running_solver.set(self)
        try:
            def objective(x: np.ndarray) -> float:
                # 1. Send new candidate parameters to the main thread
                self._from_solver.put(("value", x))
                # 2. Wait for the user to call s.aim(error)
                return self._to_solver.get()

            x_start = np.array(self.init_values, dtype=float)
            err_init = objective(x_start)
            self._cache_key_parts.append(("err_init", round(err_init, 6)))
            
            # Re-evaluate a cached vector before trusting it: the key describes the
            # declared problem shape, while the current run supplies its objective.
            if self.cache_key in Solver._CACHE:
                x_cache = Solver._CACHE[self.cache_key]
                err_cache = objective(x_cache)
                
                if err_cache < err_init:
                    x_start = x_cache
                    current_val = err_cache
                else:
                    current_val = err_init
                    
                if current_val <= self.tol:
                    self._best_x = x_start
                    self._terminate_loop()
                    return
            
            # 2. Execute Strategy (Single or Pipeline)
            # The strategy itself handles the logic of sequential calls if it's a Pipeline
            input = SolverInput(
                objective=objective,
                x0=x_start,
                bounds=self.bounds,
                steps=self.steps_list,
                tol=self.tol,
                max_steps=self.max_steps
            )
            result = self.method._solve(input)

            if self.debug:
                print(f"[Solver Debug] Optimization Finished. Best error: {result.fun:.6f} via {self.method.__class__.__name__}")

            self._best_x = result.x
            self._terminate_loop()
        except Exception as e:
            self._terminate_loop()
            raise e
        finally:
            _current_running_solver.reset(token)

    def _terminate_loop(self) -> None:
        self._from_solver.put(("break",))

def _merge_bounds(b_base: List[Tuple], b_user: List[Tuple]) -> List[Tuple]:
    """Combines instance bounds with user-defined bounds, prioritizing restrictions."""
    res = []
    for (min1, max1), (min2, max2) in zip(b_base, b_user):
        final_min = min1 if min2 is None else (min2 if min1 is None else max(min1, min2))
        final_max = max1 if max2 is None else (max2 if max1 is None else min(max1, max2))
        res.append((final_min, final_max))
    return res

def _parse_param(val: Any) -> Tuple[List[float], List[Tuple], Any, int]:
    """
    Recursively parses complex objects, lists, and bound tuples.
    Returns: (flat_values, flat_bounds, reconstruction_function, num_values)
    """
    # 1. Scalar (Float / Int)
    if isinstance(val, (int, float)):
        return [float(val)], [(None, None)], lambda x: x[0], 1
        
    # 2. Bound Tuple format: (Value, (Min, Max))
    if isinstance(val, tuple) and len(val) == 2 and isinstance(val[1], tuple) and len(val[1]) == 2:
        obj, bnds = val
        obj_vals, obj_bnds, obj_build, obj_len = _parse_param(obj)
        min_bnd, max_bnd = bnds
        
        # Expand Min Bound (supports scalar broadcasting to all DoFs, e.g., None or 10)
        if min_bnd is None:
            min_vals = [None] * obj_len
        elif isinstance(min_bnd, (int, float)):
            min_vals = [float(min_bnd)] * obj_len
        else:
            min_vals, _, _, min_len = _parse_param(min_bnd)
            if min_len != obj_len: raise ValueError("Min bound structure mismatch.")
                
        # Expand Max Bound
        if max_bnd is None:
            max_vals = [None] * obj_len
        elif isinstance(max_bnd, (int, float)):
            max_vals = [float(max_bnd)] * obj_len
        else:
            max_vals, _, _, max_len = _parse_param(max_bnd)
            if max_len != obj_len: raise ValueError("Max bound structure mismatch.")
                
        user_bounds = list(zip(min_vals, max_vals))
        final_bounds = _merge_bounds(obj_bnds, user_bounds)
        return obj_vals, final_bounds, obj_build, obj_len
    
    if isinstance(val, Vector) and not isinstance(val, SVector):
        val = SVector(val)

    # 3. Geometric / Custom Objects (Pos, Rot, Transform, SVector)
    if hasattr(val, 'values') and hasattr(val, 'bounds') and hasattr(val, 'copy'):
        v = list(val.values)
        b = list(val.bounds)
        length = len(v)
        obj_copy = val.copy() # Store a safe instance for the builder
        
        def obj_builder(flat_arr: List[float]):
            new_obj = obj_copy.copy()
            new_obj.values = flat_arr[:length]
            return new_obj
            
        return v, b, obj_builder, length
        
    # 4. Nested Iterables (List / Tuple)
    if isinstance(val, (list, tuple)):
        v_all, b_all, builders = [], [], []
        total_len = 0
        for item in val:
            v, b, build, length = _parse_param(item)
            v_all.extend(v)
            b_all.extend(b)
            builders.append((build, length))
            total_len += length
            
        is_tuple = isinstance(val, tuple)
        def list_builder(flat_arr: List[float]):
            res = []
            offset = 0
            for build, length in builders:
                res.append(build(flat_arr[offset:offset+length]))
                offset += length
            return tuple(res) if is_tuple else res
            
        return v_all, b_all, list_builder, total_len
        
    raise ValueError(f"Unsupported parameter type in Solver: {type(val)}")

_solver_session_stack: ContextVar[List['Solver.Session']] = ContextVar('_session_stack', default=[])

def solver() -> 'Solver.Session':
    """Returns the most nested active Solver Session."""
    stack = _solver_session_stack.get()
    if not stack:
        return Solver.Session.dummy()
    return stack[-1]
