from typing import List
from blender_cad import *
from tests.test_base import BaseCADTest
from mathutils import Vector

class TestSolverOptimization(BaseCADTest):
    """
    Category: Solver & Optimization
    Tests for the scipy-based parameter auto-tuning and caching system.
    """

    def test_single_parameter_optimization(self):
        """
        Test finding a single parameter Y such that Y^2 = 25.
        Initial guess is 0.0, target is 5.0 (or -5.0).
        """
        final_y = 0.0
        
        # We expect the solver to find Y=5.0 to satisfy the aim
        for s in Solver(sm.nelder_mead()):
            Y = s.param(0.0)
            # Y is managed by scipy.optimize
            current_value = Y ** 2
            s.aim_equal(current_value, 25.0)
            
            # We can access the current active solver instance using solver()
            cur_solver = solver()
            if cur_solver.is_final:
                final_y = Y

        self.assertAlmostEqual(abs(final_y), 5.0, places=4, msg="Solver failed to find single parameter root.")

    def test_single_integer_parameter_optimization(self):
        """
        Test finding an integer parameter Y such that Y^2 is close to 24.
        Brute force search is appropriate here.
        """
        final_y = 0
        
        for s in Solver(sm.brute()):
            # Round to nearest integer
            Y = round(s.param(0.0, min=0, max=10, step=1.0))
            
            # Target is 24. Math sqrt(24) is 4.8989...
            current_sq = Y ** 2
            s.aim_equal(current_sq, 24.0)
            
            if s.is_final:
                final_y = Y

        # Verify that the result is a strict integer and rounded to the nearest 'best' int
        self.assertIsInstance(final_y, int, "Final result should be cast back to int.")
        self.assertEqual(final_y, 5, f"Solver should have snapped to 5, but got {final_y}")
        

    def test_nested_solver_optimization(self):
        """
        Test nested solver capability:
        Outer solver tries to find X such that X + inner_best_Y = 6.
        Inner solver tries to find Y such that Y^2 = X.
        X, Y should be close to 4 and 2, respectively.
        """
        results = []

        # Outer solver: searching for X
        for s_outer in Solver(sm.nelder_mead(tol=1e-3), max_steps=20):
            # Using the global solver() function to define parameter
            x_val = solver().param(2.0, min=0.0, max=20.0)
            
            inner_final_y = 0.0
            # Inner solver: searching for Y based on current X
            # Each iteration of s_outer triggers a full execution of s_inner
            for s_inner in Solver(sm.nelder_mead(tol=1e-3), max_steps=20):
                y_val = solver().param(1.0, min=0.0, max=10.0)
                
                # Inner objective: Y^2 should reach X
                s_inner.aim_equal(y_val ** 2, x_val)
                
                if s_inner.is_final:
                    inner_final_y = y_val

            # Outer objective: X + optimized Y should reach 6
            # (e.g., if X=4, then inner Y=2, 4+2=6. Success!)
            s_outer.aim_equal(x_val + inner_final_y, 6.0)
            
            if s_outer.is_final:
                results = [x_val, inner_final_y]

        final_x, final_y = results
        
        # Assertions
        self.assertAlmostEqual(final_x, 4.0, places=2, 
                            msg=f"Outer solver failed. Expected X=4, got {final_x}")
        self.assertAlmostEqual(final_y, 2.0, places=2, 
                            msg=f"Inner solver failed. Expected Y=2, got {final_y}")

    def test_multi_parameter_optimization(self):
        """
        Test solving a simple system of equations:
        1) A + B = 10
        2) A - B = 2
        Result should be A=6, B=4.
        """
        results = {}

        for s in Solver():
            A = s.param(0.0)
            B = s.param(0.0)
            # Aim for multiple objectives
            s.aim_equal(A + B, 10.0)
            s.aim_equal(A - B, 2.0)
            
            if s.is_final:
                results['A'] = A
                results['B'] = B

        self.assertAlmostEqual(results['A'], 6.0, places=4)
        self.assertAlmostEqual(results['B'], 4.0, places=4)

    def test_solver_caching(self):
        """
        Verify that the solver uses cached results for identical inputs.
        The second run should perform minimal iterations (<= 2).
        """
        def run_optimization():
            iters = 0
            for s in Solver():
                X = s.param(10.0)
                iters += 1
                # Simple target: X = 2.0
                s.aim_equal(X, 2.0)
            return iters

        # First run: Actual optimization happens
        first_run_iters = run_optimization()
        
        # Second run: Should hit the cache
        second_run_iters = run_optimization()

        self.assertGreater(first_run_iters, second_run_iters, "Cache failed: First run took more iterations than second run.")
        self.assertLessEqual(second_run_iters, 4, f"Cache failed: Second run took {second_run_iters} iterations.")

    def test_bounds_limit(self):
        """
        Test that the solver respects the explicitly provided bounds.
        Target is 100, but Y is bounded to (0, 10). Result should be ~10.
        """
        final_y = 0.0
        # Target 100, but bounded to 10
        for s in Solver():
            Y = s.param(5.0, min=0.0, max=10.0)
            s.aim_equal(Y, 100.0)
            if s.is_final:
                final_y = Y

        self.assertLessEqual(final_y, 10.0001)
        self.assertGreater(final_y, 9.9)

    def test_mixed_bounded_and_free(self):
        """
        One parameter is bounded, the other is free.
        A + B = 20. A is bounded to max 5. Result: A=5, B=15.
        """
        res = {}
        for s in Solver():
            A = s.param(0.0, min=0, max=5)
            B = s.param(0.0)
            s.aim_equal(A + B, 20.0)
            if s.is_final:
                res['A'], res['B'] = A, B
        
        self.assertAlmostEqual(res['A'], 5.0, places=2)
        self.assertAlmostEqual(res['B'], 15.0, places=2)

    def test_constraint_penalty(self):
        """
        Test the constraint method. 
        Find X^2 = 25, but add a constraint that X must be negative.
        Without constraint, it might find 5.0. With constraint, it must find -5.0.
        """
        final_x = 0.0
        # Start at 1.0, which is closer to the 'wrong' positive root
        for s in Solver():
            X = s.param(-0.1)
            s.aim_equal(X ** 2, 25.0)
            # Hard constraint: X must be negative
            s.constraint(X < 0)

            if s.is_final:
                final_x = X

        self.assertAlmostEqual(final_x, -5.0, places=3)

    def test_brute_mixed_int_float(self):
        """
        Test brute on a mixed integer/float problem.
        Target: minimize (X - 4)^2 + (Y - 7.5)^2 where X is int and Y is float.
        """
        for s in Solver(sm.brute().with_polish()):
            # X as integer parameter (total steps = 10)
            X = round(s.param(0.0, min=0, max=10, step=1.0))
            # Y as float parameter (total steps = 1, refined with polish)
            Y = s.param(0.0, min=0.0, max=10.0, steps=1)
            
            # Distance squared to (4, 7.5)
            # Global minimum is at X=4, Y=7.5
            s.aim_equal(X, 4)
            s.aim_equal(Y, 7.5)
            
            if s.is_final:
                final_x = X
                final_y = Y

        # Assertions to ensure the global minimum was found correctly
        self.assertEqual(final_x, 4, f"Failed: X should be 4")
        self.assertAlmostEqual(final_y, 7.5, places=3, msg=f"Failed: Y should be 7.5")

    def test_matrix_as_nested_list(self):
        """
        Test using a 2x2 matrix (nested list) as a parameter.
        The goal is to make all elements equal to 5.0, but some are bounded.
        """
        # Initial matrix [[0, 0], [0, 0]]
        # Bounds: element [0][0] is limited to max 2.0
        initial_matrix = [
            [(0.0, (None, 2.0)), 0.0],
            [0.0, 0.0]
        ]
        
        res_matrix = None
        for s in Solver(max_steps=1000):
            m: List[List[float]] = s.param(initial_matrix)
            # Target: each element should be 5.0
            for row in m:
                for val in row:
                    s.aim_equal(val, 5.0)
            
            if s.is_final:
                res_matrix = m

        # m[0][0] should hit the upper bound of 2.0
        self.assertAlmostEqual(res_matrix[0][0], 2.0, places=2)
        # Other elements should reach 5.0
        self.assertAlmostEqual(res_matrix[0][1], 5.0, places=2)
        self.assertAlmostEqual(res_matrix[1][1], 5.0, places=2)

    def test_vector_optimization(self):
        """
        Test mathutils.Vector.
        Find a vector with length 10, where X is fixed at 0.
        """
        # Start with (0, 1, 1). X is bounded to (0, 0)
        v_init = Vector((0.0, 1.0, 1.0))
        v_bounds = (Vector((0.0, -100, -100)), Vector((0.0, 100, 100)))
        
        res_v = None
        for s in Solver():
            v = s.param(v_init, min=v_bounds[0], max=v_bounds[1])
            # Aim for vector length of 10
            s.aim_equal(v.length, 10.0)
            
            if s.is_final:
                res_v = v

        self.assertAlmostEqual(res_v.x, 0.0, places=2)
        self.assertAlmostEqual(res_v.length, 10.0, places=2)
        self.assertTrue(isinstance(res_v, Vector))

    def test_pos_with_bounds(self):
        """
        Test Pos object with coordinate bounds.
        Target position is (10, 10, 10), but X and Y are restricted.
        """
        # Pos values are [x, y, z]. 
        # Limit X to [0, 5] and Y to [0, 5]. Z is free.
        pos_init = Pos(X=1, Y=1, Z=1)
        lower_b = Pos(X=0, Y=0, Z=-100)
        upper_b = Pos(X=5, Y=5, Z=100)
        
        res_pos = None
        for s in Solver():
            loc = s.param(pos_init, min=lower_b, max=upper_b)
            target = Pos((10.0, 10.0, 10.0))
            s.aim_equal(loc, target)
            
            if s.is_final:
                res_pos = loc

        self.assertAlmostEqual(res_pos.x, 5.0, places=2)
        self.assertAlmostEqual(res_pos.y, 5.0, places=2)
        self.assertAlmostEqual(res_pos.z, 10.0, places=2)

    def test_pos_array_with_bounds(self):
        """
        Test a list of Pos objects.
        Two points must be 10 units apart, but the first point is stuck at origin.
        """
        points_init = [
            Pos(X=0, Y=0, Z=0), # Will be bounded to (0,0,0)
            Pos(X=1, Y=0, Z=0)  # Free to move
        ]
        
        # Only bound the first element of the list to be exactly zero
        # The second element (None, None) means use default/free bounds
        points_spec = [
            (points_init[0], (Pos(0,0,0), Pos(0,0,0))),
            points_init[1]
        ]
        
        res_points = None
        for s in Solver():
            pts: List[Pos] = s.param(points_spec)
            s.aim_dist(pts[0], pts[1], dist=10.0)
            
            if s.is_final:
                res_points = pts

        self.assertAlmostEqual(res_points[0].x, 0.0, places=2)
        self.assertAlmostEqual((res_points[0].position - res_points[1].position).length, 10.0, places=2)

    def test_rot(self):
        """
        Test Rot object 
        """
        rot_init = Rot(X=355)
        res_rot = None
        
        for s in Solver():
            rot = s.param(rot_init)
            s.aim_equal(rot, Rot(X=365))
            
            if s.is_final:
                res_rot = rot

        self.assertAlmostEqual(res_rot.values[0], 5.0, places=1)

    def test_location(self):
        """
        Test Location object created by Pos * Rot.
        Target: Pos(10, 20, 30) and Rot(X=10)
        """
        res = None
        init_loc = Pos(0, 0, 0) * Rot(0, 0, 0)
        target_loc = Pos(10, 20, 30) * Rot(X=50)
        
        for s in Solver():
            loc = s.param(init_loc)
            s.aim_equal(loc, target_loc)
            
            if s.is_final:
                res = loc

        self.assertAlmostEqual(res.x, 10.0, places=1)
        self.assertAlmostEqual(res.y, 20.0, places=1)
        self.assertAlmostEqual(res.z, 30.0, places=1)
        self.assertAlmostEqual(res.rx, 50.0, places=1)

    def test_dynamic_param_calls_in_loop(self):
        """
        Test calling s.param multiple times inside a loop.
        The goal is to find values for a list such that their sum equals 100,
        while each element has its own specific constraints.
        """
        initial_values = [1.0, 1.0, 1.0, 1.0]
        # Specific max bounds for each index
        max_bounds = [10.0, 20.0, 40.0, 50.0]
        
        final_list = []
        
        # The first iteration registers all parameters called in the loop
        for s in Solver(max_steps=500, tol=1e-6):
            current_collection = []
            
            # Dynamically create parameters
            for i in range(len(initial_values)):
                # Each call to s.param registers a new offset in the solver
                val = s.param(
                    initial_values[i], 
                    min=0.0, 
                    max=max_bounds[i]
                )
                current_collection.append(val)
            
            # Goal: The sum of all elements should be 100
            current_sum = sum(current_collection)
            s.aim_equal(current_sum, 100.0)
            
            if s.is_final:
                final_list = current_collection

        # Verification
        # Total sum should be 100
        self.assertAlmostEqual(sum(final_list), 100.0, places=2)
