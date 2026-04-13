from blender_cad import *
from tests.test_base import BaseCADTest

class TestCurveGeometry(BaseCADTest):
    """
    Category: Curve Geometry
    Progressive tests for 1D primitives, spline evaluation, and distribution logic.
    """

    def test_single_line(self):
        """Verify a simple straight line segment and its length."""
        with BuildPart() as result:
            with BuildCurve() as bc:
                Line((0, 0, 0), (0, 0, 5))
            
            # Basic integrity checks before beveling
            self.assertEqual(bc.curve.length(), 5.0)
            self.assertEqual(bc.curve.start, Vector((0, 0, 0)))
            self.assertEqual(bc.curve.end, Vector((0, 0, 5)))
            
            bc.curve.bevel(depth=0.1)
            add(bc.curve)

        self.assertPart(result.part, "a22f6147434fe40d144958d5de0a287c9c6a58b66489cf175adf950b17d37acb", "test_single_line")

    def test_polyline_open(self):
        """Verify a polyline with multiple segments."""
        with BuildPart() as result:
            with BuildCurve() as bc:
                Polyline((0, 0, 0), (1, 0, 0), (1, 1, 0))
            
            # Length should be $1.0 + 1.0 = 2.0$
            self.assertAlmostEqual(bc.curve.length(), 2.0, places=4)
            
            bc.curve.bevel(depth=0.05)
            add(bc.curve)

        self.assertPart(result.part, "cecc5d71860623a80b41a28ed7467c33b4cb355f257d644dd8910758ebb50666", "test_polyline_open")

    def test_nurbs_spline_closed(self):
        """Verify a closed NURBS spline (circle-like square)."""
        with BuildPart() as result:
            with BuildCurve() as bc:
                Spline((0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), close=True)
            
            self.assertTrue(bc.curve.obj.data.splines[0].use_cyclic_u)
            
            bc.curve.bevel(depth=0.2)
            add(bc.curve)

        self.assertPart(result.part, "f19e56713e5a081a4e1fa9ca10293a3083a3ad1e2796ceff15090b852608367c", "test_nurbs_spline_closed")

    def test_at_evaluation_complex_spline(self):
        """
        Verify .at() evaluation using both normalized parameter (t) and meters (t_m).
        Ensures tangent/normal consistency on a 3D spline.
        """
        with BuildPart() as result:
            with BuildCurve() as curve:
                # Create a 3D Spline rising upwards
                Spline(
                    (0, 0, 0), (5, 2, 2), (2, 5, 4), (0, 0, 8),
                )
            
            # 1. Test evaluation by factor (t)
            # Center of the curve
            loc_mid = curve.at(t=0.5).x_as_z()
            Marker(loc=loc_mid)
            
            # 2. Test evaluation by distance (t_m)
            # Evaluate at 2.0 meters from the start
            loc_dist = curve.at(t_m=2.0).x_as_z()
            Marker(loc=loc_dist)

            curve.bevel(depth=0.01)
            add(curve)

        self.assertPart(result.part, "1d89a125d9098f6fb87ae7dda109045a27c0bdef9db2719a9a87d6df12d38c08", "test_at_evaluation_complex_spline")

    def test_multi_island_evaluation(self):
        """
        Verify evaluation across a curve object containing multiple disjoint islands.
        Island 1: Line, Island 2: Spline.
        """
        with BuildPart() as result:
            with BuildCurve() as curve:
                # Island 1
                Line((0, 0, 0), (0, 10, 0))
                # Island 2 (disjoint)
                Spline((5, 0, 0), (5, 5, 2), (5, 10, 0))
            
            # Length should be roughly 10 (Line) + ~10.5 (Spline)
            self.assertGreater(curve.length(), 20.0)
            
            # Evaluate at the very beginning (Island 1)
            loc_start = curve.at(t=0.1).x_as_z()
            Marker(loc=loc_start)
            
            # Evaluate at the very end (Island 2)
            loc_end = curve.at(t=0.9).x_as_z()
            Marker(loc=loc_end)

            curve.bevel(depth=0.01)
            add(curve)

        self.assertPart(result.part, "88162da725e05c2842d0b4c8d80d1bb65b39f8ed01e3ba96ea53848b6fafc1f0", "test_multi_island_evaluation")

    def test_bezier_continuity(self):
        """Verify a Bezier curve with specific control handles."""
        with BuildPart() as result:
            with BuildCurve() as bc:
                BezierCurve(
                    start=(0, 0, 0),
                    handle1=(0, 5, 0),
                    handle2=(5, 5, 0),
                    end=(5, 0, 0)
                )
            
            bc.curve.bevel(depth=0.1)
            add(bc.curve)

        self.assertPart(result.part, "dab2f9bf5e794222af2bd2928eed5161e7eced09293c92db24b85968c4fc0aa8", "test_bezier_continuity")

    def test_automatic_vertex_merging(self):
        """
        Verify that sequential lines with coincident points merge into a single spline.
        Result should be 1 spline with 3 points, not 2 splines with 2 points each.
        """
        with BuildPart() as result:
            with BuildCurve() as bc:
                l1 = Line((0, 0, 0), (0, 0, 1))
                # Endpoint of l1 is the start of l2
                l2 = Line(l1.end, (1, 0, 1))
            
            # Check internal Blender data structure
            num_splines = len(bc.curve.obj.data.splines)
            num_points = len(bc.curve.obj.data.splines[0].points)
            
            self.assertEqual(num_splines, 1, "Lines failed to merge into a single spline!")
            self.assertEqual(num_points, 3, "Merged spline should have 3 points, not 4!")
            
            bc.curve.bevel(depth=0.05)
            add(bc.curve)

        self.assertPart(result.part, "e1080b459dd928b21eddf35f84de36f5fbf0549a6e397272a843f13ec72847f4", "test_automatic_vertex_merging")

    def test_curve_locations_distribution(self):
        """Verify that CurvePositions correctly distributes locations by spacing."""
        with BuildPart() as result:
            with BuildCurve() as bc:
                Spline((0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), close=True)
            
            # Place points every 2.5 meters (should result in 5 points: 0, 2.5, 5, 7.5, 10)
            locs = CurveLocations(bc.curve, spacing=2.5)
            self.assertEqual(len(locs), 12)
            
            with locs:
                Marker(Rot(Y=90))
        
        self.assertPart(result.part, "a160e3b9135d96c77eca0637460b106782a47a9948f9cf9dd1ff2a055e1f0046", "test_curve_locations_distribution")

    def test_pipe_run_simulation(self):
        """
        Simulate a real-world scenario: 3 parallel pipes with different paths.
        Tests the transformation of curves into solid Part objects.
        """
        pipe_depth = 0.15
        
        with BuildPart() as result:
            # Pipe 1: Straight with a 90-degree turn
            with BuildCurve() as c1:
                Polyline((0, 0, 0), (10, 0, 0), (10, 5, 0))
            c1.bevel(depth=pipe_depth)
            add(c1)
            
            # Pipe 2: Elevated spline path
            with BuildCurve() as c2:
                Spline((0, 1, 0), (5, 1, 2), (10, 6, 2))
            c2.bevel(depth=pipe_depth)
            add(c2)
            
            # Pipe 3: Complex 3D route
            with BuildCurve() as c3:
                Polyline((0, -1, 0), (5, -1, 0), (5, -1, 5), (10, 4, 5))
            c3.bevel(depth=pipe_depth)
            add(c3)
            
        self.assertPart(result.part, "439f1cc2f58c80bfc71f12082a6d20ad63e69e466c3f24a50c970e113fce189f", "test_pipe_run_simulation")

    def test_sine_wave_spline(self):
        """Verify spline generation based on a sine wave function."""
        import math
        with BuildPart() as result:
            with BuildCurve() as bc:
                # Generate points for a sine wave: y = sin(x)
                # Ampltitude 2.0, Frequency over 10 meters, 40 points for smoothness
                points = []
                for i in range(41):
                    x = (i / 40) * 10  # from 0 to 10
                    y = 2.0 * math.sin(x)
                    points.append(Vector((x, y, 0)))
                
                # Create a spline passing through all points
                Spline(points)
                
                bc.curve.bevel(depth=0.1)
                add(bc.curve)

        # Hash represents the swept volume of the sine wave tube
        self.assertPart(result.part, "ef07a0938f3067bd27cb4885ca95a022b8efbf844747c318ae7c6d8fa1bb07fc", "test_sine_wave_spline")

    def test_spiral_3d_spline(self):
        """Verify a 3D spiral (helix) generated via parametric equations."""
        import math
        with BuildPart() as result:
            with BuildCurve() as bc:
                # Helix: x = cos(t), y = sin(t), z = t
                helix_points = [
                    Vector((math.cos(t), math.sin(t), t * 0.2))
                    for t in [i * 0.5 for i in range(20)]
                ]
                
                Spline(helix_points)
                
                bc.curve.bevel(depth=0.05)
                add(bc.curve)

        self.assertPart(result.part, "731ae681b333425b425f51f7e7cbe646a1b681bcb500cfcc5f1c6f2d9b48f985", "test_spiral_3d_spline")

    def test_face_parametric_curves(self):
        """Verify parametric curves generated on planar and cylindrical surfaces."""
        import math
        with BuildPart() as result:
            Cylinder(1, 2)
            top_face = faces().top()[0]
            cyl_face = faces().filter_by(GeomType.CYLINDER)[0]
            base_selector = uv.set(0,0).offset_final(Pos(Z=0.2))

            # Generate a horizontal loop around the cylinder
            # rule: t affects U (rotation), V is constant at 0.5
            rule = lambda t: (t, 0.5)
            curve = cyl_face.curve(rule=rule, selector=base_selector, close=True)
            add(curve.bevel(depth=0.1))

            # Generate a sine wave on the top flat face
            # rule: t affects U, V oscillates around 0.3
            rule = lambda t: (t, (math.sin(t * 20) + 1) / 8 + 0.3)
            curve = top_face.curve(rule=rule, limit=0.9, selector=base_selector)
            add(curve.bevel(depth=0.01))

        # The hash validates the resulting geometry of the beveled curves on the cylinder/top face
        self.assertPart(
            result.part, 
            "554da86e53c52dcec36ba15d4c9df71237010018768bb33e7a5588b461835d88", 
            "test_face_parametric_curves"
        )

    def test_tangent_arc(self):
        """Verify an arc that continues smoothly from a previous segment's tangent."""
        with BuildPart() as result:
            with BuildCurve() as bc:
                # Start with a straight line to establish a tangent
                Line((0, 0, 0), (5, 0, 0))
                # Arc should exit smoothly towards the end point
                TangentArc(end=(10, 5, 0))
                
                bc.curve.bevel(depth=0.1)
                add(bc.curve)

        self.assertPart(result.part, "9b695d09689b3b2b2026a11f160886d8cba9823f612f9d657ecf676c4405aeaa", "test_tangent_arc")

    def test_radius_arc(self):
        """Verify an arc defined by start, end, and a specific radius."""
        with BuildPart() as result:
            with BuildCurve() as bc:
                # Create a bridge-like arc
                RadiusArc(start=(0, 0, 0), end=(10, 0, 0), radius=8.0)
                
                bc.curve.bevel(depth=0.2)
                add(bc.curve)

        self.assertPart(result.part, "c851f1d69fe7585226160a5ad14b6abf460c2eab4f28b2b9b7b8c58f3ec8ec99", "test_radius_arc")

    def test_center_arc(self):
        """Verify an arc defined by center point, radius, and angular span."""
        with BuildPart() as result:
            with BuildCurve() as bc:
                # Draw a half-circle (180 degrees)
                CenterArc(center=(0, 0, 0), radius=5.0, start_angle=0, end_angle=180)
                
                bc.curve.bevel(depth=0.15)
                add(bc.curve)

        self.assertPart(result.part, "7b8dc96cea3d8f1fa26e44df945a6ffdd8bc97f284ff3635e1a7ca64f81fad73", "test_center_arc")

    def test_jiggle_line(self):
        """Verify the organic noisy line generator with a fixed seed for hashing."""
        # Seed the random generator to ensure deterministic geometry for the hash check
        import random
        random.seed(42)
        
        with BuildPart() as result:
            with BuildCurve() as bc:
                # Generate a noisy path between two points
                Jiggle(start=(0, 0, 0), end=(10, 10, 0), noise_factor=0.8, segments=15)
                
                bc.curve.bevel(depth=0.05)
                add(bc.curve)

        self.assertPart(result.part, "8506a9945fe52c73e8567381db7127e5ff9937d1c3825dc21561a7d4b1653dae", "test_jiggle_line")

    def test_curve_builder(self):
        """Verify complex nested curve operations, directional steps, smooth-to-sharp transitions, and final edge beveling."""
        with BuildPart() as result:
            # Step 1: Initialize the core curve builder sequence with a default smooth radius (forming a rounded outer profile)
            # The structure branches down into a sharp triangle, and then a vertical line segment.
            c_built = curve(
                curve.smooth(radius=10),
                curve.step(10, angle=0),
                curve.step(10, angle=90),
                # Sub-curve block: Disable smoothing to construct sharp geometric transitions
                curve(
                    curve.smooth(False),
                    curve.step(10, angle=0),
                    curve.step(10, angle=120),  # Creates a 60-degree internal angle for the triangle
                    # Deepest sub-curve block: Shift tracking position and trace a vertical path
                    curve(
                        curve.move_to_Y(),
                        curve.step(5, rot=Rot(Y=90)),  # Rotates the path direction vertically via Y-axis rotation
                        curve.step(5, angle=-90),
                        close=False
                    ),
                    curve.step(10, angle=120),
                ),
                curve.step(10, angle=90),
                curve.step(10, angle=90),
                curve.step(10, angle=90),
                trim_ends=True
            )
            
            # Step 2: Apply a high-utility edge bevel operation to stabilize and clean up the wire junctions
            beveled_curve = c_built.curve.bevel()
            
            # Step 3: Add the resulting 3D wire artifact to the active part context
            add(beveled_curve)

        self.assertPart(
            result.part,
            "bc3c43c336a769e37c6ddf9385d5bce638540ad9875cc5ba17f162065b602ab9",
            "test_curve_builder",
        )

    def test_bvh_curve_construction_and_directional_normal_assertions(self):
        """Verify BVH spatial queries on a curve using Vector, ensuring correct normal orientation relative to the query point."""
        with BuildPart() as result:
            # Step 1: Establish a closed 2D spline curve trajectory
            with BuildCurve() as bc:
                Spline((0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), close=True)
            
            # Step 2: Build the Bounding Volume Hierarchy (BVH) structure over the reference curve
            bvh = bc.curve.build_bvh()
            
            # Query 1: Evaluate a point centrally located deep within the bounding loop profile
            point = Vector((5, 5, 0))
            loc, norm, _, dist = bvh.find_nearest(point)
            Marker(Pos(loc), mat=mat.blue)
            Marker(Pos(point), mat=mat.blue)
            Marker(Pos(norm), mat=mat.blue, size=0.3)
            # Assert normal orientation points correctly relative to the vector from the nearest point to the query point
            self.assertTrue(norm.dot(point - loc) > 0)

            # Query 2: Evaluate an external coordinate completely outside the primary loop boundary
            point = Vector((-5, 5, 0))
            loc, norm, _, dist = bvh.find_nearest(point)
            Marker(Pos(loc), mat=mat.red)
            Marker(Pos(point), mat=mat.red)
            Marker(Pos(norm), mat=mat.red, size=0.3)
            self.assertFalse(norm.dot(point - loc) > 0)

            # Query 3: Evaluate a tracking node hovering directly below the top edge perimeter
            point = Vector((5, 9, 0))
            loc, norm, _, dist = bvh.find_nearest(point)
            Marker(Pos(loc), mat=mat.green)
            Marker(Pos(point), mat=mat.green)
            Marker(Pos(norm), mat=mat.green, size=0.3)
            self.assertFalse(norm.dot(point - loc) > 0)
            
            # Step 3: Append the final curve topology representation into the active block environment
            add(bc)

        self.assertPart(
            result.part,
            "fac6b84a2711e3d724ff031bf3f33688be0aa95648885fb903e0b342122a55b0",
            "test_bvh_curve_construction_and_directional_normal_assertions",
            use_materials=True,
        )

    def test_build_curve_explicit_component_tagging_and_isolation(self):
        """Verify that explicitly tagged standard primitives inside BuildCurve are independently selectable and shiftable."""
        with BuildPart() as result:
            # Step 1: Create a multi-primitive curve collection, assigning custom tags to each entity
            with BuildCurve() as c:
                Line((0, 0, 0), (0, 10, 0), tag='A')
                Spline((5, 0, 0), (5, 5, 2), (5, 10, 0), tag='B')
                BezierCurve(
                    start=(10, 0, 0),
                    handle1=(10, 5, 0),
                    handle2=(15, 5, 0),
                    end=(15, 0, 0),
                    tag='C'
                )

            # Step 2: Add the comprehensive baseline curve topology to the active context
            add(c)

            # Step 3: Extract and isolate each primitive via its systemic tag, adding them with a vertical offset
            sub_curve_a = c.curve.tagged("A")
            add(sub_curve_a, offset=Pos(Z=1))

            sub_curve_b = c.curve.tagged("B")
            add(sub_curve_b, offset=Pos(Z=1))

            sub_curve_c = c.curve.tagged("C")
            add(sub_curve_c, offset=Pos(Z=1))

        self.assertPart(
            result.part,
            "62f2f207cedf25e125ca1d1b254281ecf12946700f5765e1f585fc9169b74a05",
            "test_build_curve_explicit_component_tagging_and_isolation",
            use_materials=False,
        )


    def test_curve_builder_tag_system_and_sub_curve_isolation(self):
        """Verify that tags can be assigned to curves and specific steps, and remain selectable for sub-curve filtering."""
        with BuildPart() as result:
            # Step 1: Construct a compound curve with specific tags assigned to steps and segments
            c = curve(
                curve.smooth(radius=10),
                curve.tag("A"),
                curve.step(10, angle=0, tag="B"),
                curve.step(10, angle=90, tag="C"),
                curve.step(10, angle=90, tag="D"),
            ).build()

            # Step 2: Add the main base curve to the active context
            add(c)

            # Step 3: Isolate sub-curves via tags and add them with distinct vertical offsets
            sub_curve_a = c.tagged("A")
            add(sub_curve_a, offset=Pos(Z=1))

            sub_curve_b = c.tagged("B")
            add(sub_curve_b, offset=Pos(Z=2))

            sub_curve_c = c.tagged("C")
            add(sub_curve_c, offset=Pos(Z=3))

            # Step 4: Perform fine-grained sub-curve filtering by removing boundary tags
            # Exclude the first point of the tagged segment
            sub_curve_c_no_first = c.tagged("C").untagged(Curve.TAG_POINT_FIRST)
            add(sub_curve_c_no_first, offset=Pos(Z=4))

            # Exclude the last point of the tagged segment
            sub_curve_c_no_last = c.tagged("C").untagged(Curve.TAG_POINT_LAST)
            add(sub_curve_c_no_last, offset=Pos(Z=5))

            # Exclude both the first and last boundary points
            sub_curve_c_body_only = c.tagged("C").untagged(Curve.TAG_POINT_FIRST, Curve.TAG_POINT_LAST)
            add(sub_curve_c_body_only, offset=Pos(Z=6))

            # Step 5: Isolate the final tagged section
            sub_curve_d = c.tagged("D")
            add(sub_curve_d, offset=Pos(Z=7))

        self.assertPart(
            result.part,
            "42a4589a1f2ac1601fd78717761cfdac3744d68d787a1290a67b4279d9dec10f",
            "test_curve_builder_tag_system_and_sub_curve_isolation",
            use_materials=False,
        )

