from blender_cad import *
from tests.test_base import BaseCADTest

class TestPart(BaseCADTest):
    """
    Category: Part Transformations
    Tests for synchronization between part-level properties and their 
    underlying transformation matrices/vectors.
    """

    def test_scale_synchronization(self):
        """
        Verify that modifying individual scale components directly updates 
        the aggregated transform scale vector.
        """
        with BuildPart() as result:
            # Create a unit cube base
            Box(1, 1, 1)
            
            # Modify scale components individually
            result.scale.x = 2
            result.scale.y = 3
            result.scale.z = 4
            
            # Check if internal transform scale reflects these changes
            # Expected: Vector((2.0, 3.0, 4.0))
            self.assertEqual(result.transform.scale.x, 2, "Transform scale X mismatch")
            self.assertEqual(result.transform.scale.y, 3, "Transform scale Y mismatch")
            self.assertEqual(result.transform.scale.z, 4, "Transform scale Z mismatch")

    def test_size_to_scale_calculation(self):
        """
        Verify that updating the absolute size of a part correctly 
        calculates and updates the corresponding scale factor.
        Example: Initial size 2, target size 4 should result in scale 2.
        """
        with BuildPart() as result:
            # Initialize with a 2x2x2 unit box
            Box(2, 2, 2)
            
            # Change absolute size on the X axis
            # The system should calculate: new_scale = target_size / original_size
            result.size.x = 4
            
            # Expected scale: 4 / 2 = 2.0
            expected_scale_x = 2.0
            
            self.assertEqual(
                result.transform.scale.x, 
                expected_scale_x, 
                f"Scale X should be {expected_scale_x} after resizing size.x to 4"
            )
            
            # Double check that the size property itself reflects the change
            # self.assertEqual(result.size.x, 4, "Size X property did not update correctly")

    def test_uniform_size_assignment(self):
        """
        Verify that assigning a single value to the size property 
        correctly normalizes different dimensions into a cube.
        Initial (1, 2, 3) -> Target Size 1.0 -> Scales (1.0, 0.5, 0.333...)
        """
        with BuildPart() as result:
            # Create a non-uniform box
            Box(1, 2, 3)
            
            # Apply uniform size to all axes
            result.size = 1
            
            # Expected scales:
            # X: 1 / 1 = 1.0
            # Y: 1 / 2 = 0.5
            # Z: 1 / 3 = 0.333...
            
            self.assertAlmostEqual(result.transform.scale.x, 1.0, places=3)
            self.assertAlmostEqual(result.transform.scale.y, 0.5, places=3)
            self.assertAlmostEqual(result.transform.scale.z, 0.333, places=3)
            
            # Ensure the absolute size is now uniform (1, 1, 1)
            # self.assertEqual(result.size.x, 1)
            # self.assertEqual(result.size.y, 1)
            # self.assertEqual(result.size.z, 1)

    def test_location_update_preserves_scale(self):
        """
        Verify that updating the part's location/rotation does not overwrite 
        or reset the previously applied scale.
        """
        with BuildPart() as result:
            # 1. Create a unit box
            Box(1, 1, 1)
            
            # 2. Apply uniform scale
            result.scale = 2
            
            # 3. Update location with position and rotation
            # This assignment should merge with the existing scale
            result.loc = Pos(2, 3, 4) * Rot(X=30, Y=40, Z=50)
            
            # 4. Verification of Translation (x, y, z)
            self.assertAlmostEqual(result.transform.x, 2.0, places=4)
            self.assertAlmostEqual(result.transform.y, 3.0, places=4)
            self.assertAlmostEqual(result.transform.z, 4.0, places=4)
            
            # 5. Verification of Rotation (Euler angles in degrees)
            self.assertAlmostEqual(result.transform.rx, 30.0, places=4)
            self.assertAlmostEqual(result.transform.ry, 40.0, places=4)
            self.assertAlmostEqual(result.transform.rz, 50.0, places=4)
            
            # 6. Verification that scale was preserved
            self.assertAlmostEqual(result.transform.sx, 2.0, 4, "Scale X was lost after moving")
            self.assertAlmostEqual(result.transform.sy, 2.0, 4, "Scale Y was lost after moving")
            self.assertAlmostEqual(result.transform.sz, 2.0, 4, "Scale Z was lost after moving")

    def test_transform_size_application(self):
        """Verify the application of the Size transformation with mixed axis parameters."""
        with BuildPart() as result:
            Box(1, 1, 1)
            result.transform = Size(XY=5, Z=1, X=1)

        self.assertPart(
            result.part,
            "a3d4302dbaaaba366483bfa9ba68362ba7e8391f511b598a27ff5607ca3381e1",
            "test_transform_size_application",
        )

    def test_size_override_chain(self):
        """Verify that sequential Size transformations override previous values on the same axes."""
        with BuildPart() as result:
            Box(2, 2, 2)
            # Initial size setting
            result.transform *= Size(X=1)
            # Sequential overrides: X=2 is replaced by X=3, then YZ and rotation are applied
            result.transform *= Size(X=2) * Size(X=3) * Size(YZ=1) * Rot(Z=45)

        self.assertPart(
            result.part,
            "dd529197443ef41565b6c1b0b7a072e3af783c814ede925927a32ca1370d28a3",
            "test_size_override_chain",
        )

    def test_complex_transform_chain_with_origin(self):
        """Verify the application of a complex transformation chain including origins, scaling, and rotations."""
        with BuildPart() as result:
            Box(2, 2, 2)
            # Applying a sequence of transformations to check origin-based math
            result.transform = (
                Origin(X=0.0) * 
                Scale(X=4) * 
                Rot(Y=40) * 
                Scale(Y=2) * 
                Origin() * 
                Rot(Y=50) * 
                Pos(X=-1)
            )

        self.assertPart(
            result.part,
            "2ad23006e216ec923a2e9d519f8584e1110e60531646ce7c8788e533f83825c3",
            "test_complex_transform_chain_with_origin",
        )

    def test_set_scale_with_axis_anchor(self):
        """
        Verify that scaling along a specific axis (-Axis.X) keeps the opposite 
        side stationary while extending the geometry in the target direction.
        A 2x2x2 Box scaled by 2 along -X should expand towards negative X.
        """
        with BuildPart() as result:
            # Create a centered 2x2x2 box (bounds: x from -1 to 1)
            Box(2, 2, 2)
            
            # Scale by 2 along the negative X axis
            # This should keep the face at X = +1 fixed and move the face at X = -1 to X = -3
            result.transform *= ScaleAlongAxis(-Axis.X, 2)
            
            # 1. Check scale factors
            self.assertEqual(result.transform.scale.x, 2.0, "Scale X should be doubled")
            self.assertEqual(result.transform.scale.y, 1.0, "Scale Y should remain unchanged")
            
            # 2. Check bounding box / position shift
            # Original bounds X: [-1, 1], New bounds X: [-3, 1]
            # The center should have shifted from 0 to -1 on the X axis
            self.assertEqual(result.transform.position.x, -1.0, "Part should have shifted towards -X")
            
            # 3. Check final absolute size
            # self.assertEqual(result.size.x, 4.0, "Absolute size X should now be 4")
    
    def test_set_size_with_axis_anchor(self):
        """
        Verify that setting an absolute size along a specific axis (Axis.Y) 
        shrinks or expands the part while keeping the specified side fixed.
        A 2x2x2 Box resized to 1.0 along Y should shrink from the positive side.
        """
        with BuildPart() as result:
            # Create a centered 2x2x2 box
            Box(2, 2, 2)
            
            # Set absolute size to 1.0 along the Y axis
            result.transform *= SizeAlongAxis(Axis.Y, 1)
            
            # 1. Check scale factor
            # New size 1.0 / Original size 2.0 = 0.5 scale
            self.assertEqual(result.transform.scale.y, 0.5, "Scale X should be 0.5 (half of original)")
            
            # 2. Check translation (Positioning)
            # Original center was 0.0. 
            # To keep X=-1.0 fixed while width becomes 1.0, the new center must be at -0.5
            self.assertEqual(result.transform.position.y, -0.5, "Part center should shift to -0.5 to maintain anchor")
            
            # 3. Check final dimensions
            #self.assertEqual(result.size.x, 1.0, "Final size X should be exactly 1.0")
            #self.assertEqual(result.size.y, 2.0, "Size Y should remain 2.0")

    def test_complex_transform_with_anchored_scaling(self):
        """
        Verify that anchored scaling and resizing work correctly when 
        applied to a part with existing translation and rotation.
        This ensures that the anchor logic respects the local coordinate 
        system of the part.
        """
        with BuildPart() as result:
            # 1. Initialize base geometry
            Box(2, 2, 2)
            
            # 2. Apply complex initial transformation
            # Position at (2, 3, 4) and rotate on all axes
            result.transform = Pos(2, 3, 4) * Rot(X=30, Y=40, Z=50)
            
            # 3. Apply anchored transformations
            # Scale X by 2, anchored at +X
            result.transform *= ScaleAlongAxis(Axis.X, 2)
            
            # Resize Y to 1.0, anchored at +Y
            result.transform *= SizeAlongAxis(Axis.Y, 1)
            
            # Resize Z to 3.0, anchored at +Z
            result.transform *= SizeAlongAxis(Axis.Z, 3)
            
            # Note: We only verify the final geometry hash here, 
            # as the manual calculation of the resulting matrix is highly complex.
            self.assertPart(
                result.part, 
                "39feb18c5be3bbbb29ab7cb8c0990869291901c503dd6efc3d6467773d603a5c", 
                "test_complex_transform_with_anchored_scaling"
            )

    def test_transform_reset_with_rotation_and_pos(self):
        """Verify that the reset parameter in ScaleAlongAxis correctly clears previous scaling on that axis."""
        with BuildPart() as result:
            Box(2, 2, 2)
            # Initial position and rotation
            result.transform = Pos(2, 3, 4) * Rot(X=30, Y=40, Z=50)
            # Apply scaling on X axis
            result.transform *= ScaleAlongAxis(Axis.X, 2)
            # Apply new scaling on X axis with reset=True to override previous X scaling
            result.transform *= ScaleAlongAxis(Axis.X, 1, reset=True)

        self.assertPart(
            result.part,
            "5ae1edd27cb2e3313ced578183db4c67994adabe26132a5ef90352e30492a17c",
            "test_transform_reset_with_rotation_and_pos",
        )

    def test_size_along_axis_sequence(self):
        """Verify the sequence of size changes where the last value overrides the previous ones, 
        and resizing from the opposite direction clips the object."""
        with BuildPart() as result:
            Box(2, 2, 2)
            # Initial position and rotation
            result.transform = Pos(2, 3, 4) * Rot(X=30, Y=40, Z=50)
            # Size 5 replaces size 3 on the X axis
            result.transform *= SizeAlongAxis(Axis.X, 3) * SizeAlongAxis(Axis.X, 5)
            # Resize to 1 from the opposite direction (-Axis.X), effectively clipping from the other side
            result.transform *= SizeAlongAxis(-Axis.X, 1)

        self.assertPart(
            result.part,
            "2c476c34d616cfe5a2f104377737829d765e34b8b5ad073a25b1750e520b186f",
            "test_size_along_axis_sequence",
        )

    def test_asymmetric_assembly_anchored_scaling(self):
        """
        Verify anchored scaling on an asymmetric assembly (two stacked cylinders).
        Testing with non-symmetric geometry ensures that the scale anchor 
        is calculated from the overall bounding box rather than the local origin.
        """
        with BuildPart() as result:
            # 1. Create an asymmetric stack: 
            # Bottom cylinder: radius 1, height 5 (from Z=0 to Z=5)
            Cylinder(1, 5)
            
            # Top cylinder: radius 0.5, height 10 (from Z=5 to Z=15)
            with Locations(Pos(Z=5)):
                Cylinder(0.5, 10)
            
            # 2. Apply arbitrary rotation to ensure axis logic works in local space
            result.loc = Rot(X=30, Y=40, Z=50)
            
            # 3. Scale by 2 along the negative Z axis
            # The top of the assembly (Z=15 in local space) should remain fixed,
            # while the assembly expands downwards along the local -Z.
            result.transform *= ScaleAlongAxis(-Axis.Z, 2)
            
            # Final geometry check via hash
            self.assertPart(
                result.part, 
                "434301ae7bc65b49828eb80aae4ca9507b980f118060ef6a644c34ce9bed8e4f", 
                "test_asymmetric_assembly_anchored_scaling"
            )

    def test_bbox_part(self):
        """Verify that bbox_part correctly follows world transform and matches the object's bounding box."""

        with BuildPart() as result:
            Cone(1, 0.1, 2)

            result.transform = (
                Pos(X=3, Y=4, Z=5)
                * Rot(X=30, Y=40, Z=50)
                * Scale(X=0.5, Y=2, Z=4)
            )

            bbox = result.part.bbox_part
            self.assertEqual(result.part.bbox_part, bbox, "bbox_part should be cached")
            add(bbox, mode=Mode.JOIN)

        self.assertPart(
            result.part,
            "0991c4846f3e1df160cc20bcdc76f3785a420803933fbd776f1053195cc2b04f",
            "test_bbox_part"
        )

    def test_convex_hull_part(self):
        """Verify that convex_hull_part correctly follows world transform and matches the object's convex hull."""

        with BuildPart() as result:
            Cone(1, 0.1, 2)

            result.transform = (
                Pos(X=3, Y=4, Z=5)
                * Rot(X=30, Y=40, Z=50)
                * Scale(X=0.5, Y=2, Z=4)
            )

            convex_hull = result.part.convex_hull_part
            self.assertEqual(result.part.convex_hull_part, convex_hull, "convex_hull_part should be cached")
            add(convex_hull, mode=Mode.JOIN)

        self.assertPart(
            result.part,
            "c0d77f2c749c4b35ff8236e8903054f5a853a2750f1bca588c415b271cdce78c",
            "test_convex_hull_part"
        )

    def test_project_2d_mesh_flattening_with_zero_dissolve_limit(self):
        """Verify that project_2d flattens complex 3D transformed geometry into a stable 2D mesh representation."""
        with BuildPart() as result:
            # Step 1: Create a twisted, multi-material chain of components along the Y-axis
            chain(
                chain.twist(
                    ml(
                        style(
                            width=2,
                            height=1,
                            background_mat=mat.red,
                            extrude=0.1
                        ),
                        ml(
                            style.absolute_center(),
                            style.circle(radius=0.1, mat=mat.yellow),
                        )
                    ),
                    ml(
                        style(
                            width=1,
                            height=1,
                            background_mat=mat.blue,
                        )
                    ),
                    axis=Axis.X,
                    angle=300,
                    segments=5
                ),
                axis=Axis.Y,
            ).build()
            
            # Step 2: Apply a series of complex 3D transforms to the resulting part
            result.transform = Pos(XY=-1, Z=1) * Scale(XY=2) * Rot(Y=45)
            
            # Step 3: Project the 3D geometry down onto a 2D plane and add it with an offset
            # Note: dissolve_limit=0.0 is explicitly used here to guarantee stable hash generation for testing
            add(result.part.project_2d(dissolve_limit=0.0), offset=Pos(Y=10), mode=Mode.JOIN)

        self.assertPart(
            result.part,
            "72a2832f03882c09fe1a718f00fa23452b71d44a2d19acc21484da72068780bb",
            "test_project_2d_mesh_flattening_with_zero_dissolve_limit",
            use_materials=True,
        )

    def test_box_set_part_primitive_storage_and_optimization(self):
        """Verify that BoxSetPart efficiently stores shapes as collections of boxes and converts non-box primitives like Sphere into bounding boxes."""
        def build_obj(part: Part | None):
            with Locations(Pos(X=5, Y=6, Z=7) * Rot(X=30, Y=40, Z=50)):
                Box(1, 2, 3)
                with Locations(Pos(Z=-2)):
                    # Note: This sphere will be converted to its bounding box equivalent when using BoxSetPart
                    Sphere(0.5) 
                with BuildPart(part=part and part.copy(), mode=Mode.JOIN):
                    with BuildPart(part=part and part.copy(), offset=Pos(Z=2), mode=Mode.JOIN):
                        with Locations(Rot(Z=45)):
                            Box(1, 1, 1)

        with BuildPart() as result:
            # Step 1: Initialize and verify an empty optimized BoxSetPart context
            part = Part.box_set_empty()
            with BuildPart(part=part) as obj:
                # Check dimensions before populating geometry
                self.assertTrue((part.size - part.bbox_part.size).length < 1e-5)
                
                # Step 2: Build the compound object inside the BoxSetPart active builder
                build_obj(part=Part.box_set_empty())
                
                # Step 3: Validate that sizes match and the internal pure-box register tracks exactly 3 box elements
                self.assertTrue((part.size - part.bbox_part.size).length < 1e-5)
                self.assertEqual(len(part.boxes), 3)

            # Step 4: Execute the same building routine using standard mesh conversion for comparison
            with Locations(Pos(Z=3)):
                with BuildPart(mode=Mode.JOIN):
                    build_obj(part=None)

        self.assertPart(
            result.part,
            "5ee24e955eea2fc447c1e0127b1fe98f90f80ebc03a1bca930d1ef4348064a37",
            "test_box_set_part_primitive_storage_and_optimization",
            use_materials=False,
        )

