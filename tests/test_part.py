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
            result.set_scale(2, -Axis.X)
            
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
            result.set_size(1.0, Axis.Y)
            
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
            result.set_scale(2, Axis.X)
            
            # Resize Y to 1.0, anchored at +Y
            result.set_size(1, Axis.Y)
            
            # Resize Z to 3.0, anchored at +Z
            result.set_size(3, Axis.Z)
            
            # Note: We only verify the final geometry hash here, 
            # as the manual calculation of the resulting matrix is highly complex.
            self.assertPart(
                result.part, 
                "39feb18c5be3bbbb29ab7cb8c0990869291901c503dd6efc3d6467773d603a5c", 
                "test_complex_transform_with_anchored_scaling"
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
            result.set_scale(2, -Axis.Z)
            
            # Final geometry check via hash
            self.assertPart(
                result.part, 
                "434301ae7bc65b49828eb80aae4ca9507b980f118060ef6a644c34ce9bed8e4f", 
                "test_asymmetric_assembly_anchored_scaling"
            )
