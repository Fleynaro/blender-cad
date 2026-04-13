from blender_cad import *
from tests.test_base import BaseCADTest

class TestPrimitivesAndBooleans(BaseCADTest):
    """
    Category: Basic Geometry
    Tests for primitive creation and fundamental boolean operations (ADD, SUBTRACT).
    """

    def test_basic_box_creation(self):
        """Verify that a single Box primitive is correctly generated."""
        with BuildPart() as result:
            Box(10, 10, 1)
        
        self.assertPart(result.part, "06b77e4b2a6d9b71be79382568a36084db23c7685de17fb2507c57dc3946b6b6", "test_basic_box_creation")

    def test_primitive_subtraction(self):
        """Verify that subtracting a Sphere from a Box produces the correct geometry."""
        with BuildPart() as result:
            Box(10, 10, 1)
            Sphere(3, mode=Mode.SUBTRACT)
        
        self.assertPart(result.part, "e1db0179bdc54097dda347ba015900dc8d835724b6f953362db7c43a90d2a400", "test_primitive_subtraction")