from blender_cad import *
from tests.test_base import BaseCADTest

class TestScopingAndContext(BaseCADTest):
    """
    Category: Context Management
    Tests how BuildPart handles internal scopes, nesting, and explicit part addition.
    """

    def test_explicit_add_operation(self):
        """Test adding an externally defined part into a new BuildPart context."""
        with BuildPart() as box_context:
            Box(10, 10, 1)
        
        with BuildPart() as result:
            add(box_context.part)
            Sphere(2, mode=Mode.SUBTRACT)
            
        self.assertPart(result.part, "a00e6d1208e3c70c0540e2ece193f09fa0f486324a8e4822fd2ee695bbcd193f", "test_explicit_add_operation")

    def test_nested_builders(self):
        """Test the behavior of nested BuildPart contexts creating a single unified part."""
        with BuildPart() as result:
            with BuildPart():
                Box(10, 10, 1)
            Sphere(3, mode=Mode.SUBTRACT)
            
        self.assertPart(result.part, "e1db0179bdc54097dda347ba015900dc8d835724b6f953362db7c43a90d2a400", "test_nested_builders")

    def test_multiple_result_contexts(self):
        """Verify that the result context can be reopened multiple times to modify the same part."""
        # 1. First context call: Create the base geometry
        with BuildPart() as result:
            Box(10, 10, 1)
        
        # 2. Second context call: Subtract geometry from the existing result
        with result:
            Sphere(3, mode=Mode.SUBTRACT)
            
        self.assertPart(result.part, "e1db0179bdc54097dda347ba015900dc8d835724b6f953362db7c43a90d2a400", "test_multiple_result_contexts")
