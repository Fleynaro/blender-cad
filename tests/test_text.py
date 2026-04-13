from blender_cad import *
from tests.test_base import BaseCADTest

class TestTextGeometry(BaseCADTest):
    """
    Category: Text Geometry
    """

    def test_text_on_curve(self):
        """Verify text generation with mixed formatting and curve deformation."""
        import math
        with BuildPart() as result:
            # Create text with mixed styling and nested formatting
            text = Text(text="Hello" + " " + t("my", mat=mat.green) + " ", size=1)
            # Test bold and italic nesting
            text.text += t.b("wor" + t.i("ld", mat=mat.red))
            text.text += "!"
            
            text.align()
            text.extrude(0.1)

            # Create a semi-circle curve using a rule (parametric function)
            rule = lambda t: (0, math.sin(t - math.pi / 2) * 3, math.cos(t - math.pi / 2) * 3)
            curve = make_curve(rule, math.pi)
            
            # Deform text along the path
            text.put_on_curve(curve)
            
            # Adjust text position along curve and tilt it via rotation
            # X-location moves text along the curve path
            text.loc = Pos(X=curve.length() / 2) * Rot(X=30)
            
            # Join the resulting mesh to the part (triggers .part conversion internally)
            add(text, mode=Mode.JOIN)

        # Hash verification for geometry and material consistency
        self.assertPart(result.part, "0ec1f8c3f0abd9def680320fdf68a55e17a6676836683ae4b1ec0285d205d48b", "test_text_on_curve")