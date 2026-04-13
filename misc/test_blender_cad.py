import math
import sys
import os
import importlib

dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if dir_path not in sys.path:
    sys.path.append(dir_path)

if 'blender_cad' in sys.modules:
    importlib.reload(sys.modules['blender_cad'])
from blender_cad import *

clear_scene()

with BuildPart() as result:
    text = Text(text = "Hello" + " " + t("my", mat=mat.green) + " ", size=1)
    text.text += t.b("wor" + t.i("ld", mat=mat.red))
    text.text += "!"
    text.align()
    text.extrude(0.1)

    rule = lambda t: (0, math.sin(t - math.pi / 2) * 3, math.cos(t - math.pi / 2) * 3)
    curve = make_curve(rule, math.pi)
    text.put_on_curve(curve)
    text.loc = Pos(X=curve.length() / 2) * Rot(X=30)
    add(text, mode=Mode.JOIN)

result.part.show(name='TEST')

print('HASH = ', result.part.hash(use_materials=False))