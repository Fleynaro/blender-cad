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
    ml(
                style(
                    width=5,
                    height=5,
                    border_radius="-30%",
                    background_mat=mat.blue,
                    border_mat=mat.red,
                    border_width=0.5,
                    extrude=1,
                ),
                rl.inside().on_each(),
                # Step 2: Define centered child node constrained to remain a square (height bound to width)
                ml(
                    style.absolute_center(),
                    style(
                        width=ml.dof_p(),
                        height=lambda n: n.width,  # Maintain 1:1 square aspect ratio based on width DOF
                        background_mat=mat.green,
                        extrude=-0.5,
                    ),
                    # Size rule attempts to expand dimension towards 100 until constrained by parent boundary
                    rl.grow(),
                ),
            ).build()

result.part.show(name='TEST')

print('HASH = ', result.part.hash(use_materials=True))