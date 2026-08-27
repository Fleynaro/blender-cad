# Agent Development Rule

## Global mandatory Python execution boundary

**This rule applies to every agent task and every Python script the agent executes: no exceptions.** The agent must create each ad-hoc Python script used to develop, inspect, validate, demonstrate, or smoke-test a change in the ignored project-root [`.scratch/`](../../.scratch/) directory, then execute it **only** inside Blender 5.0 through Blender MCP by invoking [`blender_runner.py`](../../blender_runner.py) with `--script .scratch/<script-name>.py`.

The agent must never execute ad-hoc Python from another location, paste executable Python directly into a terminal or Blender console, or invoke `python`, `py`, a virtual-environment interpreter, `pytest`, or another Python-bearing terminal command. Permanent tests remain under [`tests/`](../../tests/), but must also run only through Blender MCP and [`blender_runner.py`](../../blender_runner.py) via [`test_runner.py`](../../test_runner.py), as specified in [`testing.md`](testing.md). Remove every temporary [`.scratch/`](../../.scratch/) script and generated artifact before task completion.

## Required scratch-file template and observable diagnostic output

Every scratch file must follow this standard structure: import [`importlib`](https://docs.python.org/3/library/importlib.html) and [`sys`](https://docs.python.org/3/library/sys.html), reload [`blender_cad`](../../blender_cad/__init__.py) when it is already present in `sys.modules`, import the library with `from blender_cad import *`, call [`clear_scene()`](../../blender_cad/helpers.py:3) before building geometry, create the result inside [`BuildPart`](../../blender_cad/build_part.py), and finish with the mandatory scene display and hash output. The required template is:

```python
import importlib
import sys
if 'blender_cad' in sys.modules:
    importlib.reload(sys.modules['blender_cad'])
from blender_cad import *

clear_scene()

with BuildPart() as result:
    Box(2.0, 2.0, 2.0)
    result.part.mat = mat.red

result.part.show(name="TEST")
print("HASH = ", result.part.hash(use_materials=True))
```

Adapt the geometry and material assignment to the requested task, but preserve the template's reload, wildcard library import, scene clearing, `BuildPart` result pattern, final [`result.part.show(name="TEST")`](../../misc/test_blender_cad.py), and material-inclusive [`print("HASH = ", result.part.hash(use_materials=True))`](../../misc/test_blender_cad.py). These final calls are mandatory so the agent can inspect the visible scene object and verify the generated result from Blender MCP output. If the final build result has another variable name, use that variable consistently in both calls; preserve `name="TEST"` and [`use_materials=True`](../../blender_cad/part.py:60).

For example, when asked to create a red [`Part.Box`](../../blender_cad/primitives.py) using [`blender_cad`](../../blender_cad/), the agent must first write a purpose-named scratch script, such as [`.scratch/red_part_box_smoke.py`](../../.scratch/red_part_box_smoke.py), using the template above, and run that file through Blender MCP and [`blender_runner.py`](../../blender_runner.py). Writing the feature code without this scratch-file execution is not verification.

Use this rule for implementation work that executes project Python, changes Blender-dependent behavior, or creates, assembles, or modifies a 3D object using [`blender_cad`](../../blender_cad/). Use a short feedback loop and make the implementation observable to the agent; static review or an unexecuted script is not verification of Blender geometry.

[`testing.md`](testing.md) is the single mandatory procedure and acceptance authority. It defines the Blender MCP and [`blender_runner.py`](../../blender_runner.py) execution requirement, the prohibition on direct `python`, `py`, interpreter, and `pytest` execution, the [`.scratch/`](../../.scratch/) diagnostic lifecycle including [`clear_scene()`](../../blender_cad/helpers.py:3), the test-scope ladder, result reporting, reference-hash handling, and completion cleanup.
