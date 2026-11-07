# Project Instructions

## Blender and project paths

- Blender **5.0** is the supported version. Target its `bpy` API; do not assume Blender 3.x or 4.x examples are compatible.
- When API compatibility is uncertain, verify with a minimal script inside Blender 5.0 through Blender MCP. A normal Python interpreter does not validate `bpy` code.
- Resolve project paths from the repository root and keep them relative. Do not put developer-specific absolute paths in project scripts or these instructions. Blender MCP may need an absolute path only for the `blender_runner.py` wrapper; its target paths stay project-relative.

## Python, Blender MCP, and tests

- Run **all project Python**—including diagnostics, scripts, tests, and list-only checks—only inside Blender 5.0 through Blender MCP and `blender_runner.py`. Never run `python`, `py`, a virtual-environment interpreter, `pytest`, or another Python-bearing terminal command; do not paste executable Python into a terminal or Blender console.
- Put every ad hoc Python script in the ignored project-root `.scratch/` directory and run it through `blender_runner.py --script .scratch/<name>.py` via Blender MCP. Keep paths passed to the wrapper project-relative. In an MCP `runpy` call, place `--` between wrapper arguments and target-script arguments.
- Scratch scripts must follow the project diagnostic pattern: import `importlib` and `sys`, reload `blender_cad` if it is already loaded, use `from blender_cad import *`, call `clear_scene()` before building geometry, and create results inside `BuildPart`. Finish with `result.part.show(name="TEST")` and `print("HASH = ", result.part.hash(use_materials=True))`. Adapt the result variable consistently. Omit `clear_scene()` only when the diagnostic explicitly tests an existing scene, and document why.
- Permanent tests belong in `tests/` and must run through `test_runner.py`, itself invoked by `blender_runner.py` through Blender MCP. Use the smallest useful scope: changed test, containing file, related files, then the full suite only for broad changes or when justified.
- Inspect stdout, stderr, traceback, scene output, selected/test counts, failures, skips, and exit code. A failing test must return a non-zero code. Update reference hashes only intentionally and explain why. Convert verified behavior into focused permanent tests.

## Blender MCP dependency

- Keep `blender-mcp==1.8.7` aligned in `.kilo/kilo.jsonc` and `.codex/config.toml`.
- Before changing the version, verify the candidate server with Blender 5.0 and its installed add-on/protocol, then update the add-on to the same release.

## Documentation

- For every project change, check the relevant `README.md` and `docs/` content for behavior, APIs, configuration, workflows, tests, and examples. Update affected documentation in the same change.
- Before finishing, review the diff for stale names, inaccurate examples, and broken relative links. If no separate documentation update is needed, state why there is no user-facing or contributor-facing impact.
