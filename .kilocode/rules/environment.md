# Environment Rule

## Blender Version

- The supported and authoritative Blender version for this project is **Blender 5.0**.
- Blender Python API usage must target **`bpy` from Blender 5.0**.
- Do not assume that examples written for Blender 3.x or 4.x are compatible with this project.
- When API compatibility is uncertain, run a minimal verification script inside Blender through Blender MCP before implementing the change.
- Python code that imports or uses `bpy` is not considered validated by running it in a normal system Python interpreter.

## Blender MCP Dependency

- [`.kilocode/mcp.json`](../mcp.json) pins `blender-mcp==1.8.7`; retain the exact version to keep the MCP execution boundary reproducible.
- Before updating that pin, verify the candidate server against Blender 5.0 and its installed addon/protocol, then update the addon's installed version to the same release before committing the config change.

## Project Paths

- Resolve project-relative paths from the repository root, preferably with `Path(__file__).resolve().parent`.
- Do not hard-code a developer-specific absolute path inside project scripts.
- Blender MCP may use an absolute path to [`blender_runner.py`](../../blender_runner.py), but target scripts and test paths should remain project-relative.
