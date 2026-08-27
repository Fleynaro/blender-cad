---
description: "Follow the Blender-dependent feature development and verification workflow"
agent: "code"
subtask: false
---

# Feature Development Workflow

Use this workflow for Blender-dependent feature work. [`testing.md`](../rules/testing.md) is the authority for execution commands, the diagnostic lifecycle, scope selection, acceptance criteria, and completion cleanup. [`environment.md`](../rules/environment.md) is the authority for Blender 5.0, `bpy`, and project-path constraints.

## Phases

1. **Understand:** Inspect the relevant implementation and identify the smallest reproducible scenario.
2. **Validate the baseline:** Create and run the required minimal diagnostic through Blender MCP; follow the [`.scratch/`](../../.scratch/) lifecycle in [`testing.md`](../rules/testing.md).
3. **Implement:** Make the smallest coherent change while complying with [`environment.md`](../rules/environment.md).
4. **Verify behavior:** Re-run the diagnostic and compare its observations before and after the change.
5. **Add focused coverage:** Convert the verified behavior into a permanent focused test in [`tests/`](../../tests/).
6. **Expand test scope deliberately:** Progress from the focused test through justified broader scopes, following [`testing.md`](../rules/testing.md); do not change reference hashes unless the expected output change is intentional and documented.
7. **Close out:** Remove diagnostics and generated artifacts, confirm [`.scratch/`](../../.scratch/) is empty, and report the executed scope and results as required by [`testing.md`](../rules/testing.md).
