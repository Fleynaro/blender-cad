---
description: "Follow the Blender-dependent feature development and verification workflow"
agent: "code"
subtask: false
---

# Feature Development Workflow

Use this workflow for Blender-dependent feature work. [`testing.md`](../rules/testing.md) is the authority for execution commands, the diagnostic lifecycle, scope selection, acceptance criteria, and completion cleanup. [`environment.md`](../rules/environment.md) is the authority for Blender 5.0, `bpy`, and project-path constraints.

## Input Routing

Interpret `$ARGUMENTS` before beginning diagnostics, changing files, or creating tests:

1. If `$ARGUMENTS` is exactly `#<number>` or a GitHub issue URL for `Fleynaro/blender-cad`, follow the **Issue-reference branch**.
2. If the first argument in `$ARGUMENTS` is exactly `--no-issue`, remove that flag and treat the remaining text as the work description. Follow the **No-issue branch**. An exact `--no-issue` uses the requested work from available context.
3. If `$ARGUMENTS` is non-empty and is anything else, treat it as a new work description. Follow the **New-description branch**.
4. If `$ARGUMENTS` is empty, determine whether the requested work is exempt under [`issue-first.md`](../rules/issue-first.md). For non-exempt work, require a valid issue reference in available context and follow the **Issue-reference branch**. Exempt work may continue to the phases below without an issue reference.

An explicit issue reference must always use the **Issue-reference branch**, including when the requested work would otherwise be exempt.

### New-description Branch

Apply the `/issue` preparation workflow to the supplied description exactly as defined in [`.kilo/command/issue.md`](issue.md), then terminate this command without implementation. Preserve that workflow's read-only research, duplicate detection, one-question-at-a-time clarification, draft requirements, and explicit later-message confirmation before publication. Do not silently create an issue. If an issue is optionally published after confirmation, report it and still stop; a separate `/feature #<number>` or `/feature <issue URL>` invocation is required before implementation.

### No-issue Branch

Use this branch only when the user explicitly supplies the `--no-issue` flag. This flag is the explicit opt-out from the issue-first requirement for this invocation. Treat the remaining description, or the requested work in available context when no description remains, as the implementation scope. Do not invoke `/issue`, search for or create a GitHub issue, or require an issue reference. Continue directly to the phases below, while still applying all other project rules, diagnostics, tests, and verification requirements.

### Issue-reference Branch

Accept only `#<number>` or an issue URL whose repository is exactly `Fleynaro/blender-cad`. Retrieve the referenced issue before starting diagnostics, changing files, or creating tests. Terminate this command if the reference is malformed, unreadable, cross-repository, or closed.

For an open issue, confirm that its scope matches the requested work in the invocation or available context. If the scope does not match, terminate this command and direct the request through `/issue` to prepare a new issue or revise the existing scope. Only after this validation succeeds may the workflow continue to the phases below.

## Phases

1. **Understand:** Inspect the relevant implementation and identify the smallest reproducible scenario.
2. **Validate the baseline:** Create and run the required minimal diagnostic through Blender MCP; follow the [`.scratch/`](../../.scratch/) lifecycle in [`testing.md`](../rules/testing.md).
3. **Implement:** Make the smallest coherent change while complying with [`environment.md`](../rules/environment.md).
4. **Verify behavior:** Re-run the diagnostic and compare its observations before and after the change.
5. **Add focused coverage:** Convert the verified behavior into a permanent focused test in [`tests/`](../../tests/).
6. **Expand test scope deliberately:** Progress from the focused test through justified broader scopes, following [`testing.md`](../rules/testing.md); do not change reference hashes unless the expected output change is intentional and documented.
7. **Close out:** Remove diagnostics and generated artifacts, confirm [`.scratch/`](../../.scratch/) is empty, and report the executed scope and results as required by [`testing.md`](../rules/testing.md).
