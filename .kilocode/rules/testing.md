# Testing Rule

## Mandatory Python execution

**All Python code in this project, including scripts, tests, diagnostics, list-only checks, and validation commands, must execute exclusively inside Blender 5.0 through Blender MCP.** Invoke it only through [`blender_runner.py`](../../blender_runner.py); do not run `python`, `py`, a virtual-environment interpreter, `pytest`, or any Python-bearing command directly in a terminal.

[`blender_runner.py`](../../blender_runner.py) runs a project-relative target script, captures stdout/stderr and traceback, and prints the target exit code. It is the only entry point for [`test_runner.py`](../../test_runner.py).

The `--` separator belongs between wrapper arguments and target-script arguments. The wrapper removes it before setting the target script's `sys.argv`.

Generic Blender MCP form:

```python
import runpy, sys

sys.argv = [
    "blender_runner.py",
    "--script",
    "<target-script>.py",
    "--",
    "<target-arguments>",
]
runpy.run_path(
    "<project-root>/blender_runner.py",
    run_name="__main__",
)
```

Use project-relative paths in `--script`, `--file`, and other project arguments. Do not put developer-specific absolute paths in agent rules or project scripts. The absolute path used only in Blender MCP's [`runpy.run_path()`](blender_runner.py:120) call is an execution-environment detail, not a project path embedded in source or rules.

## Temporary development diagnostics

For a Blender-dependent feature investigation or smoke check, create a minimal, purpose-named script in the ignored project-root [`.scratch/`](../../.scratch/) directory. This is temporary diagnostic code, not a committed test and not a replacement for a focused test in [`tests/`](../../tests/).

Scratch scripts that create or inspect geometry must import [`clear_scene()`](../../blender_cad/helpers.py:3) and call it near the beginning, before creating the requested object. This makes each run deterministic and prevents objects left in the Blender session from affecting the result. Omit it only when the diagnostic explicitly tests interaction with an existing scene, and document that exception in the script.

1. Create only the smallest script that reproduces or observes the behavior, for example [`.scratch/selector_smoke.py`](../../.scratch/selector_smoke.py).
2. Run it through Blender MCP using [`blender_runner.py`](../../blender_runner.py) with `--script .scratch/<script-name>.py`.
3. Inspect its stdout, stderr, traceback, scene observations, and exit code; run the same script before and after the implementation change when comparing behavior.
4. Convert the verified behavior into a focused, permanent test under [`tests/`](../../tests/) and run it through [`test_runner.py`](../../test_runner.py).
5. Delete all scripts and generated artifacts from [`.scratch/`](../../.scratch/) before completing the feature. The directory must be empty at completion.

Example diagnostic invocation, executed as the shown block through Blender MCP:

```python
import runpy, sys

sys.argv = [
    "blender_runner.py", "--script", ".scratch/selector_smoke.py",
]
runpy.run_path("<project-root>/blender_runner.py", run_name="__main__")
```

## Test commands through Blender MCP

In each example, execute the shown `runpy` block through Blender MCP. Replace only the arguments after `--` when changing the test scope.

### One test

```python
import runpy, sys

sys.argv = [
    "blender_runner.py", "--script", "test_runner.py", "--",
    "--test", "test_context_management.TestScopingAndContext.test_nested_builders",
]
runpy.run_path("<project-root>/blender_runner.py", run_name="__main__")
```

### Several tests from different files

```python
sys.argv = [
    "blender_runner.py", "--script", "test_runner.py", "--",
    "--test", "test_primitives_booleans.TestPrimitivesAndBooleans.test_basic_box_creation",
    "--test", "test_context_management.TestScopingAndContext.test_nested_builders",
]
```

### List tests without running them

```python
sys.argv = [
    "blender_runner.py", "--script", "test_runner.py", "--",
    "--pattern", "does_not_exist_*.py", "--list",
]
```

### Run one complete file

```python
sys.argv = [
    "blender_runner.py", "--script", "test_runner.py", "--",
    "--file", "tests/test_primitives_booleans.py",
]
```

### Run several complete files

```python
sys.argv = [
    "blender_runner.py", "--script", "test_runner.py", "--",
    "--file", "tests/test_primitives_booleans.py", "tests/test_part.py",
]
```

### Path-like test selector

```python
sys.argv = [
    "blender_runner.py", "--script", "test_runner.py", "--",
    "--test", "tests/test_primitives_booleans.py:TestPrimitivesAndBooleans.test_basic_box_creation",
    "--list",
]
```

`test_runner.py` also supports `--pattern "test_chain*.py"` and, without a selection option, full discovery.

## Scope and result handling

Use the smallest meaningful scope:

1. Run a minimal diagnostic script from [`.scratch/`](../../.scratch/) through [`blender_runner.py`](../../blender_runner.py).
2. Run the changed test method or methods.
3. Run the containing file.
4. Run related files only when justified.
5. Run the full suite only when the change is broad or a full run is explicitly justified.
6. Remove the temporary diagnostic script and confirm [`.scratch/`](../../.scratch/) is empty before completing the feature.

Inspect stdout, stderr, traceback, selected count, test count, failures, errors, skips, and exit code. A failing test must return a non-zero exit code; do not update reference hashes unless intentionally using `--update-hashes` and documenting why.
