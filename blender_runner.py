"""Run arbitrary project Python code inside Blender and expose its result to an agent.

This file is intentionally a thin Blender-MCP adapter, not a test runner. Blender
MCP executes this entry point, while the adapter captures the target script's
stdout/stderr and traceback so an agent receives one inspectable result. The
same mechanism can run diagnostics, feature smoke checks, or test runners.

Example from Blender MCP::

    import runpy, sys
    sys.argv = [
        "blender_runner.py",
        "--script",
        "test_runner.py",
        "--",
        "--pattern",
        "some_test*.py",
    ]
    runpy.run_path(
        r"C:\\Users\\Fleynaro\\Desktop\\models\\blender-cad\\blender_runner.py",
        run_name="__main__",
    )

Target paths are resolved relative to the project root (the directory containing
this file), unless an absolute path is supplied.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import runpy
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_target(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an arbitrary Python script through Blender MCP and capture its output."
    )
    parser.add_argument(
        "--script",
        required=True,
        help="Target script path, relative to the project root or absolute.",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed unchanged to the target script.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parser_output = io.StringIO()
    try:
        with contextlib.redirect_stderr(parser_output):
            args = _parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:
        # argparse writes the human-readable reason to stderr and raises
        # SystemExit with only the status code. Preserve both pieces of
        # information so Blender MCP receives an actionable error, not just 2.
        code = exc.code
        exit_code = code if isinstance(code, int) else (0 if code is None else 1)
        reason = parser_output.getvalue().strip()
        if reason:
            print(f"[blender_runner] argument_error: {reason}")
        if exit_code:
            print(f"[blender_runner] exit_code={exit_code}")
        return exit_code

    target = _resolve_target(args.script)

    if not target.is_file():
        print(f"[blender_runner] target script not found: {target}")
        return 2

    captured = io.StringIO()
    exit_code = 0
    original_argv = sys.argv[:]
    script_args = args.script_args
    # argparse.REMAINDER preserves the conventional wrapper/target delimiter.
    # The target script must receive only its own arguments, not that delimiter.
    if script_args[:1] == ["--"]:
        script_args = script_args[1:]

    try:
        sys.argv = [str(target), *script_args]
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            runpy.run_path(str(target), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        exit_code = code if isinstance(code, int) else (0 if code is None else 1)
        if code not in (None, 0):
            traceback.print_exc(file=captured)
    except BaseException:
        exit_code = 1
        traceback.print_exc(file=captured)
    finally:
        sys.argv = original_argv

    output = captured.getvalue()
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    print(f"[blender_runner] exit_code={exit_code}")
    return exit_code


if __name__ == "__main__":
    # Do not raise SystemExit here: Blender MCP commonly executes this file via
    # runpy, and propagating SystemExit can terminate the Blender connection.
    main()
