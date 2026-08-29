"""Dynamically select and run unittest tests for this Blender project."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TESTS_ROOT = PROJECT_ROOT / "tests"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file", nargs="+", help="Test files, relative to the project root."
    )
    parser.add_argument(
        "--test", action="append", help="Test id; repeat for multiple tests."
    )
    parser.add_argument(
        "--pattern", help="Discovery pattern, for example test_chain*.py."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List selected test ids without running them.",
    )
    parser.add_argument("--verbosity", type=int, default=2)
    parser.add_argument(
        "--update-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update expected hashes on failure (enabled by default).",
    )
    parser.add_argument(
        "--show-geometry-if-failed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show failed geometry in Blender (enabled by default).",
    )
    return parser.parse_args(argv)


def _clear_project_modules() -> None:
    prefixes = ("blender_cad", "test_", "tests")
    for name in list(sys.modules):
        if name.startswith(prefixes):
            del sys.modules[name]


def _configure_environment(args: argparse.Namespace) -> None:
    if args.update_hashes:
        os.environ["UPDATE_HASHES"] = "True"
    else:
        os.environ.pop("UPDATE_HASHES", None)
    if args.show_geometry_if_failed:
        os.environ["SHOW_GEOMETRY_IF_FAILED"] = "True"
    else:
        os.environ.pop("SHOW_GEOMETRY_IF_FAILED", None)


def _ensure_import_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _module_from_file(file_name: str) -> str:
    path = Path(file_name)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    try:
        relative = path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Test file must be inside project root: {file_name}") from exc
    if relative.parts[0] != "tests" or path.suffix != ".py":
        raise ValueError(f"Expected a Python file inside tests/: {file_name}")
    return ".".join(relative.with_suffix("").parts)


def _normalize_test_id(test_id: str) -> str:
    """Accept a path-like test id as well as unittest's dotted form."""
    value = test_id.replace("\\", "/")
    if value.endswith(".py") or value.startswith("tests/"):
        path_part, separator, member = value.partition(":")
        module = _module_from_file(path_part)
        return f"{module}.{member}" if separator else module
    return value


def _suite_from_args(args: argparse.Namespace) -> unittest.TestSuite:
    loader = unittest.TestLoader()
    selections = [
        args.test is not None,
        args.file is not None,
        args.pattern is not None,
    ]
    if sum(selections) > 1:
        raise ValueError("Use only one of --test, --file, or --pattern at a time")
    if args.test:
        suite = unittest.TestSuite()
        for test_id in args.test:
            suite.addTests(loader.loadTestsFromName(_normalize_test_id(test_id)))
        return suite
    if args.file:
        suite = unittest.TestSuite()
        for file_name in args.file:
            suite.addTests(
                loader.loadTestsFromModule(
                    importlib.import_module(_module_from_file(file_name))
                )
            )
        return suite
    return loader.discover(str(TESTS_ROOT), pattern=args.pattern or "test*.py")


def _iter_tests(suite: unittest.TestSuite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_tests(item)
        else:
            yield item


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _ensure_import_path()
    _clear_project_modules()
    _configure_environment(args)
    try:
        suite = _suite_from_args(args)
    except (ImportError, ValueError, AttributeError) as exc:
        print(f"[test_runner] selection error: {exc}", file=sys.stderr)
        return 2

    selected = list(_iter_tests(suite))
    print(f"[test_runner] selected={len(selected)}")
    if args.list:
        for test in selected:
            print(test.id())
        return 0

    result = unittest.TextTestRunner(verbosity=args.verbosity).run(suite)
    print(
        f"[test_runner] tests={result.testsRun} failures={len(result.failures)} "
        f"errors={len(result.errors)} skipped={len(result.skipped)}"
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
