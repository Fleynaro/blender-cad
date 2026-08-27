"""Regression coverage for repeated in-process test-runner selections."""

import unittest

import test_runner


class TestRunnerIsolation(unittest.TestCase):
    def test_repeated_path_selected_test_uses_fresh_test_modules(self):
        test_id = (
            "tests/test_context_management.py:"
            "TestScopingAndContext.test_nested_builders"
        )

        self.assertEqual(test_runner.main(["--test", test_id]), 0)
        self.assertEqual(test_runner.main(["--test", test_id]), 0)
