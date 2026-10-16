"""Master Test Runner for Vision-Gesture Capstone Test Suite."""

import importlib.util
import os
import sys
import unittest

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)


def load_test_module(filepath: str):
    name = os.path.splitext(os.path.basename(filepath))[0]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_suite():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    test_files = [
        os.path.join(root_dir, "tests", "test_filters.py"),
        os.path.join(root_dir, "tests", "test_features.py"),
        os.path.join(root_dir, "tests", "test_controller.py"),
        os.path.join(root_dir, "tests", "test_integration.py"),
    ]

    for tf in test_files:
        if os.path.exists(tf):
            mod = load_test_module(tf)
            tests = loader.loadTestsFromModule(mod)
            suite.addTests(tests)
            print(f"Loaded {tests.countTestCases()} test cases from {os.path.basename(tf)}")

    print(f"\nRunning total of {suite.countTestCases()} automated test cases...\n" + "-" * 60)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("-" * 60)
    if result.wasSuccessful():
        print(f"✓ ALL {suite.countTestCases()} TESTS PASSED SUCCESSFULLY!")
        return 0
    else:
        print(f"✗ {len(result.failures)} FAILURES, {len(result.errors)} ERRORS")
        return 1


if __name__ == "__main__":
    sys.exit(run_suite())
