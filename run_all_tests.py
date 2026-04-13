import unittest
import sys
import os

# --- Configuration ---
UPDATE_HASHES = True
SHOW_GEOMETRY_IF_FAILED = True
FILES = 'test_*.py'
# --- Configuration ---

if UPDATE_HASHES:
    os.environ["UPDATE_HASHES"] = "True"
if SHOW_GEOMETRY_IF_FAILED:
    os.environ["SHOW_GEOMETRY_IF_FAILED"] = "True"

# Clear module cache before importing tests to ensure fresh code execution
pkg_name = "blender_cad"
test_file_prefix = "test_"
for name in list(sys.modules.keys()):
    if name.startswith(pkg_name) or name.startswith(test_file_prefix):
        del sys.modules[name]

# Ensure the current directory is in the system path for imports
dir_path = os.path.dirname(os.path.realpath(__file__))
if dir_path not in sys.path:
    sys.path.append(dir_path)

# Discover and load all tests from the 'tests' directory
loader = unittest.TestLoader()
suite = loader.discover(start_dir='tests', pattern=FILES)

# Initialize the runner and execute the test suite
runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)