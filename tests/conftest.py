# Pytest configuration for the rule-injector tests.
# Tests are loaded as top-level scripts (not as part of the plugin
# package), avoiding the relative-import errors that arise when pytest
# tries to import the plugin's __init__.py.
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
