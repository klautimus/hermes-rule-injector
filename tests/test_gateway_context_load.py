"""Gateway-context load regression guard (2026-08-22 audit).

The plugin MUST import successfully when loaded the way
``hermes_cli.plugins._load_directory_module`` loads it: as
``hermes_plugins.rule_injector`` via ``importlib.util.spec_from_file_location``
with ``submodule_search_locations``, where sys.path does NOT include the
plugin dir and no ``hermes_plugins`` package exists on disk.

Failure history:
- 2026-07-27..31: matchers.py absolute-import death (fixed, dual-mode).
- 2026-08-09..22: propose.py reintroduced bare absolute imports
  (``from rules import RuleStore``); ``__init__.py``'s try/except masked the
  real error as "No module named 'rules'" and EVERY rule stopped firing for
  13 days while auto-create could never run. The root-level tests did not
  catch it because pytest collects this plugin in bare-module context, where
  those imports happen to work.

This test replicates the production load context exactly.
"""

import importlib.util
import sys
import types
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
NS_PARENT = "hermes_plugins"


def _load_like_gateway():
    ns_pkg = types.ModuleType(NS_PARENT)
    ns_pkg.__path__ = []
    ns_pkg.__package__ = NS_PARENT
    sys.modules[NS_PARENT] = ns_pkg

    module_name = f"{NS_PARENT}.rule_injector"
    spec = importlib.util.spec_from_file_location(
        module_name,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.__package__ = module_name
    module.__path__ = [str(PLUGIN_DIR)]
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_plugin_loads_in_gateway_package_context():
    mod = _load_like_gateway()
    # Every symbol downstream consumers rely on must resolve.
    for attr in (
        "RuleStore",
        "RuleMatcher",
        "Adjudicator",
        "AdjudicationConfig",
        "AdjudicationState",
        "propose_rule",
        "extract_rule_draft",
        "RULE_INJECTION_REVIEW_PROMPT",
    ):
        assert hasattr(mod, attr), f"gateway-context load missing {attr}"


def test_background_review_resolver_finds_gateway_module():
    """agent/background_review._import_rule_injector_module() must resolve the
    module registered under hermes_plugins.<slug> — the bare
    'from rule_injector import ...' form only worked under pytest."""
    mod = _load_like_gateway()  # registers sys.modules['hermes_plugins.rule_injector']

    repo_root = PLUGIN_DIR.parent.parent  # ~/.hermes/hermes-agent
    bg_path = repo_root / "agent" / "background_review.py"
    if not bg_path.exists():
        # Plugin installed standalone (no sibling checkout): resolver's
        # canonical-path fallback covers this; nothing further to assert.
        return
    import importlib.util as _ilu

    bgspec = _ilu.spec_from_file_location("bg_review_under_test", bg_path)
    bgmod = _ilu.module_from_spec(bgspec)
    bgspec.loader.exec_module(bgmod)
    resolved = bgmod._import_rule_injector_module()
    assert resolved is mod or resolved is not None
    loaded = bgmod._load_rule_injector_propose()
    assert loaded is not None, (
        "_load_rule_injector_propose() returned None in gateway context — "
        "auto-create pipeline would silently no-op"
    )
    propose_rule, extract_rule_draft, prompt = loaded
    assert callable(propose_rule) and callable(extract_rule_draft)
    assert "USER CORRECTION" in prompt
