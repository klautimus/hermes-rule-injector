"""Regression tests for scope-recursive-commands-off-windows-mount.

2026-10-19 audit finding: the rule's trigger was a flat conjunction over
``tool_args`` — (terminal) AND (contains "/mnt/c") AND (contains grep|find|rg|fd).
Any single shell line combining an /mnt/c path with a search tool in a
*different* pipeline segment matched, and the injected suffix then asserted
"your last terminal command searched the Windows mount", which was false.

The real case (morning briefing, 09:01:47 local, traced):

    df -h / /mnt/c 2>/dev/null; echo "=== compaction units ==="; \
      systemctl --user list-timers --all 2>/dev/null | grep -iE "compact|clean" | head -5

``df`` statfs'd the mount; ``grep`` filtered systemctl output. Nothing
recursed over /mnt/c.

The trigger is now a regex over tool_args that binds the search tool to the
path (grep -r/rg/find/fd ... /mnt/c), plus a ``cd .../mnt/c && grep -r`` form.
"""

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_matchers = _load("sr_matchers", REPO / "matchers.py")
_rules = _load("sr_rules", REPO / "rules.py")

RULE_ID = "scope-recursive-commands-off-windows-mount"

# The exact command that produced the 2026-10-19 false positive.
BRIEFING_FP_COMMAND = (
    'df -h / /mnt/c 2>/dev/null; echo "=== compaction units ==="; '
    'systemctl --user list-timers --all 2>/dev/null | grep -iE "compact|clean" | head -5; '
    'echo "=== crontab ==="; crontab -l 2>/dev/null | head -10'
)

SHOULD_MATCH = [
    'grep -rn "TODO" /mnt/c/Users/kevin/Documents',
    'grep -rIl foo /mnt/c',
    'grep --recursive foo /mnt/c/Users',
    'rg -n "pattern" /mnt/c/Users/kevin',
    'find /mnt/c/Users/kevin -name "*.jpg" | head -20',
    'fd -e jpg . /mnt/c/Users/kevin',
    'cd /mnt/c/Users/kevin && grep -rn foo .',
    'cd /mnt/c && rg foo',
]

SHOULD_NOT_MATCH = [
    BRIEFING_FP_COMMAND,
    'ls -la /mnt/c/Users/kevin | grep jpg',
    'grep -n foo README.md',
    'cd /home/kevin/.hermes && grep -rn foo .',
    'df -h /mnt/c',
]


def _compiled_rule():
    store = _rules.RuleStore()
    matcher = _matchers.RuleMatcher(store)
    matcher._ensure_compiled()
    return next(r for r in matcher._compiled if r.id == RULE_ID)


def _ctx(command):
    return _matchers.MatchContext(
        hook="transform_tool_result",
        tool_name="terminal",
        tool_result="(output)",
        tool_args={"command": command},
        conversation_history=[],
        skill_loaded_this_session=set(),
    )


def test_trigger_is_regex_bound_to_path():
    cr = _compiled_rule()
    types = [c.trigger.get("type") for c in cr._compiled_conditions]
    assert types == ["tool_name", "regex"], types


def test_recursive_searches_of_mount_match():
    cr = _compiled_rule()
    for cmd in SHOULD_MATCH:
        assert cr.matches(_ctx(cmd)), f"expected match: {cmd}"


def test_non_recursive_and_cross_segment_commands_do_not_match():
    cr = _compiled_rule()
    for cmd in SHOULD_NOT_MATCH:
        assert not cr.matches(_ctx(cmd)), f"unexpected match: {cmd}"


def test_briefing_false_positive_is_closed():
    cr = _compiled_rule()
    assert not cr.matches(_ctx(BRIEFING_FP_COMMAND))


def test_rule_still_fires_only_on_terminal():
    cr = _compiled_rule()
    ctx = _ctx('grep -rn "TODO" /mnt/c/Users/kevin')
    ctx.tool_name = "read_file"
    assert not cr.matches(ctx)
