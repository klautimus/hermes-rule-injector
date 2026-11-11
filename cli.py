"""Plugin CLI: status, enable, disable, trace, dashboard.

Invoked as `hermes rule-injector-adjudicator <subcommand>` once the
plugin's register_cli is wired. The dispatcher is installed by the
plugin's optional setup hook; if the user has not enabled CLI commands
for the plugin, the same handlers can be reached by running
`python3 -m rule_injector.cli <subcommand>`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Allow running as a script with `python3 -m rule_injector.cli`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    # Bare-mode (script context): try the package path first, then
    # fall back to a local rules import when running from inside the
    # plugin directory (e.g. tests, ad-hoc scripts).
    try:
        from plugins.rule_injector import (  # type: ignore
            adjudicator_state as _plugin_state,
        )
    except Exception:
        class _LocalState:
            def get(self, key, default=None):
                return default
        _plugin_state = _LocalState()  # type: ignore
    try:
        from plugins.rule_injector.rules import RuleStore  # type: ignore
    except Exception:
        from rules import RuleStore  # type: ignore
    # adjudicator imports (only needed for the adjudicator CLI commands
    # which are not exercised by the bare-mode CLI test path).
    try:
        from plugins.rule_injector.adjudicator import (  # type: ignore
            AdjudicationConfig,
            AdjudicationState,
            Adjudicator,
            PHASE1_MODEL,
            PHASE1_PROVIDER,
        )
    except Exception:
        try:
            from adjudicator import (  # type: ignore
                AdjudicationConfig,
                AdjudicationState,
                Adjudicator,
                PHASE1_MODEL,
                PHASE1_PROVIDER,
            )
        except Exception:
            # Adjudicator is only needed for status/enable/disable/trace
            # commands. Bare-mode local tests of list-auto don't need it.
            AdjudicationConfig = None  # type: ignore
            AdjudicationState = None  # type: ignore
            Adjudicator = None  # type: ignore
            PHASE1_MODEL = ""  # type: ignore
            PHASE1_PROVIDER = ""  # type: ignore
else:
    from .adjudicator import (  # type: ignore
        AdjudicationConfig,
        AdjudicationState,
        Adjudicator,
        PHASE1_MODEL,
        PHASE1_PROVIDER,
    )
    from .rules import RuleStore  # type: ignore
    from . import adjudicator_state as _plugin_state

logger = logging.getLogger(__name__)


def cmd_status(_args: argparse.Namespace) -> int:
    state_ref = _plugin_state
    cfg = state_ref.get("config")
    st = state_ref.get("state")
    if cfg is None or st is None:
        print("adjudicator: not initialized")
        return 1
    snap = st.snapshot()
    print(json.dumps({
        "enabled": cfg.enabled,
        "provider": cfg.provider,
        "model": cfg.model,
        "mode": cfg.mode,
        "session_calls": snap.get("session_calls", {}),
        "hourly_calls": snap.get("hourly_calls", 0),
        "circuit_tripped": snap.get("circuit_tripped", False),
        "circuit_errors": snap.get("circuit_errors", 0),
        "traces_recorded": len(snap.get("traces", [])),
    }, indent=2))
    return 0


def cmd_enable(_args: argparse.Namespace) -> int:
    cfg = _plugin_state.get("config")
    if cfg is None:
        print("adjudicator: not initialized")
        return 1
    cfg.enabled = True
    print("adjudicator: ENABLED")
    return 0


def cmd_disable(_args: argparse.Namespace) -> int:
    cfg = _plugin_state.get("config")
    if cfg is None:
        print("adjudicator: not initialized")
        return 1
    cfg.enabled = False
    print("adjudicator: DISABLED")
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    st = _plugin_state.get("state")
    if st is None:
        print("adjudicator: not initialized")
        return 1
    snap = st.snapshot()
    traces = snap.get("traces", [])[-args.limit :]
    if args.json:
        print(json.dumps(traces, indent=2))
    else:
        for t in traces:
            print(
                f"{t['timestamp']:.0f} {t['hook']:<22} "
                f"{('FB' if t['fallback'] else 'OK'):<2} "
                f"kept={t['candidates_kept']}/{t['candidates_in']} "
                f"dropped={t['candidates_dropped']} "
                f"reason={t['reason']} {t['duration_ms']}ms"
            )
    return 0


def cmd_dashboard(_args: argparse.Namespace) -> int:
    from .dashboard import build_app  # type: ignore
    import uvicorn  # type: ignore

    app = build_app()
    uvicorn.run(app, host=_args.host, port=_args.port, log_level="info")
    return 0


# ----------------------------------------------------------------------
# Auto-created-rule management (Phase 5)
# ----------------------------------------------------------------------

def cmd_list_auto(args: argparse.Namespace) -> int:
    """Show every auto-created rule (id, name, source, created_at, hook)."""
    if RuleStore is None:
        print("rule-injector: RuleStore unavailable (import failed)")
        return 1
    try:
        store = RuleStore()
    except Exception as exc:
        print(f"rule-injector: cannot open rule store: {exc}")
        return 1
    auto_rules = store.list_auto()
    if not auto_rules:
        print("No auto-created rules.")
        return 0
    if args.json:
        print(json.dumps([r.to_dict() for r in auto_rules], indent=2))
        return 0
    print(f"Auto-created rules ({len(auto_rules)}):")
    for r in auto_rules:
        age = r.auto_created_at or "?"
        print(
            f"  {r.id:<40} {r.hook:<22} prio={r.priority:<3} "
            f"source={r.auto_created_source:<20} created={age}"
        )
        if args.verbose:
            body = (r.inject or {}).get("context") or (r.inject or {}).get("result_suffix") or ""
            print(f"    body: {body[:120]}{'...' if len(body) > 120 else ''}")
    return 0


def cmd_remove_all_auto(args: argparse.Namespace) -> int:
    """Remove every auto-created rule. Hand-curated rules are untouched."""
    if RuleStore is None:
        print("rule-injector: RuleStore unavailable (import failed)")
        return 1
    try:
        store = RuleStore()
    except Exception as exc:
        print(f"rule-injector: cannot open rule store: {exc}")
        return 1
    removed = store.count_auto()
    if removed == 0:
        print("No auto-created rules to remove.")
        return 0
    if not args.yes:
        print(
            f"About to remove {removed} auto-created rule(s). "
            "Hand-curated rules are NOT touched. Pass --yes to confirm."
        )
        return 1
    actual = store.remove_all_auto()
    print(f"Removed {actual} auto-created rule(s).")
    return 0


def cmd_count_auto(_args: argparse.Namespace) -> int:
    """Print the number of auto-created rules currently in the store."""
    if RuleStore is None:
        print("rule-injector: RuleStore unavailable (import failed)")
        return 1
    try:
        store = RuleStore()
    except Exception as exc:
        print(f"rule-injector: cannot open rule store: {exc}")
        return 1
    print(f"Auto-created rules: {store.count_auto()}")
    return 0


def register_cli(subparser) -> None:
    """Register the rule-injector-adjudicator subcommand."""
    parser = subparser.add_parser(
        "rule-injector-adjudicator",
        help="Rule Injector LLM adjudicator controls",
    )
    sub = parser.add_subparsers(dest="adjudicator_command", required=True)
    sub.add_parser("status", help="Show adjudicator status").set_defaults(func=cmd_status)
    sub.add_parser("enable", help="Enable review mode in memory").set_defaults(
        func=cmd_enable
    )
    sub.add_parser("disable", help="Disable review mode in memory").set_defaults(
        func=cmd_disable
    )
    trace = sub.add_parser("trace", help="Show recent adjudicator traces")
    trace.add_argument("--limit", type=int, default=20)
    trace.add_argument("--json", action="store_true")
    trace.set_defaults(func=cmd_trace)
    dash = sub.add_parser("dashboard", help="Run the standalone adjudicator dashboard")
    dash.add_argument("--host", default="127.0.0.1")
    dash.add_argument("--port", type=int, default=9123)
    dash.set_defaults(func=cmd_dashboard)

    # Auto-created rule management (Phase 5)
    list_auto = sub.add_parser(
        "list-auto",
        help="Show every auto-created rule (use 'remove-all-auto' to bulk delete)",
    )
    list_auto.add_argument("--json", action="store_true", help="Output as JSON")
    list_auto.add_argument(
        "--verbose", "-v", action="store_true",
        help="Also print the rule body (truncated to 120 chars)",
    )
    list_auto.set_defaults(func=cmd_list_auto)

    remove_all_auto = sub.add_parser(
        "remove-all-auto",
        help="Remove every auto-created rule. Hand-curated rules are NOT touched.",
    )
    remove_all_auto.add_argument(
        "--yes", "-y", action="store_true",
        help="Confirm removal (required unless zero auto-rules exist)",
    )
    remove_all_auto.set_defaults(func=cmd_remove_all_auto)

    count_auto = sub.add_parser(
        "count-auto",
        help="Print the number of auto-created rules currently in the store",
    )
    count_auto.set_defaults(func=cmd_count_auto)


def main(argv: list = None) -> int:
    # register_cli() expects an OUTER subparser (it calls
    # subparser.add_parser(...) internally). Feeding it the top-level
    # parser crashed with 'ArgumentParser' object has no attribute
    # 'add_parser' (2026-08-22 audit).
    parser = argparse.ArgumentParser(description="Rule Injector Adjudicator CLI")
    sub = parser.add_subparsers(dest="command_group")
    register_cli(sub)
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
