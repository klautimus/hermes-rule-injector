"""Profile-aware filesystem paths for the plugin's own data.

Hermes re-homes itself per profile: ``-p <profile>`` and the desktop app swap
``HERMES_HOME``, and a plugin that hardcodes ``~/.hermes`` writes into the
wrong profile. ``hermes_constants.get_hermes_home()`` resolves the active
home (including the context-local override), so every path the plugin writes
goes through here.

The fallbacks keep the module importable in tests and in standalone scripts,
where hermes_constants may not be on the path at all.
"""

from __future__ import annotations

import os
from pathlib import Path


def hermes_home() -> Path:
    """The active Hermes home directory."""
    try:
        from hermes_constants import get_hermes_home  # type: ignore

        return Path(get_hermes_home())
    except Exception:  # noqa: BLE001 - import or resolution failure: fall back
        pass
    env = os.environ.get("HERMES_HOME", "").strip()
    if env:
        return Path(os.path.expanduser(os.path.expandvars(env)))
    return Path.home() / ".hermes"


def logs_dir() -> Path:
    return hermes_home() / "logs"


def trace_log() -> Path:
    """One line per fired rule."""
    return logs_dir() / "rule-injector-trace.jsonl"


def propose_audit_log() -> Path:
    """One line per rule-drafting event."""
    return logs_dir() / "rule-injector-propose-audit.jsonl"


def verdict_log() -> Path:
    """One line per adjudicator decision (48h retention)."""
    return logs_dir() / "rule-injector-adjudicator-verdicts.jsonl"
