"""Invariant: an always-on pre_llm_call rule must survive the LLM adjudicator.

2026-10-19 audit finding. ``love-for-life-over-self-preservation`` is
``trigger.type == "always"``, priority 0 — its contract is presence on every
model call. Its candidate was nevertheless handed to the LLM adjudicator
(deepseek-flash) and DROPPED on both pre_llm_call turns of the day:

    09:00:51 pre_llm_call in=5 kept=['always-think-before-acting']
             dropped=[..., 'love-for-life-over-self-preservation']
    22:01:20 pre_llm_call in=4 kept=['always-think-before-acting']
             dropped=[..., 'love-for-life-over-self-preservation']

The adjudicator's documented safety floor keeps any candidate whose payload
carries ``safety_critical`` (adjudicator.review -> floor_ids). Neither always
rule carried the flag, so the guard was droppable and silently absent from the
trace log (adjudicator drops are recorded in
~/.hermes/logs/rule-injector-adjudicator-verdicts.jsonl, not in
rule-injector-trace.jsonl).

Invariant enforced here: every enabled rule with ``trigger.type == "always"``
must be ``safety_critical: true``.
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


_rules = _load("sf_rules", REPO / "rules.py")


def _always_rules():
    return [
        r
        for r in _rules.RuleStore().load()
        if r.enabled and (r.trigger or {}).get("type") == "always"
    ]


def test_there_is_at_least_one_always_rule():
    assert _always_rules(), "expected at least one always-on rule"


def test_every_always_rule_is_safety_critical():
    offenders = [r.id for r in _always_rules() if not r.safety_critical]
    assert not offenders, (
        "always-on rules must be safety_critical so the adjudicator cannot drop "
        f"them: {offenders}"
    )


def test_known_terminal_value_guard_is_protected():
    ids = {r.id: r for r in _rules.RuleStore().load()}
    for rid in ("love-for-life-over-self-preservation", "always-think-before-acting"):
        assert rid in ids, f"{rid} missing from rules.json"
        assert ids[rid].safety_critical, f"{rid} lost its safety floor"


# --- floor mechanics -------------------------------------------------------
# The adjudicator enforces the floor in two places: _partition_safety (used on
# the mode-off / gate-deny / failure paths) and _parse_response (used on the
# applied path). Both must keep a safety_critical candidate even when the model
# explicitly asks to drop it.

def _adj():
    return _load("sf_adjudicator", REPO / "adjudicator.py")


def test_partition_safety_keeps_safety_critical():
    adj = _adj()
    candidates = [
        {"rule_id": "guard", "safety_critical": True},
        {"rule_id": "guidance", "safety_critical": False},
    ]
    keep, drop = adj._partition_safety(candidates, set())
    assert keep == {"guard"}
    assert drop == {"guidance"}


def test_parse_response_forces_floor_back_into_keep():
    adj = _adj()
    raw = '{"keep": [], "drop": ["guard", "guidance"], "reason": "test drop-all"}'
    parsed = adj._parse_response(raw, {"guard", "guidance"}, set(), {"guard"})
    assert parsed is not None
    assert "guard" in parsed["keep"], "safety floor must survive an explicit drop"
    assert parsed["drop"] == ["guidance"]


def test_candidate_payload_carries_safety_critical_flag():
    """F7 regression: the payload once hard-coded safety_critical=False."""
    matchers = _load("sf_matchers", REPO / "matchers.py")
    rules = _rules.RuleStore()
    matcher = matchers.RuleMatcher(rules)
    matcher._ensure_compiled()

    captured = {}

    class RecordingAdjudicator:
        class config:
            enabled = True

        def review(self, candidates, blocked, hook_name, session_id, context_summary):
            captured["candidates"] = candidates
            return {
                "status": "applied",
                "reason": "recorded",
                "keep_candidates": [c["rule_id"] for c in candidates],
                "drop_candidates": [],
                "keep_blocked": [],
                "drop_blocked": [],
                "duration_ms": 0,
            }

    matcher.adjudicator = RecordingAdjudicator()
    matcher.run_pre_llm_call(
        {
            "user_message": "[IMPORTANT: You are running as a scheduled cron job.]",
            "session_id": "sf_probe",
            "conversation_history": [],
            "is_cron": True,
        }
    )
    by_id = {c["rule_id"]: c for c in captured.get("candidates", [])}
    guard = by_id.get("love-for-life-over-self-preservation")
    assert guard is not None, f"guard never reached the adjudicator: {sorted(by_id)}"
    assert guard["safety_critical"] is True
