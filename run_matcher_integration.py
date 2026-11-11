"""Integration tests: matcher with adjudicator seam."""

import json
import os
import sys
import tempfile
from pathlib import Path

# Plan 003 (2026-08-03): redirect test contamination to a per-test log file.
# The integration test uses the real RuleMatcher which writes to the
# production verdict log; this env var keeps the test writes out of the
# forensic log at ~/.hermes/logs/rule-injector-adjudicator-verdicts.jsonl.
_TEST_LOG_DIR = Path(tempfile.mkdtemp(prefix="rule-injector-test-"))
os.environ.setdefault("RULE_INJECTOR_TEST_LOG_PATH", str(_TEST_LOG_DIR / "verdicts.jsonl"))

sys.path.insert(0, str(Path(__file__).parent))

from matchers import RuleMatcher  # noqa: E402
from rules import RuleStore  # noqa: E402
from adjudicator import (  # noqa: E402
    AdjudicationConfig,
    AdjudicationState,
    Adjudicator,
    PHASE1_MODEL,
    PHASE1_PROVIDER,
)


class ScriptedCall:
    def __init__(self, responses=None, raise_exc=None):
        self.responses = list(responses or [])
        self.raise_exc = raise_exc
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        if self.raise_exc is not None:
            raise self.raise_exc
        if not self.responses:
            return "{}"
        return self.responses.pop(0)


RULES = [
    {
        "id": "R001",
        "name": "Root Causes",
        "enabled": True,
        "hook": "pre_llm_call",
        "priority": 1,
        "trigger": {
            "type": "keyword",
            "scope": "latest_user_message",
            "keywords": ["build"],
        },
        "inject": {"context": "[root-causes] Build real, no stubs."},
    },
    {
        "id": "R002",
        "name": "No Placeholders",
        "enabled": True,
        "hook": "pre_llm_call",
        "priority": 2,
        "trigger": {
            "type": "keyword",
            "scope": "latest_user_message",
            "keywords": ["build"],
        },
        "inject": {"context": "[no-placeholders] No stub data."},
    },
    {
        "id": "R003",
        "name": "Pipeline Router",
        "enabled": True,
        "hook": "pre_llm_call",
        "priority": 3,
        "trigger": {
            "type": "keyword",
            "scope": "latest_user_message",
            "keywords": ["build"],
        },
        "inject": {"context": "[pipeline] Load skills first."},
    },
]


def _seed_rules(tmpdir: Path) -> RuleStore:
    import json as _json
    (tmpdir / "rules.json").write_text(_json.dumps({"version": 1, "rules": RULES}))
    return RuleStore(str(tmpdir / "rules.json"))


def _kwargs(user_message="please build a thing", session_id="s1"):
    return {
        "user_message": user_message,
        "conversation_history": [],
        "session_id": session_id,
    }


def _run_matcher(matcher, kwargs):
    result = matcher.run_pre_llm_call(kwargs)
    return result.get("context", "")


def test_matcher_without_adjudicator_is_baseline(tmp_path):
    store = _seed_rules(tmp_path)
    matcher = RuleMatcher(store)  # no adjudicator
    ctx = _run_matcher(matcher, _kwargs())
    assert "[root-causes]" in ctx
    assert "[no-placeholders]" in ctx
    assert "[pipeline]" in ctx


def test_matcher_disabled_adjudicator_is_baseline(tmp_path):
    store = _seed_rules(tmp_path)
    cfg = AdjudicationConfig(enabled=False, mode="review", provider=PHASE1_PROVIDER, model=PHASE1_MODEL)
    adj = Adjudicator(cfg, AdjudicationState(trace_retention=10), ScriptedCall())
    matcher = RuleMatcher(store, adjudicator=adj)
    ctx = _run_matcher(matcher, _kwargs())
    assert "[root-causes]" in ctx
    assert "[no-placeholders]" in ctx
    assert "[pipeline]" in ctx


def test_matcher_keep_all_is_baseline(tmp_path):
    store = _seed_rules(tmp_path)
    call = ScriptedCall(
        responses=[json.dumps({"keep": ["R001", "R002", "R003"], "drop": [], "reason": "ok"})]
    )
    cfg = AdjudicationConfig(enabled=True, mode="review", provider=PHASE1_PROVIDER, model=PHASE1_MODEL)
    adj = Adjudicator(cfg, AdjudicationState(trace_retention=10), call)
    matcher = RuleMatcher(store, adjudicator=adj)
    ctx = _run_matcher(matcher, _kwargs())
    assert "[root-causes]" in ctx
    assert "[no-placeholders]" in ctx
    assert "[pipeline]" in ctx
    assert call.calls  # adjudicator was called


def test_matcher_drops_filtered_rule(tmp_path):
    store = _seed_rules(tmp_path)
    call = ScriptedCall(
        responses=[json.dumps({
            "keep": ["R001", "R003"],
            "drop": ["R002"],
            "reason": "no placeholders is redundant here",
        })]
    )
    cfg = AdjudicationConfig(enabled=True, mode="review", provider=PHASE1_PROVIDER, model=PHASE1_MODEL)
    adj = Adjudicator(cfg, AdjudicationState(trace_retention=10), call)
    matcher = RuleMatcher(store, adjudicator=adj)
    ctx = _run_matcher(matcher, _kwargs())
    assert "[root-causes]" in ctx
    assert "[pipeline]" in ctx
    assert "[no-placeholders]" not in ctx


def test_matcher_adjudicator_failure_falls_back(tmp_path):
    store = _seed_rules(tmp_path)
    call = ScriptedCall(raise_exc=TimeoutError("upstream slow"))
    cfg = AdjudicationConfig(enabled=True, mode="review", provider=PHASE1_PROVIDER, model=PHASE1_MODEL)
    adj = Adjudicator(cfg, AdjudicationState(trace_retention=10), call)
    matcher = RuleMatcher(store, adjudicator=adj)
    ctx = _run_matcher(matcher, _kwargs())
    assert "[root-causes]" in ctx
    assert "[no-placeholders]" in ctx
    assert "[pipeline]" in ctx


def test_matcher_records_trace_on_review_call(tmp_path):
    store = _seed_rules(tmp_path)
    call = ScriptedCall(
        responses=[json.dumps({
            "keep": ["R001", "R002", "R003"],
            "drop": [],
            "reason": "ok",
        })]
    )
    state = AdjudicationState(trace_retention=10)
    cfg = AdjudicationConfig(enabled=True, mode="review", provider=PHASE1_PROVIDER, model=PHASE1_MODEL)
    adj = Adjudicator(cfg, state, call)
    matcher = RuleMatcher(store, adjudicator=adj)
    _run_matcher(matcher, _kwargs(session_id="sess-XYZ"))
    snap = state.snapshot()
    assert snap["traces"], "trace not recorded"
    last = snap["traces"][-1]
    assert last["session_id"] == "sess-XYZ"
    assert last["candidates_in"] == 3
    assert last["candidates_kept"] == 3
    assert last["fallback"] is False


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_matcher_without_adjudicator_is_baseline(tmp)
        print("PASS matcher without adjudicator is baseline")
        test_matcher_disabled_adjudicator_is_baseline(tmp)
        print("PASS matcher disabled adjudicator is baseline")
        test_matcher_keep_all_is_baseline(tmp)
        print("PASS matcher keep-all is baseline")
        test_matcher_drops_filtered_rule(tmp)
        print("PASS matcher drops filtered rule")
        test_matcher_adjudicator_failure_falls_back(tmp)
        print("PASS matcher adjudicator failure falls back")
        test_matcher_records_trace_on_review_call(tmp)
        print("PASS matcher records trace on review call")
    print()
    print("ALL PASS")


if __name__ == "__main__":
    main()
