"""Unit tests for the rule-injector adjudicator."""

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from adjudicator import (  # noqa: E402
    AdjudicationConfig,
    AdjudicationState,
    Adjudicator,
    AdjudicationConfigError,
    PHASE1_MODEL,
    PHASE1_PROVIDER,
)


class ScriptedCall:
    def __init__(self, responses=None, raise_exc=None):
        self.responses = list(responses or [])
        self.raise_exc = raise_exc
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_exc is not None:
            raise self.raise_exc
        if not self.responses:
            return "{}"
        return self.responses.pop(0)


def _config(**overrides):
    base: dict = {
        "enabled": True,
        "mode": "review",
        "provider": PHASE1_PROVIDER,
        "model": PHASE1_MODEL,
        "timeout_ms": 1000,
        "session_budget": 5,
        "hourly_budget": 10,
        "circuit_breaker_threshold": 3,
        "circuit_breaker_reset_ms": 200,
    }
    base.update(overrides)
    return AdjudicationConfig(**base)


def _candidates(*ids, with_safety=None):
    out = []
    for rid in ids:
        meta = {"rule_id": rid, "rule_name": rid, "rule_type": "guidance", "priority": "normal"}
        if with_safety and rid in with_safety:
            meta["safety_critical"] = True
        out.append(meta)
    return out


def _mark_safety(cands, *rids):
    safety = set(rids)
    for c in cands:
        if c["rule_id"] in safety:
            c["safety_critical"] = True
    return cands


def _blocked(*ids):
    return [{"rule_id": rid, "reason": "cooldown"} for rid in ids]


def _state():
    return AdjudicationState(trace_retention=20)


def test_config_validates_phase1_provider(caplog):
    # 2026-08-22: alternate providers are now SUPPORTED (log-warning, not
    # raise). The old contract raised AdjudicationConfigError; Plan 010
    # changed it so Kevin can route Phase 1 to any provider. Assert the new
    # contract via caplog (the code uses logger.warning, not warnings.warn).
    cfg = AdjudicationConfig(enabled=True, provider="openai", model=PHASE1_MODEL)
    with caplog.at_level("WARNING", logger="adjudicator"):
        adj = Adjudicator(cfg, _state(), ScriptedCall())
    assert adj is not None
    assert any("differs from Phase 1 default" in r.message for r in caplog.records)


def test_config_validates_phase1_model(caplog):
    # 2026-08-22: same as provider — alternate models warn instead of raise.
    cfg = AdjudicationConfig(enabled=True, provider=PHASE1_PROVIDER, model="gpt-4o")
    with caplog.at_level("WARNING", logger="adjudicator"):
        adj = Adjudicator(cfg, _state(), ScriptedCall())
    assert adj is not None
    assert any("differs from Phase 1 default" in r.message for r in caplog.records)


def test_config_rejects_unknown_mode():
    cfg = AdjudicationConfig(enabled=True, mode="recall")
    with pytest.raises(AdjudicationConfigError):
        Adjudicator(cfg, _state(), ScriptedCall())


def test_disabled_returns_full_set_with_off_reason():
    cfg = AdjudicationConfig(enabled=False)
    adj = Adjudicator(cfg, _state(), ScriptedCall())
    result = adj.review(_candidates("a", "b"), [], "pre_llm_call", "s1")
    assert result["status"] == "fallback"
    assert result["reason"] == "mode_off"
    # Plan 007 (2026-07-27) partition semantics: on fallback the adjudicator
    # keeps safety-critical candidates and drops non-safety guidance. The
    # matcher caller additionally keeps the deterministic set unchanged
    # (fail-closed) — that caller contract is tested in
    # run_matcher_integration.py. These three tests were stale (asserted the
    # pre-Plan-007 keep-all verdict) and invisible because pytest collection
    # from the plugin root was broken until 2026-08-04.
    assert result["keep_candidates"] == []
    assert sorted(result["drop_candidates"]) == ["a", "b"]


def test_review_keep_all_returns_unchanged():
    call = ScriptedCall(responses=[json.dumps({"keep": ["a", "b"], "drop": [], "reason": "ok"})])
    adj = Adjudicator(_config(), _state(), call)
    result = adj.review(_candidates("a", "b"), [], "pre_llm_call", "s1")
    assert result["status"] == "applied"
    assert result["keep_candidates"] == ["a", "b"]
    assert result["drop_candidates"] == []
    assert len(call.calls) == 1
    messages = call.calls[0]["messages"]
    payload = json.loads(messages[1]["content"])
    assert "user_message" not in payload
    assert "user_content" not in payload
    assert payload["candidates"] == _candidates("a", "b")


def test_review_drop_some_filters_set():
    call = ScriptedCall(responses=[json.dumps({
        "keep": ["a"], "drop": ["b"], "reason": "b off-topic"
    })])
    adj = Adjudicator(_config(), _state(), call)
    result = adj.review(_candidates("a", "b"), [], "pre_llm_call", "s1")
    assert result["status"] == "applied"
    assert result["keep_candidates"] == ["a"]
    assert result["drop_candidates"] == ["b"]


def test_review_safety_floor_is_preserved_when_model_tries_to_drop():
    call = ScriptedCall(responses=[json.dumps({
        "keep": ["a"], "drop": ["a", "b"], "reason": "try"
    })])
    adj = Adjudicator(_config(), _state(), call)
    cands = _mark_safety(_candidates("a", "b"), "a")
    result = adj.review(cands, [], "pre_llm_call", "s1")
    assert result["keep_candidates"] == ["a"]
    assert "a" not in result["drop_candidates"]


def test_review_malformed_response_falls_back():
    call = ScriptedCall(responses=["not json at all"])
    adj = Adjudicator(_config(), _state(), call)
    result = adj.review(_candidates("a"), [], "pre_llm_call", "s1")
    assert result["status"] == "fallback"
    assert result["reason"] == "malformed_response"
    # Plan 007 partition: non-safety guidance dropped on fallback.
    assert result["keep_candidates"] == []
    assert result["drop_candidates"] == ["a"]


def test_review_hallucinated_ids_are_ignored():
    call = ScriptedCall(responses=[json.dumps({
        "keep": ["a", "fake-rule"], "drop": ["ghost"], "reason": "ok"
    })])
    adj = Adjudicator(_config(), _state(), call)
    result = adj.review(_candidates("a"), [], "pre_llm_call", "s1")
    assert result["keep_candidates"] == ["a"]
    assert "fake-rule" not in result["keep_candidates"]


def test_review_caller_exception_falls_back():
    call = ScriptedCall(raise_exc=TimeoutError("upstream slow"))
    adj = Adjudicator(_config(), _state(), call)
    result = adj.review(_candidates("a", "b"), [], "pre_llm_call", "s1")
    assert result["status"] == "fallback"
    assert result["reason"].startswith("call_error")
    # Plan 007 partition: non-safety guidance dropped on fallback.
    assert result["keep_candidates"] == []
    assert sorted(result["drop_candidates"]) == ["a", "b"]


def test_session_budget_exhausted_blocks_call():
    call = ScriptedCall()
    state = _state()
    adj = Adjudicator(_config(session_budget=2), state, call)
    for i in range(2):
        result = adj.review(_candidates(f"r{i}"), [], "pre_llm_call", "s1")
        assert result["status"] in ("applied", "fallback")
    result = adj.review(_candidates("r3"), [], "pre_llm_call", "s1")
    assert result["status"] == "fallback"
    assert result["reason"] == "session_budget_exhausted"
    assert call.calls == [] or len(call.calls) == 2


def test_circuit_breaker_trips_and_recovers():
    call = ScriptedCall(raise_exc=RuntimeError("boom"))
    state = _state()
    adj = Adjudicator(
        _config(circuit_breaker_threshold=2, circuit_breaker_reset_ms=50), state, call
    )
    adj.review(_candidates("a"), [], "pre_llm_call", "s1")
    adj.review(_candidates("b"), [], "pre_llm_call", "s1")
    result = adj.review(_candidates("c"), [], "pre_llm_call", "s1")
    assert result["status"] == "fallback"
    assert result["reason"] == "circuit_open"
    time.sleep(0.06)
    call.raise_exc = None
    call.responses = [json.dumps({"keep": ["x"], "drop": [], "reason": "ok"})]
    result = adj.review(_candidates("x"), [], "pre_llm_call", "s1")
    assert result["status"] == "applied"


def test_trace_records_decisions_without_user_text():
    call = ScriptedCall(responses=[json.dumps({
        "keep": ["a"], "drop": ["b"], "reason": "trim"
    })])
    state = _state()
    adj = Adjudicator(_config(), state, call)
    adj.review(_candidates("a", "b"), [], "pre_llm_call", "session-1234")
    snap = state.snapshot()
    assert snap["traces"][-1]["candidates_dropped"] == ["b"]
    assert snap["traces"][-1]["fallback"] is False
    assert snap["traces"][-1]["session_id"] == "session-1234"
    assert "user_message" not in str(snap)


def test_review_drops_unmentioned_candidates():
    call = ScriptedCall(responses=[json.dumps({
        "keep": ["a"], "drop": [], "reason": "omitted b"
    })])
    adj = Adjudicator(_config(), _state(), call)
    result = adj.review(_candidates("a", "b"), [], "pre_llm_call", "s1")
    assert result["keep_candidates"] == ["a"]
    assert "b" in result["drop_candidates"]


def test_review_blocked_review_is_advisory_only():
    call = ScriptedCall(responses=[json.dumps({
        "keep": ["a"], "drop": [], "blocked_review": [
            {"rule_id": "x", "should_attempt": True, "reason": "maybe"}
        ],
        "reason": "ok",
    })])
    adj = Adjudicator(_config(), _state(), call)
    result = adj.review(_candidates("a"), _blocked("x"), "pre_llm_call", "s1")
    assert result["keep_blocked"] == []
    assert result["blocked_review"][0]["rule_id"] == "x"
    assert result["blocked_review"][0]["should_attempt"] is True


def test_review_truncates_overlong_reason():
    long = "x" * 500
    call = ScriptedCall(responses=[json.dumps({
        "keep": ["a"], "drop": [], "reason": long
    })])
    adj = Adjudicator(_config(), _state(), call)
    result = adj.review(_candidates("a"), [], "pre_llm_call", "s1")
    assert len(result["reason"]) <= 200


def test_review_self_contradictory_keep_drop_falls_back():
    call = ScriptedCall(responses=[json.dumps({
        "keep": ["a"], "drop": ["a"], "reason": "weird"
    })])
    adj = Adjudicator(_config(), _state(), call)
    result = adj.review(_candidates("a"), [], "pre_llm_call", "s1")
    assert result["status"] == "fallback"
    assert result["reason"] == "malformed_response"


def test_payload_caps_inputs():
    call = ScriptedCall(responses=[json.dumps({"keep": [], "drop": [], "reason": "x"})])
    cfg = _config(max_candidates=3, max_blocked=2)
    adj = Adjudicator(cfg, _state(), call)
    candidates = _candidates(*[f"r{i}" for i in range(10)])
    blocked = _blocked(*[f"b{i}" for i in range(10)])
    adj.review(candidates, blocked, "pre_llm_call", "s1")
    payload = json.loads(call.calls[0]["messages"][1]["content"])
    assert len(payload["candidates"]) == 3
    assert len(payload["blocked"]) == 2
