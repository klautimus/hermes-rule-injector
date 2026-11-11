"""Standalone test runner for the adjudicator module.

pytest collection of test_*.py in this directory tries to import the
plugin's __init__.py, which uses relative imports. That fails when
pytest runs outside the package context. This runner is invoked directly
to exercise the adjudicator in isolation.
"""

import json
import sys
import time
from pathlib import Path

# Ensure the plugin dir is on sys.path so we can import adjudicator.py
# as a top-level module without going through the package init.
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

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

    def __call__(self, **kw):
        self.calls.append(kw)
        if self.raise_exc is not None:
            raise self.raise_exc
        if not self.responses:
            return "{}"
        return self.responses.pop(0)


def cfg(**kw):
    base = dict(
        enabled=True,
        mode="review",
        provider=PHASE1_PROVIDER,
        model=PHASE1_MODEL,
        timeout_ms=1000,
        session_budget=5,
        hourly_budget=10,
        circuit_breaker_threshold=3,
        circuit_breaker_reset_ms=200,
    )
    base.update(kw)
    return AdjudicationConfig(**base)


def candidates(*ids, with_safety=None):
    out = []
    for rid in ids:
        meta = {"rule_id": rid, "rule_name": rid, "rule_type": "guidance", "priority": "normal"}
        if with_safety and rid in with_safety:
            meta["safety_critical"] = True
        out.append(meta)
    return out


def mark_safety(cands, *rids):
    """Tag candidate rules as safety-critical in place."""
    safety = set(rids)
    for c in cands:
        if c["rule_id"] in safety:
            c["safety_critical"] = True
    return cands


def blocked(*ids):
    return [{"rule_id": rid, "reason": "cooldown"} for rid in ids]


def state():
    return AdjudicationState(trace_retention=20)


failures = []


def expect(name, cond):
    if cond:
        print("PASS", name)
    else:
        print("FAIL", name)
        failures.append(name)


def main():
    # Plan 010 (2026-08-03): non-default provider is now allowed (warn).
    try:
        Adjudicator(
            AdjudicationConfig(enabled=True, provider="openai", model=PHASE1_MODEL),
            state(),
            ScriptedCall(),
        )
        expect("non-default provider allowed", True)
    except AdjudicationConfigError:
        expect("non-default provider allowed", False)

    # Plan 010 (2026-08-03): non-default model is now allowed (warn).
    try:
        Adjudicator(
            AdjudicationConfig(enabled=True, provider=PHASE1_PROVIDER, model="gpt-4o"),
            state(),
            ScriptedCall(),
        )
        expect("non-default model allowed", True)
    except AdjudicationConfigError:
        expect("non-default model allowed", False)

    try:
        Adjudicator(
            AdjudicationConfig(enabled=True, mode="recall"),
            state(),
            ScriptedCall(),
        )
        expect("mode validation raises", False)
    except AdjudicationConfigError:
        expect("mode validation raises", True)

    r = Adjudicator(AdjudicationConfig(enabled=False), state(), ScriptedCall()).review(
        candidates("a", "b"), [], "pre_llm_call", "s1"
    )
    expect(
        "off mode drops guidance on fallback",
        r["status"] == "fallback"
        and r["reason"] == "mode_off"
        and r["keep_candidates"] == []
        and sorted(r["drop_candidates"]) == ["a", "b"],
    )

    # Plan 007: a safety-critical candidate survives fallback (mode_off)
    r = Adjudicator(
        AdjudicationConfig(enabled=False), state(), ScriptedCall()
    ).review(
        mark_safety(candidates("a", "b"), "a"), [], "pre_llm_call", "s1"
    )
    expect(
        "off mode keeps safety candidates",
        r["status"] == "fallback"
        and r["keep_candidates"] == ["a"]
        and sorted(r["drop_candidates"]) == ["b"],
    )

    call = ScriptedCall(responses=[json.dumps({"keep": ["a", "b"], "drop": [], "reason": "ok"})])
    r = Adjudicator(cfg(), state(), call).review(candidates("a", "b"), [], "pre_llm_call", "s1")
    expect(
        "keep-all applied",
        r["status"] == "applied" and sorted(r["keep_candidates"]) == ["a", "b"],
    )
    expect(
        "payload excludes user text",
        "user_message" not in call.calls[0]["messages"][1]["content"],
    )

    call = ScriptedCall(responses=[json.dumps({"keep": ["a"], "drop": ["b"], "reason": "off"})])
    r = Adjudicator(cfg(), state(), call).review(candidates("a", "b"), [], "pre_llm_call", "s1")
    expect("drop some", sorted(r["drop_candidates"]) == ["b"])

    call = ScriptedCall(
        responses=[json.dumps({"keep": ["a"], "drop": ["a", "b"], "reason": "try"})]
    )
    cands = mark_safety(candidates("a", "b"), "a")
    r = Adjudicator(cfg(), state(), call).review(cands, [], "pre_llm_call", "s1")
    expect(
        "safety floor preserved",
        "a" in r["keep_candidates"] and "a" not in r["drop_candidates"],
    )

    call = ScriptedCall(responses=["not json"])
    r = Adjudicator(cfg(), state(), call).review(candidates("a"), [], "pre_llm_call", "s1")
    expect(
        "malformed falls back with guidance dropped",
        r["status"] == "fallback"
        and r["reason"] == "malformed_response"
        and r["keep_candidates"] == []
        and r["drop_candidates"] == ["a"],
    )

    # Plan 007: a safety-critical candidate survives malformed fallback
    call = ScriptedCall(responses=["not json"])
    r = Adjudicator(cfg(), state(), call).review(
        mark_safety(candidates("a", "b"), "a"), [], "pre_llm_call", "s1"
    )
    expect(
        "malformed keeps safety candidates",
        r["status"] == "fallback"
        and r["keep_candidates"] == ["a"]
        and sorted(r["drop_candidates"]) == ["b"],
    )

    call = ScriptedCall(responses=[json.dumps({"keep": ["a", "fake"], "drop": ["ghost"], "reason": "x"})])
    r = Adjudicator(cfg(), state(), call).review(candidates("a"), [], "pre_llm_call", "s1")
    expect("hallucinated ids ignored", r["keep_candidates"] == ["a"])

    call = ScriptedCall(raise_exc=TimeoutError("slow"))
    r = Adjudicator(cfg(), state(), call).review(candidates("a", "b"), [], "pre_llm_call", "s1")
    expect(
        "exception falls back",
        r["status"] == "fallback" and r["reason"].startswith("call_error"),
    )

    call = ScriptedCall()
    st = state()
    adj = Adjudicator(cfg(session_budget=2), st, call)
    for i in range(2):
        adj.review(candidates(f"r{i}"), [], "pre_llm_call", "s1")
    r = adj.review(candidates("r3"), [], "pre_llm_call", "s1")
    expect(
        "session budget",
        r["status"] == "fallback" and r["reason"] == "session_budget_exhausted",
    )

    call = ScriptedCall(raise_exc=RuntimeError("boom"))
    st = state()
    adj = Adjudicator(
        cfg(circuit_breaker_threshold=2, circuit_breaker_reset_ms=50), st, call
    )
    adj.review(candidates("a"), [], "pre_llm_call", "s1")
    adj.review(candidates("b"), [], "pre_llm_call", "s1")
    r = adj.review(candidates("c"), [], "pre_llm_call", "s1")
    expect(
        "circuit breaker trips",
        r["status"] == "fallback" and r["reason"] == "circuit_open",
    )
    time.sleep(0.06)
    call.raise_exc = None
    call.responses = [json.dumps({"keep": ["x"], "drop": [], "reason": "ok"})]
    r = adj.review(candidates("x"), [], "pre_llm_call", "s1")
    expect("circuit breaker recovers", r["status"] == "applied")

    call = ScriptedCall(responses=[json.dumps({"keep": ["a"], "drop": ["b"], "reason": "trim"})])
    st = state()
    Adjudicator(cfg(), st, call).review(candidates("a", "b"), [], "pre_llm_call", "session-1234")
    snap = st.snapshot()
    expect("trace records drop", snap["traces"][-1]["candidates_dropped"] == ["b"])
    expect("trace no user text", "user_message" not in str(snap))

    call = ScriptedCall(responses=[json.dumps({"keep": ["a"], "drop": [], "reason": "x"})])
    r = Adjudicator(cfg(), state(), call).review(candidates("a", "b"), [], "pre_llm_call", "s1")
    expect("omitted candidates dropped", "b" in r["drop_candidates"])

    call = ScriptedCall(
        responses=[
            json.dumps(
                {
                    "keep": ["a"],
                    "drop": [],
                    "blocked_review": [
                        {"rule_id": "x", "should_attempt": True, "reason": "maybe"}
                    ],
                    "reason": "ok",
                }
            )
        ]
    )
    r = Adjudicator(cfg(), state(), call).review(candidates("a"), blocked("x"), "pre_llm_call", "s1")
    expect(
        "blocked_review advisory only",
        r["keep_blocked"] == [] and r["blocked_review"][0]["rule_id"] == "x",
    )

    call = ScriptedCall(responses=[json.dumps({"keep": ["a"], "drop": [], "reason": "x" * 500})])
    r = Adjudicator(cfg(), state(), call).review(candidates("a"), [], "pre_llm_call", "s1")
    expect("reason truncated", len(r["reason"]) <= 200)

    call = ScriptedCall(responses=[json.dumps({"keep": ["a"], "drop": ["a"], "reason": "weird"})])
    r = Adjudicator(cfg(), state(), call).review(candidates("a"), [], "pre_llm_call", "s1")
    expect(
        "contradictory falls back",
        r["status"] == "fallback" and r["reason"] == "malformed_response",
    )

    call = ScriptedCall(responses=[json.dumps({"keep": [], "drop": [], "reason": "x"})])
    cfg2 = cfg(max_candidates=3, max_blocked=2)
    adj = Adjudicator(cfg2, state(), call)
    adj.review(
        candidates(*[f"r{i}" for i in range(10)]),
        blocked(*[f"b{i}" for i in range(10)]),
        "pre_llm_call",
        "s1",
    )
    payload = json.loads(call.calls[0]["messages"][1]["content"])
    expect("candidates capped", len(payload["candidates"]) == 3)
    expect("blocked capped", len(payload["blocked"]) == 2)

    print()
    if failures:
        print(f"FAILED: {len(failures)}")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
