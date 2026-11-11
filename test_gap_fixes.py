"""Regression tests for audit-informed rule-injector gap fixes."""

import json
from pathlib import Path

from matchers import (
    CompiledRule,
    MatchContext,
    RuleMatcher,
    _detect_cron_context,
    _is_blocklisted_message,
)
from rules import Rule, RuleStore


def _rule(rule_id="test-rule", **trigger_options):
    trigger = {
        "type": "keyword",
        "scope": "latest_user_message",
        "keywords": ["build"],
        **trigger_options,
    }
    return Rule(
        id=rule_id,
        name=rule_id,
        hook="pre_llm_call",
        priority=1,
        trigger=trigger,
        inject={"context": f"[{rule_id}]"},
    )


def _matcher(tmp_path, rules):
    path = tmp_path / "rules.json"
    store = RuleStore(str(path))
    store.save(rules)
    return RuleMatcher(store)


def test_recent_context_keyword_exclusion_is_case_insensitive():
    rule = CompiledRule(_rule(exclude_if_recent_context_keywords=["Android", "Kotlin"]))
    assert not rule.matches(MatchContext(
        hook="pre_llm_call", user_message="build the feature",
        recent_context="The Mycelia app is written in kotlin for android",
    ))
    assert rule.matches(MatchContext(
        hook="pre_llm_call", user_message="build the feature",
        recent_context="Unreal Engine animation blueprint work",
    ))


def test_cron_detection_uses_ids_explicit_flag_and_wrappers():
    assert _detect_cron_context("cron_123", "", "")
    assert _detect_cron_context("", "cron_daily", "")
    assert _detect_cron_context("", "", "IMPORTANT: You are running as a scheduled cron job.")
    assert not _detect_cron_context("bg_123", "", "build the Android app")


def test_max_fires_per_session_is_isolated(tmp_path):
    matcher = _matcher(tmp_path, [_rule(max_fires_per_session=1)])
    kwargs = {"user_message": "build it", "conversation_history": []}
    assert "test-rule" in matcher.run_pre_llm_call({**kwargs, "session_id": "s1"})["context"]
    assert matcher.run_pre_llm_call({**kwargs, "session_id": "s1"}) == {}
    assert "test-rule" in matcher.run_pre_llm_call({**kwargs, "session_id": "s2"})["context"]


def test_cooldown_turns_counts_distinct_user_turns(tmp_path):
    matcher = _matcher(tmp_path, [_rule(cooldown_turns=2)])
    base = {"conversation_history": [], "session_id": "s1"}
    assert matcher.run_pre_llm_call({**base, "user_message": "build one"})
    assert matcher.run_pre_llm_call({**base, "user_message": "build two"}) == {}
    assert matcher.run_pre_llm_call({**base, "user_message": "build three"}) == {}
    assert matcher.run_pre_llm_call({**base, "user_message": "build four"})


def test_cap_skips_oversized_rule_instead_of_breaking(tmp_path):
    # F13 (2026-08-04 audit): budget is read from the module so this test
    # survives future cap changes (was hard-coded to the old 8000 cap).
    from matchers import INJECTION_CHAR_BUDGET
    huge = _rule("huge")
    huge.inject = {"context": "H" * (INJECTION_CHAR_BUDGET - 100)}
    does_not_fit = _rule("does-not-fit")
    does_not_fit.priority = 2
    does_not_fit.inject = {"context": "X" * 300}
    small = _rule("small")
    small.priority = 3
    small.inject = {"context": "[small]"}
    matcher = _matcher(tmp_path, [huge, does_not_fit, small])
    result = matcher.run_pre_llm_call({
        "user_message": "build now", "conversation_history": [], "session_id": "cap-test"
    })["context"]
    assert len(result) <= INJECTION_CHAR_BUDGET
    assert "[small]" in result
    assert "X" * 300 not in result


def test_real_rules_have_required_gap_fixes():
    # NOTE (2026-08-22 audit): this test previously asserted enabled=True on
    # specific live rules. Those flags are OPERATIONAL STATE that Kevin
    # tunes via audits (post-compaction-reset-guidance,
    # prior-context-memory-cascade and kevin-redirects-retrust-evidence were
    # all disabled in the 2026-08-03 FP audit after dropping 100% of their
    # fires). Contracts tests assert structure, not ops tuning.
    data = json.loads((Path(__file__).parent / "rules.json").read_text())
    rules = {r["id"]: r for r in data["rules"]}
    assert "post-compaction-reset-guidance" in rules
    assert rules["post-compaction-reset-guidance"]["priority"] <= 3
    assert rules["deliver-files-via-media"]["trigger"]["max_fires_per_session"] == 1
    assert rules["proactive-advisor-consultation"]["trigger"]["cooldown_turns"] >= 3
    assert "you didn't" in rules["kevin-redirects-retrust-evidence"]["trigger"].get("keywords", []) or \
        any("you didn't" in c.get("keywords", []) for c in rules["kevin-redirects-retrust-evidence"]["trigger"].get("conditions", []))
    assert "prior-context-memory-cascade" in rules
    recall_rule = rules["prior-context-memory-cascade"]
    assert recall_rule["hook"] == "pre_llm_call"
    assert recall_rule["priority"] == 1
    assert recall_rule["trigger"]["suppress_on_meta_discussion"] is True
    inject = recall_rule["inject"]["context"]
    assert inject.index("session_search") < inject.index("hindsight_recall")
    assert inject.index("hindsight_recall") < inject.index("mcp__uaml__memory_focus_recall")
    assert _is_blocklisted_message("Write me a rule-injector plugin rule for past conversation recall")
    assert _is_blocklisted_message("Discuss the HARD RULE: memory cascade for earlier sessions")
    assert not _is_blocklisted_message("What did we decide about the Telos board last time?")
    ue_rules = [r for r in data["rules"] if r["id"].startswith("ue-") or r["id"] in {"unreal-engine-router"}]
    assert ue_rules
    assert all(r["trigger"].get("exclude_if_recent_context_keywords") for r in ue_rules)
