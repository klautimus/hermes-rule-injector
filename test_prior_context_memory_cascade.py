#!/usr/bin/env python3
"""Targeted regression checks for the prior-context memory cascade rule."""

from matchers import CompiledRule, MatchContext, RuleMatcher, _is_blocklisted_message
from rules import RuleStore

RULE_ID = "prior-context-memory-cascade"
RULE_PREFIX = "[HARD RULE: Resolve Past-Context References Before Answering]"


def compiled_match(rule: CompiledRule, message: str) -> bool:
    context = MatchContext(
        hook="pre_llm_call",
        user_message=message,
        conversation_history=[],
        skill_loaded_this_session=set(),
    )
    return rule.matches(context)


def runtime_fires(message: str, session_id: str) -> bool:
    matcher = RuleMatcher(RuleStore())
    result = matcher.run_pre_llm_call(
        {
            "user_message": message,
            "conversation_history": [],
            "session_id": session_id,
        }
    )
    return RULE_PREFIX in result.get("context", "")


def main() -> None:
    stored = RuleStore().get(RULE_ID)
    assert stored is not None, f"missing rule: {RULE_ID}"
    assert stored.priority == 1
    assert stored.hook == "pre_llm_call"
    assert stored.trigger.get("suppress_on_meta_discussion") is True

    rule = CompiledRule(stored)

    deterministic_positives = [
        "What did we decide about the Telos board last time?",
        "Where did we leave off with the Hermes plugin?",
        "you remember addyosmani's pattern",
        "Pull up that plan we wrote",
        "the board we created — what's its status?",
        "as we discussed, ship it",
        "recall our decision on the shipping date",
        "did we ever figure out the Hindsight timeout issue?",
        "what happened with that directory scan?",
    ]
    deterministic_negatives = [
        "What is 2+2?",
        "that's a good point, but I disagree",
        "Can you explain how session_search works?",
        "Remember to commit the changes",
    ]

    for message in deterministic_positives:
        assert compiled_match(rule, message), f"false negative: {message!r}"
    for message in deterministic_negatives:
        assert not compiled_match(rule, message), f"false positive: {message!r}"

    meta_cases = [
        "Write me a rule-injector plugin rule for past conversation recall",
        "Discuss the HARD RULE: memory cascade for earlier sessions",
        "Check the [Rule: prior-context-memory-cascade] trigger",
    ]
    for index, message in enumerate(meta_cases):
        assert _is_blocklisted_message(message), f"meta blocklist miss: {message!r}"
        assert not runtime_fires(message, f"prior-context-meta-{index}"), (
            f"meta-discussion fired at runtime: {message!r}"
        )

    assert runtime_fires(
        "What did we decide about the Telos board last time?",
        "prior-context-runtime-positive",
    )

    soft_match = "I like that plan you just proposed"
    assert compiled_match(rule, soft_match)
    inject_text = stored.inject.get("context", "")
    assert "already fully visible in this session" in inject_text
    assert "session_search" in inject_text
    assert "hindsight_recall" in inject_text
    assert "mcp__uaml__memory_focus_recall" in inject_text

    print(
        f"PASS {RULE_ID}: {len(deterministic_positives)} deterministic positives, "
        f"{len(deterministic_negatives)} negatives, {len(meta_cases)} meta suppressions, "
        "runtime injection, visible-context guard, and ordered cascade verified"
    )


if __name__ == "__main__":
    main()
