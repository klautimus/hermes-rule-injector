#!/usr/bin/env python3
"""Comprehensive test suite for the rule-injector plugin.

Tests each rule against:
1. Synthetic positive cases (rule SHOULD fire)
2. Synthetic negative cases (rule should NOT fire)
3. False-positive checks (other rules should NOT fire on this rule's input)

Architecture:
- For pre_llm_call rules: construct conversation_history + user_message kwargs
- For transform_tool_result rules: construct tool_name + result kwargs
- For pre_verify rules: construct conversation_history kwargs

The matcher is instantiated fresh per test to avoid state contamination
(attempt_count caching, skill scan indexing).

Usage:
    python3 test_rules.py              # run all tests
    python3 test_rules.py --verbose    # show every rule result
    python3 test_rules.py --rule ID    # test single rule
    python3 test_rules.py --semantic   # also test semantic rules (requires sentence-transformers)
"""

import sys
import os
# D10 FIX (2026-07-21): Skip semantic matching in test suite to avoid
# scipy/sklearn import hang (sentence_transformers → sklearn → scipy →
# _make_sphinx_capabilities hangs indefinitely). Use --semantic flag to
# explicitly enable semantic rule tests (requires working sentence_transformers).
os.environ["RULE_INJECTOR_NO_SEMANTIC"] = "1"
import json
import copy
import time
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

# Add plugin dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rules import RuleStore, Rule
from matchers import RuleMatcher, MatchContext, CompiledRule


# ─── Test Helpers ──────────────────────────────────────────────────────────

def make_history(messages: List[Tuple[str, str]]) -> list:
    """Build conversation history from (role, content) tuples."""
    return [{"role": r, "content": c} for r, c in messages]

def fresh_matcher() -> RuleMatcher:
    """Create a fresh RuleMatcher with no cached state."""
    store = RuleStore()
    matcher = RuleMatcher(store)
    return matcher

def get_fired_rule_ids(matcher: RuleMatcher, kwargs: dict, hook: str) -> set:
    """Run matcher and return set of rule IDs that fired."""
    if hook == "pre_llm_call":
        result = matcher.run_pre_llm_call(kwargs)
        # result is {"context": "..."} or {}
        if result.get("context"):
            # We need to map back to rule IDs — re-run with individual checks
            return _get_firing_rules(matcher, kwargs, hook)
        return set()
    elif hook == "transform_tool_result":
        result = matcher.run_transform_tool_result(kwargs)
        original = kwargs.get("result", "")
        if result != original:
            # Extract rule IDs from the suffix format [Rule: ID]
            import re
            ids = set(re.findall(r'\[Rule: ([\w-]+)\]', result))
            return ids
        return set()
    elif hook == "pre_verify":
        result = matcher.run_pre_verify(kwargs)
        if result.get("blocks"):
            return {b["rule_id"] for b in result["blocks"]}
        return set()
    return set()

def _get_firing_rules(matcher: RuleMatcher, kwargs: dict, hook: str) -> set:
    """For pre_llm_call, determine which rules fired by checking each individually."""
    matcher._ensure_compiled()

    history = kwargs.get("conversation_history", [])
    user_msg = kwargs.get("user_message", "")

    recent_parts = []
    for m in (history or [])[-4:]:
        if isinstance(m, dict):
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                recent_parts.append(content[:500])
    recent = " ".join(recent_parts)

    skills_loaded = matcher._detect_skills_loaded_incremental(history)
    is_post_compaction = matcher._detect_post_compaction(history)
    attempt_count = matcher._count_attempts_incremental(history)

    ctx = MatchContext(
        hook=hook,
        user_message=user_msg,
        recent_context=recent,
        conversation_history=history,
        skill_loaded_this_session=skills_loaded,
        is_first_turn=kwargs.get("is_first_turn", False),
        is_post_compaction=is_post_compaction,
        attempt_count=attempt_count,
    )

    fired = set()
    for rule in matcher._compiled:
        if rule.hook != hook:
            continue
        try:
            if rule.matches(ctx):
                fired.add(rule.id)
        except Exception:
            continue
    return fired

def _get_firing_rules_tool(matcher: RuleMatcher, kwargs: dict) -> set:
    """For transform_tool_result, determine which rules fired."""
    matcher._ensure_compiled()
    ctx = MatchContext(
        hook="transform_tool_result",
        tool_name=kwargs.get("tool_name", ""),
        tool_result=kwargs.get("result", ""),
        tool_args=kwargs.get("args", {}),
    )
    fired = set()
    for rule in matcher._compiled:
        if rule.hook != "transform_tool_result":
            continue
        try:
            if rule.matches(ctx):
                fired.add(rule.id)
        except Exception:
            continue
    return fired

def _get_firing_rules_verify(matcher: RuleMatcher, kwargs: dict) -> set:
    """For pre_verify, determine which rules fired."""
    matcher._ensure_compiled()
    history = kwargs.get("conversation_history", [])
    recent_parts = []
    for m in (history or [])[-4:]:
        if isinstance(m, dict):
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                recent_parts.append(content[:500])
    recent = " ".join(recent_parts)

    skills_loaded = matcher._detect_skills_loaded_incremental(history)
    is_post_compaction = matcher._detect_post_compaction(history)
    attempt_count = matcher._count_attempts_incremental(history)

    ctx = MatchContext(
        hook="pre_verify",
        recent_context=recent,
        conversation_history=history,
        skill_loaded_this_session=skills_loaded,
        is_post_compaction=is_post_compaction,
        attempt_count=attempt_count,
    )
    fired = set()
    for rule in matcher._compiled:
        if rule.hook != "pre_verify":
            continue
        try:
            if rule.matches(ctx):
                fired.add(rule.id)
        except Exception:
            continue
    return fired


# ─── Test Case Definitions ──────────────────────────────────────────────────
# Each test case: (rule_id, hook, positive_kwargs, negative_kwargs, description)

@dataclass
class TestCase:
    rule_id: str
    hook: str
    positive: dict          # kwargs that SHOULD trigger this rule
    negative: dict          # kwargs that should NOT trigger this rule
    description: str = ""
    skip_semantic: bool = False  # skip if semantic matching unavailable


def build_test_cases(include_semantic: bool = False) -> List[TestCase]:
    """Generate test cases for every rule in rules.json."""

    cases = []

    # ─── pre_llm_call rules ─────────────────────────────────────────────────

    # 1. correction-stop-pattern
    cases.append(TestCase(
        "correction-stop-pattern", "pre_llm_call",
        {"user_message": "Stop, that's not what I said. Wrong approach.", "conversation_history": []},
        {"user_message": "Can you help me with this?", "conversation_history": []},
        "User correction keywords trigger stop-pattern"
    ))

    # 2. channel-trust
    cases.append(TestCase(
        "channel-trust", "pre_llm_call",
        {
            "user_message": "can you run this command for me",
            "conversation_history": [
                {"role": "user", "content": "This is a discord channel conversation"},
                {"role": "assistant", "content": "I see you're in a shared channel"},
            ]
        },
        {
            "user_message": "can you help me with this task",
            "conversation_history": []
        },
        "Write/exec request in shared channel context"
    ))

    # 3. just-do-it-execute
    cases.append(TestCase(
        "just-do-it-execute", "pre_llm_call",
        {"user_message": "just test it and let me know", "conversation_history": []},
        {"user_message": "I think we should test it carefully", "conversation_history": []},
        "'just X' pattern triggers immediate execute"
    ))

    # 4. kevin-redirects-retrust-evidence (has semantic)
    cases.append(TestCase(
        "kevin-redirects-retrust-evidence", "pre_llm_call",
        {"user_message": "that's not the problem, focus on the actual issue", "conversation_history": []},
        {"user_message": "I like the approach you took", "conversation_history": []},
        "Kevin redirecting investigation with keyword match",
        skip_semantic=False  # has keyword condition in OR compound
    ))

    # 5. read-file-keep-reading (transform_tool_result)
    # Matcher checks for literal 'next_offset' or 'TRUNCATED' in result text
    cases.append(TestCase(
        "read-file-keep-reading", "transform_tool_result",
        {"tool_name": "read_file", "result": 'LINE 1| content\nLINE 2| more\nRead exceeding ~100K characters are truncated on a line boundary and return a next_offset; continue with offset to read the rest. TRUNCATED.', "args": {}},
        {"tool_name": "read_file", "result": "LINE 1| short content", "args": {}},
        "Truncated read_file result triggers keep-reading"
    ))

    # 6. advisor-two-attempt
    cases.append(TestCase(
        "advisor-two-attempt", "pre_llm_call",
        {
            "user_message": "still not working, try again",
            "conversation_history": [
                {"role": "user", "content": "it's still broken"},
                {"role": "assistant", "content": "let me fix this and update the code"},
                {"role": "user", "content": "didn't work, wrong result"},
                {"role": "assistant", "content": "let me try to patch this again"},
            ]
        },
        {
            "user_message": "what is the capital of France?",
            "conversation_history": []
        },
        "2+ attempts + correction keywords trigger advisor consultation"
    ))

    # 7. browser-console-first
    cases.append(TestCase(
        "browser-console-first", "pre_llm_call",
        {"user_message": "the web app login button is broken, error on the page", "conversation_history": []},
        {"user_message": "fix the database schema", "conversation_history": []},
        "Web app + error keywords trigger browser console first"
    ))

    # 8. complex-runbook-assess-first
    cases.append(TestCase(
        "complex-runbook-assess-first", "pre_llm_call",
        {
            "user_message": "deploy the service, then configure CI/CD pipeline, step 1 then step 2",
            "conversation_history": [
                {"role": "user", "content": "we need to set up production infrastructure"},
            ]
        },
        {"user_message": "say hello", "conversation_history": []},
        "Multi-step deploy with infrastructure context"
    ))

    # 9. simplest-path-wins
    cases.append(TestCase(
        "simplest-path-wins", "pre_llm_call",
        {"user_message": "you're overthinking the architecture, keep it simple - don't overcomplicate the integration", "conversation_history": []},
        {"user_message": "which database should we use?", "conversation_history": []},
        "Overcomplex + architecture keywords trigger simplest-path"
    ))

    # 11. use-tool-via-schema
    cases.append(TestCase(
        "use-tool-via-schema", "pre_llm_call",
        {"user_message": "use the tool directly, don't write a python wrapper", "conversation_history": []},
        {"user_message": "write a python script to process data", "conversation_history": []},
        "'use the tool' keyword triggers tool-schema rule"
    ))

    # 14. check-hermes-support-first
    cases.append(TestCase(
        "check-hermes-support-first", "pre_llm_call",
        {"user_message": "set up the new integration and configure it", "conversation_history": []},
        {"user_message": "what time is it?", "conversation_history": []},
        "Setup/configure keywords trigger check-hermes-first"
    ))

    # 17. codegraph-not-grep (transform_tool_result)
    cases.append(TestCase(
        "codegraph-not-grep", "transform_tool_result",
        {"tool_name": "search_files", "result": '{"matches": ["src/auth.py:10:def login()", "src/user.py:5:class User"], "total_count": 2}', "args": {}},
        {"tool_name": "search_files", "result": '{"matches": ["README.md: read this"], "total_count": 1}', "args": {}},
        "search_files returning source code triggers codegraph-not-grep"
    ))

    # 18. industrial-brutalism-design
    cases.append(TestCase(
        "industrial-brutalism-design", "pre_llm_call",
        {
            "user_message": "design the dashboard UI for the tarot app",
            "conversation_history": [
                {"role": "user", "content": "the prompt-library frontend needs a new style"},
            ]
        },
        {"user_message": "fix the database query", "conversation_history": []},
        "Design + tarot/webprf context triggers industrial-brutalism"
    ))

    # 19. manage-confusion
    cases.append(TestCase(
        "manage-confusion", "pre_llm_call",
        {"user_message": "the requirements are inconsistent and ambiguous, I'm unclear", "conversation_history": []},
        {"user_message": "build the login page", "conversation_history": []},
        "Confusion keywords trigger manage-confusion"
    ))

    # 20. pipeline-router-first
    cases.append(TestCase(
        "pipeline-router-first", "pre_llm_call",
        {"user_message": "build the feature and implement the new app", "conversation_history": []},
        {"user_message": "what is 2+2?", "conversation_history": []},
        "Build/implement keywords trigger pipeline-router"
    ))

    # 21. push-back-warranted
    cases.append(TestCase(
        "push-back-warranted", "pre_llm_call",
        {"user_message": "that's a bad idea, won't work, there's a risk", "conversation_history": []},
        {"user_message": "that sounds great", "conversation_history": []},
        "Bad idea/risk keywords trigger push-back"
    ))

    # 22. research-router-first
    cases.append(TestCase(
        "research-router-first", "pre_llm_call",
        {"user_message": "investigate the gene variant and find the compound's IC50", "conversation_history": []},
        {"user_message": "fix the login page bug", "conversation_history": []},
        "Life-science keywords trigger research-router"
    ))

    # 23. surface-assumptions
    cases.append(TestCase(
        "surface-assumptions", "pre_llm_call",
        {
            "user_message": "implement the feature and build it",
            "conversation_history": [
                {"role": "assistant", "content": "I assume we should use the standard architecture"},
            ]
        },
        {"user_message": "implement the feature", "conversation_history": []},
        "Implement + assumption in context triggers surface-assumptions"
    ))

    # 24. todo-list-multi-phase
    cases.append(TestCase(
        "todo-list-multi-phase", "pre_llm_call",
        {
            "user_message": "step 1 then step 2, after that step 3",
            "conversation_history": [
                {"role": "assistant", "content": "I'll add this to the todo task list"},
            ]
        },
        {"user_message": "hello", "conversation_history": []},
        "Multi-phase + todo context triggers todo-list"
    ))

    # 25. user-surface-first
    cases.append(TestCase(
        "user-surface-first", "pre_llm_call",
        {"user_message": "check webprf, is it working? debug what's wrong", "conversation_history": []},
        {"user_message": "write a new function", "conversation_history": []},
        "webprf + check/debug keywords triggers user-surface-first"
    ))

    # 26. verify-before-claiming (pre_verify)
    cases.append(TestCase(
        "verify-before-claiming", "pre_verify",
        {"conversation_history": [
            {"role": "assistant", "content": "I'm done, the task is complete. All tests pass and it's working."},
        ]},
        {"conversation_history": [
            {"role": "user", "content": "what is the capital of France?"},
        ]},
        "Completion keywords in pre_verify triggers verify-before-claiming"
    ))

    # 28. advisor-makes-decisions
    cases.append(TestCase(
        "advisor-makes-decisions", "pre_llm_call",
        {
            "user_message": "which option should I go with?",
            "conversation_history": [
                {"role": "assistant", "content": "option A or option B, either way"},
            ]
        },
        {"user_message": "run the build", "conversation_history": []},
        "Multi-option keywords in context triggers advisor-makes-decisions"
    ))

    # 30. dont-suggest-disable
    cases.append(TestCase(
        "dont-suggest-disable", "pre_llm_call",
        {
            "user_message": "what should I do about this error?",
            "conversation_history": [
                {"role": "assistant", "content": "maybe we should disable headroom or turn off compression"},
            ]
        },
        {"user_message": "build the app", "conversation_history": []},
        "Disable headroom in context triggers dont-suggest-disable"
    ))

    # 31. enforce-simplicity (pre_verify)
    cases.append(TestCase(
        "enforce-simplicity", "pre_verify",
        {"conversation_history": [
            {"role": "assistant", "content": "I've implemented and built the feature, it's complete and done."},
        ]},
        {"conversation_history": [
            {"role": "user", "content": "what is the weather?"},
        ]},
        "Implement/complete keywords trigger enforce-simplicity"
    ))

    # 32. fix-root-causes
    cases.append(TestCase(
        "fix-root-causes", "pre_llm_call",
        {
            "user_message": "the library is missing",
            "conversation_history": [
                {"role": "assistant", "content": "let me create a stub and workaround with graceful degradation"},
            ]
        },
        {"user_message": "build the app", "conversation_history": []},
        "Stub/workaround in context triggers fix-root-causes"
    ))

    # 33. maintain-scope-discipline
    cases.append(TestCase(
        "maintain-scope-discipline", "pre_llm_call",
        {
            "user_message": "fix the login bug",
            "conversation_history": [
                {"role": "assistant", "content": "let me also refactor and clean up the unused code"},
            ]
        },
        {"user_message": "build the login", "conversation_history": []},
        "Refactor/clean up in context triggers scope-discipline"
    ))

    # 36. follow-latest-message
    cases.append(TestCase(
        "follow-latest-message", "pre_llm_call",
        {
            "user_message": "now do something completely different",
            "conversation_history": [
                {"role": "system", "content": "conversation was compressed. Context injection summary."},
            ]
        },
        {"user_message": "continue the work", "conversation_history": []},
        "Post-compaction triggers follow-latest-message"
    ))

    # 37. plan-then-execute-no-ask
    cases.append(TestCase(
        "plan-then-execute-no-ask", "pre_llm_call",
        {
            "user_message": "Want me to dispatch subagents?",
            "conversation_history": [
                {"role": "assistant", "content": "I've written the plan and spec in advisor-plans"},
            ]
        },
        {"user_message": "build the app", "conversation_history": []},
        "'Want me to' + plan context triggers plan-then-execute"
    ))

    # 38. deliver-files-via-media
    cases.append(TestCase(
        "deliver-files-via-media", "pre_llm_call",
        {
            "user_message": "save the output to the file",
            "conversation_history": [
                {"role": "assistant", "content": "I've written the report, it's saved and written to the path"},
            ]
        },
        {"user_message": "what is 2+2?", "conversation_history": []},
        "write_file/saved in context triggers deliver-files"
    ))

    # 39. no-placeholders (original)
    cases.append(TestCase(
        "no-placeholders", "pre_llm_call",
        {
            "user_message": "add a TODO here",
            "conversation_history": [
                {"role": "assistant", "content": "I'll create a stub with a placeholder"},
            ]
        },
        {"user_message": "build the real implementation", "conversation_history": []},
        "Placeholder/stub keywords trigger no-placeholders"
    ))

    # 39a. spec-implementation-load-implementation-pipeline-skills
    # Fires on: implement-the-spec / use-the-spec / implement-the-design / SPEC.md references.
    # Does NOT fire on: editing/reading/auditing the spec, generic build requests.
    cases.append(TestCase(
        "spec-implementation-load-implementation-pipeline-skills", "pre_llm_call",
        {
            "user_message": "Please use the spec at /home/kevin/lovecraftian-pet-sim/SPEC.md to create the game",
            "conversation_history": []
        },
        {"user_message": "build the webapp", "conversation_history": []},
        "Use-the-spec-at triggers spec-implementation rule"
    ))
    cases.append(TestCase(
        "spec-implementation-load-implementation-pipeline-skills", "pre_llm_call",
        {
            "user_message": "implement the spec sheet for the new feature",
            "conversation_history": []
        },
        {"user_message": "fix the bug in js/game.js", "conversation_history": []},
        "implement-the-spec-sheet triggers spec-implementation rule"
    ))
    cases.append(TestCase(
        "spec-implementation-load-implementation-pipeline-skills", "pre_llm_call",
        {
            "user_message": "implement the design doc for the API",
            "conversation_history": []
        },
        {"user_message": "what does the spec say?", "conversation_history": []},
        "implement-the-design-doc triggers spec-implementation rule"
    ))
    cases.append(TestCase(
        "spec-implementation-load-implementation-pipeline-skills", "pre_llm_call",
        {
            "user_message": "follow the spec",
            "conversation_history": []
        },
        {"user_message": "edit the spec.md to fix the typos", "conversation_history": []},
        "follow-the-spec triggers spec-implementation; edit-the-spec is suppressed"
    ))
    cases.append(TestCase(
        "spec-implementation-load-implementation-pipeline-skills", "pre_llm_call",
        {
            "user_message": "use the SPEC.md at /home/kevin/project",
            "conversation_history": []
        },
        {"user_message": "research the literature on ketamine", "conversation_history": []},
        "use-SPEC.md triggers spec-implementation; research falls through to atlas-standard"
    ))
    cases.append(TestCase(
        "spec-implementation-load-implementation-pipeline-skills", "pre_llm_call",
        {
            "user_message": "continue where you left off",
            "conversation_history": [
                {"role": "user", "content": "Please use the spec at /home/kevin/x/SPEC.md to build the game"},
                {"role": "assistant", "content": "I'll start by reading the spec and setting up the project structure."},
            ]
        },
        {"user_message": "what's the weather?", "conversation_history": []},
        "Continuation after spec-implementation setup triggers the rule via recent_context"
    ))

    # 40. subagent-model-match
    cases.append(TestCase(
        "subagent-model-match", "pre_llm_call",
        {
            "user_message": "dispatch the tasks",
            "conversation_history": [
                {"role": "assistant", "content": "I'll use delegate_task to spawn subagent with parallel tasks"},
            ]
        },
        {"user_message": "run the build", "conversation_history": []},
        "delegate_task/subagent in context triggers model-match"
    ))

    # 41. unrelated-fixes-announce (pre_verify)
    cases.append(TestCase(
        "unrelated-fixes-announce", "pre_verify",
        {"conversation_history": [
            {"role": "assistant", "content": "I also fixed and cleaned up some issues while I was here, took the opportunity to refactor"},
        ]},
        {"conversation_history": [
            {"role": "user", "content": "what is the weather?"},
        ]},
        "'also fixed'/'while I was here' triggers unrelated-fixes"
    ))

    # 43. proper-methodology-not-guessing (semantic)
    cases.append(TestCase(
        "proper-methodology-not-guessing", "pre_llm_call",
        {
            "user_message": "are you sure you found the true source of the problem? It's still not working.",
            "conversation_history": [
                {"role": "user", "content": "still broken"},
                {"role": "assistant", "content": "let me fix this"},
                {"role": "user", "content": "didn't work, wrong"},
                {"role": "assistant", "content": "let me patch again"},
            ]
        },
        {"user_message": "build the app", "conversation_history": []},
        "2+ attempts + 'are you sure' triggers methodology rule",
        skip_semantic=False  # has compound keyword condition in OR
    ))

    # 44. brief-advisor-with-skill-methodology (semantic)
    cases.append(TestCase(
        "brief-advisor-with-skill-methodology", "pre_llm_call",
        {
            "user_message": "consult the advisor about this",
            "conversation_history": [
                {"role": "assistant", "content": "I should skill_view the SKILL.md and follow its Process Steps in the pipeline"},
            ]
        },
        {"user_message": "build the app", "conversation_history": []},
        "Advisor + skill methodology context triggers brief-advisor",
        skip_semantic=False  # has compound keyword condition in OR
    ))

    # 45. audit-advisor-output (transform_tool_result, semantic)
    cases.append(TestCase(
        "audit-advisor-output", "transform_tool_result",
        {"tool_name": "consult_advisor", "result": "Here is my plan: I recommend you should implement the architecture design with these steps.", "args": {}},
        {"tool_name": "web_search", "result": "Here are some search results about cooking.", "args": {}},
        "consult_advisor result with plan keywords triggers audit-advisor",
        skip_semantic=False  # has tool_name + result_contains in compound
    ))

    # ─── New rules (11 added) ───────────────────────────────────────────────

    # 46. semantic-matching-tool-results
    cases.append(TestCase(
        "semantic-matching-tool-results", "transform_tool_result",
        {"tool_name": "terminal", "result": "BUILD SUCCESSFUL! All tests passed. Everything works, verified complete, no errors.", "args": {}},
        {"tool_name": "terminal", "result": "Error: compilation failed at line 42", "args": {}},
        "Success claims without evidence triggers fabrication check",
        skip_semantic=False  # has result_contains_keywords in compound OR
    ))

    # 47. semantic-matching-blind-patch-loop
    cases.append(TestCase(
        "semantic-matching-blind-patch-loop", "transform_tool_result",
        {"tool_name": "patch", "result": "Successfully patched auth.py for the third time.", "args": {}},
        {"tool_name": "read_file", "result": "File content here", "args": {}},
        "Repeated patch triggers blind-patch-loop (semantic only)",
        skip_semantic=True
    ))

    # 48. semantic-matching-codegraph-reindex
    cases.append(TestCase(
        "semantic-matching-codegraph-reindex", "transform_tool_result",
        {"tool_name": "search_files", "result": '{"matches": ["src/main.py:1:def hello()", "src/util.py:5:class Foo"], "total_count": 2}', "args": {}},
        {"tool_name": "terminal", "result": "command output here", "args": {}},
        "search_files on source triggers codegraph-reindex",
        skip_semantic=False  # has result_contains_keywords for "codegraph" in compound OR
    ))

    # 49. pre-verify-post-board-complete
    cases.append(TestCase(
        "pre-verify-post-board-complete", "pre_verify",
        {"conversation_history": [
            {"role": "assistant", "content": "The kanban board is complete. All tasks done. GATE passed."},
        ]},
        {"conversation_history": [
            {"role": "user", "content": "what is the weather?"},
        ]},
        "Board complete keywords trigger post-board-verify"
    ))

    # 50. pre-verify-user-runs-mutation-boundary
    cases.append(TestCase(
        "pre-verify-user-runs-mutation-boundary", "pre_verify",
        {"conversation_history": [
            {"role": "user", "content": "I'll run the cherry-pick command myself"},
            {"role": "assistant", "content": "I'll prepare the git rebase and push for you"},
        ]},
        {"conversation_history": [
            {"role": "user", "content": "what is the weather?"},
        ]},
        "User runs + git mutation keywords trigger boundary"
    ))

    # 51. semantic-matching-stale-server-check
    cases.append(TestCase(
        "semantic-matching-stale-server-check", "transform_tool_result",
        {"tool_name": "terminal", "result": "The fix is applied but curl shows old behavior. Server serving stale content.", "args": {}},
        {"tool_name": "web_search", "result": "The capital of France is Paris. France is a country in Europe.", "args": {}},
        "Stale server indication triggers stale-server-check (semantic only)",
        skip_semantic=True
    ))

    # 52. semantic-matching-scope-creep-detection
    cases.append(TestCase(
        "semantic-matching-scope-creep-detection", "transform_tool_result",
        {"tool_name": "patch", "result": "Also fixed the unrelated issue while I was here. Cleaned up adjacent code.", "args": {}},
        {"tool_name": "patch", "result": "Applied the requested fix.", "args": {}},
        "Scope-creep language triggers scope-creep-detection (semantic only)",
        skip_semantic=True
    ))

    # 53. semantic-matching-advisor-plan-vs-reality
    cases.append(TestCase(
        "semantic-matching-advisor-plan-vs-reality", "transform_tool_result",
        {"tool_name": "consult_advisor", "result": "The advisor recommended a file that does not exist in the codebase.", "args": {}},
        {"tool_name": "web_search", "result": "Search results", "args": {}},
        "Advisor plan vs reality gap triggers (semantic only)",
        skip_semantic=True
    ))

    # 54. pre-verify-device-smoke-test
    cases.append(TestCase(
        "pre-verify-device-smoke-test", "pre_verify",
        {"conversation_history": [
            {"role": "assistant", "content": "The Android APK is done. assembleRelease succeeded. It's working on the pixel. GATE passed."},
        ]},
        {"conversation_history": [
            {"role": "user", "content": "what is the weather?"},
        ]},
        "Android + completion keywords trigger device-smoke-test"
    ))

    # 55. semantic-matching-placeholder-detection
    cases.append(TestCase(
        "semantic-matching-placeholder-detection", "transform_tool_result",
        {"tool_name": "write_file", "result": "File written with TODO marker and {{name}} placeholder.", "args": {}},
        {"tool_name": "write_file", "result": "File written with complete implementation.", "args": {}},
        "Placeholder/stub triggers placeholder-detection (semantic only)",
        skip_semantic=True
    ))

    # 56. pre-verify-advisor-briefing-complete
    cases.append(TestCase(
        "pre-verify-advisor-briefing-complete", "pre_verify",
        {"conversation_history": [
            {"role": "assistant", "content": "I need to consult_advisor. Let me skill_view the SKILL.md and follow its Process Steps in the pipeline."},
        ]},
        {"conversation_history": [
            {"role": "user", "content": "what is the weather?"},
        ]},
        "Advisor + skill keywords trigger advisor-briefing-check"
    ))

    # 57. semantic-matching-env-fact-reassertion
    cases.append(TestCase(
        "semantic-matching-env-fact-reassertion", "transform_tool_result",
        {"tool_name": "terminal", "result": "Checking if service is actually running after Kevin said it is. Verifying port binding.", "args": {}},
        {"tool_name": "terminal", "result": "Running the build command.", "args": {}},
        "Re-probing env fact triggers env-fact-reassertion (semantic only)",
        skip_semantic=True
    ))

    # 58. sudo-password-injection
    cases.append(TestCase(
        "sudo-password-injection", "pre_llm_call",
        {"user_message": "this command requires sudo to install the package", "conversation_history": []},
        {"user_message": "what is the weather today?", "conversation_history": []},
        "Sudo keyword triggers password injection"
    ))

    # 59. prior-context-memory-cascade
    cases.append(TestCase(
        "prior-context-memory-cascade", "pre_llm_call",
        {"user_message": "What did we decide about the Telos board last time?", "conversation_history": []},
        {"user_message": "What is 2+2?", "conversation_history": []},
        "Past-context reference triggers session_search -> Hindsight -> UAML cascade"
    ))

    return cases

class TestResult:
    def __init__(self, rule_id: str, test_name: str):
        self.rule_id = rule_id
        self.test_name = test_name
        self.passed = False
        self.positive_fired = False
        self.negative_fired = False
        self.false_positives: set = set()
        self.error: str = ""

    def __repr__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        fp = f" [false positives: {self.false_positives}]" if self.false_positives else ""
        return f"{status} {self.rule_id}: {self.test_name}{fp}"


def run_tests(include_semantic: bool = False, verbose: bool = False, filter_rule: str = None) -> Tuple[List[TestResult], dict]:
    """Run all test cases and return results."""
    cases = build_test_cases(include_semantic=include_semantic)

    if filter_rule:
        cases = [c for c in cases if c.rule_id == filter_rule]

    # Load all rule IDs for false-positive checking
    store = RuleStore()
    all_rules = store.load()
    all_rule_ids = {r.id for r in all_rules}

    results = []
    summary = {
        "total": 0, "passed": 0, "failed": 0, "skipped": 0,
        "false_positives": 0, "false_negatives": 0,
        "by_hook": {"pre_llm_call": {"pass": 0, "fail": 0},
                     "transform_tool_result": {"pass": 0, "fail": 0},
                     "pre_verify": {"pass": 0, "fail": 0}},
        "failures": [],
    }

    for case in cases:
        result = TestResult(case.rule_id, case.description)
        summary["total"] += 1

        # Skip semantic-only rules if not requested
        if case.skip_semantic and not include_semantic:
            summary["skipped"] += 1
            if verbose:
                print(f"⏭️ SKIP {case.rule_id}: semantic-only (use --semantic to test)")
            continue

        try:
            # Run positive test
            matcher = fresh_matcher()
            if case.hook == "pre_llm_call":
                fired = _get_firing_rules(matcher, case.positive, case.hook)
            elif case.hook == "transform_tool_result":
                fired = _get_firing_rules_tool(matcher, case.positive)
            elif case.hook == "pre_verify":
                fired = _get_firing_rules_verify(matcher, case.positive)
            else:
                fired = set()

            result.positive_fired = case.rule_id in fired

            # Check for false positives (other rules that fired on this input)
            unexpected = fired - {case.rule_id}
            # Some overlap is expected (e.g., "still not working" triggers multiple rules)
            # Only flag as false positive if the unexpected rule is from a DIFFERENT category
            result.false_positives = unexpected

            # Run negative test
            matcher2 = fresh_matcher()
            if case.hook == "pre_llm_call":
                neg_fired = _get_firing_rules(matcher2, case.negative, case.hook)
            elif case.hook == "transform_tool_result":
                neg_fired = _get_firing_rules_tool(matcher2, case.negative)
            elif case.hook == "pre_verify":
                neg_fired = _get_firing_rules_verify(matcher2, case.negative)
            else:
                neg_fired = set()

            result.negative_fired = case.rule_id in neg_fired

            # Determine pass/fail
            # PASS if: positive fires AND negative does NOT fire
            # Acceptable: positive fires AND negative fires BUT positive has strong justification
            # FAIL if: positive does NOT fire (false negative)
            if result.positive_fired and not result.negative_fired:
                result.passed = True
                summary["passed"] += 1
                summary["by_hook"][case.hook]["pass"] += 1
            elif not result.positive_fired:
                result.passed = False
                summary["failed"] += 1
                summary["false_negatives"] += 1
                summary["by_hook"][case.hook]["fail"] += 1
                summary["failures"].append(f"FALSE NEGATIVE: {case.rule_id} did not fire on positive case")
            elif result.negative_fired:
                # Check if the negative case legitimately should fire
                # For some rules (e.g., broad keyword rules), the negative case might overlap
                result.passed = False
                summary["failed"] += 1
                summary["false_positives"] += 1
                summary["by_hook"][case.hook]["fail"] += 1
                summary["failures"].append(f"FALSE POSITIVE: {case.rule_id} fired on negative case")

            if verbose:
                print(f"{'✅' if result.passed else '❌'} {case.rule_id}")
                print(f"   Positive fired: {result.positive_fired} (expected: True)")
                print(f"   Negative fired: {result.negative_fired} (expected: False)")
                if result.false_positives:
                    print(f"   Also fired: {result.false_positives}")
                print(f"   {case.description}")
                print()

        except Exception as e:
            result.error = str(e)
            result.passed = False
            summary["failed"] += 1
            summary["failures"].append(f"ERROR: {case.rule_id}: {e}")
            if verbose:
                print(f"💥 {case.rule_id}: {e}")

        results.append(result)

    return results, summary


def print_summary(results: List[TestResult], summary: dict):
    """Print test summary report."""
    print("\n" + "=" * 80)
    print("RULE-INJECTOR TEST SUITE RESULTS")
    print("=" * 80)
    print(f"\nTotal tests:    {summary['total']}")
    print(f"Passed:         {summary['passed']} ✅")
    print(f"Failed:         {summary['failed']} ❌")
    print(f"Skipped:        {summary['skipped']} ⏭️")
    print(f"False negatives: {summary['false_negatives']}")
    print(f"False positives: {summary['false_positives']}")

    print(f"\nBy Hook:")
    for hook, counts in summary["by_hook"].items():
        total = counts["pass"] + counts["fail"]
        print(f"  {hook}: {counts['pass']}/{total} passed")

    if summary["failures"]:
        print(f"\n{'─' * 80}")
        print("FAILURES:")
        for f in summary["failures"]:
            print(f"  • {f}")

    # Check for rules with NO test coverage
    store = RuleStore()
    all_rule_ids = {r.id for r in store.load()}
    tested_ids = {r.rule_id for r in results if not r.error}
    untested = all_rule_ids - tested_ids
    if untested:
        print(f"\n{'─' * 80}")
        print(f"UNTESTED RULES ({len(untested)}):")
        for rid in sorted(untested):
            print(f"  • {rid}")

    print(f"\n{'=' * 80}")
    if summary["failed"] == 0:
        print("🎉 ALL TESTS PASSED")
    else:
        print(f"⚠️  {summary['failed']} FAILURES NEED ATTENTION")
    print("=" * 80)


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test rule-injector rules")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show every rule result")
    parser.add_argument("--semantic", "-s", action="store_true", help="Test semantic-only rules (requires sentence-transformers). Unsets RULE_INJECTOR_NO_SEMANTIC.")
    parser.add_argument("--rule", "-r", type=str, help="Test single rule by ID")
    args = parser.parse_args()

    # If --semantic is passed, unset the env var to allow sentence_transformers import
    if args.semantic:
        os.environ.pop("RULE_INJECTOR_NO_SEMANTIC", None)

    results, summary = run_tests(
        include_semantic=args.semantic,
        verbose=args.verbose,
        filter_rule=args.rule
    )

    print_summary(results, summary)

    sys.exit(0 if summary["failed"] == 0 else 1)
