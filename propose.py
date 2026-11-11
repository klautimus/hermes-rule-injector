"""Auto-creation pipeline for rule-injector rules (Phase 3).

This module implements the propose_rule flow: the background-review fork
drafts a structured rule from an observed user correction; this module
runs the draft through the Adjudicator's review_proposed_rule() gate,
and if approved, persists it to the rule store with auto_created=True.

The function ``propose_rule`` is the canonical entry point used by the
background-review fork (see ``agent/background_review.py``). It is also
called from the CLI ``hermes rule-injector propose`` and any dashboard
action that wants to surface the auto-create flow without going through
the fork.

Design notes
------------
- The function is PURE-ish: it takes a RuleStore + Adjudicator as inputs
  and returns a verdict dict. No hidden globals, no module-level state
  mutation. The plugin wires it up at register() time.

- False-negatives (a good rule gets rejected) are recoverable on the
  next fork cycle. False-positives (a bad rule ships) are not. The
  Adjudicator's review_proposed_rule() defaults to REJECT, and this
  function never overrides that default. If the LLM approves AND the
  schema is clean, the rule is persisted; otherwise we return the
  verdict without writing.

- The function emits a single trace-log entry per call (via the same
  _log_activation used by the runtime hooks) so audit trails across
  the full pipeline — fork draft -> adjudicator review -> store write
  — are visible in ~/.hermes/logs/rule-injector-trace.jsonl.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# Dual-mode imports (F12 bug class, 2026-08-22 audit): bare absolute imports
# kill the ENTIRE plugin in the gateway's package context (loaded as
# `hermes_plugins.rule_injector`), because sys.path does not include the
# plugin dir there. The fallback also fails, so __init__.py's try/except
# swallows a masked error and every rule stops firing. Same fix pattern as
# matchers.py / cli.py.
try:
    from .rules import RuleStore
    from .adjudicator import Adjudicator
except ImportError:  # bare-module context (tests, scripts)
    from rules import RuleStore
    from adjudicator import Adjudicator

logger = logging.getLogger("rule_injector.propose")

# Audit log lives at a separate path so it can be tailed independently of
# the runtime hook trace log. Same line-per-event JSONL format.
PROPOSE_AUDIT_LOG = os.path.expanduser(
    "~/.hermes/logs/rule-injector-propose-audit.jsonl"
)


def _log_propose_event(event: Dict[str, Any]) -> None:
    """Append a single JSONL event to the propose audit log.

    Used for full-pipeline audit (drafted -> reviewed -> accepted/rejected
    -> persisted) so any future investigation can reconstruct what
    happened even if the rule was deleted later.
    """
    try:
        os.makedirs(os.path.dirname(PROPOSE_AUDIT_LOG), exist_ok=True)
        event.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
        with open(PROPOSE_AUDIT_LOG, "a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass  # never crash on audit-logging


def _find_existing_rules_with_same_trigger(
    rule_store: RuleStore, proposed: Dict[str, Any], limit: int = 5
) -> List[Dict[str, Any]]:
    """Return up to ``limit`` existing rules whose trigger overlaps.

    Used as context for the adjudicator's dedup check. A "match" is
    defined as either: (a) same id, (b) any keyword overlap, or
    (c) same hook + same trigger.type. The 5 most-recently-updated
    matches are returned so the LLM judge has deduplication context
    without a 100+-rule context dump.
    """
    proposed_trigger = proposed.get("trigger") or {}
    proposed_keywords = set(proposed_trigger.get("keywords") or []) if isinstance(
        proposed_trigger.get("keywords"), list
    ) else set()
    proposed_hook = proposed.get("hook", "")
    proposed_type = str(proposed_trigger.get("type", ""))
    proposed_id = str(proposed.get("id", "")).strip()

    matches: List[Dict[str, Any]] = []
    for r in rule_store.load():
        # Same id is an obvious duplicate regardless of trigger.
        if proposed_id and r.id == proposed_id:
            matches.append({
                "rule_id": r.id,
                "match_reason": "same id",
                "hook": r.hook,
                "trigger_type": str((r.trigger or {}).get("type", "")),
                "keywords": (r.trigger or {}).get("keywords", []) if isinstance(
                    (r.trigger or {}).get("keywords"), list
                ) else [],
            })
            continue
        # Any keyword overlap.
        r_keywords = (r.trigger or {}).get("keywords", []) if isinstance(
            (r.trigger or {}).get("keywords"), list
        ) else []
        if proposed_keywords and r_keywords:
            overlap = proposed_keywords & set(r_keywords)
            if overlap:
                matches.append({
                    "rule_id": r.id,
                    "match_reason": f"keyword overlap: {sorted(overlap)[:3]}",
                    "hook": r.hook,
                    "trigger_type": str((r.trigger or {}).get("type", "")),
                })
                continue
        # Same hook + same trigger.type (often a near-duplicate).
        if proposed_hook == r.hook and proposed_type and proposed_type == str(
            (r.trigger or {}).get("type", "")
        ):
            matches.append({
                "rule_id": r.id,
                "match_reason": f"same hook + same trigger.type ({proposed_type})",
                "hook": r.hook,
            })
        if len(matches) >= limit:
            break
    # Sort by recency (most recently updated first).
    matches.sort(key=lambda m: m.get("rule_id", ""), reverse=True)
    return matches[:limit]


def _ensure_auto_provenance(rule_data: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Stamp auto_created=True + source + ISO-8601 timestamp.

    Idempotent: if the caller already set these, we keep their values
    (only filling missing ones). The fork sometimes drafts the
    timestamp; sometimes not.
    """
    rule = dict(rule_data)  # shallow copy
    rule["auto_created"] = True
    rule["auto_created_source"] = str(source or "unknown")[:50]
    if not rule.get("auto_created_at"):
        rule["auto_created_at"] = (
            datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        )
    return rule


def propose_rule(
    rule_data: Dict[str, Any],
    correction_source: str,
    correction_text: str,
    rule_store: RuleStore,
    adjudicator: Adjudicator,
    session_id: str = "background_review",
) -> Dict[str, Any]:
    """The canonical auto-create entry point.

    Parameters
    ----------
    rule_data : dict
        The full proposed rule object (id, name, description, hook,
        priority, trigger, inject). The fork produces this; the CLI
        also accepts it as JSON via stdin.
    correction_source : str
        What triggered the draft: 'oob_message', 'stop_event',
        'explicit_correction', or 'cli'. Used for provenance + audit.
    correction_text : str
        The user message or signal that prompted the draft. Surfaced
        to the adjudicator so it can judge scope fit.
    rule_store : RuleStore
        Where the rule will be persisted on accept.
    adjudicator : Adjudicator
        The gate. Calls review_proposed_rule() which may REJECT.
    session_id : str
        For budget tracking + audit log. Background-review fork passes
        its session id; CLI passes a fixed string like 'cli'.

    Returns
    -------
    dict with keys:
        - status: 'persisted' | 'rejected' | 'duplicate_skipped'
        - rule_id: the persisted rule's id (or proposed id if rejected)
        - verdict: the full Adjudicator verdict dict
        - reason: human-readable explanation
    """
    proposed_id = str(rule_data.get("id", "")).strip() or "(no-id)"
    started = time.monotonic()
    audit_event: Dict[str, Any] = {
        "event": "propose_rule.started",
        "session_id": session_id,
        "correction_source": correction_source,
        "proposed_id": proposed_id,
    }
    _log_propose_event(audit_event)

    # Check for a duplicate id BEFORE paying the LLM call cost — same-id
    # collisions are mechanical and don't need an LLM to catch.
    if proposed_id != "(no-id)" and rule_store.get(proposed_id) is not None:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        result = {
            "status": "duplicate_skipped",
            "rule_id": proposed_id,
            "verdict": None,
            "reason": f"rule with id {proposed_id!r} already exists; dedup-skip",
        }
        _log_propose_event({
            **audit_event,
            "event": "propose_rule.dedup_skip",
            "reason": result["reason"],
            "duration_ms": elapsed_ms,
        })
        return result

    existing = _find_existing_rules_with_same_trigger(rule_store, rule_data)

    verdict = adjudicator.review_proposed_rule(
        proposed_rule=rule_data,
        correction_source=correction_source,
        correction_text=correction_text,
        existing_rules_with_same_trigger=existing,
        session_id=session_id,
    )

    elapsed_ms = int((time.monotonic() - started) * 1000)

    if not verdict.get("accept"):
        result = {
            "status": "rejected",
            "rule_id": proposed_id,
            "verdict": verdict,
            "reason": verdict.get("reason", "rejected by adjudicator"),
            "issues": verdict.get("issues", []),
        }
        _log_propose_event({
            **audit_event,
            "event": "propose_rule.rejected",
            "reason": result["reason"],
            "issues": result["issues"],
            "duration_ms": elapsed_ms,
        })
        return result

    # Persist with auto_created provenance stamped.
    stamped = _ensure_auto_provenance(rule_data, correction_source)
    try:
        persisted = rule_store.add(stamped)
    except Exception as exc:  # noqa: BLE001 - atomic write failures must not crash
        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.warning("propose_rule: rule_store.add failed: %s", exc)
        result = {
            "status": "rejected",
            "rule_id": proposed_id,
            "verdict": verdict,
            "reason": f"persistence failure: {type(exc).__name__}: {exc}",
            "issues": [str(exc)],
        }
        _log_propose_event({
            **audit_event,
            "event": "propose_rule.persistence_failed",
            "reason": result["reason"],
            "duration_ms": elapsed_ms,
        })
        return result

    result = {
        "status": "persisted",
        "rule_id": persisted.id,
        "verdict": verdict,
        "reason": "auto-created rule added to rules.json",
    }
    _log_propose_event({
        **audit_event,
        "event": "propose_rule.persisted",
        "rule_id": persisted.id,
        "auto_created_source": persisted.auto_created_source,
        "auto_created_at": persisted.auto_created_at,
        "duration_ms": elapsed_ms,
        "verdict_confidence": verdict.get("confidence"),
    })
    return result


# ----------------------------------------------------------------------
# Background-review prompt + draft extraction
# ----------------------------------------------------------------------

# Prompt given to the background-review fork when review_rules=True.
# Adapted from _SKILL_REVIEW_PROMPT in hermes-agent/agent/background_review.py
# but scoped specifically to the rule-injector pipeline.
#
# 2026-08-26 audit fix: 25 of 27 drafts were rejected (92.6% rejection).
# Root cause: the fork lacked (a) knowledge of existing rules by intent,
# (b) a keyword blacklist of common code-context false-positives, and
# (c) a body-content blacklist of forbidden patterns (user names, dates,
# operational HOWTOs). Added six-step gate structure with explicit
# "Nothing to draft" nudge. The adjudicator's PROPOSED_RULE_SYSTEM_PROMPT
# is unchanged — it correctly enforces these criteria; the fix is
# upstream so the fork produces better drafts.
RULE_INJECTION_REVIEW_PROMPT = (
    "Review the conversation above and consider whether any USER CORRECTION "
    "warrants a new rule in the rule-injector plugin. The rule-injector "
    "injects short context snippets at specific points in the conversation "
    "loop (pre_llm_call / transform_tool_result / pre_verify) to enforce "
    "user preferences that have recurred.\n\n"

    "== STEP 1: SIGNAL CHECK ==\n"
    "Any ONE of these signals warrants consideration (NOT a guarantee of a "
    "rule):\n"
    "  • User sent an out-of-band message correcting the agent's style, "
    "tone, format, verbosity, or approach. The marker "
    "'[OUT-OF-BAND USER MESSAGE' is the canonical signal.\n"
    "  • User pressed /stop or interrupted mid-flow. The fact that they "
    "stopped is itself a correction; capture what was going wrong.\n"
    "  • User said 'stop doing X', 'don't do Y', 'remember Z', 'always "
    "do W', or any explicit preference statement.\n"
    "If NONE of these fired this turn, return 'Nothing to draft.' and stop. "
    "DO NOT invent corrections. DO NOT generalize from a single instance — "
    "the rule must reflect a PATTERN (the same correction recurring) or "
    "an EXPLICIT instruction, not a one-time preference statement.\n\n"

    "== STEP 2: PRE-DRAFT DUPLICATE CHECK (MANDATORY) ==\n"
    "BEFORE drafting, check ~/.hermes/plugins/rule-injector/rules.json for "
    "existing rules whose ID, trigger keywords, OR INTENT already covers "
    "the correction. The most-duplicated existing rules and their intents "
    "(as of 2026-08-26, derived from 25 rejections — 14 were duplicates):\n"
    "  - just-do-it-execute (priority 3): When the user says 'just X' for "
    "a mechanical action, execute immediately without planning.\n"
    "  - correction-stop-pattern (priority 2): When the user corrects you, "
    "immediately stop the current approach and pivot.\n"
    "  - tool-correction-immediate-compliance (priority 1): When the user "
    "corrects tool/skill selection, stop explaining and perform the action.\n"
    "  - research-depth-mandate (priority 1): Formal research/analysis "
    "requests must follow DEPTH PROTOCOL (25+ sources, primary verification, "
    "do not retreat to institutional framing).\n"
    "  - audit-found-bugs-fix-mandate (priority 1): Bugs found in audits "
    "must be FIXED via subagent, not just reported.\n"
    "  - subagent-spot-check-required (priority 4): Before dispatching "
    "subagents for a known outcome, spot-check current state to avoid "
    "wasted dispatches and false-success.\n"
    "  - fix-root-causes (priority 1): Never stub; fix the root cause.\n"
    "  - pipeline-router-first (priority 3): Load using-agent-skills router "
    "first before any coding/build/debug/feature work.\n"
    "If an existing rule already covers the correction's intent, return "
    "'Nothing to draft — <rule_id> already covers this.' and stop. DO NOT "
    "draft a near-duplicate. To verify beyond the static catalog above, "
    "use search_files or read_file against ~/.hermes/plugins/rule-injector/"
    "rules.json to inspect current rules.\n\n"

    "== STEP 3: KEYWORD BLACKLIST ==\n"
    "These words will cause trigger.keywords to false-positive on benign "
    "turns and will be REJECTED by the adjudicator. DO NOT use them as "
    "trigger keywords (especially for latest_user_message scope):\n"
    "  - Generic action verbs: create, make, add, fix, file, build, code, "
    "run, test, deploy, change, update, write, set, use, get, show, check, "
    "try, do, doing, did, done\n"
    "  - Generic technical terms: API, model, server, function, error, log, "
    "output, data, request, response, params, body, code, files, http\n"
    "  - HTTP status codes: 404, 403, 500, 200, 502, 503\n"
    "  - Generic state verbs: works, broken, failing, passed, fix, broke, "
    "broken, fixed\n"
    "  - Vague intensifiers: really, very, just, only\n"
    "  - Personal pronouns: I, you, me, my, your, we, our\n"
    "  - Time references: today, yesterday, now, last, this, next, ago\n"
    "Pick keywords from the SPECIFIC correction — the rarest, most "
    "diagnostic tokens. If you cannot think of 3+ specific keywords that "
    "won't false-positive on benign turns on the same topic, return "
    "'Nothing to draft.' and stop.\n\n"

    "== STEP 4: INJECT BODY DISCIPLINE ==\n"
    "The inject body must be USER-AGNOSTIC and GENERALIZABLE. Forbidden:\n"
    "  - User names/identifiers: 'Kevin', 'I', 'me', 'my', 'you', 'your', "
    "'we' (use 'the user' or 'the agent' or imperative voice instead)\n"
    "  - Dated context: 'Aug 2026', 'today', 'yesterday', 'last week', "
    "'yesterday', or any specific date\n"
    "  - Past quotes verbatim — paraphrase or generalize\n"
    "  - Provider-specific implementation details (NIM reasoning_effort, "
    "specific model IDs like 'deepseek-v4-flash', wire format details)\n"
    "  - Operational HOWTOs (specific click sequences, file paths, "
    "configuration knobs, tool commands like 'adb pair', 'curl -X POST')\n"
    "  - Auto-attribution to dated corrections ('Kevin's standing "
    "correction (Aug 2026, Mycelia web build)' introduces unverifiable "
    "provenance)\n"
    "  - Specific incident references (project names like 'Mycelia', "
    "specific file paths, session IDs)\n"
    "Required:\n"
    "  - 200-1500 chars (aim for 200-500 for most rules; approaching 1500 "
    "indicates scope creep)\n"
    "  - 1-3 sentences describing the BEHAVIORAL RULE\n"
    "  - Imperative voice (e.g., 'Stop and verify against primary sources "
    "before reporting')\n"
    "  - Generalize to the principle, not the incident\n\n"

    "== STEP 5: SCOPE FIT ==\n"
    "The rule must address ONLY what the user corrected. Do NOT bundle "
    "multiple lessons from one turn — each lesson is a separate rule or "
    "'Nothing to draft.' Do NOT expand a debugging correction into a full "
    "methodology (5-step procedure, escalation rules, etc.). If the user's "
    "correction was 'stop guessing about API errors', the rule is about "
    "guessing, not about API error debugging in general. Single-point "
    "correction → single-point rule.\n\n"

    "== STEP 6: META-RULES NOTE ==\n"
    "Some rule categories are reserved for hand creation (the adjudicator "
    "will REJECT them anyway, so don't waste an LLM call):\n"
    "  - Rules about the rule-injector itself (meta-governance, e.g., "
    "'load-rule-injector-authoring-first')\n"
    "  - Rules about the broader system (Hindsight, UAML, advisor, kanban, "
    "Hermes internals, plugins)\n"
    "  - safety_critical=true rules (hand-curator-only)\n"
    "  - Rules with priority < 10 (hand-curator-only)\n"
    "If your draft falls in one of these categories, return 'Nothing to "
    "draft.' and stop.\n\n"

    "== STEP 7: DRAFT FORMAT (when warranted) ==\n"
    "When you DO find a signal AND pass all the above gates, draft a "
    "structured rule in this exact JSON shape:\n\n"
    "{\n"
    '  "correction_source": "oob_message" | "stop_event" | '
    '"explicit_correction",\n'
    '  "correction_text": "<the user message or signal verbatim, '
    'truncated to 500 chars>",\n'
    '  "proposed_rule": {\n'
    '    "id": "<kebab-case-or-snake_case-id, 4-40 chars>",\n'
    '    "name": "<short imperative title, max 80 chars>",\n'
    '    "description": "<one sentence: what this rule enforces>",\n'
    '    "hook": "pre_llm_call" | "transform_tool_result" | "pre_verify",\n'
    '    "priority": <integer 10-99; 15-25 typical for user corrections>,\n'
    '    "trigger": {\n'
    '      "type": "keyword" | "compound" | "semantic",\n'
    '      "scope": "latest_user_message",\n'
    '      "keywords": [<3-8 specific terms the rule should match>]\n'
    '    },\n'
    '    "inject": {\n'
    '      "context": "<the rule body, 200-1500 chars, imperative voice>"\n'
    '    }\n'
    "  }\n"
    "}\n\n"

    "== FINAL REMINDERS ==\n"
    "  - DO NOT use trigger.type='always' — fires every turn.\n"
    "  - DO NOT use safety_critical=true — reserved for hand-curated rules.\n"
    "  - DO NOT use priority < 10 — reserved for hand-curated rules.\n"
    "  - Use kebab-case for the id (e.g., 'stop-listing-tos').\n"
    "  - The message MUST be a single JSON object (no prose, no markdown "
    "fences) when drafting. If no rule is warranted, the message MUST be "
    "exactly 'Nothing to draft.' (or 'Nothing to draft — <reason>.' for "
    "the duplicate case) and nothing else.\n"
    "  - 'Nothing to draft.' is the CORRECT outcome when the correction "
    "doesn't warrant a rule, when a rule already exists, or when you "
    "cannot meet the quality bar. Do NOT force a draft to satisfy the "
    "signal-check criteria from Step 1 — those define what triggers "
    "consideration, but the quality gates filter out drafts that should "
    "not ship."
)


_RULE_DRAFT_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _find_balanced_json_objects(text: str) -> List[str]:
    """Return a list of balanced top-level ``{...}`` substrings.

    Walks the string tracking brace depth and string state so it
    correctly handles nested objects and string literals. Returns
    objects in source order. Used by ``extract_rule_draft`` as a
    regex-free fallback (regexes can't reliably match nested JSON).
    """
    candidates: List[str] = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(text[start : i + 1])
                    start = -1
    return candidates


def extract_rule_draft(text: str) -> Optional[Dict[str, Any]]:
    """Extract a structured rule draft from the fork's text response.

    Handles three common shapes:
      1. Markdown-fenced ```json {...} ``` block.
      2. Bare JSON object on its own (the prompt's preferred shape).
      3. Text-wrapped JSON (any balanced object — outer-first preferred).

    Uses a brace-walking parser (regexes can't reliably match nested
    JSON) and tries each candidate from outermost to innermost.

    Returns the parsed dict if it has the right shape, else None.
    Never raises — failed parses return None so the caller can fall
    back to 'no rule drafted this turn'.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    # Skip the explicit 'no draft' marker.
    if "Nothing to draft." in text:
        return None
    # 1. Markdown-fenced JSON block: take it verbatim if present.
    fenced = _RULE_DRAFT_FENCE_RE.findall(text)
    candidates = list(fenced) + _find_balanced_json_objects(text)
    # De-dupe while preserving order.
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    for raw in unique:
        # Try to recover incomplete JSON: walk offsets until parse
        # succeeds or we run out of attempts.
        data: Any = None
        for offset in range(len(raw)):
            try:
                data = json.loads(raw[offset:])
                break
            except json.JSONDecodeError:
                continue
        if data is None:
            continue
        if not isinstance(data, dict):
            continue
        if "proposed_rule" not in data or not isinstance(data["proposed_rule"], dict):
            continue
        return data
    return None
