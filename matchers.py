"""Matchers for the rule-injector plugin.

3-stage pipeline: fast reject -> scope extraction -> match evaluation.
Rules are compiled at load time for performance — keyword matching is O(n_keywords)
per rule, negligible for <100 rules.

Semantic matching layer: uses sentence-transformers (all-MiniLM-L6-v2, 384-dim,
CPU) for cosine-similarity matching against reference phrases. Lazy-loaded singleton
— model only imports when a semantic rule is first evaluated. Graceful degradation
if sentence-transformers is unavailable.

Compound triggers support "operator" field: "and" (default, backward compat) or
"or" (short-circuit, cost-ordered conditions — keyword checks before semantic).
"""

import re
import os
import sys
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Set, Dict, List

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(name)s [%(levelname)s] %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = True

try:
    from . import domain_inference
except ImportError:  # standalone/legacy import path (tests, scripts run from plugin dir)
    import domain_inference

# Persistent verdict log: every adjudicator decision is appended to a 48h
# JSONL file at ~/.hermes/logs/rule-injector-adjudicator-verdicts.jsonl so
# we can audit "what did the LLM see and decide" without relying on the
# in-memory deque. The script `audit-adjudicator.py` lives next to this
# module under ~/.hermes/scripts/.
_ADJUDICATOR_LOG_DIR = Path.home() / ".hermes" / "scripts"
try:
    if str(_ADJUDICATOR_LOG_DIR) not in sys.path:
        sys.path.insert(0, str(_ADJUDICATOR_LOG_DIR))
    import adjudicator_verdict_log as _adj_verdict_log  # type: ignore
except Exception as _e:  # noqa: BLE001
    # Don't block plugin load if the helper script isn't present yet.
    _adj_verdict_log = None  # type: ignore
    logger.warning("adjudicator_verdict_log helper not importable: %s", _e)

INJECTION_CHAR_BUDGET = 50000

# F3 (2026-08-04 audit): per-tool-result cap for transform_tool_result
# suffixes. Smaller than the pre_llm_call budget because this hook fires on
# EVERY tool call; 6000 chars of appended guidance is already a lot for one
# tool result. The transform path previously had NO budget at all.
TRANSFORM_SUFFIX_BUDGET = 6000


def _sanitize_preview(text: Any, max_len: int = 500) -> str:
    """Sanitize untrusted preview text before it reaches the adjudicator LLM.

    Strips control characters (including null bytes and newline-injection
    payloads), collapses whitespace, and caps length. Preview text comes from
    tool results/args which are untrusted data; sanitizing limits both the
    injection surface and token cost.
    """
    if text is None:
        return ""
    s = str(text)
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len]


def _detect_cron_context(session_id: str, task_id: str, user_message: str,
                         explicit: bool = False) -> bool:
    """Identify cron runs using stable IDs, explicit metadata, or known wrappers."""
    if explicit or session_id.startswith("cron_") or task_id.startswith("cron_"):
        return True
    return bool(re.search(
        r"^(?:IMPORTANT:\s*)?You are (?:running as )?a scheduled cron job\b",
        (user_message or "")[:500], re.IGNORECASE,
    ))

# D4/D5 FIX (2026-07-14): Message-pattern blocklist.
# System-generated messages (cron intros, reflection instructions, Discord
# wrappers) contain development keywords ("stop", "app", "configure", "audit",
# "integrate", etc.) that trigger dozens of false-positive keyword rules.
# When the latest user message matches one of these patterns, skip ALL
# keyword-based pre_llm_call rules. Only non-keyword triggers (post_compaction,
# attempt_count) remain active.
# NOTE: The "^\[Triggering message id:" pattern was removed (2026-07-21).
# Discord gateway messages that start with this wrapper were being incorrectly
# blocklisted, suppressing rules on legitimate user content. The wrapper prefix
# is stripped by the gateway before the message reaches the agent, so this
# pattern was always wrong for Discord sessions.
_BLOCKLIST_PATTERNS = [
    re.compile(r"^Review the conversation above", re.IGNORECASE),
    re.compile(r"^IMPORTANT: You are running as a scheduled cron", re.IGNORECASE),
    re.compile(r"^You are a scheduled cron job", re.IGNORECASE),
    re.compile(r"^You are the orchestrator of", re.IGNORECASE),
    re.compile(r"^\[The user sent a text document:", re.IGNORECASE),
    re.compile(r"^\[Note: model was just switched", re.IGNORECASE),
    # D6 FIX (2026-07-21): Meta-discussion detector.
    # When Kevin talks about the rule injector itself ("you just got 1000× smarter",
    # "rule injector working", "HARD RULE", "[Rule: ...]"), the META conversation
    # contains the same keywords the rules fire on, causing every rule to fire
    # simultaneously — injecting irrelevant guidance.
    # Example real FP (2026-07-20, Discord): Kevin said "I was seriously struggling
    # getting you to change the Hermes config in another session and you did it first
    # try with the rule injector working" — triggered correction-stop-pattern,
    # pipeline-router-first, brief-advisor-with-skill-methodology, research-router-first.
    # The agent rationalized it as "self-referential rule activation" but the
    # guidance was irrelevant to the conversation. Now suppressed.
    re.compile(r"\brule[- ]injector\b", re.IGNORECASE),
    re.compile(r"\brule injector working\b", re.IGNORECASE),
    re.compile(r"\brule.json\b", re.IGNORECASE),
    re.compile(r"\bHARD RULE:", re.IGNORECASE),
    re.compile(r"\[Rule: [a-z][\w-]+\]", re.IGNORECASE),
    re.compile(r"\btrace[ -]?logs?\b", re.IGNORECASE),
]


def _is_blocklisted_message(user_message: str) -> bool:
    """Check if a user message is a system-generated wrapper that should bypass keyword rules."""
    if not user_message:
        return False
    # Check the first 500 chars for system-wrapper patterns (anchored ^ regexes)
    head = user_message[:500]
    head_match = any(p.search(head) for p in _BLOCKLIST_PATTERNS)
    if head_match:
        return True
    # D6 FIX (2026-07-21): Also check the rest of the message for meta-discussion
    # patterns, but cap at 4KB total scan to avoid huge tool results / files.
    # Meta-discussion keywords ("rule-injector", "[Rule: ...]") typically appear
    # mid-message; if they're past the first 4KB, the agent has been reading a LOT
    # of content and the keyword is probably incidental (e.g., a comment in source).
    scan_target = user_message[:4096] if len(user_message) > 4096 else user_message
    full_match = any(
        not p.pattern.startswith("^") and p.search(scan_target)
        for p in _BLOCKLIST_PATTERNS
    )
    return full_match


def _extract_recent_assistant(conversation_history, max_turns: int = 3) -> str:
    """Extract last N assistant text turns from conversation history.

    Used by both pre_llm_call and transform_tool_result so the recent_assistant
    scope is consistent across hooks. Filters out tool-call assistant messages
    (which have empty content) — only captures narrative reasoning text.
    """
    if not conversation_history:
        return ""
    parts = []
    for m in conversation_history:
        if isinstance(m, dict) and m.get("role") == "assistant":
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                parts.append(content[:500])
    return " ".join(parts[-max_turns:]) if parts else ""


# Cost ordering for compound OR short-circuit — cheapest triggers first
# so semantic (~30ms) is never evaluated if a keyword (~0ms) already matches.
_TRIGGER_COST = {
    "keyword": 0,
    "keyword_flexible": 1,
    "regex": 2,
    "tool_name": 3,
    "result_truncated": 4,
    "empty_result": 5,
    "tool_result_contains": 6,
    "result_contains_keywords": 7,
    "attempt_count": 8,
    "post_compaction": 9,
    "semantic": 10,  # most expensive — ~30ms per query encode
}


@dataclass
class MatchContext:
    """Normalized context passed to matchers."""
    hook: str
    user_message: str = ""
    recent_context: str = ""  # last ~4 messages concatenated
    recent_assistant: str = ""  # NEW (2026-07-21): last 2-3 assistant turns only
    tool_name: str = ""
    tool_result: str = ""
    tool_args: dict = None
    conversation_history: list = None
    attempt_count: int = 0  # computed for attempt_count rules
    is_first_turn: bool = False
    is_post_compaction: bool = False
    is_cron: bool = False  # NEW (2026-07-21): True for cron-triggered sessions
    skill_loaded_this_session: set = None  # set of skill names loaded
    final_response: str = ""  # agent's draft response (pre_verify only)
    # Plan 005 (2026-08-03): set of tool names used earlier in this session,
    # used to enforce rule.trigger.required_tools. Populated from
    # conversation_history in each hook.
    tools_used_this_session: set = None


class CompiledRule:
    """Pre-compiled rule for fast matching."""

    def __init__(self, rule):
        self.id = rule.id
        self.hook = rule.hook
        self.priority = rule.priority
        self.trigger = rule.trigger
        self.inject = rule.inject
        # F7 (2026-08-04 audit): carry Plan 007 metadata through to the
        # compiled form — the adjudicator candidate payload reads these.
        self.domain = getattr(rule, "domain", "")
        self.applicable_when = getattr(rule, "applicable_when", None) or []
        self.not_applicable_when = getattr(rule, "not_applicable_when", None) or []
        self.safety_critical = bool(getattr(rule, "safety_critical", False))
        self._compile()

    def _compile(self):
        t = self.trigger
        # Keyword patterns for word-boundary matching
        self._keyword_patterns: List[re.Pattern] = []
        for kw in t.get("keywords", []):
            # Escape special regex chars, add word boundaries
            pattern = r'\b' + re.escape(kw) + r'\b'
            self._keyword_patterns.append(re.compile(pattern, re.IGNORECASE if t.get("case_insensitive", True) else 0))
        
        # For keyword_flexible trigger type - reads from result_contains_keywords
        for kw in t.get("result_contains_keywords", []):
            # Escape special regex chars, NO word boundaries for flexible matching
            pattern = re.escape(kw)
            self._keyword_patterns.append(re.compile(pattern, re.IGNORECASE if t.get("case_insensitive", True) else 0))
        
        # Regex patterns (pre-compiled)
        ci = t.get("case_insensitive", True)
        self._regex_patterns = [
            re.compile(p, re.IGNORECASE if ci else 0)
            for p in t.get("patterns", [])
        ]
        # Tool names for tool-based triggers
        self._tool_names = set(t.get("tool_names", []))
        # Match all keywords (AND) vs any (OR)
        self._match_all = t.get("match_all", False)
        # Min attempts
        self._min_attempts = t.get("min_attempts", 0)
        # Exclude if skill loaded
        self._exclude_skills = set(t.get("exclude_if_skill_loaded", []))
        self._exclude_context_keywords = tuple(
            kw.lower() for kw in t.get("exclude_if_recent_context_keywords", [])
            if isinstance(kw, str) and kw
        )
        self._cooldown_turns = max(0, int(t.get("cooldown_turns", 0) or 0))
        self._max_fires_per_session = max(0, int(t.get("max_fires_per_session", 0) or 0))
        # NEW (2026-07-21): Exclude rule in cron sessions
        self._exclude_if_cron: bool = t.get("exclude_if_cron", False)
        # NEW (2026-07-21): Per-rule opt-in for meta-discussion suppression.
        # When True, this rule is skipped when the user message matches the
        # meta-discussion blocklist (talking about rule-injector itself).
        # When False (default), the rule fires normally even during meta-discussion.
        # This replaces the blanket suppression that was too aggressive (D6→D11 fix).
        self._suppress_on_meta: bool = t.get("suppress_on_meta_discussion", False)
        # Compliance metadata is declarative and used by injected directives.
        self._required_tools = set(t.get("required_tools", []))
        self._required_skills = set(t.get("required_skills", []))
        # Result conditions
        self._result_truncated = t.get("result_truncated", False)
        self._empty_result = t.get("empty_result", False)
        # For result_contains_keywords trigger
        self._result_contains_patterns: List[re.Pattern] = []
        for kw in t.get("result_contains_keywords", []):
            pattern = r'\b' + re.escape(kw) + r'\b'
            self._result_contains_patterns.append(re.compile(pattern, re.IGNORECASE))
        
        # Semantic trigger — store references as raw text, embeddings back-filled lazily
        self._semantic_references: List[str] = t.get("references", [])
        self._semantic_threshold: float = t.get("threshold", 0.25)
        self._semantic_embeddings = None  # lazy: computed on first match
        # Store full trigger for _match_semantic access
        if t.get("type") == "semantic":
            self._semantic_trigger = {
                "references": self._semantic_references,
                "threshold": self._semantic_threshold,
            }
        else:
            self._semantic_trigger = None

        # Pre-compile compound conditions
        self._compiled_conditions = []
        self._compound_operator = t.get("operator", "and")  # default AND for backward compat
        conditions = t.get("conditions", [])
        # For OR compounds, sort by cost so cheapest conditions short-circuit first
        if self._compound_operator == "or" and conditions:
            conditions = sorted(conditions, key=lambda c: _TRIGGER_COST.get(c.get("type", "keyword"), 5))
        for cond in conditions:
            temp_rule = CompiledRule.__new__(CompiledRule)
            temp_rule.trigger = cond
            temp_rule._compile()
            self._compiled_conditions.append(temp_rule)

    def matches(self, ctx: MatchContext) -> bool:
        """Check if this rule's trigger matches the current context."""
        # Fast reject: if skill loaded and exclusion applies
        if self._exclude_skills and ctx.skill_loaded_this_session:
            if self._exclude_skills & ctx.skill_loaded_this_session:
                return False
        if self._exclude_context_keywords and ctx.recent_context:
            recent_lower = ctx.recent_context.lower()
            if any(keyword in recent_lower for keyword in self._exclude_context_keywords):
                return False
        # NEW (2026-07-21): Skip rule entirely in cron sessions if exclude_if_cron is set
        if self._exclude_if_cron and ctx.is_cron:
            return False
        # Plan 005 (2026-08-03): required_tools — rule fires only if all listed
        # tools have been used this session. Empty list = no constraint.
        if self._required_tools and ctx.tools_used_this_session:
            if not self._required_tools.issubset(ctx.tools_used_this_session):
                return False
        elif self._required_tools and not ctx.tools_used_this_session:
            return False
        # Plan 005 (2026-08-03): required_skills — rule fires only if all listed
        # skills have been loaded this session. Reuses the existing
        # skill_loaded_this_session field.
        if self._required_skills and ctx.skill_loaded_this_session:
            if not self._required_skills.issubset(ctx.skill_loaded_this_session):
                return False
        elif self._required_skills and not ctx.skill_loaded_this_session:
            return False

        trigger_type = self.trigger.get("type", "keyword")
        scope = self.trigger.get("scope", "latest_user_message")

        if trigger_type == "keyword":
            return self._match_keywords(ctx, scope)
        elif trigger_type == "regex":
            return self._match_regex(ctx, scope)
        elif trigger_type == "tool_name":
            return self._match_tool_name(ctx)
        elif trigger_type == "tool_result_contains":
            return self._match_tool_result(ctx)
        elif trigger_type == "result_truncated":
            return self._match_result_truncated(ctx)
        elif trigger_type == "empty_result":
            return self._match_empty_result(ctx)
        elif trigger_type == "attempt_count":
            return ctx.attempt_count >= self._min_attempts
        elif trigger_type == "post_compaction":
            return ctx.is_post_compaction
        elif trigger_type == "result_contains_keywords":
            return self._match_result_contains_keywords(ctx)
        elif trigger_type == "keyword_flexible":
            return self._match_keyword_flexible(ctx, scope)
        elif trigger_type == "semantic":
            return self._match_semantic(ctx, scope)
        elif trigger_type == "compound":
            return self._match_compound(ctx)
        elif trigger_type == "always":
            # Unconditional trigger — always matches regardless of context.
            # Used for rules that must fire on every single turn (e.g.,
            # interleaved thinking enforcement).  Subject to exclude_skills,
            # exclude_context_keywords, and exclude_if_cron guards above.
            return True
        return False

    def _get_scope_text(self, ctx: MatchContext, scope: str) -> str:
        if scope == "latest_user_message":
            return ctx.user_message or ""
        elif scope == "recent_context":
            # For pre_verify, include the agent's draft response so completion
            # claims ("Done", "complete", "finished") are visible to keyword
            # matching. Without this, pre_verify rules can't see the text they
            # are supposed to gate (it hasn't entered history yet).
            parts = []
            if ctx.recent_context:
                parts.append(ctx.recent_context)
            if ctx.hook == "pre_verify" and ctx.final_response:
                parts.append(ctx.final_response)
            return " ".join(parts) if parts else ""
        elif scope == "recent_assistant":
            # NEW (2026-07-21): Last 2-3 assistant turns only. For agent-autonomous
            # triggering of UE-skill rules that the agent should follow while
            # working independently. The agent's own narration ("I'll write a
            # UCLASS for the creature", "this needs a Blueprint subclass") is the
            # signal surface these rules need.
            if ctx.recent_assistant:
                return ctx.recent_assistant
            # Fall back to extracting from conversation history if not pre-computed
            parts = []
            for m in (ctx.conversation_history or [])[-6:]:
                if isinstance(m, dict) and m.get("role") == "assistant":
                    content = m.get("content", "")
                    if isinstance(content, str) and content.strip():
                        parts.append(content[:500])
            return " ".join(parts[-3:]) if parts else ""
        elif scope == "tool_result":
            return ctx.tool_result or ""
        elif scope == "tool_args":
            # NEW (2026-07-21): Extract text from tool call arguments so
            # transform_tool_result rules can match on file content, paths,
            # and patch content — not just the tool result string.
            # This addresses Discrepancy #3: semantic-matching-placeholder-detection
            # and fix-root-causes/no-placeholders could never see what was
            # being written because content is in args, not result.
            args = ctx.tool_args
            if not args or not isinstance(args, dict):
                return ""
            # NEW (2026-08-03): Skip tool_args keyword matching for markdown
            # files. The placeholder-content-guard and semantic-matching-
            # placeholder-detection rules fire on English words like "TODO",
            # "stub", "FIXME" that legitimately appear in markdown plans,
            # READMEs, and audit reports. Placeholder detection is meant
            # for SOURCE CODE only — not documentation. C++ rules (cpp-reminder,
            # new-cpp-class-content-check) match on .cpp/.h extensions, not
            # markdown, so this skip does not affect them.
            _path = args.get("path") or ""
            if isinstance(_path, str) and _path.lower().endswith(
                (".md", ".markdown", ".mdx", ".txt", ".rst")
            ):
                return ""
            parts = []
            # write_file: {"path": "...", "content": "..."}
            for key in ("path", "content", "new_string", "old_string", "patch", "code", "command", "text", "url"):
                val = args.get(key)
                if isinstance(val, str) and val:
                    parts.append(val)
                elif isinstance(val, (list, dict)) and val:
                    import json as _json
                    try:
                        parts.append(_json.dumps(val)[:2000])
                    except Exception:
                        pass
            return " ".join(parts)
        elif scope == "conversation":
            return " ".join(
                m.get("content", "") if isinstance(m, dict) else str(m)
                for m in (ctx.conversation_history or [])[-6:]
            )
        return ""

    def _match_keywords(self, ctx, scope) -> bool:
        text = self._get_scope_text(ctx, scope).lower()
        if not text:
            return False
        if self._match_all:
            return all(p.search(text) for p in self._keyword_patterns)
        return any(p.search(text) for p in self._keyword_patterns)

    def _match_regex(self, ctx, scope) -> bool:
        text = self._get_scope_text(ctx, scope)
        if not text:
            return False
        return any(p.search(text) for p in self._regex_patterns)

    def _match_tool_name(self, ctx) -> bool:
        if not self._tool_names:
            return bool(ctx.tool_name)
        return ctx.tool_name in self._tool_names

    def _match_tool_result(self, ctx) -> bool:
        if self._tool_names and ctx.tool_name not in self._tool_names:
            return False
        text = (ctx.tool_result or "").lower()
        if not text:
            return False
        if self._match_all:
            return all(p.search(text) for p in self._keyword_patterns)
        return any(p.search(text) for p in self._keyword_patterns) if self._keyword_patterns else False

    def _match_result_truncated(self, ctx) -> bool:
        if ctx.tool_name != "read_file":
            return False
        result = ctx.tool_result or ""
        if "next_offset" in result:
            return True
        # Match the *value* of the truncation flag, never the bare key. The previous
        # form (`"TRUNCATED" in result.upper()`) uppercased the whole payload, so it
        # matched the JSON key ``"truncated": false`` and fired on EVERY read_file,
        # injecting a false "this file is truncated" claim. (Audit 2026-10-30.)
        return bool(re.search(r'"truncated(_lines|_by)?"\s*:\s*true', result, re.IGNORECASE))

    def _match_empty_result(self, ctx) -> bool:
        result = ctx.tool_result or ""
        # Check for empty/zero results in JSON tool results
        return ('"matches": []' in result or
                '"total_count": 0' in result or
                '"matches": []' in result or
                '"files_only": []' in result or
                '"results": []' in result or
                '"content": ""' in result or
                result.strip() == "" or
                result.strip() == "{}" or
                '"matches_text": ""' in result)

    def _match_result_contains_keywords(self, ctx) -> bool:
        """Match if tool result contains source code keywords with word boundaries."""
        if self._tool_names and ctx.tool_name not in self._tool_names:
            return False
        text = (ctx.tool_result or "").lower()
        if not text:
            return False
        return any(p.search(text) for p in self._result_contains_patterns)

    def _match_keyword_flexible(self, ctx, scope) -> bool:
        """Match keywords without strict word boundaries - for JSON/content matching."""
        # Honor tool_names gate (Pattern 2 fix 2026-07-31): keyword_flexible
        # previously ignored tool_names, letting write-scoped rules fire on
        # read-only tools whose args contained the keyword.
        if self._tool_names and ctx.tool_name not in self._tool_names:
            return False
        text = self._get_scope_text(ctx, scope).lower()
        if not text:
            return False
        if self._match_all:
            return all(p.search(text) for p in self._keyword_patterns)
        return any(p.search(text) for p in self._keyword_patterns)

    def _match_semantic(self, ctx, scope) -> bool:
        """Semantic matching via sentence-transformer embeddings.

        Uses max-similarity across multiple reference phrases. Returns False
        gracefully if model is unavailable or import fails.
        """
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from semantic_matcher import SemanticMatcher

        matcher = SemanticMatcher.get_instance()
        if matcher is None or not matcher.is_available():
            return False

        # Get the trigger configuration (stored in self._semantic_trigger)
        trig = getattr(self, "_semantic_trigger", None)
        if trig is None:
            return False

        references = trig.get("references", [])
        threshold = trig.get("threshold", 0.25)

        if not references:
            return False

        # Lazy back-fill: compute reference embeddings on first match attempt
        if getattr(self, "_semantic_embeddings", None) is None:
            embeddings = matcher.encode_references(references)
            if embeddings is None:
                return False
            self._semantic_embeddings = embeddings

        # Encode the user message (scope-based)
        text = self._get_scope_text(ctx, scope)
        if not text:
            return False

        query_emb = matcher.encode_query(text)
        if query_emb is None:
            return False

        # Max-similarity across all references
        # Both reference and query embeddings are normalized -> matmul = cosine sim
        try:
            similarities = self._semantic_embeddings @ query_emb
            max_sim = float(similarities.max())
            # Log for threshold tuning
            import logging
            logging.getLogger(__name__).debug(
                "semantic match: rule=%s score=%.3f threshold=%.3f match=%s",
                getattr(self, "id", "unknown"), max_sim, threshold, max_sim >= threshold
            )
            return max_sim >= threshold
        except Exception:
            return False

    def _match_compound(self, ctx) -> bool:
        """Compound trigger — supports both AND (default) and OR operators.

        Operator is specified in the trigger as "operator": "and" | "or".
        For OR: conditions are evaluated in cost order (cheapest first) with
        short-circuit evaluation. If any condition matches, return True.
        """
        operator = getattr(self, "_compound_operator", "and")

        if operator == "or":
            # Short-circuit OR: evaluate in cost order, return True on first match
            for temp_rule in self._compiled_conditions:
                if temp_rule.matches(ctx):
                    return True
            return False
        else:
            # AND (default, backward compatible): all must match
            for temp_rule in self._compiled_conditions:
                if not temp_rule.matches(ctx):
                    return False
            return True


class RuleMatcher:
    """Orchestrates rule matching and injection across all hooks."""

    def __init__(self, store, adjudicator=None):
        self.store = store
        self.adjudicator = adjudicator
        self._compiled: list = []
        self._compile_version = -1
        # Incremental detection state
        self._last_skill_scan_index = 0
        self._last_attempt_scan_index = 0
        self._cached_skills_loaded = set()
        self._cached_attempt_count = 0
        # D9 FIX: Cooldown for keyword rules — cache the last user message hash
        # and the rule IDs that fired on it. Skip re-injecting the same rule on
        # subsequent LLM calls with the same user message (multi-turn tool rounds).
        # F4 (2026-08-04 audit): state is keyed PER SESSION — the gateway serves
        # multiple sessions in one process, and a flat set meant session B's new
        # message cleared session A's fired-rule set, defeating the cooldown.
        self._cooldown_state: Dict[str, dict] = {}
        # F10 (2026-08-04 audit): hot-reload version cache (mtime-gated md5).
        self._last_version_mtime: float = 0
        self._last_version: str = ""
        # Ephemeral per-session fire limits. A gateway restart starts a clean epoch.
        self._session_fire_state: Dict[str, dict] = {}

    def _advance_session_turn(self, session_id: str, user_message: str) -> int:
        """Return a monotonically increasing distinct-user-message turn."""
        if not session_id:
            return 0
        state = self._session_fire_state.setdefault(
            session_id, {"turn": 0, "message_hash": "", "fires": {}}
        )
        msg_hash = hashlib.md5((user_message or "").encode()).hexdigest()[:16]
        if msg_hash != state["message_hash"]:
            state["turn"] += 1
            state["message_hash"] = msg_hash
        return state["turn"]

    def _session_rule_can_fire(self, rule, session_id: str, turn: int) -> bool:
        if not session_id:
            return True
        entry = self._session_fire_state[session_id]["fires"].get(rule.id)
        if not entry:
            return True
        if rule._max_fires_per_session and entry["count"] >= rule._max_fires_per_session:
            return False
        if rule._cooldown_turns and turn - entry["last_turn"] <= rule._cooldown_turns:
            return False
        return True

    def _record_session_rule_fire(self, rule, session_id: str, turn: int) -> None:
        if not session_id:
            return
        fires = self._session_fire_state[session_id]["fires"]
        entry = fires.setdefault(rule.id, {"count": 0, "last_turn": -1})
        entry["count"] += 1
        entry["last_turn"] = turn

    def _adjudicate_injections(
        self,
        injections: list,
        hook_name: str,
        session_id: str,
        user_message: str = "",
        recent_context: str = "",
        recent_assistant: str = "",
        extra_context: Optional[Dict[str, Any]] = None,
        log_path=None,  # Plan 003 (2026-08-03): test-only override
    ) -> list:
        """Run the optional adjudicator and return the filtered injection list.

        Returns the original list unchanged if no adjudicator is wired, the
        adjudicator is disabled, or the call returns a fallback verdict.

        ``extra_context`` (2026-07-31) carries hook-specific context —
        tool_name/tool_args_preview/tool_result_preview for
        transform_tool_result, final_response_preview for pre_verify. All
        preview text is sanitized (control chars stripped, length-capped)
        before it reaches the adjudicator LLM, which is instructed to treat
        it as untrusted data.
        """
        if self.adjudicator is None or not injections:
            return injections
        if not getattr(self.adjudicator.config, "enabled", False):
            return injections
        try:
            # F7 (2026-08-04 audit): surface each rule's explicit domain and
            # safety_critical flag (RuleStore now preserves them). Previously
            # the payload hard-coded safety_critical=False, so the adjudicator
            # safety floor could never engage for any rule.
            rule_by_id = {r.id: r for r in self._compiled}
            candidate_payload = [
                {
                    "rule_id": rid,
                    "rule_name": rid,
                    "rule_type": "guidance",
                    "priority": prio,
                    "injection_length_chars": len(text),
                    "injection_text_preview": text[:400],
                    "injection_text_truncated": len(text) > 400,
                    "domain": domain_inference.resolve_domain(
                        rid, getattr(rule_by_id.get(rid), "domain", "") or None
                    ),
                    "applicable_when": list(
                        getattr(rule_by_id.get(rid), "applicable_when", None) or []
                    ),
                    "not_applicable_when": list(
                        getattr(rule_by_id.get(rid), "not_applicable_when", None) or []
                    ),
                    "safety_critical": bool(
                        getattr(rule_by_id.get(rid), "safety_critical", False)
                    ),
                }
                for text, prio, rid in injections
            ]
            context_summary = {
                "hook": hook_name,
                "latest_user_message": (user_message or "")[:1000],
                "recent_context": (recent_context or "")[:1500],
                "recent_assistant": (recent_assistant or "")[:1500],
            }
            if extra_context:
                for key, value in extra_context.items():
                    context_summary[key] = _sanitize_preview(value, 500)
            verdict = self.adjudicator.review(
                candidates=candidate_payload,
                blocked=[],
                hook_name=hook_name,
                session_id=session_id or "unknown",
                context_summary=context_summary,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("adjudicator: review call failed: %s", exc)
            return injections

        # Persist the verdict for offline auditing. Best-effort: a logging
        # failure must never break the rule pipeline. See
        # ~/.hermes/scripts/adjudicator_verdict_log.py for the on-disk
        # schema and ~/.hermes/scripts/audit-adjudicator.py for the CLI.
        if _adj_verdict_log is not None:
            try:
                cfg = getattr(self.adjudicator, "config", None)
                # F2 (2026-08-04 audit): on fallback this caller keeps the full
                # deterministic set (fail-closed, documented behavior). The
                # adjudicator's own fallback partition (keep-safety-only) is
                # NOT applied by this caller, so the on-disk verdict must
                # record what actually happened — everything was injected.
                # Without this, the audit trail claims rules were dropped
                # when they were appended to the context.
                log_verdict = dict(verdict)
                if verdict.get("status") != "applied":
                    log_verdict["keep_candidates"] = sorted(
                        {str(c.get("rule_id", "")) for c in candidate_payload}
                    )
                    log_verdict["drop_candidates"] = []
                # Plan 003 (2026-08-03): log_path override (test-only).
                # Honor caller-provided log_path, then env var (RULE_INJECTOR_TEST_LOG_PATH),
                # then module default. Production callers pass None → env var
                # not set → production log file.
                import os
                _env_path = os.environ.get("RULE_INJECTOR_TEST_LOG_PATH")
                resolved_log_path = (
                    log_path
                    or (Path(_env_path) if _env_path else None)
                    or _adj_verdict_log.DEFAULT_LOG_PATH
                )
                _adj_verdict_log.append_verdict(
                    verdict={
                        **log_verdict,
                        "session_id": session_id or "unknown",
                        "hook_name": hook_name,
                        "blocked_in": 0,
                        "provider": getattr(cfg, "provider", "") if cfg else "",
                        "model": getattr(cfg, "model", "") if cfg else "",
                    },
                    candidates=candidate_payload,
                    context_summary={
                        **context_summary,
                        "session_id": session_id or "unknown",
                    },
                    log_path=resolved_log_path,
                )
            except Exception as _e:  # noqa: BLE001
                logger.debug("adjudicator_verdict_log append failed: %s", _e)

        if verdict.get("status") != "applied":
            return injections

        drop = set(verdict.get("drop_candidates", []) or [])
        if not drop:
            return injections
        return [(text, prio, rid) for text, prio, rid in injections if rid not in drop]

    def _ensure_compiled(self):
        """Recompile if rules changed (hot-reload). Uses file mtime + content hash."""
        rules = self.store.load()
        version = self._get_rules_version()
        if version != self._compile_version:
            compiled = []
            for r in rules:
                if not r.enabled:
                    continue
                try:
                    compiled.append(CompiledRule(r))
                except Exception as exc:  # noqa: BLE001
                    # F1 (2026-08-04 audit): one malformed rule must not take
                    # down the whole injector. Previously a non-string keyword
                    # or invalid regex raised here, the hook wrapper swallowed
                    # it, and ALL rules silently stopped firing with zero logs.
                    logger.warning(
                        "rule-injector: skipping malformed rule %r: %s: %s",
                        getattr(r, "id", "<unknown>"), type(exc).__name__, exc,
                    )
            self._compiled = sorted(compiled, key=lambda r: r.priority)
            # Plan 008 (2026-08-03): O(1) lookup map for the contradiction check
            # in run_pre_llm_call. The previous linear scan was O(n) per
            # injection per turn (137 rules × per-turn scan).
            self._by_id = {r.id: r for r in self._compiled}
            self._compile_version = version

    def _get_rules_version(self) -> str:
        """Get a version string based on file mtime and content hash.

        F10 (2026-08-04 audit): the full-file md5 is only recomputed when the
        mtime changed. Previously every hook call read the entire rules.json
        (~376KB) just to hash it.
        """
        path = self.store.path
        try:
            mtime = os.path.getmtime(path) if os.path.exists(path) else 0
        except OSError:
            return str(len(self.store.load()))
        if mtime == self._last_version_mtime:
            return self._last_version
        try:
            with open(path, 'rb') as f:
                content_hash = hashlib.md5(f.read()).hexdigest()[:8]
        except (OSError, IOError):
            return str(len(self.store.load()))
        self._last_version_mtime = mtime
        self._last_version = f"{mtime}:{content_hash}"
        return self._last_version

    def run_pre_llm_call(self, kwargs) -> dict:
        """Run all pre_llm_call rules. Returns {"context": "..."} or {}."""
        self._ensure_compiled()

        history = kwargs.get("conversation_history", [])
        user_msg = kwargs.get("user_message", "")
        session_id = kwargs.get("session_id", "") or ""
        task_id = kwargs.get("task_id", "") or ""

        # Extract recent context (last 4 messages)
        recent_parts = []
        for m in (history or [])[-4:]:
            if isinstance(m, dict):
                content = m.get("content", "")
                if isinstance(content, str) and content.strip():
                    recent_parts.append(content[:500])  # cap each msg
        recent = " ".join(recent_parts)

        # NEW (2026-07-21): Extract recent assistant turns only (last 3 assistant messages).
        # Powers the recent_assistant scope for agent-autonomous rule triggering.
        recent_assistant = _extract_recent_assistant(history)

        # Detect cron from explicit hook metadata, stable IDs, or the cron wrapper.
        is_cron = _detect_cron_context(
            session_id, task_id, user_msg, bool(kwargs.get("is_cron", False))
        )
        session_turn = self._advance_session_turn(session_id, user_msg)

        # Detect skills loaded this session (incremental)
        skills_loaded = self._detect_skills_loaded_incremental(history)

        # Detect post-compaction
        is_post_compaction = self._detect_post_compaction(history)

        # Count fix attempts (incremental)
        attempt_count = self._count_attempts_incremental(history)

        ctx = MatchContext(
            hook="pre_llm_call",
            user_message=user_msg,
            recent_context=recent,
            recent_assistant=recent_assistant,
            conversation_history=history,
            skill_loaded_this_session=skills_loaded,
            is_first_turn=kwargs.get("is_first_turn", False),
            is_post_compaction=is_post_compaction,
            is_cron=is_cron,
            attempt_count=attempt_count,
            tools_used_this_session=self._extract_tools_used(history),
        )

        injections = []  # tuples of (text, priority, rule_id)
        blocklisted = _is_blocklisted_message(user_msg)

        # D9 FIX: Cooldown for keyword-based rules across multi-turn LLM calls.
        # When a user message triggers keyword rules, the same rules fire on every
        # subsequent LLM call (tool round) until the user sends a new message.
        # After the first injection, skip re-injecting the same keyword rule.
        # F4 (2026-08-04 audit): per-session cooldown state so interleaved
        # sessions in the same gateway process don't reset each other's set.
        cooldown = self._cooldown_state.setdefault(
            session_id or "default", {"msg_hash": "", "fired": set(), "calls": 0}
        )
        msg_hash_source = f"{session_id}\0{user_msg}"
        msg_hash = hashlib.md5(msg_hash_source.encode()).hexdigest()[:16] if user_msg else ""
        is_new_message = msg_hash != cooldown["msg_hash"]
        if is_new_message:
            cooldown["msg_hash"] = msg_hash
            cooldown["fired"] = set()
            cooldown["calls"] = 0
        else:
            cooldown["calls"] += 1

        for rule in self._compiled:
            if rule.hook != "pre_llm_call":
                continue
            # D11 FIX (2026-07-21): Per-rule meta-discussion suppression (replaces D6 blanket).
            # When the user message is blocklisted (talking about rule-injector internals),
            # only skip rules that opted in to meta-discussion suppression via
            # "suppress_on_meta_discussion": true in rules.json. Other rules fire normally.
            if blocklisted and rule._suppress_on_meta:
                continue
            try:
                # D9 FIX: Skip keyword rules that already fired on this user message
                # (cooldown — prevents redundant injections on multi-turn tool rounds)
                trigger_type = rule.trigger.get("type", "keyword")
                scope = rule.trigger.get("scope", "latest_user_message")
                is_keyword_on_user = (
                    trigger_type == "keyword" and scope == "latest_user_message"
                ) or (
                    trigger_type == "compound"
                    and rule.trigger.get("operator", "and") == "or"
                    and any(
                        c.get("type") == "keyword" and c.get("scope") == "latest_user_message"
                        for c in rule.trigger.get("conditions", [])
                    )
                )
                if not is_new_message and is_keyword_on_user and rule.id in cooldown["fired"]:
                    continue
                if not self._session_rule_can_fire(rule, session_id, session_turn):
                    continue

                if rule.matches(ctx):
                    # Track fired keyword rules for cooldown
                    if is_keyword_on_user:
                        cooldown["fired"].add(rule.id)
                    text = rule.inject.get("context", "")
                    if text:
                        injections.append((text, rule.priority, rule.id))
                        self._record_session_rule_fire(rule, session_id, session_turn)
            except Exception:
                continue  # never crash the conversation

        # Optional LLM adjudication: deterministic candidates become a
        # reviewable set; on any failure or with a missing adjudicator, the
        # original injection list is used unchanged.
        injections = self._adjudicate_injections(
            injections,
            hook_name="pre_llm_call",
            session_id=session_id,
            user_message=user_msg,
            recent_context=recent,
            recent_assistant=recent_assistant,
        )

        # Post-verdict contradiction check (Plan 007 Step 3): a candidate
        # whose domain resolves to software-development but whose user
        # message has NO software-context signal is force-dropped regardless
        # of the LLM adjudicator's verdict. Catches the case where the trigger
        # fired incorrectly AND the LLM kept it (the Failure A pattern).
        if injections:
            contradiction_dropped: List[str] = []
            kept_injections: List[tuple] = []
            for text, prio, rid in injections:
                rule = self._by_id.get(rid)
                if rule is None:
                    kept_injections.append((text, prio, rid))
                    continue
                rule_domain = domain_inference.resolve_domain(rid)
                if (
                    rule_domain == "software-development"
                    and not domain_inference.has_software_context(user_msg)
                ):
                    contradiction_dropped.append(rid)
                    continue
                kept_injections.append((text, prio, rid))
            if contradiction_dropped:
                logger.debug(
                    "contradiction check force-dropped %d software-domain candidates: %s",
                    len(contradiction_dropped),
                    contradiction_dropped,
                )
                injections = kept_injections

        if injections:
            # Priority ordered, but skip an oversized rule rather than breaking;
            # smaller lower-priority directives can still use the remaining budget.
            ordered = sorted(injections, key=lambda item: item[1])
            kept = []
            used = 0
            for text, _, _ in ordered:
                separator = 2 if kept else 0
                if used + separator + len(text) <= INJECTION_CHAR_BUDGET:
                    kept.append(text)
                    used += separator + len(text)
            total = "\n\n".join(kept)
            if len(kept) < len(ordered):
                marker = "\n[...some rules truncated...]"
                if len(total) + len(marker) <= INJECTION_CHAR_BUDGET:
                    total += marker
            return {"context": total}
        return {}

    def run_transform_tool_result(self, kwargs) -> str:
        """Run all transform_tool_result rules. Returns modified result or original.

        Adjudication (2026-07-31): matched rules are now reviewed by the LLM
        adjudicator when one is wired, using per-hook budgets. Tool context
        (tool name, args, result preview) is passed so the adjudicator can
        judge applicability against what the tool actually returned. On any
        fallback the deterministic set is used unchanged (fail-closed).
        """
        self._ensure_compiled()

        # NEW (2026-07-21): Detect cron session for rule exclusion
        session_id = kwargs.get("session_id", "") or ""
        task_id = kwargs.get("task_id", "") or ""
        user_msg = kwargs.get("user_message", "")
        is_cron = _detect_cron_context(
            session_id, task_id, user_msg,
            bool(kwargs.get("is_cron", False)),
        )

        history = kwargs.get("conversation_history", [])
        # Extract recent context BEFORE building ctx so transform rules can
        # use recent_context scope AND exclude_if_recent_context_keywords.
        # (2026-07-31 fix: these were silently dead on this hook — the ctx
        # never received recent_context, so exclusions never matched and
        # recent_context-scoped rules like cherry-pick-scope-judgment could
        # never fire.)
        recent_parts = []
        for m in (history or [])[-4:]:
            if isinstance(m, dict):
                content = m.get("content", "")
                if isinstance(content, str) and content.strip():
                    recent_parts.append(content[:500])
        recent = " ".join(recent_parts)
        # F3 (2026-08-04 audit): session fire limits (max_fires_per_session,
        # cooldown_turns) were previously ignored on this hook — the session
        # state exists but only run_pre_llm_call consulted it.
        session_turn = self._advance_session_turn(session_id, user_msg)
        ctx = MatchContext(
            hook="transform_tool_result",
            tool_name=kwargs.get("tool_name", ""),
            tool_result=kwargs.get("result", ""),
            tool_args=kwargs.get("args", {}),
            recent_context=recent,
            recent_assistant=_extract_recent_assistant(history),
            is_cron=is_cron,
            tools_used_this_session=self._extract_tools_used(history),
        )

        original = kwargs.get("result", "")
        suffixes = []  # (text, priority, rule_id)
        for rule in self._compiled:
            if rule.hook != "transform_tool_result":
                continue
            try:
                # D11 FIX (2026-07-21): Per-rule meta-discussion suppression for transform_tool_result.
                # When the tool result contains rule-injector source/trace logs, only skip
                # rules that opted in to meta-discussion suppression. Other rules evaluate normally.
                if ctx.tool_result and _is_blocklisted_message(ctx.tool_result) and rule._suppress_on_meta:
                    continue
                if not self._session_rule_can_fire(rule, session_id, session_turn):
                    continue
                if rule.matches(ctx):
                    suffix = rule.inject.get("result_suffix", rule.inject.get("context", ""))
                    if suffix:
                        suffixes.append((f"[Rule: {rule.id}] {suffix}", rule.priority, rule.id))
                        self._record_session_rule_fire(rule, session_id, session_turn)
            except Exception:
                continue

        if suffixes:
            # Optional LLM adjudication: review the deterministic suffix set.
            # Extra context gives the adjudicator the tool identity and a
            # preview of what the tool returned so it can judge rule fit.
            extra_context = {
                "tool_name": (kwargs.get("tool_name", "") or "")[:80],
                "tool_args_preview": str(kwargs.get("args", {}))[:400],
                "tool_result_preview": (original or "")[:500],
            }
            suffixes = self._adjudicate_injections(
                suffixes,
                hook_name="transform_tool_result",
                session_id=session_id,
                user_message=user_msg,
                recent_context=recent,
                recent_assistant=_extract_recent_assistant(history),
                extra_context=extra_context,
            )
            if suffixes:
                # F3 (2026-08-04 audit): cap total appended guidance. The
                # transform path previously had no char budget — N fired rules
                # appended unbounded text to every tool result. Priority-ordered
                # greedy with skip-on-overflow (same policy as pre_llm_call).
                ordered = sorted(suffixes, key=lambda item: item[1])
                kept = []
                used = 0
                for text, _, _ in ordered:
                    separator = 2 if kept else 0
                    if used + separator + len(text) <= TRANSFORM_SUFFIX_BUDGET:
                        kept.append(text)
                        used += separator + len(text)
                if kept:
                    return original + "\n\n---\n" + "\n\n".join(kept)
        return original

    def run_pre_verify(self, kwargs) -> dict:
        """Run all pre_verify rules. Returns {"action": "continue", "blocks": [...]} or {}.

        Adjudication (2026-07-31): completion-gate blocks are reviewed by the
        LLM adjudicator with a KEEP-BIASED policy — a gate block is only
        dropped when the adjudicator is highly confident it does not apply.
        Dropping a gate is safety-reducing, so the deterministic set is kept
        on ANY fallback (fail-closed). The safety floor still applies.
        """
        self._ensure_compiled()

        history = kwargs.get("conversation_history", [])
        session_id = kwargs.get("session_id", "") or ""
        task_id = kwargs.get("task_id", "") or ""
        user_msg = kwargs.get("user_message", "")
        is_cron = _detect_cron_context(
            session_id, task_id, user_msg,
            bool(kwargs.get("is_cron", False)),
        )
        # Extract recent context (last 4 messages) for keyword matching
        recent_parts = []
        for m in (history or [])[-4:]:
            if isinstance(m, dict):
                content = m.get("content", "")
                if isinstance(content, str) and content.strip():
                    recent_parts.append(content[:500])
        recent = " ".join(recent_parts)

        # Detect skills loaded this session (incremental)
        skills_loaded = self._detect_skills_loaded_incremental(history)

        # Detect post-compaction
        is_post_compaction = self._detect_post_compaction(history)

        # Count fix attempts (incremental)
        attempt_count = self._count_attempts_incremental(history)

        # Plan 004 (2026-08-03): session_turn needed for the
        # _session_rule_can_fire / _record_session_rule_fire calls below.
        session_turn = self._advance_session_turn(session_id, user_msg)

        ctx = MatchContext(
            hook="pre_verify",
            recent_context=recent,
            conversation_history=history,
            skill_loaded_this_session=skills_loaded,
            is_post_compaction=is_post_compaction,
            is_cron=is_cron,
            attempt_count=attempt_count,
            final_response=kwargs.get("final_response", ""),
            tools_used_this_session=self._extract_tools_used(history),
        )

        blocks = []  # (message, priority, rule_id)
        for rule in self._compiled:
            if rule.hook != "pre_verify":
                continue
            try:
                # Plan 004 (2026-08-03): mirror the transform_tool_result fix
                # at line 1034 — consult _session_rule_can_fire so
                # max_fires_per_session and cooldown_turns are honored on the
                # pre_verify hook too. The audit found this was asymmetric.
                if not self._session_rule_can_fire(rule, session_id, session_turn):
                    continue
                if rule.matches(ctx):
                    msg = rule.inject.get("message", rule.inject.get("context", ""))
                    if msg:
                        blocks.append((msg, rule.priority, rule.id))
                        self._record_session_rule_fire(rule, session_id, session_turn)
            except Exception:
                continue

        if blocks:
            # Optional LLM adjudication (KEEP-BIASED via SYSTEM_PROMPT policy):
            # only drop gate blocks the adjudicator is confident are
            # inapplicable. Extra context includes the draft final response.
            extra_context = {
                "final_response_preview": (kwargs.get("final_response", "") or "")[:500],
            }
            blocks = self._adjudicate_injections(
                blocks,
                hook_name="pre_verify",
                session_id=session_id,
                user_message=user_msg,
                recent_context=recent,
                recent_assistant=_extract_recent_assistant(history),
                extra_context=extra_context,
            )
            if blocks:
                return {
                    "action": "continue",
                    "blocks": [{"rule_id": rid, "message": text} for text, _, rid in blocks],
                }
        return {}

    def _detect_skills_loaded_incremental(self, history) -> set:
        """Scan conversation for skill_view() calls incrementally."""
        skills = set(self._cached_skills_loaded)
        if not history:
            return skills
        
        # Only scan new messages since last check
        new_messages = history[self._last_skill_scan_index:]
        self._last_skill_scan_index = len(history)
        
        for m in new_messages:
            if not isinstance(m, dict):
                continue
            content = m.get("content", "")
            if not isinstance(content, str):
                continue
            # Look for skill_view(name="X") in tool calls or mentions
            import re as _re
            for match in _re.finditer(r'skill_view\((?:name=)?["\']([^"\']+)["\']', content):
                skill_name = match.group(1)
                # Normalize: strip category prefix if present
                if "/" in skill_name:
                    skill_name = skill_name.split("/")[-1]
                skills.add(skill_name)
        
        self._cached_skills_loaded = skills
        return skills

    def _detect_post_compaction(self, history) -> bool:
        """Detect if the conversation just went through compaction."""
        if not history:
            return False
        for m in history[-3:]:
            if not isinstance(m, dict):
                continue
            content = (m.get("content") or "").lower()
            if "compaction" in content or "compressed" in content or "summary" in content:
                # Check if it's recent (last 3 messages)
                if "conversation was compressed" in content or "context injection" in content:
                    return True
        return False

    def _extract_tools_used(self, history) -> set:
        """Plan 005 (2026-08-03): extract tool names used this session
        from conversation_history. Permissive regex over assistant message
        content — supports two common wire formats:
          - {"name": "tool_name", ...}
          - <tool_name>tool_name</tool_name>
        Returns an empty set if no tools detected.
        """
        import re
        if not history:
            return set()
        names = set()
        for m in history:
            if not isinstance(m, dict):
                continue
            if m.get("role") != "assistant":
                continue
            txt = m.get("content", "")
            if not isinstance(txt, str):
                continue
            for match in re.finditer(r'"name":\s*"([a-zA-Z_][a-zA-Z0-9_]*)"', txt):
                names.add(match.group(1))
            for match in re.finditer(r'<tool_name>([a-zA-Z_][a-zA-Z0-9_]*)</tool_name>', txt):
                names.add(match.group(1))
        return names

    def _count_attempts_incremental(self, history) -> int:
        """Count repeated fix attempts incrementally.

        D1 FIX (2026-07-14): Previous implementation used a LOCAL variable
        (prev_was_correction) that reset to False on every call. Since this
        method scans incrementally (only new messages since last call), the
        user's correction signal was in a PREVIOUS call's scan window.
        Subsequent assistant messages in later calls never saw
        prev_was_correction=True, so the count stayed at 0.

        Fix: use persistent instance state (_cached_in_correction) that
        survives across calls. Stay in correction window until a new
        non-correction user message arrives. Count ALL assistant messages
        and tool results containing fix actions while in the window.
        """
        if not history:
            return 0

        fix_verbs = {"fix", "correct", "update", "modify", "change",
                     "retry", "patch", "adjust", "tweak", "attempt", "try", "redo", "re-try"}
        correction_signals = {"still", "not working", "doesn't work", "wrong",
                             "try again", "still broken", "broke", "failed",
                             "didn't", "are you sure", "did you verify", "error"}
        # Tool call patterns that count as fix attempts when found in assistant messages
        tool_fix_patterns = {"patch", "write_file", "execute_code"}

        count = self._cached_attempt_count
        # Persistent correction window state — survives across incremental calls
        in_correction = getattr(self, "_cached_in_correction", False)

        # Only scan new messages
        new_messages = history[self._last_attempt_scan_index:]
        self._last_attempt_scan_index = len(history)

        for m in new_messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "")
            content = (m.get("content") or "")
            if not isinstance(content, str):
                content = str(content) if content else ""
            content_lower = content.lower()

            if role == "user":
                if any(sig in content_lower for sig in correction_signals):
                    # Re-affirm we are in a fix loop. Do NOT reset count —
                    # a 2nd correction means the prior attempts still count.
                    in_correction = True
                elif not content.startswith("[Rule:") and not content.startswith("[HARD RULE"):
                    # A genuine non-correction, non-injected user message
                    # ends the loop and clears accumulated attempt state.
                    if in_correction:
                        in_correction = False
                        count = 0

            if in_correction and role == "assistant":
                # Count tool-call mentions (patch/write_file/execute_code) first
                tool_count = sum(1 for p in tool_fix_patterns if p in content_lower)
                if tool_count > 0:
                    count += tool_count
                elif any(v in content_lower for v in fix_verbs):
                    count += 1

        self._cached_attempt_count = count
        self._cached_in_correction = in_correction
        return count