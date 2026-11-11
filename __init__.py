"""Rule-Injector Plugin

Injects contextual hard rules into the conversation at the point of need,
rather than burying them in the system prompt where they wash out.

Based on the Nemotron harness optimization finding: "rules in tool results
worked, same words in tool descriptions did nothing" and "injected messages
at the point of need beat standing rules in the system prompt."

Three hooks:
  - pre_llm_call: inject {"context": "..."} into the user message
  - transform_tool_result: append guidance to tool results
  - pre_verify: gate completion with {"action": "continue", "message": "..."}
"""

import os
import json
import logging
from datetime import datetime
# Dual-mode imports (F12, 2026-08-04 audit): the plugin must load both as a
# package (gateway: `import rule_injector`) and as a bare module (pytest
# collection from the plugin root, standalone scripts). The hard relative
# import previously made `pytest` from the plugin root fail with 30
# collection errors, and silently killed ALL rules if rules.json was ever
# edited to include a malformed entry that broke module import.
try:
    from .rules import RuleStore
    from .matchers import RuleMatcher
    from .adjudicator import AdjudicationConfig, AdjudicationState, Adjudicator, PHASE1_MODEL, PHASE1_PROVIDER, PHASE1_TASK
    from .propose import propose_rule, extract_rule_draft, RULE_INJECTION_REVIEW_PROMPT  # Phase 3
except ImportError:  # bare-module context (tests, scripts)
    from rules import RuleStore
    from matchers import RuleMatcher
    from adjudicator import AdjudicationConfig, AdjudicationState, Adjudicator, PHASE1_MODEL, PHASE1_PROVIDER, PHASE1_TASK
    from propose import propose_rule, extract_rule_draft, RULE_INJECTION_REVIEW_PROMPT  # Phase 3

LOGGER = logging.getLogger("rule_injector")
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(name)s [%(levelname)s] %(message)s"))
    LOGGER.addHandler(_h)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = True

# Module-level so the dashboard/API can find the same adjudicator instance
# that the matcher is using. Initialized in register() so the test
# environments that import the module without a running gateway get a
# benign default.
adjudicator_state: dict = {
    "config": None,
    "state": None,
    "adjudicator": None,
}

# Trace log path
TRACE_LOG = os.path.expanduser("~/.hermes/logs/rule-injector-trace.jsonl")

def _log_activation(hook: str, rule_id: str, matched: bool, context_summary: str = "", session_id: str = ""):
    """Log rule activation for auditability."""
    try:
        os.makedirs(os.path.dirname(TRACE_LOG), exist_ok=True)
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": session_id or "unknown",
            "hook": hook,
            "rule_id": rule_id,
            "matched": matched,
            "context_summary": context_summary[:200] if context_summary else ""
        }
        with open(TRACE_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # never crash on logging


def _get_session_id() -> str:
    """Best-effort session ID extraction from Hermes runtime context."""
    try:
        # Try environment variable first (Hermes sets this for plugins)
        sid = os.environ.get("HERMES_SESSION_ID", "")
        if sid:
            return sid
        # Try reading from the state file
        state_path = os.path.expanduser("~/.hermes/state.json")
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
            sid = state.get("session_id", "")
            if sid:
                return sid
    except Exception:
        pass
    return "unknown"


def _build_adjudicator(enabled: bool) -> Adjudicator:
    cfg = AdjudicationConfig(enabled=enabled, mode="review", provider=PHASE1_PROVIDER, model=PHASE1_MODEL, task=PHASE1_TASK)
    _apply_adjudicator_overrides(cfg, _read_adjudicator_overrides())
    state = AdjudicationState(trace_retention=100)
    auxiliary_client = _load_auxiliary_client()
    return Adjudicator(cfg, state, auxiliary_client)


def _load_auxiliary_client():
    """Construct a callable that invokes auxiliary_client.call_llm.

    Imported lazily so the plugin still loads in environments where the
    hermes-agent package is not importable (the rule-injector test suite).
    A failure to import falls back to a deterministic offline client.
    """
    try:
        from agent.auxiliary_client import call_llm as _call_llm

        def _client(**kwargs):
            return _call_llm(**kwargs)

        return _client
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("adjudicator: auxiliary_client unavailable, using offline fallback: %s", exc)

        def _offline(**kwargs):
            return json.dumps(
                {
                    "keep": [],
                    "drop": [],
                    "blocked_review": [],
                    "reason": "offline_fallback",
                    "overall_confidence": 0.0,
                }
            )

        return _offline


def _read_adjudicator_config_from_yaml() -> bool:
    """Read enabled flag from config.yaml's auxiliary.adjudicator.enabled.

    No file I/O is performed in test environments that have not yet
    initialized hermes_cli; we just return False.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception:
        return False
    try:
        return bool(cfg.get("auxiliary", {}).get("adjudicator", {}).get("enabled", False))
    except Exception:
        return False


# ── Operator config for auxiliary.adjudicator ────────────────────────────
# The plugin declared these keys from the start but only ever read `enabled`,
# so every tuned value in config.yaml was INERT and the hardcoded
# AdjudicationConfig dataclass defaults applied instead (transform_tool_result
# got 15 calls/session, 60/hour, and a 10s per-call timeout). Reading the
# block makes the documented knobs real.
_ADJ_STR_KEYS = ("mode", "provider", "model", "task")
_ADJ_INT_KEYS = (
    "timeout_ms", "session_budget", "hourly_budget", "circuit_breaker_threshold",
    "circuit_breaker_reset_ms", "max_candidates", "max_blocked", "context_chars",
    "trace_retention",
)
_ADJ_HOOK_BUDGET_KEYS = ("hook_session_budget", "hook_hourly_budget")


def _read_adjudicator_overrides() -> dict:
    """config.yaml's ``auxiliary.adjudicator`` block (``{}`` when unavailable).

    No file I/O in test environments that have not initialized hermes_cli —
    the import raises and we fall through to the dataclass defaults.
    """
    try:
        from hermes_cli.config import load_config

        block = (load_config() or {}).get("auxiliary", {})
    except Exception:
        return {}
    block = block.get("adjudicator") if isinstance(block, dict) else None
    return block if isinstance(block, dict) else {}


def _apply_adjudicator_overrides(config, overrides: dict) -> None:
    """Overlay operator config onto the dataclass defaults, in place.

    Per-hook budgets MERGE rather than replace: ``AdjudicationConfig.normalized()``
    requires an entry for every wired hook, so a partial operator map must not
    drop (e.g.) ``propose_rule`` and trip validation.
    """
    for key in _ADJ_STR_KEYS:
        value = overrides.get(key)
        if isinstance(value, str) and value.strip():
            setattr(config, key, value.strip())
    for key in _ADJ_INT_KEYS:
        value = overrides.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        setattr(config, key, int(value))
    for key in _ADJ_HOOK_BUDGET_KEYS:
        value = overrides.get(key)
        if not isinstance(value, dict):
            continue
        merged = dict(getattr(config, key))
        for hook, budget in value.items():
            if isinstance(budget, bool) or not isinstance(budget, (int, float)):
                continue
            merged[str(hook)] = int(budget)
        setattr(config, key, merged)


def register(ctx):
    """Register rule-injector hooks."""
    store = RuleStore()
    enabled = _read_adjudicator_config_from_yaml()
    config = AdjudicationConfig(
        enabled=enabled,
        mode="review",
        provider=PHASE1_PROVIDER,
        model=PHASE1_MODEL,
        task=PHASE1_TASK,
    )
    _apply_adjudicator_overrides(config, _read_adjudicator_overrides())
    state = AdjudicationState(trace_retention=100)
    adjudicator = Adjudicator(config, state, _load_auxiliary_client())
    matcher = RuleMatcher(store, adjudicator=adjudicator)
    # Make the live state reachable from the dashboard/API.
    adjudicator_state["config"] = config
    adjudicator_state["state"] = state
    adjudicator_state["adjudicator"] = adjudicator
    _fallback_sid = _get_session_id()  # best-effort at register time
    # Subagent-session verification: each plugin load (main OR subagent
    # process) emits one log line. Use this to confirm rules are firing
    # inside subagent sessions: dispatch a subagent and grep
    # ~/.hermes/logs/rule-injector-trace.jsonl for two distinct
    # session_ids with a single "register" entry each.
    try:
        import logging
        _reg_logger = logging.getLogger("rule_injector")
        if not _reg_logger.handlers:
            _reg_logger.addHandler(logging.StreamHandler())
        _reg_logger.info(
            "rule-injector loaded fallback_session=%s pid=%d rules=%d",
            _fallback_sid, os.getpid(), len(store.load())
        )
    except Exception:
        pass  # never crash on logging
    # F11 (2026-08-04 audit): also emit a trace-log entry at register. The
    # agent.log probe line above has been unreliable since the module-level
    # StreamHandler was added (it diverts the line to stderr), so the trace
    # log is now the durable "plugin loaded in process" signal.
    try:
        _log_activation("register", "plugin", True,
                        f"rules={len(store.load())} pid={os.getpid()}", _fallback_sid)
    except Exception:
        pass

    def _resolve_sid(kw):
        """Get session_id from per-call hook kwargs, fall back to register-time value."""
        return kw.get("session_id") or _fallback_sid

    def on_pre_llm_call(**kw):
        try:
            result = matcher.run_pre_llm_call(kw)
            sid = _resolve_sid(kw)
            # Log which rules fired
            if result.get("context"):
                # Parse injected context to find rule IDs
                injected = result["context"]
                for rule in matcher._compiled:
                    if rule.hook == "pre_llm_call" and rule.inject.get("context", "") in injected:
                        _log_activation("pre_llm_call", rule.id, True, kw.get("user_message", "")[:100], sid)
            # D11 FIX (2026-07-21): Log suppressed rules for auditability.
            # When the user message is blocklisted, rules with suppress_on_meta_discussion
            # are skipped. Log them so the trace log shows the full picture.
            user_msg = kw.get("user_message", "")
            if user_msg:
                from .matchers import _is_blocklisted_message, MatchContext
                if _is_blocklisted_message(user_msg):
                    # Plan 009 (2026-08-03): only emit SUPPRESSED for rules
                    # that WOULD have matched. The previous loop emitted
                    # one SUPPRESSED entry per meta-suppressed rule per
                    # blocklisted turn regardless of trigger — pure noise.
                    history = kw.get("conversation_history", [])
                    recent_parts = []
                    for m in (history or [])[-4:]:
                        if isinstance(m, dict):
                            content_str = m.get("content", "")
                            if isinstance(content_str, str) and content_str.strip():
                                recent_parts.append(content_str[:500])
                    recent = " ".join(recent_parts)
                    skills_loaded = matcher._detect_skills_loaded_incremental(history)
                    is_post_compaction = matcher._detect_post_compaction(history)
                    attempt_count = matcher._count_attempts_incremental(history)
                    suppress_ctx = MatchContext(
                        hook="pre_llm_call",
                        user_message=user_msg,
                        recent_context=recent,
                        conversation_history=history,
                        skill_loaded_this_session=skills_loaded,
                        is_post_compaction=is_post_compaction,
                        is_cron=kw.get("is_cron", False),
                        attempt_count=attempt_count,
                    )
                    for rule in matcher._compiled:
                        if (rule.hook == "pre_llm_call"
                            and getattr(rule, "_suppress_on_meta", False)
                            and rule.matches(suppress_ctx)):
                            _log_activation("pre_llm_call", rule.id, False,
                                          f"SUPPRESSED(meta-discussion): {user_msg[:80]}", sid)
            return result
        except Exception as exc:
            # F1 (2026-08-04 audit): never crash the conversation, but DO log
            # the failure — the previous silent swallow made a dead injector
            # indistinguishable from a working one.
            LOGGER.warning("rule-injector: pre_llm_call hook failed: %s: %s", type(exc).__name__, exc)
            return {}

    def on_transform_tool_result(**kw):
        try:
            result = matcher.run_transform_tool_result(kw)
            sid = _resolve_sid(kw)
            # Log transform rules that fired
            original = kw.get("result", "")
            tool_name = kw.get("tool_name", "")
            if isinstance(result, str) and result != original:
                for rule in matcher._compiled:
                    if rule.hook == "transform_tool_result" and f"[Rule: {rule.id}]" in result:
                        _log_activation("transform_tool_result", rule.id, True, f"{tool_name}:{original[:80]}", sid)
            return result
        except Exception as exc:
            LOGGER.warning("rule-injector: transform_tool_result hook failed: %s: %s", type(exc).__name__, exc)
            return kw.get("result", "")

    def on_pre_verify(**kw):
        try:
            result = matcher.run_pre_verify(kw)
            sid = _resolve_sid(kw)
            # Log verify rules that fired
            if result.get("blocks"):
                for block in result["blocks"]:
                    _log_activation("pre_verify", block["rule_id"], True, "completion attempt", sid)
            return result
        except Exception as exc:
            LOGGER.warning("rule-injector: pre_verify hook failed: %s: %s", type(exc).__name__, exc)
            return {}

    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("transform_tool_result", on_transform_tool_result)
    ctx.register_hook("pre_verify", on_pre_verify)
