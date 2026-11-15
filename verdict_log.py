"""Persistent verdict log for the rule-injector LLM adjudicator.

Why this exists
---------------
The in-memory `AdjudicationState` (in `adjudicator.py`) stores verdicts in a
`deque(maxlen=trace_retention)` where `trace_retention: 100` per `config.yaml`.
That means:

* At most the last 100 verdicts are kept
* All verdicts vanish on every gateway restart
* There is no way to audit "what context did the LLM see when it dropped rule X
  three hours ago?" — the answer is gone

This module appends every adjudicator verdict to a JSONL file with the full
context that was sent to the LLM, so an audit script (see
`audit-adjudicator.py`) can review the LLM's decisions and judge whether it's
working as intended.

Format
------
Each line is a JSON object with:

    {
      "timestamp": <unix epoch seconds>,
      "iso_time": <ISO 8601 UTC>,
      "session_id": <str>,
      "hook": <str, e.g. "pre_llm_call">,
      "verdict_status": "applied" | "fallback",
      "verdict_reason": <str, e.g. "applied", "malformed_response", "mode_off">,
      "candidates_in": <int>,
      "candidates_kept": <list[str]>,
      "candidates_dropped": <list[str]>,
      "blocked_in": <int>,
      "blocked_kept": <list[str]>,
      "duration_ms": <int>,
      "provider": <str>,
      "model": <str>,
      "fallback": <bool>,
      "user_message": <str, first 1000 chars>,
      "recent_context": <str, first 1500 chars>,
      "recent_assistant": <str, first 1500 chars>,
      "candidates": [
          {
            "rule_id": <str>,
            "priority": <int>,
            "injection_length_chars": <int>,
            "injection_text_preview": <str, first 400 chars>,
            "injection_text_truncated": <bool>,
            "verdict": "kept" | "dropped" | "not_in_candidate_set"
          },
          ...
      ]
    }

Retention
---------
Default retention is 48 hours. On every append, the file is scanned for lines
with `timestamp` older than the cutoff and removed in place. The atomic-rename
pattern is used to avoid corruption mid-write.

File location
-------------
`~/.hermes/logs/rule-injector-adjudicator-verdicts.jsonl`

This is intentional: it lives next to the rule-injector trace file so it
shows up in the same `logs/` audit workflows.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


try:  # plugin context: resolve the active profile's home
    from .paths import verdict_log as _verdict_log_path
except ImportError:  # bare-module context (tests, scripts run from the plugin root)
    try:
        from paths import verdict_log as _verdict_log_path
    except ImportError:  # no plugin tree on the path at all
        _verdict_log_path = None

DEFAULT_LOG_PATH = (
    _verdict_log_path()
    if _verdict_log_path is not None
    else Path.home() / ".hermes" / "logs" / "rule-injector-adjudicator-verdicts.jsonl"
)
DEFAULT_RETENTION_SECONDS = 48 * 60 * 60  # 48 hours


def _now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_verdict(
    verdict: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    context_summary: Dict[str, Any],
    *,
    log_path: Path = DEFAULT_LOG_PATH,
    retention_seconds: int = DEFAULT_RETENTION_SECONDS,
) -> None:
    """Append one verdict record to the persistent log, then prune old entries.

    Safe to call from any process. The function is best-effort: any IO error
    is swallowed and logged but does not raise, so a logging failure cannot
    break the rule-injector pipeline.

    Args:
        verdict: The dict returned by `Adjudicator.review()`.
        candidates: The candidate payload that was sent to the LLM (full
            schema, with `rule_id`, `priority`, `injection_text_preview`,
            etc.).
        context_summary: The `context_summary` dict built in
            `matchers.py:_adjudicate_injections` — contains
            `latest_user_message`, `recent_context`, `recent_assistant`,
            `hook`.
        log_path: Override the log file location (default: standard path).
        retention_seconds: How long to keep entries (default: 48 hours).
    """
    try:
        _ensure_parent(log_path)
        keep_set = set(verdict.get("keep_candidates", ()) or ())
        drop_set = set(verdict.get("drop_candidates", ()) or ())
        candidate_ids = {str(c.get("rule_id", "")) for c in candidates}

        # Build per-candidate record with verdict status
        per_candidate = []
        for c in candidates:
            rid = str(c.get("rule_id", ""))
            if rid in keep_set:
                v = "kept"
            elif rid in drop_set:
                v = "dropped"
            else:
                v = "not_in_candidate_set"
            per_candidate.append({
                "rule_id": rid,
                "priority": c.get("priority"),
                "injection_length_chars": c.get("injection_length_chars"),
                "injection_text_preview": c.get("injection_text_preview", ""),
                "injection_text_truncated": c.get("injection_text_truncated", False),
                "verdict": v,
            })

        ts = time.time()
        record = {
            "timestamp": ts,
            "iso_time": _now_iso(),
            "session_id": verdict.get("session_id") or context_summary.get("session_id", ""),
            "hook": context_summary.get("hook") or verdict.get("hook_name", ""),
            "verdict_status": verdict.get("status", "fallback"),
            "verdict_reason": verdict.get("reason", ""),
            "candidates_in": len(candidates),
            "candidates_kept": sorted(keep_set),
            "candidates_dropped": sorted(drop_set),
            "blocked_in": verdict.get("blocked_in", 0),
            "blocked_kept": verdict.get("keep_blocked", []),
            "duration_ms": verdict.get("duration_ms", 0),
            "provider": verdict.get("provider", ""),
            "model": verdict.get("model", ""),
            "fallback": verdict.get("status", "") != "applied",
            "user_message": context_summary.get("latest_user_message", "")[:1000],
            "recent_context": context_summary.get("recent_context", "")[:1500],
            "recent_assistant": context_summary.get("recent_assistant", "")[:1500],
            "candidates": per_candidate,
            "overall_confidence": verdict.get("overall_confidence"),
        }

        line = json.dumps(record, separators=(",", ":")) + "\n"
        # Append + prune atomically
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass

        # Prune in place. Cheap because file is small (a few thousand lines max
        # at 48h × 1 verdict per minute). Use atomic temp+rename.
        _prune(log_path, retention_seconds, now=ts)
    except Exception as exc:  # noqa: BLE001
        # Logging failures must never break the rule pipeline.
        try:
            import logging
            logging.getLogger("adjudicator_verdict_log").warning(
                "failed to append verdict: %s", exc
            )
        except Exception:
            pass


def _prune(log_path: Path, retention_seconds: int, *, now: float) -> None:
    cutoff = now - retention_seconds
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return
    kept: List[str] = []
    removed = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            kept.append(line + "\n")
            continue
        ts = rec.get("timestamp")
        if not isinstance(ts, (int, float)) or ts >= cutoff:
            kept.append(line + "\n")
        else:
            removed += 1
    if removed == 0:
        return
    # Atomic temp+rename
    fd, tmp = tempfile.mkstemp(prefix=".adjudicator-verdicts-", suffix=".tmp",
                               dir=str(log_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.writelines(kept)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, log_path)
        try:
            import logging
            logging.getLogger("adjudicator_verdict_log").info(
                "pruned %d adjudicator verdict(s) older than %ds", removed, retention_seconds
            )
        except Exception:
            pass
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_recent(
    limit: int = 50,
    *,
    log_path: Path = DEFAULT_LOG_PATH,
    only_applied: bool = False,
    only_fallbacks: bool = False,
    session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return up to `limit` most recent verdict records (newest first).

    Args:
        limit: Maximum number of records to return.
        only_applied: Only return verdicts where `status == "applied"`.
        only_fallbacks: Only return verdicts where `status != "applied"`.
        session_id: Filter to a specific session.
    """
    if not log_path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if only_applied and rec.get("verdict_status") != "applied":
                continue
            if only_fallbacks and rec.get("verdict_status") == "applied":
                continue
            if session_id and rec.get("session_id") != session_id:
                continue
            out.append(rec)
    out.sort(key=lambda r: r.get("timestamp", 0), reverse=True)
    return out[:limit]


def summarize(
    log_path: Path = DEFAULT_LOG_PATH,
    retention_seconds: int = DEFAULT_RETENTION_SECONDS,
) -> Dict[str, Any]:
    """Aggregate stats over the retention window."""
    if not log_path.exists():
        return {"file": str(log_path), "exists": False}
    cutoff = time.time() - retention_seconds
    total = 0
    within = 0
    applied = 0
    fallback_reasons: Dict[str, int] = {}
    rule_drop_count: Dict[str, int] = {}
    rule_keep_count: Dict[str, int] = {}
    durations: List[int] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            ts = rec.get("timestamp", 0)
            if ts < cutoff:
                continue
            within += 1
            if rec.get("verdict_status") == "applied":
                applied += 1
            else:
                reason = rec.get("verdict_reason", "")
                fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
            for rid in rec.get("candidates_kept", []):
                rule_keep_count[rid] = rule_keep_count.get(rid, 0) + 1
            for rid in rec.get("candidates_dropped", []):
                rule_drop_count[rid] = rule_drop_count.get(rid, 0) + 1
            d = rec.get("duration_ms")
            if isinstance(d, (int, float)):
                durations.append(int(d))
    durations.sort()
    n = len(durations)
    return {
        "file": str(log_path),
        "exists": True,
        "retention_seconds": retention_seconds,
        "total_lines_in_file": total,
        "lines_within_retention": within,
        "applied_count": applied,
        "fallback_count": within - applied,
        "fallback_reasons": fallback_reasons,
        "drop_rate_per_rule": _drop_rates(rule_drop_count, rule_keep_count),
        "duration_ms": {
            "count": n,
            "p50": durations[n // 2] if n else None,
            "p95": durations[int(n * 0.95)] if n else None,
            "max": durations[-1] if n else None,
        },
    }


def _drop_rates(drop: Dict[str, int], keep: Dict[str, int]) -> Dict[str, Dict[str, Any]]:
    rates = {}
    for rid in set(drop) | set(keep):
        d = drop.get(rid, 0)
        k = keep.get(rid, 0)
        total = d + k
        if total == 0:
            continue
        rates[rid] = {"dropped": d, "kept": k, "drop_rate": round(d / total, 3)}
    return dict(sorted(rates.items(), key=lambda kv: -kv[1]["drop_rate"]))


if __name__ == "__main__":
    # Allow `python3 -m adjudicator_verdict_log` for a quick summary
    import sys
    import json as _json
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        print(_json.dumps(summarize(), indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "recent":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        for r in read_recent(limit):
            print(_json.dumps(r, indent=2))
    else:
        print("usage: adjudicator_verdict_log.py summarize|recent [N]")
