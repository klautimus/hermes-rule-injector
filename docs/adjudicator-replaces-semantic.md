# Adjudicator Replaces Semantic Matching in Production

**Status:** Active as of 2026-07-21 (commit `ebeaae7c` "feat(rule-injector):
add review-only LLM adjudicator core"; wiring commit `c501410c` "feat
(rule-injector): wire review-only adjudicator into matcher").
**Last verified:** 2026-07-27, commit `b64c62d`, session trace
`20260727_211218_cdc0c945` (agent.log shows
`Auxiliary rule_injector_adjudicator: using minimax (minimax-m3)
at https://api.minimax.io/anthropic` and
`rule-injector-adjudicator-verdicts.jsonl` contains verdict_id=58 and
60 with `verdict_status="applied"`).

## What changed

The **review-only LLM adjudicator** (`auxiliary.adjudicator.enabled:
true` in `~/.hermes/config.yaml`) is the live context-preservation and
rule-selection mechanism in this install. Provider `minimax`, model
`minimax-m3`, hard-coded at `adjudicator.py:30-31` (`PHASE1_PROVIDER`,
`PHASE1_MODEL`). Verdicts are persisted to
`~/.hermes/logs/rule-injector-adjudicator-verdicts.jsonl` (48h rolling
retention via `adjudicator_verdict_log.py:append_verdict` and prune).

The **sentence-transformers semantic matcher** (`semantic_matcher.py`,
`matchers.py:_match_semantic`, model
`sentence-transformers/all-MiniLM-L6-v2`) is **inert in this
environment.** The venv at
`/home/kevin/.hermes/hermes-agent/venv/bin/python3` does not have
`sentence_transformers` or `numpy` installed. `SemanticMatcher._ensure_model`
catches `ImportError` and sets `_import_failed=True` permanently for the
process. Semantic rule branches silently never fire. They degrade
gracefully — no error, no log entry, no fallback.

## Why this is the architecture, not a bug

Kevin confirmed on 2026-07-27: *"The adjudicator replaces the sentence
transformer."* The adjudicator performs the semantic work the
sentence-transformers layer was designed for — judging whether a
rule's intent matches the current conversation — with the advantage of
running **after** candidate rules are collected, so it sees
keyword/regex/semantic matches together and can adjudicate the full
set holistically. The sentence-transformers layer ran *before* injection
and could only cosine-score individual rules against the message in
isolation.

The LLM-side implementation (`auxiliary_client.call_llm`) bypasses the
main conversation's prompt cache, has its own per-session and per-hour
budgets, enforces a safety floor that prevents dropping
`safety_critical: true` rules, and falls back to the full deterministic
candidate set on any failure (timeout, malformed JSON, provider error,
circuit-tripped). See `llm-adjudication-architecture.md` for the
contract.

## What this means for rule authors

- **Keyword and regex branches are the only determinative matchers** in
  this install. Write them so they catch user intent directly without
  assuming semantic cosine similarity will rescue them.
- **Semantic branches in rules are latent capability.** They will not
  fire today. Keep them only if you intend to install sentence-transformers
  later and want the rule ready. Do not rely on them for correctness.

## 2026-07-31 update: all 7 semantic rules converted to keyword_flexible

The seven rules that used `trigger.type: semantic` were dead weight —
they encode real lessons (blind-patch-loop, stale-server-check,
env-fact-reassertion, scope-creep-detection, branch-base verification,
subagent spot-check, cherry-pick scope judgment) but could never fire
with the inert semantic matcher. They are now `keyword_flexible`
triggers with curated multi-word keyword lists scoped to their original
intent (tool_result / latest_user_message / recent_context). The LLM
adjudicator (now wired to ALL three hooks as of 2026-07-31) filters any
residual over-fire. No rule uses `trigger.type: semantic` anymore.
- **The adjudicator sees all matched candidates and decides which to
  keep.** A rule that matches on keyword but is semantically irrelevant
  to the current message will be suppressed by the adjudicator with a
  `DOMAIN_MISMATCH` verdict reason. Conversely, a rule that doesn't
  match any branch but *should* apply will not be seen by the adjudicator
  at all — the adjudicator only reviews candidates the matcher produced.
- **Do not add sentence-transformers back** without explicit instruction
  from Kevin. If the adjudicator is performing well (check
  `rule-injector-adjudicator-verdicts.jsonl`), there is no reason to
  pull the torch/sentence-transformers dependency in. If the adjudicator
  is missing context that a real semantic layer would have caught, that
  is the trigger to revisit — not a quiet default.

## How to verify the adjudicator is live

```bash
# The auxiliary client announces the provider/model at session start.
grep -m1 'rule_injector_adjudicator: using' ~/.hermes/logs/agent.log

# Recent verdicts show applied decisions.
tail -5 ~/.hermes/logs/rule-injector-adjudicator-verdicts.jsonl | python3 -m json.tool

# One-shot summary CLI (sourced from the persist log).
python3 ~/.hermes/scripts/audit-adjudicator.py summary
```

If the first command returns a line naming the provider and the second
shows recent `verdict_status="applied"` entries with `status="applied"`
in the persisted schema, the adjudicator is active and persisting.

## What is NOT changed

- `rules.json` semantic branches remain in place as latent capability.
- `semantic_matcher.py` and `matchers.py` code is unchanged — the
  `ImportError` guard is the correct behavior when the dependency is
  absent (graceful degradation, no spurious errors).
- `INJECTION_CHAR_BUDGET = 8000` (`matchers.py:48`) is unchanged.
- `auxiliary.adjudicator` config block in `~/.hermes/config.yaml`
  (`provider: minimax`, `model: minimax-m3`) is unchanged and required.

## Cross-references

- `docs/llm-adjudicator-quickstart.md` — provider/model contract and
  enable / disable commands.
- `docs/llm-adjudication-architecture.md` — design rationale for
  review-only mode and safety floor.
- `~/.hermes/skills/devops/rule-injector-audit/SKILL.md` — audit
  playbook (updated to treat semantic-matcher absence as inert, not a
  bug).
- `~/.hermes/skills/devops/rule-injector-authoring/SKILL.md` —
  authoring workflow notes updated for current architecture.
- `~/.hermes/skills/devops/rule-injector-testing/SKILL.md` — test
  guidance for the post-semantic world.
- `matchers.py`, `semantic_matcher.py` docstrings — in-module
  architecture notes cross-link back to this runbook.
