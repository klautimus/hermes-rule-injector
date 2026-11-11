# Optional LLM adjudicator — Phase 1 (review-only, minimax/minimax-m3)

> **This is the live semantic layer in production.** The review-only
> LLM adjudicator replaces sentence-transformers as the
> context-preservation mechanism. sentence-transformers is inert
> (the dependency is not installed in the active venv). See
> `adjudicator-replaces-semantic.md` for the full architecture,
> including why Phase 1 IS the architecture rather than a transition.

The adjudicator runs an extra LLM pass on the rules selected by the
deterministic matcher, and may drop candidates that are off-topic or
not needed. It is **strictly** opt-in. Default is `off`.

The implementation lives in `~/.hermes/plugins/rule-injector/`:

- `adjudicator.py` — `AdjudicationConfig`, `AdjudicationState`, `Adjudicator`
- `matchers.py` — matcher seam (`RuleMatcher(..., adjudicator=adj)`)
- `dashboard.py` — standalone FastAPI app
- `cli.py` — `hermes rule-injector-adjudicator {status,enable,disable,trace,dashboard}`

The web server dashboard at `/rule-injector` exposes the same state at:

- `GET  /api/rule-injector/adjudicator/status`
- `POST /api/rule-injector/adjudicator/toggle`

## Enabling

Add the following to `~/.hermes/config.yaml` under `auxiliary:`:

```yaml
auxiliary:
  adjudicator:
    enabled: true
    mode: review         # review only; recall is Phase 2
    provider: minimax
    model: minimax-m3     # must match PHASE1_MODEL in adjudicator.py or AdjudicatorConfig.normalized() raises
    timeout_ms: 10000     # raised from 5000 after observed 6026ms call
    session_budget: 50   # max adjudication calls per session
    hourly_budget: 200   # max adjudication calls per rolling hour
    circuit_breaker_threshold: 5
    circuit_breaker_reset_ms: 300000
    max_candidates: 20
    max_blocked: 20
    trace_retention: 100
```

The judge then requires `MINIMAX_API_KEY` to be set in `~/.hermes/.env`.

## Verification

```bash
# 1. Unit tests
RULE_INJECTOR_NO_SEMANTIC=1 python3 ~/.hermes/plugins/rule-injector/run_adjudicator_tests.py
python3 ~/.hermes/plugins/rule-injector/run_matcher_integration.py
python3 ~/.hermes/plugins/rule-injector/test_replay.py

# 2. CLI status
hermes rule-injector-adjudicator status
hermes rule-injector-adjudicator trace --limit 5

# 3. Dashboard
hermes rule-injector-adjudicator dashboard --port 9123
# open http://127.0.0.1:9123

# 4. Web dashboard tab
# open http://127.0.0.1:8642/rule-injector (adjudicator card on the page)
```

## Safety guarantees

- **Review-only.** The LLM may only keep or drop existing candidate
  rules. It cannot invent new rules, modify injection text, or unblock
  rules that the deterministic engine suppressed.
- **Safety floor.** Rules tagged `safety_critical: true` are always
  kept, regardless of the LLM decision. No safety rule can be dropped
  via the adjudicator.
- **Deterministic fallback.** Any failure (timeout, malformed JSON,
  provider error, circuit open, budget exhausted) falls back to the
  complete deterministic candidate set. The agent never sees fewer
  rules than the deterministic engine selected.
- **No user text in prompts.** The adjudicator sends bounded rule
  metadata only — no conversation history, no raw injection payloads,
  no user messages.
- **No prompt-cache pollution.** The adjudicator uses the same
  auxiliary-client path as the advisor tool, which is independent of
  the main conversation's prompt cache.
- **No recursion.** The auxiliary client's `call_llm` does not re-enter
  the rule-injector hooks. The plugin is never called by itself.
- **Trace auditability.** Every adjudication call (applied or fallback)
  is recorded with timestamp, session_id, decision counts, dropped
  IDs, reason, and latency. Raw user content is never stored.

## Future work (deferred)

- **Recall mode.** Allow the LLM to add rules from a bounded near-miss
  shortlist. Deferred until we have a release cycle of review-mode
  traces to tune retrieval.
- **Per-rule safety classification.** Currently the safety floor is
  detected by `safety_critical: true` or `rule_type == "safety"`. The
  rule-injector author can add either flag to rules. A future
  release may auto-classify by rule category.
