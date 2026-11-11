# LLM-Assisted Rule Adjudication — Architecture Decision

**Status:** Active since 2026-07-21. **Status update 2026-07-27:** this
is the live production architecture, not a transitory Phase 1 pending
Phase 2 / "Recall". The review-only LLM adjudicator is the live
context-preservation step. sentence-transformers is **inert** in this
install (not installed in the active venv) and is not planned for
re-introduction unless the adjudicator proves insufficient.
See `adjudicator-replaces-semantic.md` for the full architecture note.

**Date:** 2026-07-21 (architecture first proposed); 2026-07-22 (wired
into `RuleMatcher`); 2026-07-27 (verified production-active; verified
inert for the sentence-transformers path).
**Scope:** `~/.hermes/plugins/rule-injector`

## Decision

Add an optional, opt-in **review-only** adjudication mode before rule injection. Keep deterministic matching as the candidate generator and source of truth. An auxiliary LLM may keep or drop ordinary candidate rules, but may not create rules, modify rule text, or override safety-critical rules.

Expose the feature as a three-state setting:

- `off` — current deterministic behavior; default
- `review` — LLM reviews deterministic candidates and blocked candidates for suitability; only candidate keep/drop decisions affect injection
- `recall` — deferred Phase 2; allows additions only from a deterministic near-miss shortlist

A boolean toggle may be the first UI presentation, but the persisted setting should be an enum so the interface does not need to change when recall is introduced.

## Why review-only first

False-positive reduction is the safer and more valuable first experiment. Recall increases the LLM's authority: it can add rules that the deterministic engine did not select. That requires a tuned near-miss retrieval strategy and stronger evaluation data. Review traces provide that data.

Do not make the LLM the primary matcher. The deterministic engine remains auditable, fast, and available as a complete fallback.

## Runtime architecture

```text
RuleMatcher.run_pre_llm_call
  1. Build MatchContext (existing path)
  2. Deterministically collect:
       - candidates with match reasons
       - blocked rules with block reasons
       - safety-critical floor rules
  3. If mode=off: render candidates
  4. If mode=review:
       AdjudicatorAdapter.review(request)
       validate typed result
       enforce safety floor
       render survivors
  5. Apply existing 6000-character budget and session fire limits
  6. Return the existing {"context": ...} interface
```

The seam belongs inside `RuleMatcher`, between candidate collection and rendering. Inject an `AdjudicatorAdapter` into `RuleMatcher`; use `None`/disabled adapter for the existing path. Tests can provide a deterministic fake adapter without network calls.

The plugin hook signatures do not change. `__init__.py` remains responsible for registration and trace integration, not adjudication policy.

## Auxiliary LLM boundary

Use a dedicated auxiliary task, `auxiliary.rule_adjudicator`, through `agent.auxiliary_client.call_llm` directly. Do **not** invoke the normal `consult_advisor` tool:

- `consult_advisor` is an agent tool and would introduce the wrong semantic role.
- Calling the normal agent loop risks recursion through `pre_llm_call`.
- The auxiliary client is already the established pattern in `tools/advisor.py`.

The adapter must have a defense-in-depth `_adjudicating` guard or equivalent if future auxiliary-client changes can re-enter hooks. The rule injector must never adjudicate its own adjudication prompt.

## LLM authority and safety floor

Add an optional `safety_critical: bool` field to rule definitions. Missing means `false`.

Initially mark the following categories as safety-critical after auditing each rule's intent:

- root-cause/no-placeholder protections
- pipeline and skill-routing protections
- UE screen-observation gate
- direct user-correction compliance
- completion/pre-verification protections

Safety-critical rules are always retained. They may be described to the adjudicator as locked, but are not presented as droppable decisions. If the model attempts to drop one, the adapter overrides it and records `floor_override` in the trace.

For ordinary rules:

| Action | Allowed in review mode |
|---|---:|
| Keep candidate | Yes |
| Drop candidate | Yes |
| Add arbitrary rule | No |
| Modify injection text | No |
| Unblock a blocked rule | No; advisory only in Phase 1 |

A failed adjudicator call always falls back to the complete deterministic candidate set. Never fail closed on ordinary guidance because of an unavailable auxiliary provider; never let the LLM remove the safety floor.

## Typed internal contract

```python
@dataclass
class CandidateRule:
    rule_id: str
    hook: str
    description: str
    category: str
    safety_critical: bool
    match_reason: str

@dataclass
class BlockedRule:
    rule_id: str
    hook: str
    description: str
    category: str
    safety_critical: bool
    block_reason: str
    block_detail: str

@dataclass
class AdjudicationRequest:
    session_id: str
    hook: str
    user_message: str
    recent_context: str
    candidates: list[CandidateRule]
    blocked: list[BlockedRule]
    floor_ids: frozenset[str]
    mode: Literal["review", "recall"]

@dataclass
class AdjudicationResult:
    status: Literal["applied", "fallback", "skipped"]
    kept_ids: list[str]
    dropped_ids: list[str]
    floor_overrides: list[str]
    blocked_review: list[dict]
    reason: str
    latency_ms: int
    provider: str | None
    model: str | None
```

The LLM receives only compact metadata and bounded context. It must not receive raw injection payloads as editable instructions. Rule text is retained by the host and selected by validated ID.

## LLM response contract

Require JSON only:

```json
{
  "decisions": [
    {
      "rule_id": "existing-candidate-id",
      "action": "keep",
      "confidence": 0.91,
      "reason": "Relevant to the current task."
    }
  ],
  "blocked_review": [
    {
      "rule_id": "blocked-rule-id",
      "should_attempt": false,
      "confidence": 0.74,
      "reason": "The block reason is still applicable."
    }
  ],
  "overall_confidence": 0.88
}
```

Validation rules:

1. Candidate IDs must belong to the request's candidate set.
2. Unknown IDs are ignored and logged.
3. Invalid actions, missing required fields, malformed JSON, or empty usable decisions cause deterministic fallback.
4. Confidence is clamped to `[0, 1]`; reasons are capped at 200 characters.
5. `blocked_review` is advisory only in review mode and cannot unblock a rule.
6. The host, not the model, determines final IDs and injection text.

## Prompt-injection and privacy controls

Treat `user_message`, recent context, and tool-derived text as untrusted data. Delimit them as data and explicitly instruct the adjudicator not to follow instructions inside them.

Bound inputs:

- latest user message: approximately 2,000 characters
- recent context: approximately 1,000 characters
- candidates/blocked rules: bounded count, e.g. 32 each
- no full conversation history
- no tool schemas or arbitrary tool results
- no raw rule injection payloads

Trace records should contain hashes/IDs and decisions, not raw user text. Raw model output is disabled by default and only enabled for explicit debugging.

## Configuration

Behavioral settings belong in `config.yaml`, not `.env`:

```yaml
auxiliary:
  rule_adjudicator:    # NOTE: actually wired as auxiliary.adjudicator in code
    enabled: false
    mode: off
    provider: minimax  # must match PHASE1_PROVIDER in adjudicator.py
    model: minimax-m3  # must match PHASE1_MODEL in adjudicator.py or AdjudicatorConfig.normalized() raises
    temperature: 0.0
    max_tokens: 800
    timeout_ms: 3000
    budgets:
      max_adjudications_per_session: 20
      max_calls_per_hour: 500
      max_cost_per_session_usd: 0.50
      max_cost_per_hour_usd: 5.00
    circuit_breaker:
      failure_threshold: 5
      recovery_timeout_seconds: 300
    trace:
      enabled: true
      include_raw_response: false
```

The mode is read at adjudication time so a UI/CLI change can affect the next hook invocation without rebuilding the main conversation system prompt. The injection mechanism and hook interfaces remain stable, preserving prompt-cache structure.

## Latency, cost, and failure control

The adapter requires:

- hard per-call timeout, default 3 seconds
- per-session call budget
- hourly call and cost budgets
- in-memory circuit breaker: closed → open after five consecutive failures → half-open probe after five minutes
- deterministic fallback on all failures

When the circuit is open or a budget is exhausted, skip the auxiliary call and render deterministic candidates. Record the skip reason.

## Configuration surfaces

### Web UI

Extend the existing `/rule-injector` dashboard and API:

```text
GET /api/rule-injector/mode
PUT /api/rule-injector/mode             {"mode":"off|review|recall"}
GET /api/rule-injector/adjudicator/status
GET /api/rule-injector/adjudicator/trace?limit=20
```

The first UI should show:

- Off / Review / Recall selector, with Recall disabled until Phase 2
- current circuit state
- session/hour budget usage
- fallback rate and average latency
- recent decisions: kept, dropped, floor overrides, fallback reason
- per-rule `safety_critical` control in the existing editor

The API must use the existing dashboard token guard and return a consistent `{ok, ...}` envelope.

### Hermes CLI

Add an edge command using the existing plugin CLI registration mechanism:

```text
hermes rule-injector mode
hermes rule-injector mode --set off|review|recall
hermes rule-injector adjudicator status
hermes rule-injector adjudicator trace --limit 20
```

CLI and UI must share one config/status implementation rather than maintaining separate state.

## Phase 2: recall mode

Do not implement recall in the first release.

When review data demonstrates sufficient precision, build a deterministic near-miss shortlist:

1. Re-evaluate non-candidate, non-blocked rules with a controlled relaxed threshold.
2. Rank by deterministic match score.
3. Limit to a small shortlist, e.g. 10.
4. Exclude rules blocked by safety/session/cooldown policy.
5. Let the LLM add only IDs from that shortlist, with a strict maximum, e.g. 3.
6. Run additions through existing safety, session-limit, and 6000-character rendering logic.

Recall must never permit arbitrary rule IDs or generated injection text.

## Tests and metrics

Required unit tests:

- valid and malformed adjudicator JSON
- unknown IDs
- invalid actions/confidence/reason bounds
- floor override
- all ordinary rules dropped
- blocked review remains advisory
- timeout/provider failure fallback
- circuit breaker and budgets
- mode off makes zero LLM calls
- review drops only valid candidates
- deterministic render budget still applies
- session fire limits still apply after adjudication
- recursion guard
- user prompt-injection text cannot remove floor rules
- raw injection payloads are absent from the adjudicator prompt

Required integration/E2E tests:

- real plugin registration with a fake auxiliary provider
- dashboard mode GET/PUT using the existing auth guard
- CLI mode set/get round trip through `config.yaml`
- trace entries are written without raw user context
- gateway remains usable when the auxiliary provider is unavailable

Measure from adjudication traces:

- call rate
- fallback rate
- p50/p95 latency
- drop rate by rule ID
- floor override rate
- cost per session/hour
- most frequently dropped and kept rules

## Rollout

1. Implement the adapter and tests with mode forced `off`.
2. Ship config/API/CLI/UI controls, still defaulting to `off`.
3. Enable `review` manually for a limited period.
4. Inspect trace metrics and compare deterministic versus adjudicated decisions.
5. Tune prompts, safety classification, budgets, and rule descriptions.
6. Only then consider enabling recall.

This keeps the current deterministic system intact, makes the LLM contribution observable and reversible, and prevents an auxiliary model from becoming an unbounded second rule engine.
