# Rule Injector

**Rules that arrive when they're needed, instead of sitting in a system prompt getting ignored.**

A Hermes Agent plugin that moves hard behavioural rules out of SOUL.md / the system prompt and injects them at the exact moment they apply — on the tool call, the tool result, or the completion gate where the failure would actually happen.

---

## The problem

Every serious agent setup accumulates rules. "Never ship a placeholder." "Verify before claiming." "Read the config, don't guess it." They get pasted into the system prompt, and then one of two things happens: the prompt grows until the rules are diluted noise, or the rules are condensed until they're too vague to change behaviour.

The research finding this plugin is built on is blunt: **rules in tool results worked. The same words in tool descriptions did nothing. Injected messages at the point of need beat standing rules in the system prompt.** Context position is not a detail — it's most of the effect.

Rule Injector treats a rule as a small program with a trigger, not a paragraph of prose.

## What it does

A rule is `{trigger, inject}`. Triggers are matched against the live turn, and only the rules that match get injected, only where they land.

Three hooks, three different jobs:

| Hook | Position | Good for |
|---|---|---|
| `pre_llm_call` | Before the model runs | Behavioural guards: "think before acting", "no placeholders", "read the source" |
| `transform_tool_result` | Appended to a tool result | Corrective nudges where the model is already reading: repeated errors, empty results |
| `pre_verify` | Completion gate | Hard stops: "you have no evidence for that claim", "that's a limitations list, not a delivery" |

Trigger types include `always`, `keyword`, `regex`, `tool_name`, `tool_error`, `path`, `session_scope` and boolean `compound` (and/or/not over any of the above), so a rule can be as narrow as *"only when the user's message mentions a threshold AND the last tool was a file read."*

## Install

```bash
# from a checkout
hermes plugins install /path/to/rule-injector

# or from the catalog (once listed)
hermes plugins install rule-injector
```

Then enable it and restart the gateway:

```bash
hermes plugins enable rule-injector
```

Verify it loaded and is firing:

```bash
hermes plugins list
tail -f ~/.hermes/logs/rule-injector-trace.jsonl
```

## Write a rule

`rules.json` next to the plugin is a plain list — no code, no restart:

```json
{
  "id": "no-placeholders",
  "name": "No Placeholders",
  "enabled": true,
  "hook": "pre_llm_call",
  "priority": 1,
  "trigger": {
    "type": "compound",
    "operator": "or",
    "conditions": [
      { "type": "keyword", "scope": "latest_user_message",
        "keywords": ["implement", "build", "code", "stub"] },
      { "type": "tool_name", "tool_names": ["write_file", "patch"] }
    ]
  },
  "inject": {
    "context": "Never ship a stub or a TODO body. Build the real data path."
  }
}
```

The file is re-read on modification (mtime-checked), the loader is fault-isolated per rule, and a malformed rule is skipped rather than taking the whole set down.

This repo ships six starter rules in `rules.json` to show the shape. Replace them with yours — the point of the plugin is that your rules are yours.

## How matching stays cheap

- **Compiled keyword sets**, not a regex sweep over every rule per turn.
- **Per-turn injection budget** — a per-turn character cap applied while rules are collected by priority, so a large rule library can't flood the context. High-priority always-on rules are exempt and cannot be dropped; low-priority noise is.
- **Scope guards** — rules can be scoped to the latest user message, the whole conversation, or the tool result, so a keyword in an old turn doesn't re-fire forever.

## The adjudicator (optional, off by default)

Keyword matching is precise and dumb: it misses paraphrases and it fires on incidental mentions. The optional adjudicator adds a second, semantic pass for the ambiguous cases — a cheap auxiliary model reviews the candidate rule against the turn and decides fire / don't fire, with the decisions trace-logged so you can audit why a rule did or didn't apply.

It is **disabled by default** and costs one auxiliary model call per ambiguous turn when enabled. Enable it in `~/.hermes/config.yaml`:

```yaml
auxiliary:
  adjudicator:
    provider: deepseek        # any configured provider
    model: deepseek-flash
```

## Auto-drafting rules from corrections

`propose.py` turns a correction ("no, that's wrong because…") into a draft rule with a trigger and an inject block, which you review before it lands. The useful part isn't generation, it's the review step: seeing a rule as `{trigger, inject}` makes vague rules obviously vague.

## CLI and dashboard

```bash
python cli.py list                     # every rule, hook, priority, enabled
python cli.py trace --tail 50          # what fired this session and why
```

`dashboard/` ships a small read-only web view of the rule set and the trace log.

## Limitations — read these before installing

- **Keyword triggers are literal.** A rule written for the word "deploy" does not fire on "ship it". That is the trade for being free. Use the adjudicator or compound triggers when a rule must be robust to paraphrase.
- **Injections cost tokens.** Every fired rule adds text to the turn. The budget keeps it bounded, but a rule set of 150+ rules with loose triggers will visibly raise input tokens.
- **It changes model behaviour, not model capability.** A bad rule produces confident wrong behaviour more reliably.
- **Rules are global to the profile.** Per-project rule sets need a separate profile or a path-scoped trigger; there is no first-class project scoping yet.
- **The adjudicator is a second model reading your turn.** If you enable it, that content goes to whichever provider you configured. Leave it off if that matters.
- **No rule versioning/rollback UI.** `rules.json` is a file; use git.
- Tested on Hermes 0.21.x, Linux/WSL. Not tested on macOS or the Desktop app.

## Disclosure

- **Reads:** `rules.json` in the plugin directory, and the plugin's own trace log.
- **Writes:** `~/.hermes/logs/rule-injector-trace.jsonl` (one line per activation), and `rules.json` when a rule is created or edited through the CLI/propose path.
- **Network:** none by default. If the adjudicator or the semantic matcher is enabled, it calls the provider you configured, using Hermes's own auxiliary client and credentials — the plugin stores no keys and reads no `.env`, `auth.json` or other credential store.
- **Shell:** none. The plugin never spawns commands.
- **Telemetry:** none. Nothing leaves the machine except the adjudicator call you opt into.

## Feedback that would help

- Rule sets where keyword triggers produced obvious false positives or false negatives — the trigger shapes that survive real use are the interesting part.
- Whether a 150-rule set stays under your acceptable token overhead with the budget as shipped.
- `pre_verify` gate ideas that catch a failure *before* it reaches the user, especially around "done" claims without evidence.
- Anyone running it on macOS or the Desktop app.

## License

MIT.
