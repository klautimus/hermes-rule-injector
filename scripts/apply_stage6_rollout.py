"""Stage 6 — Rollout: cap rebalance + per-UE-rule throttles + telemetry.

Code changes:
  1. matchers.py: raise INJECTION_CHAR_BUDGET from 6000 to 8000.
  2. rules.json: add cooldown_turns=2 and max_fires_per_session=6 inside
     the trigger dict of every UE-family rule (the 65 rules we touched
     across Stages 2-5, plus the 4 transform_tool_result rules).
  3. rules.json: trim inject text on priority-<=3 rules that exceed
     350 chars.

Operational:
  - Wipe plugin __pycache__ for clean hot-reload.
"""
import json
import os
import re
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

UE_RULE_PREFIXES = (
    'ue-skill-',
    'unreal-engine-router',
    'ue-workflow-skill-router',
    'ue-windows-observation-gate',
    'ue-tool-and-screenshot-combo',
    'ue-continuation-active-guard',
)

# Per-rule cooldown_turns / max_fires_per_session targets. Apply inside
# the trigger dict.
def apply_throttles(rules):
    changed = 0
    for r in rules:
        if not r.get('enabled', True):
            continue
        if not any(r['id'].startswith(p) or r['id'] == p for p in UE_RULE_PREFIXES):
            continue
        trigger = r.get('trigger')
        if not isinstance(trigger, dict):
            continue
        before_ct = trigger.get('cooldown_turns')
        before_mf = trigger.get('max_fires_per_session')
        trigger['cooldown_turns'] = 2
        trigger['max_fires_per_session'] = 6
        if before_ct != 2 or before_mf != 6:
            changed += 1
    return changed


def raise_budget():
    """Bump INJECTION_CHAR_BUDGET from 6000 to 8000 in matchers.py."""
    p = REPO / 'matchers.py'
    text = p.read_text()
    new = re.sub(r'^(INJECTION_CHAR_BUDGET\s*=\s*)\d+',
                 r'\g<1>8000', text, count=1, flags=re.MULTILINE)
    if new == text:
        print('  WARN: INJECTION_CHAR_BUDGET replacement did not match; no change')
        return False
    p.write_text(new)
    return True


def trim_inject_text(rules, max_chars=350):
    """For priority <= 3 rules, truncate inject.context to max_chars and
    append an ellipsis to indicate truncation. This is a coarse trim;
    a real implementation would rewrite the inject text per rule.
    """
    changed = 0
    for r in rules:
        if r.get('priority', 99) > 3:
            continue
        inject = r.get('inject', {})
        if not isinstance(inject, dict):
            continue
        text = inject.get('context', '') or inject.get('message', '')
        if len(text) <= max_chars:
            continue
        # Trim at last full sentence before max_chars, append ' [...]' marker
        end = text[:max_chars].rfind('. ')
        if end < 100:
            end = max_chars - 4
        trimmed = text[:end].rstrip() + ' [...]'
        if 'context' in inject:
            inject['context'] = trimmed
        elif 'message' in inject:
            inject['message'] = trimmed
        changed += 1
    return changed


def main():
    # 1. raise INJECTION_CHAR_BUDGET
    if raise_budget():
        print('  matchers.py: INJECTION_CHAR_BUDGET -> 8000')
    else:
        print('  matchers.py: budget already updated or not found')

    # 2-3. update rules.json
    with open(REPO / 'rules.json') as f:
        data = json.load(f)
    rules = data['rules']

    n_throttle = apply_throttles(rules)
    print(f'  rules.json: applied throttles to {n_throttle} rule(s)')

    n_trim = trim_inject_text(rules, max_chars=350)
    print(f'  rules.json: trimmed inject text on {n_trim} priority-<=3 rule(s)')

    if n_throttle or n_trim:
        with open(REPO / 'rules.json', 'w') as f:
            json.dump(data, f, indent=2)
        new_mtime = time.time() + 5
        os.utime(REPO / 'rules.json', (new_mtime, new_mtime))
    cache = REPO / '__pycache__'
    if cache.exists():
        for entry in cache.iterdir():
            if entry.is_file() and entry.suffix == '.pyc':
                entry.unlink()
        print('  Wiped stale __pycache__/*.pyc')


if __name__ == '__main__':
    main()
