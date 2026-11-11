"""Move per-rule fire-control fields (cooldown_turns, max_fires_per_session,
exclude_if_recent_context_keywords) from the top-level dict into the
trigger dict. The matcher's CompiledRule reads these from the trigger
field, not the rule's top level. Fixes the "unrecognized top-level
options" warnings on rules we added in Stages 2-5.
"""
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

FIRE_CONTROL_KEYS = (
    'cooldown_turns',
    'max_fires_per_session',
    'exclude_if_recent_context_keywords',
    'exclude_if_skill_loaded',
    'exclude_if_cron',
    'suppress_on_meta_discussion',
)


def main():
    with open(REPO / 'rules.json') as f:
        data = json.load(f)
    rules = data['rules']
    moved = 0
    for r in rules:
        trigger = r.get('trigger')
        if not isinstance(trigger, dict):
            continue
        changed = False
        for k in FIRE_CONTROL_KEYS:
            if k in r and k not in trigger:
                trigger[k] = r.pop(k)
                changed = True
                moved += 1
        if changed:
            print(f'  {r["id"]}: moved fire-control fields into trigger')
    if moved == 0:
        print('No fields needed moving.')
        return
    with open(REPO / 'rules.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f'\nWrote rules.json with {moved} field(s) moved.')
    new_mtime = time.time() + 5
    os.utime(REPO / 'rules.json', (new_mtime, new_mtime))


if __name__ == '__main__':
    main()
