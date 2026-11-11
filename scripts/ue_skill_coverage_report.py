"""Stage 6 telemetry: write a daily UE-rule coverage report.

Run nightly. Reads the last 24h of rule-injector trace log, counts per-rule
firings broken down by source session, and writes a 1-page Markdown report
to ~/.hermes/audits/ue-skill-coverage-YYYY-MM-DD.md.
"""
import collections
import datetime
import json
import os
import sys
from pathlib import Path

TRACE_LOG = Path(os.path.expanduser('~/.hermes/logs/rule-injector-trace.jsonl'))
AUDIT_DIR = Path(os.path.expanduser('~/.hermes/audits'))
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def analyze(window_hours=24):
    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = (now - datetime.timedelta(hours=window_hours)).isoformat().replace('+00:00', 'Z')
    per_rule = collections.Counter()
    per_session = collections.defaultdict(collections.Counter)
    per_hook = collections.Counter()
    total = 0
    matched = 0
    if not TRACE_LOG.exists():
        return None
    for line in open(TRACE_LOG):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get('timestamp', '') < cutoff:
            continue
        total += 1
        if not r.get('matched'):
            continue
        matched += 1
        rid = r.get('rule_id', '?')
        per_rule[rid] += 1
        per_session[r.get('session_id', '?')][rid] += 1
        per_hook[r.get('hook', '?')] += 1
    return {
        'window_hours': window_hours,
        'cutoff': cutoff,
        'total_entries': total,
        'matched_entries': matched,
        'per_rule': per_rule,
        'per_session': per_session,
        'per_hook': per_hook,
    }


def render(data, date_label):
    lines = [
        f'# UE-Skill Routing Coverage — {date_label}',
        '',
        f'**Window:** last {data["window_hours"]} h (since {data["cutoff"]})',
        f'**Trace entries:** {data["total_entries"]} total / {data["matched_entries"]} matched',
        f'**Per-hook:** {dict(data["per_hook"])}',
        '',
        '## Top 30 fired rules (UE-family)',
    ]
    ue_rules = [(rid, n) for rid, n in data['per_rule'].most_common()
                if rid.startswith('ue-') or rid in {
                    'unreal-engine-router', 'ue-workflow-skill-router',
                    'ue-windows-observation-gate', 'ue-tool-and-screenshot-combo',
                    'ue-continuation-active-guard',
                }]
    for rid, n in ue_rules[:30]:
        lines.append(f'- `{rid}`: {n}')
    lines += ['', '## Sessions with UE-rule firings']
    for sid, rid_counts in sorted(data['per_session'].items(), key=lambda kv: -sum(kv[1].values()))[:10]:
        ue_only = {k: v for k, v in rid_counts.items() if k.startswith('ue-') or k in {'unreal-engine-router'}}
        if ue_only:
            lines.append(f'- `{sid}`: {sum(ue_only.values())} UE firings ({len(ue_only)} distinct rules)')
    return '\n'.join(lines) + '\n'


def main():
    today = datetime.date.today().isoformat()
    data = analyze(window_hours=24)
    if data is None:
        print(f'Trace log not found at {TRACE_LOG}')
        return 0
    out = render(data, today)
    out_path = AUDIT_DIR / f'ue-skill-coverage-{today}.md'
    out_path.write_text(out)
    print(f'Wrote {out_path} ({data["matched_entries"]} matched entries, '
          f'{len([r for r, _ in data["per_rule"].most_common() if r.startswith("ue-") or r == "unreal-engine-router"])} UE rules fired)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
