"""Author the two missing per-skill rules (Stage 3):

  - ue-skill-unreal-engine-build-workflow
  - ue-skill-unreal-engine-game-development

Both rules follow the established pattern: priority 5, pre_llm_call,
compound(keyword OR semantic), with the universal exclude block. The
recent_assistant branch is included so the rules fire on mid-task
narrations.

Idempotent: re-running produces the same rules.json (add() is no-op
when an ID already exists, but we use upsert-by-id to be safe).
"""
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

NEW_RULES = [
    {
        'id': 'ue-skill-unreal-engine-build-workflow',
        'name': 'UE WSL workflow skill (UE-MCP, portproxy, Live Coding)',
        'description': 'Load skill_view(name=unreal-engine-build-workflow) when working on WSL→Windows UE editor bridge, MCP server transport, portproxy, firewall, or Live Coding workarounds.',
        'enabled': True,
        'hook': 'pre_llm_call',
        'priority': 5,
        'trigger': {
            'type': 'compound',
            'operator': 'or',
            'conditions': [
                {
                    'type': 'keyword',
                    'scope': 'latest_user_message',
                    'keywords': [
                        'unreal-engine-wsl-workflow', 'unreal-engine build workflow',
                        'WSL UE', 'WSL Unreal', 'WSL bridge', 'WSL→Windows',
                        'UE MCP bridge', 'MCP bridge UE', 'UE-MCP', 'eldritch-ue',
                        'portproxy unreal', 'netsh portproxy', 'cmd.exe unreal',
                        'wsl.localhost', 'wsl.localhost Unreal', 'WSL→UE',
                        'hermes mcp', 'MCP server UE', '127.0.0.1:8000', '8001 port',
                        'portproxy 8001', 'httplistener unable to bind',
                    ],
                    'match_all': False,
                    'case_insensitive': True,
                },
                {
                    'type': 'semantic',
                    'scope': 'latest_user_message',
                    'references': [
                        'driving Unreal Editor from WSL via MCP bridge',
                        'setting up WSL→Windows portproxy for UE',
                        'handling MCP server transport errors for UE',
                        'killing zombie UE processes and port cleanup',
                        'Live Coding workarounds for UE on Windows',
                        'binding UE HttpListener fails because port is taken',
                    ],
                    'threshold': 0.55,
                },
                {
                    'type': 'keyword',
                    'scope': 'recent_assistant',
                    'keywords': [
                        'WSL', 'portproxy', 'cmd.exe', 'wsl.localhost', 'svchost',
                        'netsh', 'MCP server', 'MCP bridge', 'ClosedResourceError',
                        'httplistener', 'Live Coding', '172.30.160.1', '127.0.0.1:8000',
                        'UE editor process', 'VibeUE', 'eldritch-ue', 'port 8000',
                        'port 8001', 'Windows MCP', 'terminal', 'cmd.exe', 'UnrealEditor.exe',
                    ],
                    'match_all': False,
                    'case_insensitive': True,
                },
            ],
        },
        'inject': {
            'context': (
                '[HARD RULE: UE Skill] You are working on the Unreal Engine WSL workflow. '
                'Load skill_view(name=unreal-engine-build-workflow) FIRST. This skill covers '
                'WSL→Windows MCP bridge setup (portproxy, firewall, UE HttpListener binding), '
                'editor process lifecycle (kill zombies, restart cleanly), and Live Coding '
                'workarounds (close editor, disable Live Coding before build).'
            )
        },
        'exclude_if_recent_context_keywords': [
            'Kotlin', 'Android', 'Compose', 'gradle', 'Mycelia',
            'tarot-app', '.kt', 'Activity', 'ViewModel', 'iOS',
            'Swift', 'Python', 'FastAPI', 'Django', 'React',
            'Next.js', 'TypeScript', 'Unity', 'Godot',
        ],
    },
    {
        'id': 'ue-skill-unreal-engine-game-development',
        'name': 'UE game project planning + research skill',
        'description': 'Load skill_view(name=unreal-engine-game-development) when planning, scoping, or researching an Unreal Engine game project (phase plans, feasibility, existing implementations, vibe coding).',
        'enabled': True,
        'hook': 'pre_llm_call',
        'priority': 5,
        'trigger': {
            'type': 'compound',
            'operator': 'or',
            'conditions': [
                {
                    'type': 'keyword',
                    'scope': 'latest_user_message',
                    'keywords': [
                        'unreal-engine-game-development', 'game project plan',
                        'game feasibility', 'phase 1 plan unreal', 'phase plan unreal',
                        'game architecture unreal', 'game scope', 'palworld unreal',
                        'vibe coding unreal', 'vibeue', 'monster catcher plan',
                        'existing game implementation unreal', 'open world plan',
                        'procedural generation plan', 'crafting system plan',
                        'monster breeding unreal', 'rpg mechanics plan unreal',
                    ],
                    'match_all': False,
                    'case_insensitive': True,
                },
                {
                    'type': 'semantic',
                    'scope': 'latest_user_message',
                    'references': [
                        'planning an Unreal Engine game project',
                        'researching existing Unreal Engine game implementations',
                        'designing a phase-1 vertical slice for an Unreal game',
                        'evaluating feasibility of building a full game with UE',
                        'crafting game systems like combat, monsters, building',
                        'procedural generation for game content',
                    ],
                    'threshold': 0.55,
                },
                {
                    'type': 'keyword',
                    'scope': 'recent_assistant',
                    'keywords': [
                        'Phase 1 plan', 'feasibility', 'existing implementation',
                        'GitHub search', 'game architecture', 'scope decision',
                        'vertical slice', 'procedural generation', 'crafting',
                        'monster catcher', 'creature system', 'palworld',
                        'open world', 'monster breeding', 'NPC AI', 'quest system',
                    ],
                    'match_all': False,
                    'case_insensitive': True,
                },
            ],
        },
        'inject': {
            'context': (
                '[HARD RULE: UE Skill] You are planning or scoping an Unreal Engine game '
                'project. Load skill_view(name=unreal-engine-game-development) FIRST. This '
                'skill covers project-level research (existing GitHub implementations, '
                'feasibility, vibe coding workflow), phase planning (Phase 1 vertical slice, '
                'successor phases), and architectural decision patterns for monster-catcher / '
                'open-world / survival game loops.'
            )
        },
        'exclude_if_recent_context_keywords': [
            'Kotlin', 'Android', 'Compose', 'gradle', 'Mycelia',
            'tarot-app', '.kt', 'Activity', 'ViewModel', 'iOS',
            'Swift', 'Python', 'FastAPI', 'Django', 'React',
            'Next.js', 'TypeScript', 'Unity', 'Godot',
        ],
    },
]


def main():
    with open(REPO / 'rules.json') as f:
        data = json.load(f)
    rules = data['rules']
    rule_by_id = {r['id']: r for r in rules}
    added = 0
    for new in NEW_RULES:
        if new['id'] in rule_by_id:
            print(f'  SKIP {new["id"]}: already exists')
            continue
        rules.append(new)
        added += 1
        print(f'  OK   added {new["id"]}')
    if added == 0:
        print('\nNo changes needed.')
        return
    with open(REPO / 'rules.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f'\nWrote rules.json with {added} new rule(s).')
    # Bump mtime to force hot-reload
    new_mtime = time.time() + 5
    os.utime(REPO / 'rules.json', (new_mtime, new_mtime))
    cache = REPO / '__pycache__'
    if cache.exists():
        for entry in cache.iterdir():
            if entry.is_file() and entry.suffix == '.pyc':
                entry.unlink()
        print('Wiped stale __pycache__/*.pyc')


if __name__ == '__main__':
    main()
