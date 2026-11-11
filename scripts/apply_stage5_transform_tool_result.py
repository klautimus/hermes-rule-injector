"""Stage 5 — Author 4 transform_tool_result UE-aware rules.

The matcher's transform_tool_result hook scans ctx.tool_result for
keywords using result_contains_keywords. We add rules that fire when
real UE work outputs surface in tool results.

Each rule:
  - hook: transform_tool_result
  - trigger: result_contains_keywords (with tool_name filter optional)
  - inject.result_suffix: a short, specific guidance hint

This closes C12 (transform_tool_result gap) — 0 of 1,509 prior
transform_tool_result entries are UE.
"""
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

RULES = [
    {
        'id': 'ue-tool-result-asset-path',
        'name': 'UE asset path in tool result',
        'description': 'Tool result contains /Game/... path or UBlueprint/AssetData reference. Remind to load asset-management skill before mutating.',
        'enabled': True,
        'hook': 'transform_tool_result',
        'priority': 7,
        'trigger': {
            'type': 'result_contains_keywords',
            'result_contains_keywords': [
                '/Game/', '/Engine/', 'UBlueprintGeneratedClass', 'UBlueprint',
                'USkeletalMesh', 'UStaticMesh', 'UMaterialInstance',
                'AssetImportTask', 'Interchange', 'AssetRegistry', 'AssetData',
                '.uasset', 'asset_path', 'FAssetData',
            ],
        },
        'inject': {
            'result_suffix': (
                '[UE-tool-result-asset-path] The tool result returned UE asset paths or class '
                'references. Load skill_view(name=asset-management) before mutating these assets '
                '(TSoftObjectPtr vs hard reference, async load via FStreamableManager, etc.).'
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
        'id': 'ue-tool-result-editor-log',
        'name': 'UE editor log pattern in tool result',
        'description': 'Tool result contains UE editor log output (LogCompile/LogBlueprint/LogAssetTools). Remind to load build-workflow skill for common failure modes.',
        'enabled': True,
        'hook': 'transform_tool_result',
        'priority': 7,
        'trigger': {
            'type': 'result_contains_keywords',
            'result_contains_keywords': [
                'LogCompile:', 'LogBlueprint:', 'LogAssetTools:', 'LogInterchangeEngine:',
                'LogPython:', 'LogSkeletalMesh', 'LogStaticMesh', 'LogTexture',
                'LogMaterial', 'LogLiveCoding', 'LogHttpListener',
                'UAT: ', 'BuildCookRun', 'unable to bind', 'unresolved external',
                'Live Coding', 'Hot Reload', 'LogProperty',
                'LogModuleManager: Warning',
            ],
        },
        'inject': {
            'result_suffix': (
                '[UE-tool-result-editor-log] UE editor log output detected. Common failure '
                'modes: "unable to bind" -> check portproxy; "unresolved external" -> check '
                'Build.cs PublicDependencyModuleNames; "Live Coding active" -> close editor '
                'before build. Load skill_view(name=unreal-engine-build-workflow) for cross-WSL '
                'recipes if needed.'
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
        'id': 'ue-tool-result-cpp-error',
        'name': 'UE C++ build/compile error in tool result',
        'description': 'Tool result contains UE C++ build error. Remind to load module-and-build-system + cpp-fundamentals + importing-content for header issues.',
        'enabled': True,
        'hook': 'transform_tool_result',
        'priority': 7,
        'trigger': {
            'type': 'result_contains_keywords',
            'result_contains_keywords': [
                'error C', 'error LNK', 'unresolved external symbol',
                'cannot open include file', 'syntax error:', 'fatal error C',
                'LNK2019:', 'LNK2001:', 'LNK1120:', 'MSB',
                'UnrealBuildTool', 'UBT', 'Compile Module',
                'Build.cs', 'header file for module',
            ],
        },
        'inject': {
            'result_suffix': (
                '[UE-tool-result-cpp-error] C++ build/compile error in UE build output. Load '
                'skill_view(name=module-and-build-system) FIRST for the canonical fix patterns. '
                'If header issue, also load skill_view(name=importing-content) for FBX/asset '
                'header references. If class declaration issue, load '
                'skill_view(name=cpp-fundamentals).'
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
        'id': 'ue-tool-result-fbx-doc',
        'name': 'FBX document attached in tool result',
        'description': 'Tool result indicates an FBX file was attached (Kaydara FBX, or document cache path). Remind to load importing-content + animation-system BEFORE Python or C++ import.',
        'enabled': True,
        'hook': 'transform_tool_result',
        'priority': 7,
        'trigger': {
            'type': 'result_contains_keywords',
            'result_contains_keywords': [
                'Kaydara FBX', 'FBX Binary', 'fbx', '.fbx',
                'documents/', 'untrusted_tool_result',
            ],
        },
        'inject': {
            'result_suffix': (
                '[UE-tool-result-fbx-doc] FBX document attached. Load '
                'skill_view(name=importing-content) BEFORE attempting any Python or C++ '
                'import. If skeleton/anim planned, also load skill_view(name=animation-system). '
                'Check the file for valid SkeletalMesh + AnimSequence data, not just the mesh.'
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
    for new in RULES:
        if new['id'] in rule_by_id:
            rule_by_id[new['id']] = new
            print(f'  OK updated {new["id"]} (idempotent)')
        else:
            rules.append(new)
            added += 1
            print(f'  OK added {new["id"]}')
    with open(REPO / 'rules.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f'\nWrote rules.json with {added} new rule(s).')
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
