"""Stage 4 — Author the ue-continuation-active-guard rule.

Design choices (after advisor audit correction):
  - hook: pre_llm_call
  - priority: 4 (alongside the router)
  - trigger: compound(operator=or) with three branches:
      1. recent_assistant keyword (narration super-terms)
      2. recent_context keyword (fallback: 4-message wider scan)
      3. recent_assistant semantic (model picks up on UE-work phrases)
  - cooldown_turns: 3, max_fires_per_session: 8
  - inject: a real, working tool-signal->UE-skill catalog (no placeholders)

The active guard is an active-guard pattern (per `rule-injector-authoring`
step 5): it fires whenever the agent is mid-task on UE work, NOT only when
the user types a specific bad word. The agent reads the inject and uses
it to select the right per-skill skill_view call.
"""
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

INJECT_TEXT = (
    "[UE_CONTINUATION_GUARD] You are mid-UE-task. Based on your recent tool activity, "
    "load the matching skill BEFORE proceeding:\n\n"
    "| Tool/Path Signal in your recent work | Load this skill |\n"
    "|---|---|\n"
    "| mcp__eldritch_ue__* tool, /Game/ asset path, .uasset result | skills/unreal-engine/asset-management.md |\n"
    "| AssetImportTask, FBX, Interchange, import dialogue | skills/unreal-engine/importing-content.md |\n"
    "| ULevel, World Partition, /Game/Maps/, ALevelInstance | skills/unreal-engine/levels-and-world-partition.md |\n"
    "| USkeletalMesh, AnimSequence, AnimBP, montage, Bone | skills/unreal-engine/animation-system.md |\n"
    "| UMaterial, UMaterialInstanceDynamic, FMaterialParameterInfo | skills/unreal-engine/materials-and-shaders.md |\n"
    "| UCLASS, UFUNCTION, UPROPERTY, GENERATED_BODY, AActor | skills/unreal-engine/cpp-fundamentals.md + coding-standards.md |\n"
    "| Blueprint, UBlueprint, UBlueprintGeneratedClass, UWidget | skills/unreal-engine/blueprint-fundamentals.md + umg-and-slate.md |\n"
    "| UGameplayTag, FGameplayTag, FGameplayTagContainer | skills/unreal-engine/gameplay-tags.md |\n"
    "| AbilitySystemComponent, UGameplayAbility, UAttributeSet | skills/unreal-engine/gameplay-ability-system.md |\n"
    "| BehaviorTree, Blackboard, AIController, UBTTaskNode | skills/unreal-engine/ai-and-navigation.md |\n"
    "| UCharacter, UCharacterMovementComponent, movement mode | skills/unreal-engine/character-and-movement.md |\n"
    "| FStreamableManager, TSoftObjectPtr, async load unreal | skills/unreal-engine/asset-management.md |\n"
    "| physics, collision, SweepSingleByChannel, FHitResult | skills/unreal-engine/physics-and-chaos.md |\n"
    "| mcp__windows_mcp__* tool (file ops, screenshots) | skills/unreal-engine/unreal-engine-build-workflow.md |\n"
    "| WSL, portproxy, svchost, wsl.localhost, netsh | skills/unreal-engine/unreal-engine-build-workflow.md |\n"
    "| Phase 1 plan, feasibility, monster catcher, palworld | skills/unreal-engine/unreal-engine-game-development.md |\n\n"
    "If no specific signal matches, load skills/unreal-engine/unreal-engine-router.md for guidance."
)

RULE = {
    'id': 'ue-continuation-active-guard',
    'name': 'UE continuation active-guard (fires on mid-task continuations)',
    'description': (
        'Active guard that fires during UE coding sessions when the user types '
        'short continuation messages ("continue", "please continue", "try now", "is '
        'it working now") and the agent is mid-task on UE work. Injects a '
        'tool-signal->UE-skill catalog so the agent selects the right per-skill '
        'skill_view call. Cooldown 3 turns, max 8 fires per session.'
    ),
    'enabled': True,
    'hook': 'pre_llm_call',
    'priority': 4,
    'cooldown_turns': 3,
    'max_fires_per_session': 8,
    'trigger': {
        'type': 'compound',
        'operator': 'or',
        'conditions': [
            {
                'type': 'keyword',
                'scope': 'recent_assistant',
                'keywords': [
                    'mcp__eldritch_ue__', 'mcp__windows_mcp__', 'AssetImportTask',
                    'FBX', 'SubobjectData', '/Game/', '/Script/Engine.', 'Blueprint',
                    'UAsset', 'BulkData', 'UCLASS', 'UFUNCTION', 'UPROPERTY',
                    'GENERATED_BODY', 'BehaviorTree', 'Blackboard', 'AIController',
                    'UBTTaskNode', 'AnimSequence', 'AnimBP', 'montage', 'UMaterial',
                    'UMaterialInstance', 'MaterialInstance', 'FMaterialParameter',
                    'AbilitySystemComponent', 'UGameplayAbility', 'UAttributeSet',
                    'FGameplayTag', 'UCharacterMovementComponent', 'ULevel',
                    'World Partition', 'ALevelInstance', 'AInstancedFoliageActor',
                    'FStreamableManager', 'TSoftObjectPtr', 'SweepSingleByChannel',
                    'FHitResult', 'OnComponentHit', 'OnComponentBeginOverlap',
                    'Live Coding', 'HttpListener', 'Wwise', 'portproxy', 'svchost',
                    'wsl.localhost', 'cmd.exe', 'MCP bridge', 'MCP server',
                    'UE editor', 'UE-MCP', 'eldritch-ue', 'VibeUE',
                    'Lovecraftian', 'LoveMonster', 'Mireling', 'Beholder', 'Eldritch',
                    'BP_CreatureSpawner', 'EldritchPlayerController',
                    'UEldritchPhase1Bootstrap',
                ],
                'match_all': False,
                'case_insensitive': True,
            },
            {
                'type': 'keyword',
                'scope': 'recent_context',
                'keywords': [
                    'mcp__eldritch_ue__', 'mcp__windows_mcp__', '/Game/', '.uasset',
                    '.fbx', 'AssetImportTask', 'UCLASS', 'UFUNCTION', 'UPROPERTY',
                    'BehaviorTree', 'Blackboard', 'Blueprint', 'UnrealEditor.exe',
                    'Live Coding', 'portproxy', 'Wwise', 'Wrangle',
                    'Lovecraftian', 'LoveMonster', 'Mireling', 'Beholder',
                ],
                'match_all': False,
                'case_insensitive': True,
            },
            {
                'type': 'semantic',
                'scope': 'recent_assistant',
                'references': [
                    'agent is mid-task on Unreal Engine C++ or Blueprint work',
                    'agent is driving the Unreal Editor from WSL via MCP',
                    'agent is debugging an Unreal Engine build error or compile failure',
                    'agent is importing assets into Unreal Engine',
                    'agent is configuring Unreal Engine project settings',
                    'agent is wiring Unreal Engine C++ classes to Blueprint subclasses',
                ],
                'threshold': 0.55,
            },
        ],
    },
    'inject': {
        'context': INJECT_TEXT
    },
    'exclude_if_recent_context_keywords': [
        'Kotlin', 'Android', 'Compose', 'gradle', 'Mycelia',
        'tarot-app', '.kt', 'Activity', 'ViewModel', 'iOS',
        'Swift', 'Python', 'FastAPI', 'Django', 'React',
        'Next.js', 'TypeScript', 'Unity', 'Godot',
    ],
}


def main():
    with open(REPO / 'rules.json') as f:
        data = json.load(f)
    rules = data['rules']
    rule_by_id = {r['id']: r for r in rules}
    if 'ue-continuation-active-guard' in rule_by_id:
        # Update in place
        rule_by_id['ue-continuation-active-guard'] = RULE
        print('  OK updated ue-continuation-active-guard (idempotent)')
    else:
        rules.append(RULE)
        print('  OK added ue-continuation-active-guard')

    with open(REPO / 'rules.json', 'w') as f:
        json.dump(data, f, indent=2)
    print('  Wrote rules.json')
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
