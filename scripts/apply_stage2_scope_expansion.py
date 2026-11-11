"""Apply Stage 2 rule modifications to rules.json.

Modifies the 55 single-scope ue-skill-* rules + ue-tool-and-screenshot-combo
to add a `recent_assistant` keyword branch. Adds natural-prose keywords to
the vendor (UDS/UDW) rules. Disambiguates the F12 keyword collision
between uds-setup-and-modes and uds-performance-mobile-troubleshooting.

This script is idempotent: re-running it produces the same rules.json.

Side-effects: writes to rules.json via RuleStore (atomic temp+rename +
mtime bump). Also wipes plugin __pycache__ so the next gateway start
recompiles.

Run::

    python3 scripts/apply_stage2_scope_expansion.py
"""
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Map rule_id -> list of 4-8 high-signal class/API names to add to the
# recent_assistant keyword branch. These mirror the existing latest_user_message
# keyword list but DROP generic "unreal" / "UE5" tokens that are too common
# in any assistant narration.
RECENT_ASSISTANT_KEYWORDS = {
    'ue-skill-actors-and-components':          ['AActor', 'UActorComponent', 'USceneComponent', 'UPrimitiveComponent', 'BeginPlay', 'EndPlay', 'Destroyed', 'root component', 'RegisterComponent', 'UChildActorComponent'],
    'ue-skill-ai-and-navigation':              ['Behavior Tree', 'behavior tree', 'Blackboard', 'AIController', 'UBTTaskNode', 'UBTDecorator', 'UBTService', 'NavMesh', 'EQS', 'FEnvQueryRequest', 'AI Perception', 'UAIPerceptionComponent', 'OnTargetPerceptionUpdated', 'StateTree', 'UStateTreeAIComponent', 'MoveTo'],
    'ue-skill-asset-management':              ['TSoftObjectPtr', 'TSoftClassPtr', 'FSoftObjectPath', 'FStreamableManager', 'FStreamableHandle', 'UAssetRegistry', 'AssetRegistry', 'UAssetManager', 'primary data asset', 'asset bundle', 'async load', 'load hitch', 'ConstructorHelpers', 'FObjectFinder'],
    'ue-skill-audio-and-metasounds':          ['MetaSound', 'SoundCue', 'SoundWave', 'UAudioComponent', 'PlaySound2D', 'PlaySoundAtLocation', 'spatial attenuation', 'submix', 'audio concurrency', 'Quartz', 'USoundAttenuation', 'USoundClass'],
    'ue-skill-automation-and-testing':        ['IMPLEMENT_SIMPLE_AUTOMATION_TEST', 'IMPLEMENT_COMPLEX_AUTOMATION_TEST', 'DEFINE_SPEC', 'AFunctionalTest', 'Catch2', 'EAutomationTestFlags', 'FAutomationTestBase', 'TestTrue', 'TestEqual', 'TestNotNull', 'AddError', 'latent command'],
    'ue-skill-blueprint-fundamentals':        ['UBlueprint', 'UBlueprintGeneratedClass', 'event graph', 'construction script', 'Blueprint variable', 'Blueprint component', 'BP graph', 'macro library', 'blueprint function', 'blueprint compilation'],
    'ue-skill-character-and-movement':        ['ACharacter', 'UCharacterMovementComponent', 'character movement', 'movement mode', 'capsule component', 'root motion', 'air control', 'crouch', 'JumpMaxHoldTime', 'PhysCustom', 'Mover plugin'],
    'ue-skill-coding-standards':              ['Epic coding standard', 'UE naming convention', 'unreal coding style', 'class prefix', 'PascalCase', 'bBool prefix', 'Allman brace', 'TEXT() macro', 'IWYU', 'forward declaration'],
    'ue-skill-control-rig-and-ik':            ['Control Rig', 'RigVM', 'URigHierarchy', 'FRigUnit', 'IK Rig', 'UIKRigDefinition', 'IK Retargeter', 'UIKRetargeter', 'Two Bone IK', 'FABRIK', 'CCDIK', 'FAnimNode_ControlRig', 'FAnimNode_IKRig', 'UControlRigComponent', 'UIKRigComponent'],
    'ue-skill-core-types-and-containers':     ['TArray', 'TMap', 'TSet', 'TQueue', 'FString', 'FName', 'FText', 'FVector', 'FRotator', 'FQuat', 'FTransform', 'FMath', 'TOptional', 'TVariant', 'TTuple', 'TStringBuilder', 'Large World Coordinates', 'LWC', 'double precision'],
    'ue-skill-data-driven-design':            ['DataTable', 'UDataTable', 'FTableRowBase', 'UCompositeDataTable', 'UDataAsset', 'UPrimaryDataAsset', 'UCurveFloat', 'UCurveTable', 'FRuntimeFloatCurve', 'UDeveloperSettings', 'UPROPERTY(config)', 'GConfig', 'ini config', 'CSV import'],
    'ue-skill-debugging-techniques':          ['DrawDebug', 'DrawDebugLine', 'DrawDebugSphere', 'DrawDebugString', 'DrawDebugPoint', 'Visual Logger', 'UE_VLOG', 'Gameplay Debugger', 'FGameplayDebuggerCategory', 'AddOnScreenDebugMessage', 'debug overlay', 'natvis', 'Live Coding', 'debugger unreal c++'],
    'ue-skill-delegates-and-events':          ['DECLARE_DELEGATE', 'DECLARE_MULTICAST_DELEGATE', 'DECLARE_DYNAMIC_MULTICAST_DELEGATE', 'BlueprintAssignable', 'AddDynamic', 'RemoveDynamic', 'BindUObject', 'AddUObject', 'BindLambda', 'AddWeakLambda', 'FDelegateHandle', 'ExecuteIfBound', 'Broadcast', 'DECLARE_EVENT'],
    'ue-skill-editor-scripting-and-python':   ['Blutility', 'UEditorUtilityWidget', 'UEditorUtilityObject', 'UEditorUtilityWidgetBlueprint', 'UAssetActionUtility', 'commandlet', 'batch process assets', 'editor automation', 'UEditorSubsystem', 'UEditorActorSubsystem', 'UEditorAssetSubsystem', 'ULevelEditorSubsystem', 'unreal module python', 'unreal python'],
    'ue-skill-enhanced-input':               ['Enhanced Input', 'UInputAction', 'UInputMappingContext', 'UEnhancedInputComponent', 'UEnhancedInputLocalPlayerSubsystem', 'AddMappingContext', 'RemoveMappingContext', 'BindAction', 'ETriggerEvent', 'UInputModifier', 'UInputTrigger', 'FInputActionValue', 'input mapping context', 'legacy input migration'],
    'ue-skill-gameplay-ability-system':       ['AbilitySystemComponent', 'UAbilitySystemComponent', 'UGameplayAbility', 'UGameplayEffect', 'UAttributeSet', 'FGameplayAttributeData', 'ATTRIBUTE_ACCESSORS', 'ability task', 'UAbilityTask', 'GameplayCue', 'InstancedPerActor', 'LocalPredicted', 'replication mode', 'cooldown ability', 'cost ability', 'buff debuff', 'GameplayAbilities plugin'],
    'ue-skill-gameplay-framework':            ['AGameModeBase', 'AGameMode', 'AGameStateBase', 'AGameState', 'APlayerController', 'APlayerState', 'UGameInstance', 'game framework', 'login flow', 'possession', 'RestartPlayer', 'PostLogin', 'PreLogin', 'HandleStartingNewPlayer', 'match state', 'HUD class'],
    'ue-skill-gameplay-tags':                 ['FGameplayTag', 'FGameplayTagContainer', 'Gameplay Tag', 'UE_DECLARE_GAMEPLAY_TAG', 'UE_DEFINE_GAMEPLAY_TAG', 'RequestGameplayTag', 'UGameplayTagsManager', 'FGameplayTagQuery', 'IGameplayTagAssetInterface', 'DefaultGameplayTags.ini', 'tag query', 'tag container', 'MatchesTag', 'HasTag'],
    'ue-skill-landscape-and-foliage':        ['ALandscape', 'ULandscapeComponent', 'heightmap', 'UFoliageType', 'AInstancedFoliageActor', 'landscape spline', 'PCG', 'UPCGComponent', 'UPCGGraph', 'procedural generation', 'terrain sculpt', 'landscape material layer', 'edit layer'],
    'ue-skill-levels-and-world-partition':    ['World Partition', 'UWorldPartition', 'data layer', 'UDataLayerAsset', 'level instance', 'ALevelInstance', 'APackedLevelActor', 'One File Per Actor', 'OFPA', 'streaming level', 'ULevelStreaming', 'LoadStreamLevel', 'streaming source'],
    'ue-skill-lighting-and-lumen':            ['Lumen', 'global illumination', 'UDirectionalLightComponent', 'UPointLightComponent', 'USpotLightComponent', 'URectLightComponent', 'USkyLightComponent', 'light mobility', 'Stationary light', 'Movable light', 'virtual shadow maps', 'USkyAtmosphereComponent', 'UExponentialHeightFogComponent', 'auto exposure', 'reflection capture', 'post process volume'],
    'ue-skill-logging-and-assertions':        ['UE_LOG', 'DECLARE_LOG_CATEGORY', 'DEFINE_LOG_CATEGORY', 'log category', 'UE_LOGFMT', 'structured logging', 'check();', 'checkf();', 'verify();', 'verifyf();', 'ensure();', 'ensureMsgf', 'ensureAlways', 'FMsg', 'FDebug', 'verbosity level'],
    'ue-skill-materials-and-shaders':        ['UMaterial', 'UMaterialInstanceConstant', 'UMaterialInstanceDynamic', 'material instance', 'material parameter', 'scalar parameter', 'vector parameter', 'texture parameter', 'SetScalarParameterValue', 'SetVectorParameterValue', 'material function', 'material domain', 'shading model', 'blend mode', 'material parameter collection', 'CreateDynamicMaterialInstance'],
    'ue-skill-memory-and-gc':                ['garbage collection', 'TWeakObjectPtr', 'TSharedPtr', 'TSharedRef', 'TUniquePtr', 'TObjectPtr', 'UPROPERTY pointer', 'GC root', 'dangling pointer', 'MarkAsGarbage', 'AddToRoot', 'RemoveFromRoot', 'FGCObject', 'AddReferencedObjects', 'TStrongObjectPtr'],
    'ue-skill-meshes-static-and-skeletal':   ['UStaticMesh', 'UStaticMeshComponent', 'USkeletalMesh', 'USkeletalMeshComponent', 'USkinnedMeshComponent', 'instanced static mesh', 'ISM', 'HISM', 'UHierarchicalInstancedStaticMeshComponent', 'material slot', 'mesh collision', 'UBodySetup', 'Nanite mesh', 'LOD', 'mesh section', 'PhysicsAsset'],
    'ue-skill-module-and-build-system':      ['Build.cs', 'Target.cs', 'ModuleRules', 'TargetRules', 'UnrealBuildTool', 'UBT', 'IMPLEMENT_MODULE', 'IMPLEMENT_PRIMARY_GAME_MODULE', 'IModuleInterface', 'unresolved external', 'module not found', 'PublicDependencyModuleNames'],
    'ue-skill-nanite-and-rendering':         ['Nanite', 'virtualized geometry', 'FMeshNaniteSettings', 'Nanite displacement', 'Nanite tessellation', 'deferred rendering', 'forward rendering', 'GPU Scene', 'TSR', 'temporal super resolution', 'post process', 'FPostProcessSettings', 'scene capture', 'render target', 'scalability cvar', 'virtual texture'],
    'ue-skill-navigating-engine-source':     ['Engine/Source', 'engine source', 'find API', 'UClass hierarchy', 'module lookup', 'include path', 'API signature', 'verify API exists', 'Public/Private/Classes', 'source tree layout', 'UHT generated file'],
    'ue-skill-networking-and-replication':   ['replication', 'replicate', 'RPC', 'UFUNCTION(Server', 'UFUNCTION(Client', 'UFUNCTION(NetMulticast', 'Reliable', 'Unreliable', 'Replicated', 'ReplicatedUsing', 'OnRep_', 'DOREPLIFETIME', 'DOREPLIFETIME_CONDITION', 'Push Model', 'MARK_PROPERTY_DIRTY_FROM_NAME', 'HasAuthority', 'GetLocalRole', 'GetRemoteRole', 'FFastArraySerializer', 'Iris', 'net dormancy', 'NetUpdateFrequency', 'bReplicates'],
    'ue-skill-niagara-vfx':                  ['Niagara', 'UNiagaraSystem', 'UNiagaraComponent', 'UNiagaraFunctionLibrary', 'particle', 'VFX', 'emitter', 'SpawnSystemAtLocation', 'SpawnSystemAttached', 'Niagara module', 'Niagara Data Interface', 'Cascade migration', 'GPU emitter', 'CPU emitter', 'NiagaraDataChannel', 'OnSystemFinished'],
    'ue-skill-packaging-and-deployment':     ['cook', 'package my game', 'package game', 'package project', 'shipping build', 'UAT', 'IoStore', '.utoc', '.ucas', 'pak file', 'ProjectPackagingSettings', 'bUseIoStore', 'bGenerateChunks', 'DLC', 'platform deployment', 'build configuration', 'DebugGame', 'DirectoriesToAlwaysCook', 'content-on-demand'],
    'ue-skill-physics-and-chaos':            ['Chaos physics', 'collision channel', 'ECollisionChannel', 'ECollisionResponse', 'ECR_Block', 'ECR_Overlap', 'collision profile', 'line trace', 'sweep', 'SweepSingleByChannel', 'LineTraceSingleByChannel', 'OverlapMultiByChannel', 'FHitResult', 'FCollisionQueryParams', 'OnComponentHit', 'OnComponentBeginOverlap', 'UPhysicalMaterial', 'AddForce', 'AddImpulse', 'ragdoll', 'UPhysicsAsset', 'FConstraintInstance', 'UPhysicsConstraintComponent', 'SetSimulatePhysics'],
    'ue-skill-plugins-and-modules':          ['.uplugin', 'plugin descriptor', 'EHostType', 'ELoadingPhase', 'IModuleInterface', 'StartupModule', 'ShutdownModule', 'engine plugin', 'project plugin', 'IPluginManager', 'IPlugin', 'content-only plugin', 'explicit-load plugin', 'plugin dependency', 'EnabledByDefault', 'CanContainContent'],
    'ue-skill-profiling-and-optimization':   ['Unreal Insights', 'stat unit', 'stat fps', 'stat game', 'stat gpu', 'stat scenerendering', 'stat memory', 'profiling', 'SCOPE_CYCLE_COUNTER', 'QUICK_SCOPE_CYCLE_COUNTER', 'TRACE_CPUPROFILER_EVENT_SCOPE', 'DECLARE_STATS_GROUP', 'CSV_SCOPED_TIMING_STAT', 'memory profiling', 'LLM', 'MemReport', 'frame rate', 'hitch', 'optimization'],
    'ue-skill-project-structure':           ['.uproject', 'EngineAssociation', 'FProjectDescriptor', 'Config/Default', 'plugin reference', 'unreal project structure', 'Binaries/', 'Intermediate/', 'DerivedDataCache', 'content virtual path', '/Game/', 'primary game module', 'IMPLEMENT_PRIMARY_GAME_MODULE'],
    'ue-skill-save-and-load':               ['USaveGame', 'SaveGameToSlot', 'LoadGameFromSlot', 'AsyncSaveGameToSlot', 'AsyncLoadGameFromSlot', 'DoesSaveGameExist', 'DeleteGameInSlot', 'FMemoryWriter', 'FMemoryReader', 'FObjectAndNameAsStringProxyArchive', 'save file', 'serialize actor', 'save slot', 'save versioning', 'ISaveGameSystem', 'ULocalPlayerSaveGame', 'ArIsSaveGame'],
    'ue-skill-sequencer-and-cinematics':   ['Sequencer', 'ULevelSequence', 'ALevelSequenceActor', 'ULevelSequencePlayer', 'CreateLevelSequencePlayer', 'Cine Camera', 'UCineCameraComponent', 'ACineCameraActor', 'Movie Render Queue', 'Movie Render Pipeline', 'cutscene', 'cinematic', 'camera cuts', 'possessable', 'spawnable', 'UMovieScene', 'sequence binding'],
    'ue-skill-subsystems':                  ['USubsystem', 'UGameInstanceSubsystem', 'UWorldSubsystem', 'UTickableWorldSubsystem', 'UEditorSubsystem', 'UEngineSubsystem', 'ULocalPlayerSubsystem', 'Initialize subsystem', 'Deinitialize', 'ShouldCreateSubsystem', 'InitializeDependency', 'subsystem framework'],
    'ue-skill-timers-and-async':             ['FTimerManager', 'SetTimer', 'FTimerHandle', 'ClearTimer', 'SetTimerForNextTick', 'PauseTimer', 'UnPauseTimer', 'Async', 'EAsyncExecution', 'AsyncTask', 'ENamedThreads', 'TFuture', 'TPromise', 'FNonAbandonableTask', 'FAutoDeleteAsyncTask', 'FAsyncTask', 'FRunnable', 'FRunnableThread', 'UE Tasks System', 'UE::Tasks::Launch', 'FTSTicker', 'latent action'],
    'ue-tool-and-screenshot-combo':         ['Unreal', 'UE5', 'Blueprint', 'Lovecraftian', 'Lovecraft', 'UE MCP', 'MCP', 'screenshot', 'screen', 'editor', 'viewport', 'working'],
}

# Natural-prose keyword extensions for vendor rules (Stage 002 + F11).
# These are added to the EXISTING latest_user_message branch (which the
# current scope already uses), not to the new recent_assistant branch.
VENDOR_NATURAL_PROSE_KEYWORDS = {
    'ue-skill-uds-cinematics-rendering':    ['cinematic', 'path tracer', 'movie render', 'MRQ', 'cutscene', 'sequencer'],
    'ue-skill-uds-clouds':                  ['clouds', 'wispy', 'cloudy', 'overcast', 'cloud performance'],
    'ue-skill-uds-fog-and-atmosphere':     ['volumetric fog', 'atmospheric scattering'],
    'ue-skill-uds-lighting-and-shadows':    ['cloud shadows', 'cubemap sky light', 'cloud shadow'],
    'ue-skill-uds-simulation':             ['real sun', 'real moon', 'latitude', 'longitude', 'city preset', 'north yaw', 'simulate real'],
    'ue-skill-uds-sun-moon-stars':         ['sun position', 'moon orbit', 'aurora', 'light pollution', 'planets', 'nebula', 'moon phase', 'sun color'],
    'ue-skill-uds-time':                  ['day night cycle', 'sunrise event', 'sunset event', 'midnight', 'day length', 'night length'],
    'ue-skill-udw-material-and-screen-effects': ['puddle', 'rainbow', 'frost', 'footprints', 'wetness', 'snow material', 'wet ground', 'screen droplets', 'heat haze', 'breath'],
    'ue-skill-udw-particles-lightning-wind-sounds': ['rain', 'snow', 'lightning', 'wind', 'dust', 'thunder', 'puddle', 'rainbow', 'flash', 'strike', 'wind sound', 'rain sound'],
    'ue-skill-udw-random-seasons-temperature': ['temperature', 'season', 'weather variation', 'seasonal'],
}

# F12: uds-setup-and-modes has 8 bare bigrams that trigger on non-UDS
# UE discussions. Remove them; the remaining UDS-prefixed forms preserve
# accuracy.
UDS_SETUP_DROP_KEYWORDS = [
    'sky mode', 'static cloud', '2D dynamic cloud', 'voxel cloud',
    'aurora sky', 'space sky', 'volumetric cloud mode', 'UDS mobile',
]


def load_rules():
    with open(REPO / 'rules.json') as f:
        data = json.load(f)
    return data


def find_keyword_branch(rule):
    """Return the keyword condition with scope == 'latest_user_message' in a
    compound rule, or None if not found.
    """
    for c in rule.get('trigger', {}).get('conditions', []):
        if c.get('type') == 'keyword' and c.get('scope', '') == 'latest_user_message':
            return c
    return None


def has_recent_assistant_keyword(rule):
    for c in rule.get('trigger', {}).get('conditions', []):
        if c.get('type') == 'keyword' and c.get('scope') == 'recent_assistant':
            return True
    return False


def main():
    data = load_rules()
    rules = data['rules']
    rule_by_id = {r['id']: r for r in rules}
    changes = 0

    # Stage 2A: add recent_assistant keyword branch to 55 single-scope rules
    for rid, keywords in RECENT_ASSISTANT_KEYWORDS.items():
        r = rule_by_id.get(rid)
        if r is None:
            print(f'  SKIP {rid}: rule not found')
            continue
        if has_recent_assistant_keyword(r):
            print(f'  SKIP {rid}: already has recent_assistant branch')
            continue
        new_branch = {
            'type': 'keyword',
            'scope': 'recent_assistant',
            'keywords': keywords,
            'match_all': False,
            'case_insensitive': True,
        }
        # Make sure trigger type is compound
        if r.get('trigger', {}).get('type') != 'compound':
            print(f'  WARN {rid}: trigger type is {r.get("trigger", {}).get("type")} not compound; skipping')
            continue
        r['trigger'].setdefault('conditions', []).append(new_branch)
        changes += 1
        print(f'  OK   {rid}: added recent_assistant branch with {len(keywords)} keywords')

    # Stage 2B: add natural-prose keywords to vendor latest_user_message branch
    for rid, additions in VENDOR_NATURAL_PROSE_KEYWORDS.items():
        r = rule_by_id.get(rid)
        if r is None:
            print(f'  SKIP {rid}: rule not found')
            continue
        branch = find_keyword_branch(r)
        if branch is None:
            print(f'  WARN {rid}: no latest_user_message keyword branch found')
            continue
        existing = set(branch.get('keywords', []))
        new_kws = [k for k in additions if k not in existing]
        if not new_kws:
            print(f'  SKIP {rid}: all natural-prose keywords already present')
            continue
        branch['keywords'] = list(existing) + new_kws
        changes += 1
        print(f'  OK   {rid}: added {len(new_kws)} natural-prose keywords ({new_kws!r})')

    # Stage 2C: F12 disambiguation — remove 8 bare bigrams from
    # uds-setup-and-modes latest_user_message branch.
    r = rule_by_id.get('ue-skill-uds-setup-and-modes')
    if r is not None:
        branch = find_keyword_branch(r)
        if branch is not None:
            before = list(branch.get('keywords', []))
            branch['keywords'] = [k for k in before if k not in UDS_SETUP_DROP_KEYWORDS]
            dropped = [k for k in before if k in UDS_SETUP_DROP_KEYWORDS]
            if dropped:
                changes += 1
                print(f'  OK   ue-skill-uds-setup-and-modes: dropped {len(dropped)} ambiguous keywords ({dropped!r})')

    if changes == 0:
        print('\nNo changes needed (already applied).')
        return

    with open(REPO / 'rules.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f'\nWrote rules.json with {changes} change(s).')

    # Bump mtime so the live plugin hot-reloads (RuleStore._check_reload
    # uses mtime + content hash; content hash changes too because we
    # modified the file).
    new_mtime = time.time() + 5
    os.utime(REPO / 'rules.json', (new_mtime, new_mtime))

    # Wipe plugin __pycache__ so the next gateway start recompiles.
    cache = REPO / '__pycache__'
    if cache.exists():
        import shutil
        for entry in cache.iterdir():
            if entry.is_file() and entry.suffix == '.pyc':
                entry.unlink()
        print('Wiped stale __pycache__/*.pyc')


if __name__ == '__main__':
    main()
