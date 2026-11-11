#!/usr/bin/env python3
"""Replay representative turns from the audited UE and Mycelia sessions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rules import RuleStore
from matchers import MatchContext, RuleMatcher, _extract_recent_assistant


def fired(matcher, user_message, history, session_id):
    matcher._ensure_compiled()
    recent = " ".join(
        m.get("content", "")[:500] for m in history[-4:]
        if isinstance(m, dict) and isinstance(m.get("content"), str)
    )
    ctx = MatchContext(
        hook="pre_llm_call",
        user_message=user_message,
        recent_context=recent,
        recent_assistant=_extract_recent_assistant(history),
        conversation_history=history,
    )
    return {r.id for r in matcher._compiled if r.hook == "pre_llm_call" and r.matches(ctx)}


def main():
    matcher = RuleMatcher(RuleStore())

    ue_history = [
        {"role": "user", "content": "Continue the Lovecraftian creature setup."},
        {"role": "assistant", "content": "I will import the FBX animations with AssetImportTask, build the AnimBP state machine, then verify the Unreal viewport."},
    ]
    ue = fired(matcher, "Make the import dialogue appear and I will select Snap to Frame", ue_history, "telegram-ue")
    assert "unreal-engine-router" in ue, ue
    assert "ue-windows-observation-gate" in ue, ue

    correction = fired(
        matcher,
        "It worked the first time but you didn't check with me or screenshot the desktop so you missed that",
        ue_history,
        "telegram-ue",
    )
    assert "kevin-redirects-retrust-evidence" in correction, correction

    android_history = [
        {"role": "user", "content": "The Mycelia Android Kotlin tarot app uses Jetpack Compose and Gradle."},
        {"role": "assistant", "content": "I will update the ViewModel and Compose screen in the .kt files."},
    ]
    android = fired(matcher, "Now when shuffling the deck, 78 cards feels slow", android_history, "discord-mycelia")
    ue_fps = sorted(r for r in android if r.startswith("ue-") or r == "unreal-engine-router")
    assert not ue_fps, ue_fps

    print("Replay PASS: UE routing + proactive observation + correction firing; Mycelia UE false positives=0")


if __name__ == "__main__":
    main()
