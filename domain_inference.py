"""Domain inference for rule-injector candidates.

Resolution order for a rule's `domain` field:
  1. Explicit `domain` field on the rule's JSON object, if present and non-empty.
  2. Inferred from `rule_id` via DOMAIN_INFERENCE_TABLE (first match wins).
  3. Default: "general".

Used by matchers.py to enrich the candidate payload sent to the LLM
adjudicator, and by the post-verdict contradiction check in
run_pre_llm_call.
"""
from __future__ import annotations

import re
from typing import List, Tuple, Optional

# Order matters: first matching pattern wins.
DOMAIN_INFERENCE_TABLE: List[Tuple[str, str]] = [
    (r"^pipeline-.*-first$",          "software-development"),
    (r"^pipeline-.*-cpp-reminder$",   "software-development"),
    (r"^pipeline-.*-reminder$",       "software-development"),
    (r"^using-agent-skills$",         "software-development"),
    (r"^research-.*",                 "research"),
    (r"^creative-.*|^draft-.*",       "creative-writing"),
    (r"^safety-.*|^dangerous-.*",     "safety"),
    (r"^doc-.*|^readme-.*",           "documentation"),
]

DOMAIN_DEFAULT = "general"

# Keywords that signal genuine software-development intent in a user message.
# Used by the contradiction check (Step 4 of plan 007) — a candidate whose
# domain resolves to software-development but whose user message contains
# NONE of these is force-dropped regardless of the LLM adjudicator's verdict.
SOFTWARE_CONTEXT_KEYWORDS = (
    "code", "coding", "build", "compile", "debug", "feature",
    "function", "class", "module", "method", "API", "endpoint",
    "repo", "repository", "commit", "branch", "merge", "PR",
    "source", "test", "refactor", "implement", "patch", "fix",
    # File extensions
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
    ".cpp", ".c", ".h", ".hpp", ".rb", ".sh", ".sql", ".yml",
    ".yaml", ".toml", ".kt", ".swift",
    # Frameworks / runtimes / package managers
    "react", "vue", "django", "flask", "fastapi", "express",
    "spring", "rails", "docker", "kubernetes", "npm", "pip",
    "cargo", "go build", "make", "cmake", "pytest", "jest",
    "gradle", "mvn",
    # Dev-ops / OS / installation signals (Plan 007 audit fix, 2026-07-27):
    # these were missing from the original list and caused T3 false negatives
    # on legitimate coding requests like "do the patching and get it setup
    # on my windows system" — the message has generic verbs but no dev-ops
    # signal. Adding these terms restores the firing path without weakening
    # the T2 research-suppression guarantee (research messages don't contain
    # these terms in their semantic surface).
    "windows", "macos", "linux", "ubuntu", "debian", "arch",
    "install", "uninstall", "reinstall", "setup", "set up",
    "configure", "patch", "patching", "deploy", "rollout",
    "wsl", "wsl2", "powershell", "bash", "zsh", "cli",
    "executable", "binary", "daemon", "shell",
    # F6 (2026-08-04 audit): audit/review/bug vocabulary was missing, so the
    # contradiction check force-dropped software-domain rules on legitimate
    # software requests phrased with these words (e.g. "audit the rule
    # injector"). "fix" was already present — these close the gap without
    # adding generic words like "review"/"log" that appear in non-software
    # messages.
    "audit", "inspect", "analyze", "analyse", "diagnos",
    "bug", "plugin", "integration", "codebase", "config",
)

_COMPILED = [(re.compile(pat), dom) for pat, dom in DOMAIN_INFERENCE_TABLE]


def resolve_domain(rule_id: str, explicit: Optional[str] = None) -> str:
    """Resolve a candidate's domain from explicit field + rule_id inference."""
    if explicit and isinstance(explicit, str) and explicit.strip():
        return explicit.strip().lower()
    for pattern, dom in _COMPILED:
        if pattern.match(rule_id):
            return dom
    return DOMAIN_DEFAULT


def has_software_context(user_message: str) -> bool:
    """True iff user_message contains any SOFTWARE_CONTEXT_KEYWORDS token."""
    if not user_message:
        return False
    lower = user_message.lower()
    return any(kw in lower for kw in SOFTWARE_CONTEXT_KEYWORDS)
