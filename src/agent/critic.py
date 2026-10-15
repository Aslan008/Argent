"""The Critic: an independent, adversarial reviewer of a plan / idea / answer.

Run as an isolated sub-agent (cleared context, no tools by default), so it judges
the artifact cold instead of defending the reasoning that produced it. Same model
by default; a stronger/different model is a later option. The prompt is kept
here, separate from the agent wiring, so it can be unit-tested on its own.
"""

import re
from enum import Enum
from typing import Tuple


class CriticVerdict(str, Enum):
    SOLID = "SOLID"
    FIXABLE = "FIXABLE"
    RETHINK = "RETHINK"
    UNKNOWN = "UNKNOWN"


def parse_critic_verdict(text: str) -> CriticVerdict:
    """Extract VERDICT: SOLID | FIXABLE | RETHINK from critic output.
    
    Robust against markdown bolding (**VERDICT: SOLID**), trailing punctuation,
    or mixed casing.
    """
    if not text:
        return CriticVerdict.UNKNOWN
    match = re.search(r"(?:VERDICT|\*\*VERDICT\*\*):\s*(SOLID|FIXABLE|RETHINK)\b", text, re.IGNORECASE)
    if match:
        val = match.group(1).upper()
        if val in CriticVerdict.__members__:
            return CriticVerdict(val)
    return CriticVerdict.UNKNOWN


def sanitize_and_truncate_diff(diff_text: str, max_chars: int = 24000) -> str:
    """Filter out noise (lockfiles, minified bundles) and truncate giant diffs
    before feeding to the Critic subagent.
    """
    if not diff_text:
        return ""
    lines = diff_text.splitlines()
    filtered_lines = []
    skip_file = False
    
    # Skip noisy files that bloat context without architectural value
    noise_patterns = (
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
        "Cargo.lock", "composer.lock", ".min.js", ".min.css", ".map",
    )
    
    for line in lines:
        if line.startswith("diff --git"):
            skip_file = any(pat in line for pat in noise_patterns)
        if not skip_file:
            filtered_lines.append(line)
            
    filtered_text = "\n".join(filtered_lines)
    if len(filtered_text) > max_chars:
        half = max_chars // 2
        filtered_text = (
            filtered_text[:half]
            + f"\n\n[... Diff truncated ({len(filtered_text)} chars total, showing first & last {half} chars) ...]\n\n"
            + filtered_text[-half:]
        )
    return filtered_text


CRITIC_SYSTEM = (
    "You are a ruthless but fair Critic. You did NOT write what you are reviewing "
    "and you have no stake in it — your only job is to find what is wrong with it "
    "before it costs real work.\n\n"
    "Treat the plan, idea or answer you are given as GUILTY UNTIL PROVEN INNOCENT. "
    "Hunt for: the false assumption, the missing or mishandled case, the step that "
    "won't actually work, the hidden dependency, the simpler approach that was "
    "overlooked, the way it fails under load / edge input / on this platform.\n\n"
    "RULES:\n"
    "- Report AT MOST 5 findings, hardest-hitting first. For each: a one-line title, "
    "one line on why it is a real problem, and a concrete fix or counterexample. Be "
    "specific — name the exact step / file / assumption. No vague 'could be improved'.\n"
    "- Judge what you were given. If read-only tools are available, you may use them "
    "to check if files, methods or classes actually exist before claiming they do or do not.\n"
    "- If, after honest scrutiny, there is no significant problem, say exactly: "
    "'No significant issues found.' Never invent problems to look useful — a false "
    "alarm is also a failure.\n"
    "- Do not rewrite the whole thing; point precisely at what to change.\n"
    "- End with one line: VERDICT: SOLID | FIXABLE | RETHINK.\n\n"
    "CRITICAL LANGUAGE RULE: Respond and reason in the EXACT SAME LANGUAGE as the material "
    "you are reviewing (if the plan/goal is in Russian, answer completely in Russian)."
)


def build_critique_task(target: str, goal: str = None, what: str = "plan / idea", max_chars: int = 32000) -> str:
    """The task message handed to the Critic sub-agent."""
    g = (goal or "").strip() or "n/a"
    raw_target = (target or "").strip()
    if len(raw_target) > max_chars:
        half = max_chars // 2
        raw_target = (
            raw_target[:half]
            + f"\n\n[... Content truncated ({len(raw_target)} chars total) ...]\n\n"
            + raw_target[-half:]
        )
    return (
        f"GOAL (context only — do not critique the goal itself): {g}\n\n"
        f"Independently critique the {what} below. You did not write it.\n"
        f"--- BEGIN ---\n{raw_target}\n--- END ---"
    )


_KNOWN_PROVIDERS = ("ollama", "zai", "koboldcpp", "openrouter", "claude", "anthropic", "deepseek", "xkiro", "wormsoft")


def parse_critic_model(spec: str) -> Tuple[str, str]:
    """Parse a critic-model spec into (provider, model).

    'provider:model' splits when the prefix is a known provider — so
    'openrouter:anthropic/claude-sonnet-4-6' → ('openrouter', 'anthropic/...').
    Otherwise provider is '' (use the current one), which keeps Ollama tags like
    'qwen2.5:3b' intact instead of mistaking 'qwen2.5' for a provider.
    """
    spec = (spec or "").strip()
    if ":" in spec:
        head, rest = spec.split(":", 1)
        if head.strip().lower() in _KNOWN_PROVIDERS and rest.strip():
            return head.strip().lower(), rest.strip()
    return "", spec
