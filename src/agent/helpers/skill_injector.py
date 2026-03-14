"""
skill_injector.py — Injects equipment spec + skills into worker prompts.

Pattern: the agent reads loaded instructions BEFORE responding,
similar to how Claude Code reads SKILL.md files before acting.
"""
from typing import Optional


def build_equipment_context_block(
    state: dict,
    categories: Optional[list[str]] = None,
) -> str:
    """
    Build a markdown block with equipment spec + filtered skills for prompt injection.

    Args:
        state: AgentState dict.
        categories: Filter skills by category (e.g. ["troubleshoot"]).
                    None = include all loaded skills.

    Returns:
        Formatted markdown string ready to prepend to a prompt. Empty if no context.
    """
    parts = []

    # 1. Equipment spec (always included if present)
    spec = state.get("equipment_spec", "")
    if spec.strip():
        parts.append(f"## Equipment Context\n\n{spec}")

    # 2. Skills filtered by category
    skills = state.get("loaded_skills", [])
    skills_meta = state.get("loaded_skills_meta", [])

    if categories and skills_meta:
        skills_to_inject = [
            content
            for content, meta in zip(skills, skills_meta)
            if meta.get("category") in categories
        ]
    else:
        skills_to_inject = skills

    if skills_to_inject:
        skills_block = "\n\n---\n\n".join(skills_to_inject)
        parts.append(f"## Loaded Skills\n\n{skills_block}")

    if not parts:
        return ""

    return (
        "\n\n═══════════════════════════════════════\n"
        "EQUIPMENT CONTEXT — Read before responding\n"
        "═══════════════════════════════════════\n\n"
        + "\n\n".join(parts)
        + "\n\n═══════════════════════════════════════\n"
    )
