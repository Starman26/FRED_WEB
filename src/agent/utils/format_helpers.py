"""
format_helpers.py - Format complex data structures into readable text for LLM prompts.
"""
from typing import Dict, Any


def format_user_profile(profile: Dict[str, Any]) -> str:
    """Format user profile data into readable text."""
    if not profile:
        return "Usuario sin perfil configurado"

    parts = []

    if full_name := profile.get("full_name"):
        parts.append(f"Nombre: {full_name}")

    if career := profile.get("career"):
        parts.append(f"Carrera: {career}")

    if semester := profile.get("semester"):
        parts.append(f"Semestre: {semester}")

    if skills := profile.get("skills"):
        if isinstance(skills, list) and skills:
            parts.append(f"Habilidades: {', '.join(skills)}")

    if goals := profile.get("goals"):
        if isinstance(goals, list) and goals:
            parts.append(f"Objetivos: {', '.join(goals)}")

    if interests := profile.get("interests"):
        if isinstance(interests, list) and interests:
            parts.append(f"Intereses: {', '.join(interests)}")

    return "\n".join(parts) if parts else "Usuario sin perfil configurado"
