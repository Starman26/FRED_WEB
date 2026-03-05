"""
format_helpers.py - Helpers para formatear datos complejos a texto legible

Este módulo contiene funciones para convertir estructuras de datos complejas
a texto descriptivo para usar en prompts del LLM.
"""
from typing import Dict, Any


def format_user_profile(profile: Dict[str, Any]) -> str:
    """
    Formatea información relevante del perfil del usuario.

    Args:
        profile: Dict con los datos del perfil

    Returns:
        String con información del perfil formateada
    """
    if not profile:
        return "Usuario sin perfil configurado"

    parts = []

    # Nombre
    if full_name := profile.get("full_name"):
        parts.append(f"Nombre: {full_name}")

    # Carrera
    if career := profile.get("career"):
        parts.append(f"Carrera: {career}")

    # Semestre
    if semester := profile.get("semester"):
        parts.append(f"Semestre: {semester}")

    # Skills
    if skills := profile.get("skills"):
        if isinstance(skills, list) and skills:
            parts.append(f"Habilidades: {', '.join(skills)}")

    # Goals
    if goals := profile.get("goals"):
        if isinstance(goals, list) and goals:
            parts.append(f"Objetivos: {', '.join(goals)}")

    # Interests
    if interests := profile.get("interests"):
        if isinstance(interests, list) and interests:
            parts.append(f"Intereses: {', '.join(interests)}")

    return "\n".join(parts) if parts else "Usuario sin perfil configurado"
