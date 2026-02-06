"""
format_helpers.py - Helpers para formatear datos complejos a texto legible

Este módulo contiene funciones para convertir estructuras de datos complejas
(como JSONB) a texto descriptivo para usar en prompts del LLM.
"""
import json
from typing import Dict, Any, Union


def format_learning_style(learning_style: Union[Dict[str, Any], str, None]) -> str:
    """
    Convierte el JSONB de learning_style del profile a texto descriptivo.

    Args:
        learning_style: Dict con las preferencias de aprendizaje del usuario,
                       o string JSON que se parseará automáticamente

    Returns:
        String con descripción del estilo de aprendizaje

    Ejemplos:
        >>> format_learning_style({"type": "visual", "pace": "fast"})
        "Visual (ritmo rápido)"

        >>> format_learning_style('{"type": "visual"}')
        "Visual - Aprende mejor con diagramas, esquemas y ejemplos gráficos"

        >>> format_learning_style({})
        "General (adaptativo)"
    """
    # Si learning_style es un string (JSON de Supabase), parsearlo
    if isinstance(learning_style, str):
        try:
            learning_style = json.loads(learning_style)
        except (json.JSONDecodeError, ValueError, TypeError):
            return "General (adaptativo)"

    # Validar que sea un dict válido
    if not learning_style or not isinstance(learning_style, dict):
        return "General (adaptativo)"

    parts = []

    # Tipo principal de aprendizaje
    learning_type = learning_style.get("type", "").lower()
    if learning_type:
        type_descriptions = {
            "visual": "Visual - Aprende mejor con diagramas, esquemas y ejemplos gráficos",
            "auditory": "Auditivo - Prefiere explicaciones paso a paso y narrativas",
            "auditivo": "Auditivo - Prefiere explicaciones paso a paso y narrativas",
            "kinesthetic": "Kinestésico - Aprende mejor con ejercicios prácticos y experimentación",
            "kinestésico": "Kinestésico - Aprende mejor con ejercicios prácticos y experimentación",
            "reading": "Lectura/Escritura - Prefiere documentación detallada y referencias",
            "lectura": "Lectura/Escritura - Prefiere documentación detallada y referencias",
            "mixed": "Mixto - Combina varios estilos de aprendizaje",
            "mixto": "Mixto - Combina varios estilos de aprendizaje",
        }
        parts.append(type_descriptions.get(learning_type, learning_type.capitalize()))

    # Ritmo de aprendizaje
    pace = learning_style.get("pace", "").lower()
    if pace:
        pace_descriptions = {
            "slow": "ritmo pausado con más explicaciones",
            "lento": "ritmo pausado con más explicaciones",
            "medium": "ritmo moderado",
            "moderado": "ritmo moderado",
            "fast": "ritmo rápido, directo al grano",
            "rápido": "ritmo rápido, directo al grano",
        }
        if pace in pace_descriptions:
            parts.append(pace_descriptions[pace])

    # Nivel de profundidad
    depth = learning_style.get("depth", "").lower()
    if depth:
        depth_descriptions = {
            "basic": "nivel básico",
            "básico": "nivel básico",
            "intermediate": "nivel intermedio",
            "intermedio": "nivel intermedio",
            "advanced": "nivel avanzado con detalles técnicos",
            "avanzado": "nivel avanzado con detalles técnicos",
        }
        if depth in depth_descriptions:
            parts.append(depth_descriptions[depth])

    # Preferencia de ejemplos
    examples = learning_style.get("examples", "").lower()
    if examples:
        example_descriptions = {
            "theoretical": "prefiere explicaciones teóricas",
            "teórico": "prefiere explicaciones teóricas",
            "practical": "prefiere ejemplos prácticos del laboratorio",
            "práctico": "prefiere ejemplos prácticos del laboratorio",
            "both": "combina teoría y práctica",
            "ambos": "combina teoría y práctica",
        }
        if examples in example_descriptions:
            parts.append(example_descriptions[examples])

    # Si hay otros campos personalizados
    if "notes" in learning_style and learning_style["notes"]:
        parts.append(f"Notas: {learning_style['notes']}")

    if not parts:
        return "General (adaptativo)"

    # Unir todas las partes
    result = parts[0]
    if len(parts) > 1:
        result += " (" + ", ".join(parts[1:]) + ")"

    return result


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
