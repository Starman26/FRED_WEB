"""
practices - Sistema de Prácticas Guiadas de Laboratorio

Este módulo proporciona:
- Schema para definir prácticas estructuradas (YAML/JSON)
- Gestión de sesiones de usuario
- Tracking de progreso y evaluación
- Generación de resúmenes de aprendizaje

Uso básico:
    from src.agent.practices import get_practice_manager
    
    manager = get_practice_manager()
    
    # Ver prácticas disponibles
    practices = manager.get_available_practices()
    
    # Iniciar una práctica
    session = manager.get_or_create_session(user_id, "abb_pick_and_place")
    
    # Obtener contexto para el tutor
    context = manager.get_step_context_for_tutor(session)
    
    # Completar paso
    next_step = manager.complete_step(session, step_id, response, score)
    
    # Finalizar práctica
    summary = manager.complete_practice(session)
"""

from .practice_schema import (
    Practice,
    PracticeSession,
    PracticeStep,
    PracticeSection,
    ComprehensionQuestion,
    StepType,
    QuestionType,
    load_practice_from_file,
)

from .practice_manager import (
    PracticeManager,
    get_practice_manager,
    reset_practice_manager,
)

__all__ = [
    # Schema
    "Practice",
    "PracticeSession",
    "PracticeStep",
    "PracticeSection",
    "ComprehensionQuestion",
    "StepType",
    "QuestionType",
    "load_practice_from_file",
    # Manager
    "PracticeManager",
    "get_practice_manager",
    "reset_practice_manager",
]
