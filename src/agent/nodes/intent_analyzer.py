"""
intent_analyzer.py - DEPRECATED

Este módulo ha sido reemplazado por src.agent.nodes.planner que fusiona
análisis de intención + planificación en un solo nodo.

Se mantiene por backward-compatibility. Todos los imports se redirigen a planner.py.
"""
import warnings

warnings.warn(
    "intent_analyzer is deprecated. Use src.agent.nodes.planner instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from planner for backward compatibility
from src.agent.nodes.planner import (
    planner_node as intent_analyzer_node,
    get_intent_analysis,
    is_command_intent,
    get_detected_action,
    get_detected_entities,
    needs_clarification,
)

__all__ = [
    "intent_analyzer_node",
    "get_intent_analysis",
    "is_command_intent",
    "get_detected_action",
    "get_detected_entities",
    "needs_clarification",
]
