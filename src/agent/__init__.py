"""
 Multi-Agent System con Orchestration Multi-Step

Este módulo implementa un sistema multi-agente con:
- Orchestration multi-step (plan → execute → route → synthesize)
- Human-in-the-loop para solicitar clarificación
- Workers especializados (research, tutor, troubleshooting, summarizer)
- Contrato JSON universal (WorkerOutput)
"""
from src.agent.graph import graph, supervisor_agent, create_graph, create_graph_with_verification
from src.agent.state import AgentState, STATE_DEFAULTS

__all__ = [
    "graph",
    "supervisor_agent",
    "create_graph",
    "create_graph_with_verification",
    "AgentState",
    "STATE_DEFAULTS",
]
