"""
graph.py - Definición del grafo principal con orchestration multi-step

ARQUITECTURA:
START → bootstrap → intent_analyzer → plan → [worker1 → route → worker2 → ...] → synthesize → END
                                                        ↓
                                                   human_input (si necesita clarificación)
"""
from typing import Optional

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import AgentState
from src.agent.bootstrap import bootstrap_node
from src.agent.orchestrator import orchestrator_plan_node, orchestrator_route_node, synthesize_node
from src.agent.nodes.human_input import human_input_node
from src.agent.nodes.verify_info import verify_info_node
from src.agent.nodes.intent_analyzer import intent_analyzer_node
from src.agent.workers.chat_node import chat_node
from src.agent.workers.research_node import research_node
from src.agent.workers.tutor_node import tutor_node
from src.agent.workers.troubleshooter_node import troubleshooter_node
from src.agent.workers.summarizer_node import summarizer_node


def route_after_bootstrap(state: AgentState) -> str:
    """Decide si ir a verificación o plan"""
    import os
    if os.getenv("REQUIRE_VERIFICATION", "false").lower() == "true" and not state.get("customer_id"):
        return "verify_info"
    return "plan"


def route_after_verify(state: AgentState) -> str:
    if state.get("customer_id"):
        return "plan"
    elif state.get("needs_human_input"):
        return "human_input"
    return "plan"


def route_from_plan(state: AgentState) -> str:
    next_node = state.get("next", "chat")
    valid_workers = {"chat", "research", "tutor", "troubleshooting", "summarizer"}
    if next_node in valid_workers:
        return next_node
    elif next_node in ("END", "__end__", "end"):
        return "END"
    return "chat"


def route_from_orchestrator(state: AgentState) -> str:
    if state.get("needs_human_input"):
        return "human_input"
    
    next_node = state.get("next", "END")
    valid_destinations = {
        "chat": "chat", "research": "research", "tutor": "tutor", "troubleshooting": "troubleshooting",
        "summarizer": "summarizer", "synthesize": "synthesize", "human_input": "human_input", "END": "END",
    }
    return valid_destinations.get(next_node, "END")


def route_after_human_input(state: AgentState) -> str:
    """Después de human_input, continuar con el flujo."""
    # Si hay verificación pendiente
    if not state.get("customer_id") and state.get("clarification_questions"):
        return "verify_info"
    
    # Si hay plan, continuar con route
    if state.get("orchestration_plan"):
        return "route"
    
    return "plan"


def create_graph(enable_verification: bool = False, checkpointer: Optional[MemorySaver] = None) -> StateGraph:
    """Crea el grafo principal con orchestration multi-step."""
    workflow = StateGraph(AgentState)
    
    # Añadir nodos
    workflow.add_node("bootstrap", bootstrap_node)
    workflow.add_node("intent_analyzer", intent_analyzer_node)  # NUEVO: Análisis de intención
    if enable_verification:
        workflow.add_node("verify_info", verify_info_node)
    workflow.add_node("plan", orchestrator_plan_node)
    workflow.add_node("route", orchestrator_route_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("human_input", human_input_node)
    workflow.add_node("chat", chat_node)
    workflow.add_node("research", research_node)
    workflow.add_node("tutor", tutor_node)
    workflow.add_node("troubleshooting", troubleshooter_node)
    workflow.add_node("summarizer", summarizer_node)
    
    # Entry point
    workflow.set_entry_point("bootstrap")
    
    # Edges - NUEVO FLUJO: bootstrap → intent_analyzer → plan
    if enable_verification:
        workflow.add_conditional_edges("bootstrap", route_after_bootstrap, {"verify_info": "verify_info", "plan": "intent_analyzer"})
        workflow.add_conditional_edges("verify_info", route_after_verify, {"plan": "intent_analyzer", "human_input": "human_input"})
    else:
        workflow.add_edge("bootstrap", "intent_analyzer")
    
    # Intent analyzer siempre va a plan
    workflow.add_edge("intent_analyzer", "plan")
    
    workflow.add_conditional_edges("plan", route_from_plan, {
        "chat": "chat", "research": "research", "tutor": "tutor", "troubleshooting": "troubleshooting", "summarizer": "summarizer", "END": END
    })
    
    # Workers → route
    workflow.add_edge("chat", "route")
    workflow.add_edge("research", "route")
    workflow.add_edge("tutor", "route")
    workflow.add_edge("troubleshooting", "route")
    workflow.add_edge("summarizer", "route")
    
    # route → siguiente
    workflow.add_conditional_edges("route", route_from_orchestrator, {
        "chat": "chat", "research": "research", "tutor": "tutor", "troubleshooting": "troubleshooting",
        "summarizer": "summarizer", "synthesize": "synthesize", "human_input": "human_input", "END": END
    })
    
    workflow.add_edge("synthesize", END)
    
    if enable_verification:
        workflow.add_conditional_edges("human_input", route_after_human_input, {"verify_info": "verify_info", "route": "route", "plan": "plan"})
    else:
        workflow.add_conditional_edges("human_input", route_after_human_input, {"route": "route", "plan": "plan"})
    
    return workflow.compile(checkpointer=checkpointer) if checkpointer else workflow.compile()


# Grafo por defecto
_checkpointer = MemorySaver()
graph = create_graph(enable_verification=False, checkpointer=_checkpointer)
supervisor_agent = graph  # Alias para langgraph.json


def create_graph_with_verification() -> StateGraph:
    """Crea grafo con verificación de usuario"""
    return create_graph(enable_verification=True, checkpointer=MemorySaver())


def get_graph_structure() -> dict:
    """Retorna estructura del grafo para debugging"""
    return {
        "nodes": list(graph.nodes.keys()),
        "entry_point": "bootstrap",
        "analysis": ["intent_analyzer"],
        "workers": ["chat", "research", "tutor", "troubleshooting", "summarizer"],
        "orchestration": ["plan", "route", "synthesize"],
    }
