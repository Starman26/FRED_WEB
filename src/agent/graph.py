"""
graph.py - Definición del grafo principal con orchestration multi-step

ARQUITECTURA:
START → bootstrap → planner → [worker₁ → route → worker₂ → ...] → synthesize → END
                                        ↓
                                   human_input (si necesita clarificación)

El planner fusiona intent analysis + plan generation en un solo nodo.
El adaptive_router evalúa outputs de workers y adapta el plan dinámicamente.
"""
from langgraph.graph import StateGraph, END, START

from src.agent.state import AgentState
from src.agent.bootstrap import bootstrap_node
from src.agent.nodes.planner import planner_node
from src.agent.orchestrator import adaptive_router_node, synthesize_node
from src.agent.nodes.human_input import human_input_node
from src.agent.nodes.verify_info import verify_info_node
from src.agent.workers.chat_node import chat_node
from src.agent.workers.research_node import research_node
from src.agent.workers.tutor_node import tutor_node
from src.agent.workers.troubleshooter_node import troubleshooter_node
from src.agent.workers.summarizer_node import summarizer_node
from src.agent.workers.robot_operator_node import robot_operator_node
from src.agent.workers.analysis_node import analysis_node


# ============================================
# VALID WORKERS (centralizado, una sola fuente de verdad)
# ============================================
VALID_WORKERS = {
    "chat", "research", "tutor", "troubleshooting",
    "summarizer", "robot_operator", "analysis"
}


def route_after_bootstrap(state: AgentState) -> str:
    import os
    if os.getenv("REQUIRE_VERIFICATION", "false").lower() == "true" and not state.get("customer_id"):
        return "verify_info"
    return "planner"


def route_after_verify(state: AgentState) -> str:
    if state.get("customer_id"):
        return "planner"
    elif state.get("needs_human_input"):
        return "human_input"
    return "planner"


def route_from_planner(state: AgentState) -> str:
    """Routes from planner to first worker or END."""
    next_node = state.get("next", "chat")
    if next_node in VALID_WORKERS:
        return next_node
    elif next_node in ("END", "__end__", "end"):
        return "END"
    import logging
    logging.getLogger("graph").warning(
        f"route_from_planner: '{next_node}' NOT in VALID_WORKERS {VALID_WORKERS}, falling back to 'chat'"
    )
    return "chat"


def route_from_orchestrator(state: AgentState) -> str:
    """Routes from adaptive_router to next worker, synthesize, human_input, or END."""
    if state.get("needs_human_input"):
        return "human_input"

    next_node = state.get("next", "END")

    valid_destinations = {w: w for w in VALID_WORKERS}
    valid_destinations.update({
        "synthesize": "synthesize",
        "human_input": "human_input",
        "END": "END",
    })

    return valid_destinations.get(next_node, "END")


def route_after_human_input(state: AgentState) -> str:
    if not state.get("customer_id") and state.get("clarification_questions"):
        return "verify_info"
    if state.get("orchestration_plan"):
        return "route"
    return "planner"


def create_graph(enable_verification: bool = False) -> StateGraph:
    """Crea el grafo principal con orchestration multi-step."""
    workflow = StateGraph(AgentState)

    # ==========================================
    # NODOS
    # ==========================================
    workflow.add_node("bootstrap", bootstrap_node)
    workflow.add_node("planner", planner_node)
    if enable_verification:
        workflow.add_node("verify_info", verify_info_node)
    workflow.add_node("route", adaptive_router_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("human_input", human_input_node)

    # Workers
    workflow.add_node("chat", chat_node)
    workflow.add_node("research", research_node)
    workflow.add_node("tutor", tutor_node)
    workflow.add_node("troubleshooting", troubleshooter_node)
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("robot_operator", robot_operator_node)
    workflow.add_node("analysis", analysis_node)

    # ==========================================
    # EDGES
    # ==========================================
    workflow.set_entry_point("bootstrap")

    # Bootstrap → planner (o verify_info)
    if enable_verification:
        workflow.add_conditional_edges("bootstrap", route_after_bootstrap,
            {"verify_info": "verify_info", "planner": "planner"})
        workflow.add_conditional_edges("verify_info", route_after_verify,
            {"planner": "planner", "human_input": "human_input"})
    else:
        workflow.add_edge("bootstrap", "planner")

    # planner → worker (o END)
    planner_edges = {w: w for w in VALID_WORKERS}
    planner_edges["END"] = END
    workflow.add_conditional_edges("planner", route_from_planner, planner_edges)

    # Todos los workers → route (adaptive_router)
    for worker in VALID_WORKERS:
        workflow.add_edge(worker, "route")

    # route → siguiente destino
    route_edges = {w: w for w in VALID_WORKERS}
    route_edges.update({
        "synthesize": "synthesize",
        "human_input": "human_input",
        "END": END,
    })
    workflow.add_conditional_edges("route", route_from_orchestrator, route_edges)

    # synthesize → END
    workflow.add_edge("synthesize", END)

    # human_input → route/planner (o verify_info)
    hi_edges = {"route": "route", "planner": "planner"}
    if enable_verification:
        hi_edges["verify_info"] = "verify_info"
    workflow.add_conditional_edges("human_input", route_after_human_input, hi_edges)

    return workflow.compile()


# Grafo por defecto
graph = create_graph(enable_verification=False)
supervisor_agent = graph  # Alias para langgraph.json


def create_graph_with_verification() -> StateGraph:
    return create_graph(enable_verification=True)


def get_graph_structure() -> dict:
    return {
        "nodes": list(graph.nodes.keys()),
        "entry_point": "bootstrap",
        "planning": ["planner"],
        "workers": sorted(VALID_WORKERS),
        "orchestration": ["planner", "route", "synthesize"],
    }
