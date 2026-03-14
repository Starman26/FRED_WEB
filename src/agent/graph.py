"""
graph.py - Main orchestration graph.

START -> bootstrap -> planner -> [worker -> route -> ...] -> synthesize -> END
"""
import logging
from enum import StrEnum

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

_log = logging.getLogger("graph")

# Practice worker: fall back to placeholder if import fails
_practice_available = False
try:
    from src.agent.nodes.practice_worker import practice_worker_node
    _practice_available = True
except Exception as _exc:
    _log.warning("practice_worker_node import failed (%s), using placeholder", _exc)

if not _practice_available:
    from langchain_core.messages import AIMessage as _AIMsg

    def practice_worker_node(state):  # type: ignore[misc]  # noqa: D103
        return {
            "messages": [_AIMsg(content="Practice mode not yet implemented.")],
            "worker_outputs": [],
        }


class Node(StrEnum):
    """All graph node names. Single source of truth."""
    BOOTSTRAP = "bootstrap"
    PLANNER = "planner"
    ROUTE = "route"
    SYNTHESIZE = "synthesize"
    HUMAN_INPUT = "human_input"
    VERIFY_INFO = "verify_info"
    CHAT = "chat"
    RESEARCH = "research"
    TUTOR = "tutor"
    TROUBLESHOOTING = "troubleshooting"
    SUMMARIZER = "summarizer"
    ROBOT_OPERATOR = "robot_operator"
    ANALYSIS = "analysis"
    PRACTICE = "practice"


# Worker registry: add new workers here + import above
WORKER_REGISTRY = {
    Node.CHAT: chat_node,
    Node.RESEARCH: research_node,
    Node.TUTOR: tutor_node,
    Node.TROUBLESHOOTING: troubleshooter_node,
    Node.SUMMARIZER: summarizer_node,
    Node.ROBOT_OPERATOR: robot_operator_node,
    Node.ANALYSIS: analysis_node,
    Node.PRACTICE: practice_worker_node,
}

CORE_WORKERS = {Node.CHAT, Node.RESEARCH, Node.TUTOR, Node.TROUBLESHOOTING,
                Node.SUMMARIZER, Node.ROBOT_OPERATOR, Node.ANALYSIS}
SPECIAL_WORKERS = {Node.PRACTICE}
VALID_WORKERS = CORE_WORKERS | SPECIAL_WORKERS

ALL_NODES = {n.value for n in Node}


def _normalize_destination(next_node: str, allowed: set[str], fallback: str) -> str:
    """Validate and normalize a graph destination."""
    if next_node in allowed:
        return next_node
    if next_node in ("END", "__end__", "end"):
        return "END" if "END" in allowed else fallback
    _log.warning(f"Unknown destination '{next_node}', allowed={allowed}, falling back to '{fallback}'")
    return fallback


def route_after_bootstrap(state: AgentState) -> str:
    import os
    if os.getenv("REQUIRE_VERIFICATION", "false").lower() == "true" and not state.get("customer_id"):
        return Node.VERIFY_INFO
    return Node.PLANNER


def route_after_verify(state: AgentState) -> str:
    """Route based on verification outcome."""
    status = state.get("verification_status", "unknown")

    if status == "verified" or state.get("customer_id"):
        return Node.PLANNER
    if status == "needs_human_input" or state.get("needs_human_input"):
        return Node.HUMAN_INPUT
    # failed or unknown: proceed with whatever we have
    return Node.PLANNER


def route_from_planner(state: AgentState) -> str:
    """Routes from planner to first worker or END."""
    next_node = state.get("next", Node.CHAT)
    return _normalize_destination(next_node, VALID_WORKERS | {"END"}, Node.CHAT)


def route_from_orchestrator(state: AgentState) -> str:
    """Routes from adaptive_router to next worker, synthesize, human_input, or END."""
    if state.get("needs_human_input"):
        return Node.HUMAN_INPUT

    next_node = state.get("next", "END")
    allowed = VALID_WORKERS | {Node.SYNTHESIZE, Node.HUMAN_INPUT, "END"}
    return _normalize_destination(next_node, allowed, "END")


def route_after_human_input(state: AgentState) -> str:
    """Route after human_input based on why the input was requested."""
    reason = state.get("human_input_reason", "")

    if reason == "verification":
        return Node.VERIFY_INFO
    if reason == "worker_clarification" and state.get("orchestration_plan"):
        return Node.ROUTE

    # Legacy fallback for flows without human_input_reason
    if not state.get("customer_id") and state.get("clarification_questions"):
        return Node.VERIFY_INFO
    if state.get("orchestration_plan"):
        return Node.ROUTE
    return Node.PLANNER


def _register_nodes(workflow: StateGraph, enable_verification: bool):
    """Register all nodes in the graph."""
    workflow.add_node(Node.BOOTSTRAP, bootstrap_node)
    workflow.add_node(Node.PLANNER, planner_node)
    workflow.add_node(Node.ROUTE, adaptive_router_node)
    workflow.add_node(Node.SYNTHESIZE, synthesize_node)
    workflow.add_node(Node.HUMAN_INPUT, human_input_node)

    if enable_verification:
        workflow.add_node(Node.VERIFY_INFO, verify_info_node)

    for name, node_fn in WORKER_REGISTRY.items():
        workflow.add_node(name, node_fn)


def _register_edges(workflow: StateGraph, enable_verification: bool):
    """Register all edges in the graph."""
    workflow.set_entry_point(Node.BOOTSTRAP)

    if enable_verification:
        workflow.add_conditional_edges(Node.BOOTSTRAP, route_after_bootstrap, {
            Node.VERIFY_INFO: Node.VERIFY_INFO,
            Node.PLANNER: Node.PLANNER,
        })
        workflow.add_conditional_edges(Node.VERIFY_INFO, route_after_verify, {
            Node.PLANNER: Node.PLANNER,
            Node.HUMAN_INPUT: Node.HUMAN_INPUT,
        })
    else:
        workflow.add_edge(Node.BOOTSTRAP, Node.PLANNER)

    planner_edges = {w: w for w in VALID_WORKERS}
    planner_edges["END"] = END
    workflow.add_conditional_edges(Node.PLANNER, route_from_planner, planner_edges)

    for worker in VALID_WORKERS:
        workflow.add_edge(worker, Node.ROUTE)

    route_edges = {w: w for w in VALID_WORKERS}
    route_edges.update({
        Node.SYNTHESIZE: Node.SYNTHESIZE,
        Node.HUMAN_INPUT: Node.HUMAN_INPUT,
        "END": END,
    })
    workflow.add_conditional_edges(Node.ROUTE, route_from_orchestrator, route_edges)

    workflow.add_edge(Node.SYNTHESIZE, END)

    hi_edges = {
        Node.ROUTE: Node.ROUTE,
        Node.PLANNER: Node.PLANNER,
    }
    if enable_verification:
        hi_edges[Node.VERIFY_INFO] = Node.VERIFY_INFO
    workflow.add_conditional_edges(Node.HUMAN_INPUT, route_after_human_input, hi_edges)


def create_graph(enable_verification: bool = False) -> StateGraph:
    """Create the main orchestration graph (no checkpointer)."""
    workflow = StateGraph(AgentState)
    _register_nodes(workflow, enable_verification)
    _register_edges(workflow, enable_verification)
    return workflow.compile()


def create_graph_with_checkpointer(checkpointer=None, enable_verification: bool = False):
    """Create the main orchestration graph with an external checkpointer."""
    workflow = StateGraph(AgentState)
    _register_nodes(workflow, enable_verification)
    _register_edges(workflow, enable_verification)
    return workflow.compile(checkpointer=checkpointer)


graph = create_graph(enable_verification=False)
supervisor_agent = graph  # Alias for langgraph.json


def create_graph_with_verification() -> StateGraph:
    return create_graph(enable_verification=True)


def get_graph_structure() -> dict:
    """Return known graph structure from constants (not from compiled object)."""
    return {
        "nodes": sorted(ALL_NODES),
        "entry_point": Node.BOOTSTRAP,
        "planning": [Node.PLANNER],
        "workers": sorted(VALID_WORKERS),
        "core_workers": sorted(CORE_WORKERS),
        "special_workers": sorted(SPECIAL_WORKERS),
        "orchestration": [Node.PLANNER, Node.ROUTE, Node.SYNTHESIZE],
    }