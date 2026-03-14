"""
state.py
Shared agent state with multi-step orchestration support.
"""
from typing import TypedDict, Annotated, Sequence, Any, List, Dict, Optional
from langchain_core.messages import BaseMessage
import operator


# Sentinel value to clear worker_outputs (use instead of [])
RESET_WORKER_OUTPUTS = "__RESET_WORKER_OUTPUTS__"


def merge_worker_outputs(left: List[Dict], right: Any) -> List[Dict]:
    """Reducer for worker_outputs. Deduplicates by task_id, supports sentinel reset."""
    if right == RESET_WORKER_OUTPUTS:
        return []

    if right is None:
        return left or []
    if isinstance(right, list) and len(right) == 0:
        return left or []

    if not left:
        return right if isinstance(right, list) else []

    if not isinstance(right, list):
        return left

    existing_ids = {o.get("task_id") for o in left if o.get("task_id")}
    merged = list(left)

    for output in right:
        task_id = output.get("task_id")
        if task_id and task_id not in existing_ids:
            merged.append(output)
            existing_ids.add(task_id)
        elif not task_id:
            merged.append(output)

    return merged


def merge_dicts(left: Dict, right: Dict) -> Dict:
    """Reducer for pending_context. Empty dict resets, None preserves left."""
    # Explicit empty dict means reset
    if right is not None and len(right) == 0:
        return {}

    if right is None:
        return left or {}
    
    if not left:
        return right or {}
    
    return {**left, **right}


class AgentState(TypedDict):
    """Shared state across all graph nodes. Initialized in bootstrap."""

    # Messages & events (accumulated via operator.add)
    messages: Annotated[Sequence[Any], operator.add]
    events: Annotated[List[Dict[str, Any]], operator.add]

    # Flow control
    next: str
    done: bool

    # Multi-step orchestration
    orchestration_plan: List[str]
    current_step: int
    worker_outputs: Annotated[List[Dict[str, Any]], merge_worker_outputs]
    pending_context: Annotated[Dict[str, Any], merge_dicts]
    _route_count: int  # Anti-loop counter for adaptive_router

    # Human-in-the-loop
    needs_human_input: bool
    clarification_questions: List[Any]
    follow_up_suggestions: List[str]
    # Wizard state lives in pending_context["wizard_state"]

    # Planning
    intent_analysis: Dict[str, Any]
    plan_reasoning: str
    planner_method: str  # "fast" (regex) or "llm" (chain-of-thought)

    # Memory
    rolling_summary: str
    window_count: int
    loaded_memory: str

    # Metadata
    task_type: str
    user_name: str
    user_id: Optional[str]
    team: str
    interaction_mode: str  # 'chat', 'code', 'agent', 'voice'
    llm_model: str
    auth_user_id: Optional[str]  # Used for RLS filtering in analysis
    team_id: Optional[str]
    customer_id: Optional[str]
    token_usage: Annotated[int, operator.add]
    image_attachments: List[Dict[str, Any]]  # Kept out of messages to avoid .lower() crashes

    # Practice / Automation mode
    automation_id: Optional[str]
    automation_md_content: str
    automation_step: int
    automation_type: str
    automation_context: str
    practice_status: str  # "in_progress", "completed", "paused"
    practice_chunks: List[Dict[str, Any]]  # Multi-message chunks for SSE streaming
    last_tool_step: int  # Prevents re-execution of same step
    user_profile_md: str
    robot_ids: List[str]
    robot_state: Annotated[Dict[str, Any], merge_dicts]

    # BITL (Bridge-in-the-Loop)
    bridge_report: Optional[dict]
    practice_session_active: bool
    current_practice_step: int
    total_practice_steps: int
    practice_results: list
    practice_expected_steps: list
    target_robot_id: Optional[str]

    # Tool execution & devices
    tool_execution_log: Annotated[List[Dict[str, Any]], operator.add]
    active_devices: Dict[str, Any]

    # Streaming (not serializable, injected per-request)
    _stream_session_id: Optional[str]

    # Loaded Context (Agent Skills System)
    equipment_spec: str          # spec_md from selected equipment profile
    loaded_skills: List[str]     # content_md of applicable skills
    loaded_skills_meta: List[Dict[str, Any]]  # [{slug, title, category}]

    # Legacy worker results
    research_result: Any
    tutor_result: Any
    troubleshooting_result: Any
    summarizer_result: Any


STATE_DEFAULTS: Dict[str, Any] = {
    "next": "supervisor",
    "done": False,

    "orchestration_plan": [],
    "current_step": 0,
    "worker_outputs": [],
    "pending_context": {},
    "_route_count": 0,

    "needs_human_input": False,
    "clarification_questions": [],
    "follow_up_suggestions": [],

    "intent_analysis": {},
    "plan_reasoning": "",
    "planner_method": "",

    "rolling_summary": "",
    "window_count": 0,
    "loaded_memory": "",

    "task_type": "",
    "user_name": "Usuario",
    "user_id": None,
    "team": "",
    "interaction_mode": "chat",
    "llm_model": "",
    "auth_user_id": None,
    "team_id": None,
    "customer_id": None,
    "token_usage": 0,
    "image_attachments": [],

    "automation_id": None,
    "automation_md_content": "",
    "automation_step": 1,
    "automation_type": "",
    "automation_context": "",
    "practice_status": "in_progress",
    "practice_chunks": [],
    "last_tool_step": 0,
    "user_profile_md": "",
    "robot_ids": [],
    "robot_state": {},

    "bridge_report": None,
    "practice_session_active": False,
    "current_practice_step": 0,
    "total_practice_steps": 0,
    "practice_results": [],
    "practice_expected_steps": [],
    "target_robot_id": None,

    "tool_execution_log": [],
    "active_devices": {},

    "_stream_session_id": None,

    "equipment_spec": "",
    "loaded_skills": [],
    "loaded_skills_meta": [],

    "research_result": None,
    "tutor_result": None,
    "troubleshooting_result": None,
    "summarizer_result": None,

    "events": [],
}


def get_state_defaults() -> Dict[str, Any]:
    """Return a copy of the default state values."""
    return STATE_DEFAULTS.copy()


def validate_state(state: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate state structure. Returns (is_valid, error_message)."""
    if "messages" not in state:
        return False, "Campo obligatorio faltante: messages"
    
    if not isinstance(state.get("messages"), (list, tuple)):
        return False, "messages debe ser list o tuple"
    
    return True, None
