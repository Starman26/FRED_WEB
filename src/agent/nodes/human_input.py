"""
human_input.py

Human-in-the-Loop node. Supports v2 payloads (dict with type="clarification")
and legacy format (list of strings/dicts).
"""
from typing import Dict, Any, List

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from src.agent.state import AgentState
from src.agent.contracts.question_schema_v2 import AnswerSet
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_read, event_report


def human_input_node(state: AgentState) -> Dict[str, Any]:
    """Interrupts workflow to collect user input, then parses the response."""
    questions = state.get("clarification_questions", [])
    pending_context = state.get("pending_context", {}) or {}

    logger.node_start("human_input", {
        "payload_type": type(questions).__name__,
    })

    events = [event_read("human_input", "Solicitando información al usuario...")]

    if isinstance(questions, dict) and questions.get("type") == "clarification":
        # v2 payload — send as-is to the frontend
        interrupt_payload = questions
        display_title = questions.get("title", "Información necesaria")
    elif isinstance(questions, list) and questions:
        formatted = _format_legacy_questions(questions)
        interrupt_payload = {
            "type": "clarification",
            "worker": pending_context.get("current_worker", "unknown"),
            "title": "Información necesaria",
            "context": "",
            "questions": formatted,
            "wizard_mode": False,
        }
        display_title = "Información necesaria"
    else:
        interrupt_payload = {
            "type": "clarification",
            "worker": pending_context.get("current_worker", "unknown"),
            "title": "Más información necesaria",
            "context": "",
            "questions": [{
                "id": "generic",
                "question": "¿Podrías darme más contexto sobre tu solicitud?",
                "type": "text",
                "required": True,
            }],
            "wizard_mode": False,
        }
        display_title = "Más información necesaria"

    logger.info("human_input", f"Interrupt: {display_title} ({len(interrupt_payload.get('questions', []))} questions)")

    user_response = interrupt(interrupt_payload)

    logger.info("human_input", f"Usuario respondió: {str(user_response)[:100]}...")
    print(f"[HITL DEBUG] user_response type={type(user_response).__name__}, value={str(user_response)[:300]}", flush=True)
    events.append(event_report("human_input", "Recibí tu respuesta, continuando..."))

    answer_set = AnswerSet.from_resume(user_response)

    updated_context = pending_context.copy()
    updated_context["answers"] = answer_set.answers
    updated_context["user_clarification"] = answer_set.to_user_clarification()
    updated_context["wizard_completed"] = answer_set.completed
    updated_context["_hitl_consumed"] = False  # reset so router picks up this new round

    next_destination = "route"
    logger.info("human_input", f"Contexto actualizado, continuando a: {next_destination}")

    display_prompt = f"## {display_title}\n\n" + "\n".join(
        f"- {q.get('question', str(q))}"
        for q in interrupt_payload.get("questions", [])[:5]
    )

    return {
        "messages": [
            AIMessage(content=display_prompt),
            HumanMessage(content=answer_set.to_user_clarification() or str(user_response)),
        ],
        "needs_human_input": False,
        "clarification_questions": [],
        "pending_context": updated_context,
        "next": next_destination,
        "events": events,
    }


def _format_legacy_questions(questions: List[Any]) -> List[dict]:
    """Convert legacy question formats to v2 question dicts."""
    formatted = []
    for i, q in enumerate(questions[:5], 1):
        if isinstance(q, dict):
            formatted.append({
                "id": q.get("id", f"q{i}"),
                "question": q.get("question", str(q)),
                "type": q.get("type", "text"),
                "required": q.get("required", True),
                "options": q.get("options", []),
            })
        elif isinstance(q, str):
            formatted.append({
                "id": f"q{i}",
                "question": q,
                "type": "text",
                "required": True,
            })
        else:
            formatted.append({
                "id": getattr(q, "id", f"q{i}"),
                "question": getattr(q, "question", str(q)),
                "type": getattr(q, "type", "text"),
                "required": getattr(q, "required", True),
            })
    return formatted
