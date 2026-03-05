"""
human_input.py - Nodo de Human-in-the-Loop (v2)

Soporta payloads v2 (dict con type="clarification") y legacy (lista de strings/dicts).

Para reanudar:
    from langgraph.types import Command
    # v2 format:
    result = graph.invoke(Command(resume={"answers": {"q1": "val"}, "completed": True}), config)
    # Legacy format (still supported):
    result = graph.invoke(Command(resume=[{"question": "...", "answer": "..."}]), config)
"""
from typing import Dict, Any, List

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from src.agent.state import AgentState
from src.agent.contracts.question_schema_v2 import AnswerSet
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_read, event_report


def human_input_node(state: AgentState) -> Dict[str, Any]:
    """
    Nodo Human-in-the-Loop que interrumpe el workflow para pedir input al usuario.

    - Si clarification_questions es un dict v2 (type="clarification"),
      lo envía directo como interrupt payload.
    - Si es una lista legacy, formatea como string y envía.
    - Parsea la respuesta del usuario con AnswerSet.from_resume().
    """
    questions = state.get("clarification_questions", [])
    pending_context = state.get("pending_context", {}) or {}

    logger.node_start("human_input", {
        "payload_type": type(questions).__name__,
    })

    events = [event_read("human_input", "Solicitando información al usuario...")]

    # ==========================================
    # BUILD INTERRUPT PAYLOAD
    # ==========================================
    if isinstance(questions, dict) and questions.get("type") == "clarification":
        # v2 payload — send as-is to the frontend
        interrupt_payload = questions
        display_title = questions.get("title", "Información necesaria")
    elif isinstance(questions, list) and questions:
        # Legacy: list of dicts or strings — wrap in a v2-like structure
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
        # No questions — generic prompt
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

    # ==========================================
    # INTERRUPT: Pausa el grafo y espera input
    # ==========================================
    user_response = interrupt(interrupt_payload)

    # ==========================================
    # REANUDACIÓN: Se ejecuta cuando el usuario responde
    # ==========================================
    logger.info("human_input", f"Usuario respondió: {str(user_response)[:100]}...")
    print(f"[HITL DEBUG] user_response type={type(user_response).__name__}, value={str(user_response)[:300]}", flush=True)
    events.append(event_report("human_input", "Recibí tu respuesta, continuando..."))

    # Parse response with AnswerSet
    answer_set = AnswerSet.from_resume(user_response)

    # Build updated context — backward compat
    updated_context = pending_context.copy()
    updated_context["answers"] = answer_set.answers
    updated_context["user_clarification"] = answer_set.to_user_clarification()
    updated_context["wizard_completed"] = answer_set.completed
    updated_context["_hitl_consumed"] = False  # Reset so router picks up this new HITL round

    # Determine where to go next
    next_destination = "route"
    logger.info("human_input", f"Contexto actualizado, continuando a: {next_destination}")

    # Build display text for conversation history
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
        "clarification_questions": [],  # Limpiar preguntas
        "pending_context": updated_context,
        "next": next_destination,
        "events": events,
    }


def _format_legacy_questions(questions: List[Any]) -> List[dict]:
    """Convert legacy question formats to v2 question dicts."""
    formatted = []
    for i, q in enumerate(questions[:5], 1):
        if isinstance(q, dict):
            # Already a dict — normalize it
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
            # Pydantic model or other — try to extract
            formatted.append({
                "id": getattr(q, "id", f"q{i}"),
                "question": getattr(q, "question", str(q)),
                "type": getattr(q, "type", "text"),
                "required": getattr(q, "required", True),
            })
    return formatted
