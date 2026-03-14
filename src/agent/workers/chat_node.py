"""
chat_node.py - General conversation worker.

Handles greetings, casual questions, programming help, conceptual explanations.
Not a RAG engine, diagnostics system, or deep research worker.
"""
import os
from typing import Dict, Any
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage

from src.agent.state import AgentState
from src.agent.contracts.worker_contract import WorkerOutputBuilder
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_execute, event_report
from src.agent.interaction_modes import get_mode_instructions
from src.agent.prompts.format_rules import MARKDOWN_FORMAT_RULES

try:
    from src.agent.knowledge import get_lab_knowledge_summary
    LAB_KNOWLEDGE_AVAILABLE = True
except ImportError:
    LAB_KNOWLEDGE_AVAILABLE = False


_QUICK_REPLIES = {
    "hola": "Hola {name}, ¿en qué te puedo ayudar?",
    "hello": "Hello {name}, how can I help you?",
    "hi": "Hi {name}, what do you need?",
    "hey": "Hey {name}, what's up?",
    "buenos días": "Buenos días {name}, ¿qué necesitas?",
    "buenos dias": "Buenos días {name}, ¿qué necesitas?",
    "buenas tardes": "Buenas tardes {name}, ¿en qué te ayudo?",
    "buenas noches": "Buenas noches {name}, ¿qué necesitas?",
    "gracias": "De nada {name}. ¿Algo más en lo que pueda ayudar?",
    "thanks": "You're welcome {name}. Anything else?",
    "thank you": "You're welcome {name}. Need anything else?",
    "bye": "See you later {name}!",
    "adiós": "¡Hasta luego {name}!",
    "adios": "¡Hasta luego {name}!",
}


def _try_quick_reply(message: str, user_name: str) -> str | None:
    """Return a quick reply if the message is a simple greeting/thanks, else None."""
    normalized = message.strip().lower().rstrip("!.,?¡¿ ")
    template = _QUICK_REPLIES.get(normalized)
    if template:
        return template.format(name=user_name)
    return None


CHAT_SYSTEM_PROMPT = """You are ORION, the conversational assistant for the FrED Factory team.

Your role is to handle general conversation, lightweight technical help, greetings, casual questions, quick explanations, and broad programming support.

You are NOT the real-time diagnostics engine, NOT the documentation retrieval system, and NOT the autonomous troubleshooting worker.

You are familiar with the FrED Factory laboratory context when available:
{lab_knowledge}

## CORE BEHAVIOR
- Be clear, useful, and direct
- Be professional but natural
- Match the technical depth to the user's question
- Use examples when they genuinely help
- Never use emojis
- Always respond in the same language as the user

## BOUNDARIES
- If the user asks for general knowledge, casual help, conceptual explanation, brainstorming, or programming support, answer directly
- If the user asks about live equipment state, current faults, PLC connectivity, active alarms, or any operational condition requiring verification, do not pretend it has been checked unless that information is explicitly available in context
- For lab-specific uncertainty, speak in terms of guidance, likely causes, or recommended checks — not confirmed facts
- Do not fabricate diagnostics, real-time status, sensor values, sources, or tool results
- Do not act like a troubleshooting tool or a document retrieval system

## RESPONSE STRATEGY
- Greeting or casual message → short, natural, warm
- Simple factual question → answer immediately
- Programming question → practical answer with concrete examples if useful
- Conceptual lab question → explain clearly using lab context when relevant
- Unverified operational lab question → be explicit about what is known vs not confirmed
- Opinion or brainstorming → be concrete, not vague

## STYLE
- Prefer direct answers over long introductions
- Avoid repetitive filler ("Certainly", "I'd be happy to", "Great question")
- Be honest about uncertainty
- Do not over-format unless it helps readability

{format_rules}

At the end, include exactly 3 brief follow-up suggestions:
---SUGGESTIONS---
1. [suggestion]
2. [suggestion]
3. [suggestion]
---END_SUGGESTIONS---

User name: {user_name}
"""


def _get_lab_context() -> str:
    if not LAB_KNOWLEDGE_AVAILABLE:
        return ""
    try:
        return get_lab_knowledge_summary()
    except Exception:
        return ""


def _get_last_user_message(state: AgentState) -> str:
    for m in reversed(state.get("messages", []) or []):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
        if isinstance(m, dict) and m.get("role") in ("human", "user"):
            return (m.get("content") or "").strip()
    return ""


def _extract_suggestions(text: str) -> tuple[str, list[str]]:
    """Extract suggestions block from LLM output."""
    suggestions = []
    content = text

    if "---SUGGESTIONS---" in text and "---END_SUGGESTIONS---" in text:
        parts = text.split("---SUGGESTIONS---")
        content = parts[0].strip()

        if len(parts) > 1:
            block = parts[1].split("---END_SUGGESTIONS---")[0]
            for line in block.strip().split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    cleaned = line.lstrip("0123456789.-) ").strip()
                    if cleaned:
                        suggestions.append(cleaned)

    return content, suggestions[:3]


def _build_conversation_history(state, max_turns: int = 4):
    """Build recent conversation history, excluding the current user message."""
    from langchain_core.messages import HumanMessage as HM, AIMessage as AIM

    raw_messages = state.get("messages", []) or []
    history = []

    for m in raw_messages:
        if hasattr(m, "type") and hasattr(m, "content"):
            if m.type == "human":
                history.append(HM(content=m.content))
            elif m.type == "ai" and m.content and m.content.strip():
                content = m.content[:500] + "..." if len(m.content) > 500 else m.content
                history.append(AIM(content=content))
        elif isinstance(m, dict):
            role = m.get("role", m.get("type", ""))
            content = m.get("content", "")
            if role in ("human", "user") and content:
                history.append(HM(content=content))
            elif role in ("ai", "assistant") and content and content.strip():
                content = content[:500] + "..." if len(content) > 500 else content
                history.append(AIM(content=content))

    if history and isinstance(history[-1], HM):
        history = history[:-1]

    if len(history) > max_turns * 2:
        history = history[-(max_turns * 2):]

    return history


def chat_node(state: AgentState) -> Dict[str, Any]:
    """General conversation worker. Boundary-aware: won't pretend to be diagnostics/research."""
    start_time = datetime.utcnow()
    logger.node_start("chat_node", {})
    events = [event_execute("chat", "Processing request...")]

    from src.agent.utils.stream_utils import get_worker_stream
    stream = get_worker_stream(state, "chat")

    user_message = _get_last_user_message(state)
    user_name = state.get("user_name", "User")
    model_name = state.get("llm_model") or os.getenv("DEFAULT_MODEL", "gemini-2.0-flash")

    if not user_message:
        output = WorkerOutputBuilder.chat(
            content=f"Hola {user_name}, ¿en qué te puedo ayudar?",
            summary="Initial greeting",
            confidence=1.0,
        )
        return {
            "worker_outputs": [output.model_dump()],
            "events": events + [event_report("chat", "Response ready")],
            "follow_up_suggestions": [],
        }

    quick = _try_quick_reply(user_message, user_name)
    if quick:
        output = WorkerOutputBuilder.chat(
            content=quick,
            summary="Quick reply",
            confidence=1.0,
        )
        output.metadata.completed_at = datetime.utcnow().isoformat()
        output.metadata.model_used = "quick_reply"
        logger.node_end("chat_node", {"quick_reply": True})
        events.append(event_report("chat", "Quick reply"))
        return {
            "worker_outputs": [output.model_dump()],
            "events": events,
            "follow_up_suggestions": [],
            "token_usage": 0,
        }

    try:
        from src.agent.utils.llm_factory import get_llm, invoke_and_track
        llm = get_llm(state, temperature=0.4)
    except Exception as e:
        logger.error("chat_node", f"LLM init error: {e}")
        output = WorkerOutputBuilder.chat(
            content=f"{user_name}, estoy teniendo problemas técnicos. Intenta de nuevo en un momento.",
            summary="LLM error fallback",
            confidence=0.3,
        )
        return {
            "worker_outputs": [output.model_dump()],
            "events": events,
            "follow_up_suggestions": [],
        }

    lab_knowledge = _get_lab_context()
    prompt = CHAT_SYSTEM_PROMPT.format(
        user_name=user_name,
        lab_knowledge=lab_knowledge,
        format_rules=MARKDOWN_FORMAT_RULES,
    )

    mode_instr = get_mode_instructions(state)
    if mode_instr:
        prompt += mode_instr

    team = state.get("team", "")
    if team:
        prompt += f"\nUser team: {team}"

    # Inject equipment spec (no skills — chat doesn't need detailed instructions)
    spec = state.get("equipment_spec", "")
    if spec.strip():
        prompt += f"\n\n## Equipment Context\n\n{spec}"

    llm_messages = [SystemMessage(content=prompt)]
    history = _build_conversation_history(state, max_turns=4)
    llm_messages.extend(history)

    image_attachments = state.get("image_attachments") or []
    if image_attachments:
        multimodal_content = [{"type": "text", "text": user_message}] + image_attachments
        llm_messages.append(HumanMessage(content=multimodal_content))
    else:
        llm_messages.append(HumanMessage(content=user_message))

    suggestions = []
    try:
        stream.status("Pensando...")
        response, tokens_used = invoke_and_track(llm, llm_messages, "chat")

        raw_result = (response.content or "").strip()
        result_text, suggestions = _extract_suggestions(raw_result)
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

    except Exception as e:
        logger.error("chat_node", f"LLM invocation error: {e}")
        result_text = f"{user_name}, tuve un error procesando tu mensaje. ¿Podrías intentar de nuevo?"
        suggestions = []
        processing_time = 0
        tokens_used = 0

    output = WorkerOutputBuilder.chat(
        content=result_text,
        summary="General conversation",
        confidence=0.9,
    )
    output.metadata.completed_at = datetime.utcnow().isoformat()
    output.metadata.processing_time_ms = processing_time
    output.metadata.model_used = model_name

    logger.node_end("chat_node", {"content_length": len(result_text)})
    events.append(event_report("chat", "Response ready"))

    return {
        "worker_outputs": [output.model_dump()],
        "events": events,
        "follow_up_suggestions": suggestions,
        "token_usage": tokens_used,
    }