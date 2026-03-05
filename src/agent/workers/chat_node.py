"""
chat_node.py - Worker para conversación general y saludos

Este worker maneja:
- Saludos y presentaciones
- Preguntas casuales
- Agradecimientos y despedidas
- Conversación general que no requiere RAG ni tutorías
- Preguntas básicas sobre el laboratorio

NO usa WorkerOutput contract completo para mantener respuestas ligeras.
"""
import os
from typing import Dict, Any
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from src.agent.state import AgentState
from src.agent.contracts.worker_contract import WorkerOutputBuilder
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_execute, event_report

# Importar modos de interacción
from src.agent.interaction_modes import get_mode_instructions
from src.agent.prompts.format_rules import MARKDOWN_FORMAT_RULES

# Importar conocimiento del laboratorio
try:
    from src.agent.knowledge import (
        get_lab_knowledge_summary,
        get_robot_info,
        get_station_info,
        get_terminology_definition,
        TERMINOLOGY,
        ROBOTS,
    )
    LAB_KNOWLEDGE_AVAILABLE = True
except ImportError:
    LAB_KNOWLEDGE_AVAILABLE = False


def get_lab_context_for_chat() -> str:
    """Obtiene contexto del laboratorio para el chat"""
    if not LAB_KNOWLEDGE_AVAILABLE:
        return ""
    
    try:
        return get_lab_knowledge_summary()
    except:
        return ""



def _build_conversation_history(state, max_turns: int = 4):
    """
    Builds recent conversation history from state messages.
    Returns list of LangChain message objects (excluding the last user message).
    max_turns = number of recent user+assistant pairs to include.
    """
    from langchain_core.messages import HumanMessage as HM, AIMessage as AIM
    
    raw_messages = state.get("messages", []) or []
    history = []
    
    for m in raw_messages:
        if hasattr(m, "type") and hasattr(m, "content"):
            if m.type == "human":
                history.append(HM(content=m.content))
            elif m.type == "ai" and m.content and m.content.strip():
                # Truncate long AI responses to save tokens
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
    
    # Remove the last message (current user message - will be added separately)
    if history and isinstance(history[-1], HM):
        history = history[:-1]
    
    # Keep only last N turns (each turn = user + assistant = 2 messages)
    if len(history) > max_turns * 2:
        history = history[-(max_turns * 2):]
    
    return history

CHAT_SYSTEM_PROMPT = """You are ORION, an advanced AI assistant for the FrED Factory team.

You are a GENERAL-PURPOSE assistant who can help with ANY topic, including:
- Programming (Python, JavaScript, React, Vue, CSS, HTML, APIs, databases)
- Software architecture, design patterns, code review
- AI/ML, prompt engineering, LLM integration
- DevOps, deployment, debugging
- General questions, opinions, advice, brainstorming

You ALSO have specialized expertise in:
- PLCs and industrial automation (Siemens S7-1200, TIA Portal)
- Collaborative robotics (Universal Robots UR3e/UR5e/UR10e)
- Industry 4.0 and technical education
- The FrED Factory laboratory systems

{lab_knowledge}

COMMUNICATION STYLE:
- Professional but approachable
- Concise and precise responses
- Technical terminology when appropriate
- Helpful and constructive
- NEVER use emojis in responses

RULES:
- **LANGUAGE: ALWAYS respond in the same language the user writes in. If they write in English, respond in English. If in Spanish, respond in Spanish.**
- Help with ANY topic the user asks about - you are NOT limited to lab topics
- When the question is about code/programming, give concrete examples and solutions
- When the question is about the lab, reference equipment and stations
- Be honest about limitations

{format_rules}

Always end your response with exactly 3 follow-up suggestions in this format:
---SUGGESTIONS---
1. [First suggestion for continuing the conversation]
2. [Second suggestion]
3. [Third suggestion]
---END_SUGGESTIONS---

User name: {user_name}
"""


def get_last_user_message(state: AgentState) -> str:
    """Extrae el último mensaje del usuario"""
    for m in reversed(state.get("messages", []) or []):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
        if isinstance(m, dict) and m.get("role") in ("human", "user"):
            return (m.get("content") or "").strip()
    return ""


def extract_suggestions(text: str) -> tuple[str, list[str]]:
    """
    Extrae las sugerencias del texto de respuesta.
    
    Returns:
        (content_without_suggestions, list_of_suggestions)
    """
    suggestions = []
    content = text
    
    # Buscar el bloque de sugerencias
    if "---SUGGESTIONS---" in text and "---END_SUGGESTIONS---" in text:
        parts = text.split("---SUGGESTIONS---")
        content = parts[0].strip()
        
        if len(parts) > 1:
            suggestions_block = parts[1].split("---END_SUGGESTIONS---")[0]
            # Extraer líneas que empiezan con número
            for line in suggestions_block.strip().split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    # Limpiar el prefijo numérico o guión
                    clean_line = line.lstrip("0123456789.-) ").strip()
                    if clean_line:
                        suggestions.append(clean_line)
    
    return content, suggestions[:3]  # Máximo 3 sugerencias


def chat_node(state: AgentState) -> Dict[str, Any]:
    """
    Worker de conversación general.
    
    Maneja saludos, presentaciones y conversación casual de forma profesional.
    """
    start_time = datetime.utcnow()
    logger.node_start("chat_node", {})
    events = [event_execute("chat", "Processing request...")]
    
    user_message = get_last_user_message(state)
    user_name = state.get("user_name", "User")
    
    if not user_message:
        # Respuesta genérica si no hay mensaje
        output = WorkerOutputBuilder.tutor(
            content="How may I assist you?",
            summary="Initial greeting",
            confidence=1.0
        )
        output.worker = "chat"
        return {
            "worker_outputs": [output.model_dump()],
            "events": events + [event_report("chat", "Response ready")],
            "follow_up_suggestions": ["Ask about PLC diagnostics", "Request cobot status", "Inquire about station operations"],
        }
    
    # Configurar LLM
    model_name = state.get("llm_model") or os.getenv("DEFAULT_MODEL", "gemini-2.0-flash")
    
    try:
        from src.agent.utils.llm_factory import get_llm
        llm = get_llm(state, temperature=0.7)
    except Exception as e:
        logger.error("chat_node", f"Error inicializando LLM: {e}")
        output = WorkerOutputBuilder.tutor(
            content="System ready. How may I assist you?",
            summary="Fallback response",
            confidence=0.5
        )
        output.worker = "chat"
        return {
            "worker_outputs": [output.model_dump()],
            "events": events,
            "follow_up_suggestions": ["Check system status", "Report an issue", "Request technical documentation"],
        }
    
    # Generar respuesta
    lab_knowledge = get_lab_context_for_chat() if LAB_KNOWLEDGE_AVAILABLE else ""
    
    # Build user context from team
    team = state.get("team", "")
    user_context = ""
    if team:
        user_context += f"\nUser team: {team}"
    
    prompt = CHAT_SYSTEM_PROMPT.format(
        user_name=user_name,
        lab_knowledge=lab_knowledge,
        format_rules=MARKDOWN_FORMAT_RULES,
    )
    if user_context:
        prompt += f"\n{user_context}"
    
    suggestions = []
    try:
        # Inyectar instrucciones de modo (Chat/Code/Agent/Voice + Focus)
        mode_instr = get_mode_instructions(state)
        if mode_instr:
            prompt += mode_instr
        
        # Build message list with conversation history
        llm_messages = [SystemMessage(content=prompt)]
        history = _build_conversation_history(state, max_turns=4)
        llm_messages.extend(history)

        # If user attached images, build multimodal content blocks for the LLM
        image_attachments = state.get("image_attachments") or []
        if image_attachments:
            multimodal_content = [{"type": "text", "text": user_message}] + image_attachments
            llm_messages.append(HumanMessage(content=multimodal_content))
        else:
            llm_messages.append(HumanMessage(content=user_message))
        
        from src.agent.utils.llm_factory import invoke_and_track
        response, tokens_used = invoke_and_track(llm, llm_messages, "chat")

        raw_result = (response.content or "").strip()
        # Extraer sugerencias del resultado
        result_text, suggestions = extract_suggestions(raw_result)
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
    except Exception as e:
        logger.error("chat_node", f"Error generando respuesta: {e}")
        result_text = "I'm ready to assist. What do you need?"
        suggestions = ["Check station status", "Report a problem", "Ask about equipment"]
        processing_time = 0
        tokens_used = 0
    
    # Construir output
    output = WorkerOutputBuilder.tutor(
        content=result_text,
        summary="General conversation",
        confidence=1.0
    )
    output.worker = "chat"
    output.metadata.completed_at = datetime.utcnow().isoformat()
    output.metadata.processing_time_ms = processing_time
    output.metadata.model_used = model_name
    
    logger.node_end("chat_node", {"content_length": len(result_text)})
    events.append(event_report("chat", "Response ready"))
    
    return {
        "worker_outputs": [output.model_dump()],
        "events": events,
        "follow_up_suggestions": suggestions if suggestions else ["Check station status", "Report an issue", "Ask about equipment"],
        "token_usage": tokens_used,
    }