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


CHAT_SYSTEM_PROMPT = """You are SENTINEL, a technical operations assistant for the FrED Factory laboratory.

EXPERTISE:
- PLCs and industrial automation (Siemens S7-1200, TIA Portal)
- Collaborative robotics (Universal Robots UR3e/UR5e/UR10e)
- Python, AI/ML, LangGraph/LangChain
- Industry 4.0 and technical education

{lab_knowledge}

COMMUNICATION STYLE:
- Professional and direct
- Concise and precise responses
- Technical terminology when appropriate
- No emojis or casual expressions
- Maintain a serious, agent-like demeanor

RULES:
- **LANGUAGE: ALWAYS respond in the same language the user writes in. If they write in English, respond in English. If in Spanish, respond in Spanish.**
- Keep greetings brief and professional
- Offer assistance without being overly friendly
- If technical help is needed, guide them to ask specific questions
- Reference laboratory equipment and stations when relevant
- Acknowledge limitations honestly

RESPONSE FORMAT:
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
    model_name = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-20250514")
    
    try:
        if "claude" in model_name.lower():
            llm = ChatAnthropic(model=model_name, temperature=0.7)
        else:
            llm = ChatOpenAI(model=model_name, temperature=0.7)
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
    prompt = CHAT_SYSTEM_PROMPT.format(
        user_name=user_name,
        lab_knowledge=lab_knowledge
    )
    
    suggestions = []
    try:
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=user_message)
        ])
        
        raw_result = (response.content or "").strip()
        # Extraer sugerencias del resultado
        result_text, suggestions = extract_suggestions(raw_result)
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
    except Exception as e:
        logger.error("chat_node", f"Error generando respuesta: {e}")
        result_text = "I'm ready to assist. What do you need?"
        suggestions = ["Check station status", "Report a problem", "Ask about equipment"]
        processing_time = 0
    
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
    }
