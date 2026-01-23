"""
chat_node.py - Worker para conversación general y saludos

Este worker maneja:
- Saludos y presentaciones
- Preguntas casuales
- Agradecimientos y despedidas
- Conversación general que no requiere RAG ni tutorías

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


CHAT_SYSTEM_PROMPT = """Eres FrEDie, un asistente virtual amigable especializado en:
- PLCs y automatización industrial
- Cobots y robótica
- Python, AI/ML y LangGraph/LangChain
- Educación técnica e Industria 4.0

PERSONALIDAD:
- Amigable y cercano
- Profesional pero accesible
- Respuestas concisas para conversación casual
- Usa emojis ocasionalmente para ser más expresivo

REGLAS:
- Responde en español
- Para saludos: respuesta breve y amigable
- Para presentaciones: reconoce el nombre y ofrece ayuda
- Para agradecimientos: responde cordialmente
- NO hagas respuestas largas para conversación casual
- Si detectas que el usuario necesita ayuda técnica, ofrécele que pregunte

Nombre del usuario: {user_name}
"""


def get_last_user_message(state: AgentState) -> str:
    """Extrae el último mensaje del usuario"""
    for m in reversed(state.get("messages", []) or []):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
        if isinstance(m, dict) and m.get("role") in ("human", "user"):
            return (m.get("content") or "").strip()
    return ""


def chat_node(state: AgentState) -> Dict[str, Any]:
    """
    Worker de conversación general.
    
    Maneja saludos, presentaciones y conversación casual de forma ligera.
    """
    start_time = datetime.utcnow()
    logger.node_start("chat_node", {})
    events = [event_execute("chat", "Procesando...")]
    
    user_message = get_last_user_message(state)
    user_name = state.get("user_name", "Usuario")
    
    if not user_message:
        # Respuesta genérica si no hay mensaje
        output = WorkerOutputBuilder.tutor(
            content="¡Hola! ¿En qué puedo ayudarte hoy? 👋",
            summary="Saludo inicial",
            confidence=1.0
        )
        output.worker = "chat"  # Override worker name
        return {
            "worker_outputs": [output.model_dump()],
            "events": events + [event_report("chat", "✅ Respuesta lista")],
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
            content="¡Hola! Estoy aquí para ayudarte. ¿Qué necesitas?",
            summary="Respuesta fallback",
            confidence=0.5
        )
        output.worker = "chat"
        return {
            "worker_outputs": [output.model_dump()],
            "events": events,
        }
    
    # Generar respuesta
    prompt = CHAT_SYSTEM_PROMPT.format(user_name=user_name)
    
    try:
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=user_message)
        ])
        
        result_text = (response.content or "").strip()
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
    except Exception as e:
        logger.error("chat_node", f"Error generando respuesta: {e}")
        result_text = f"¡Hola{', ' + user_name if user_name != 'Usuario' else ''}! 👋 ¿En qué puedo ayudarte?"
        processing_time = 0
    
    # Construir output (usamos tutor builder pero cambiamos el worker)
    output = WorkerOutputBuilder.tutor(
        content=result_text,
        summary="Conversación general",
        confidence=1.0
    )
    output.worker = "chat"  # Override para identificar como chat
    output.metadata.completed_at = datetime.utcnow().isoformat()
    output.metadata.processing_time_ms = processing_time
    output.metadata.model_used = model_name
    
    logger.node_end("chat_node", {"content_length": len(result_text)})
    events.append(event_report("chat", "✅ Respuesta lista"))
    
    return {
        "worker_outputs": [output.model_dump()],
        "events": events,
    }
