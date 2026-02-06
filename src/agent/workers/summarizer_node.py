"""
summarizer_node.py - Worker especializado en compresión de memoria

Se activa cuando window_count >= 12. Actualiza rolling_summary y resetea window_count.
"""
import os
from typing import Dict, Any, List
from datetime import datetime

from langchain_core.messages import SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from src.agent.state import AgentState
from src.agent.contracts.worker_contract import WorkerOutputBuilder, create_error_output
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_execute, event_report


SUMMARIZER_PROMPT = """Eres un compresor de contexto conversacional.

Tu tarea: Resume los mensajes en bullet points concisos (8-12 máximo).
Prioriza: objetivos, decisiones, datos técnicos, estado actual.
Retorna SOLO el resumen, sin texto adicional."""


def get_message_content(message) -> str:
    if isinstance(message, dict):
        return (message.get("content") or "").strip()
    return (getattr(message, "content", "") or "").strip()


def get_message_type(message) -> str:
    if isinstance(message, dict):
        return message.get("type") or message.get("role") or "unknown"
    return getattr(message, "type", "unknown")


def format_messages_for_summary(messages: list, limit: int = 12) -> str:
    recent = messages[-limit:] if len(messages) > limit else messages
    formatted = []
    for msg in recent:
        content = get_message_content(msg)
        if content:
            formatted.append(f"[{get_message_type(msg)}]: {content[:500]}...")
    return "\n\n".join(formatted)


def extract_key_points(summary_text: str) -> List[str]:
    key_points = []
    for line in summary_text.split("\n"):
        line = line.strip()
        if line.startswith(("•", "-", "*")):
            point = line.lstrip("•-* ").strip()
            if point and len(point) > 10:
                key_points.append(point)
    return key_points[:10]


def summarizer_node(state: AgentState) -> Dict[str, Any]:
    """Worker summarizer que comprime la memoria de la conversación."""
    start_time = datetime.utcnow()
    logger.node_start("summarizer_node", {"window_count": state.get("window_count", 0)})
    events = [event_execute("summarizer", "Comprimiendo memoria...")]
    
    messages = state.get("messages", [])
    prior_summary = state.get("rolling_summary", "")
    messages_count = len(messages)
    messages_to_compress = min(messages_count, 12)
    
    if messages_count == 0:
        output = WorkerOutputBuilder.summarizer(content="No hay mensajes.", messages_compressed=0, summary="Sin mensajes")
        return {"worker_outputs": [output.model_dump()], "summarizer_result": output.model_dump_json(), "window_count": 0, "events": events}
    
    messages_text = format_messages_for_summary(messages, limit=12)
    
    model_name = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-20250514")
    try:
        llm = ChatAnthropic(model=model_name, temperature=0.3) if "claude" in model_name.lower() else ChatOpenAI(model=model_name, temperature=0.3)
    except Exception as e:
        error_output = create_error_output("summarizer", "LLM_INIT_ERROR", str(e))
        return {"worker_outputs": [error_output.model_dump()], "summarizer_result": error_output.model_dump_json(), "rolling_summary": prior_summary, "window_count": 0, "events": events}
    
    prior_summary_section = ""
    if prior_summary:
        prior_summary_section = f"## RESUMEN PREVIO (intégralo)\n{prior_summary}\n\n"
    
    prompt_content = f"""{SUMMARIZER_PROMPT}

{prior_summary_section}## MENSAJES A RESUMIR ({messages_to_compress})
{messages_text}"""
    
    try:
        response = llm.invoke([SystemMessage(content=prompt_content)])
        new_summary = (response.content or "").strip()
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
    except Exception as e:
        new_summary = prior_summary or "[Error en compresión]"
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
    
    compression_ratio = (len(prior_summary) + len(messages_text)) / len(new_summary) if new_summary else 0.0
    key_points = extract_key_points(new_summary)
    
    output = WorkerOutputBuilder.summarizer(
        content=new_summary,
        key_points=key_points,
        messages_compressed=messages_to_compress,
        compression_ratio=round(compression_ratio, 2),
        summary=f"Comprimidos {messages_to_compress} mensajes ({compression_ratio:.1f}x)"
    )
    output.metadata.completed_at = datetime.utcnow().isoformat()
    output.metadata.processing_time_ms = processing_time
    output.metadata.model_used = model_name
    
    logger.node_end("summarizer_node", {"compression_ratio": compression_ratio})
    events.append(event_report("summarizer", f"✅ Memoria comprimida ({compression_ratio:.1f}x)"))
    
    return {
        "worker_outputs": [output.model_dump()],
        "summarizer_result": output.model_dump_json(),
        "rolling_summary": new_summary,
        "window_count": 0,
        "events": events,
    }
