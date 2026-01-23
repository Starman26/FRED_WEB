"""
tutor_node.py - Worker especializado en tutorías y explicaciones educativas

Usa WorkerOutput contract, NO retorna done=True, usa pending_context para evidencia.
"""
import os
from typing import Dict, Any, List
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from src.agent.state import AgentState
from src.agent.contracts.worker_contract import WorkerOutputBuilder, EvidenceItem, create_error_output
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_execute, event_report, event_error


TUTOR_MULTISTEP_PROMPT = """Eres un **Tutor Técnico Especializado** experto en:
- PLCs (Controladores Lógicos Programables)
- Cobots (Robots Colaborativos)
- Python y AI/ML (LangGraph, LangChain)

## CONTEXTO IMPORTANTE
{context_section}

## EVIDENCIA DE INVESTIGACIÓN PREVIA
{evidence_section}

## INSTRUCCIONES
1. **Usa la evidencia proporcionada**: Si hay evidencia, úsala y cítala [Título, Pág. X-Y]
2. **Estructura clara**: Usa encabezados y listas cuando ayuden
3. **Sé didáctico**: Explica paso a paso, con ejemplos
4. **Responde en español**

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


def get_evidence_from_context(state: AgentState) -> tuple[str, List[EvidenceItem]]:
    """Obtiene evidencia del pending_context"""
    pending_context = state.get("pending_context", {})
    evidence_data = pending_context.get("evidence", [])
    
    if not evidence_data:
        for output in state.get("worker_outputs", []):
            if output.get("worker") == "research":
                evidence_data = output.get("evidence", [])
                break
    
    if not evidence_data:
        return "No hay evidencia de investigación previa.", []
    
    evidence_items = []
    evidence_parts = []
    for ev in evidence_data:
        if isinstance(ev, dict):
            title, page, chunk = ev.get("title", "Doc"), ev.get("page", "?"), ev.get("chunk", "")
            evidence_parts.append(f"**{title}** (Pág. {page})\n{chunk[:300]}...")
            evidence_items.append(EvidenceItem(title=title, page=page, chunk=chunk, score=ev.get("score", 0)))
    
    return "\n\n".join(evidence_parts) if evidence_parts else "No hay evidencia.", evidence_items


def get_prior_summaries(state: AgentState) -> str:
    """Obtiene resúmenes de workers anteriores"""
    prior_summaries = state.get("pending_context", {}).get("prior_summaries", [])
    if not prior_summaries:
        return "Sin contexto previo."
    return "\n".join([f"- **{ps.get('worker')}**: {ps.get('summary')}" for ps in prior_summaries if ps.get('summary')]) or "Sin contexto previo."


def tutor_node(state: AgentState) -> Dict[str, Any]:
    """Worker tutor que genera contenido educativo."""
    start_time = datetime.utcnow()
    logger.node_start("tutor_node", {"has_pending_context": bool(state.get("pending_context"))})
    events = [event_execute("tutor", "Preparando explicación educativa...")]
    
    user_message = get_last_user_message(state)
    if not user_message:
        error_output = create_error_output("tutor", "NO_MESSAGE", "No hay mensaje del usuario")
        return {"worker_outputs": [error_output.model_dump()], "tutor_result": error_output.model_dump_json(), "events": events}
    
    evidence_text, evidence_items = get_evidence_from_context(state)
    context_text = get_prior_summaries(state)
    has_evidence = len(evidence_items) > 0
    
    model_name = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-20250514")
    try:
        llm = ChatAnthropic(model=model_name, temperature=0.7) if "claude" in model_name.lower() else ChatOpenAI(model=model_name, temperature=0.7)
    except Exception as e:
        error_output = create_error_output("tutor", "LLM_INIT_ERROR", f"Error inicializando modelo: {str(e)}")
        return {"worker_outputs": [error_output.model_dump()], "tutor_result": error_output.model_dump_json(), "events": events}
    
    prompt = TUTOR_MULTISTEP_PROMPT.format(
        context_section=context_text if context_text != "Sin contexto previo." else "Primera interacción",
        evidence_section=evidence_text,
        user_name=state.get("user_name", "Usuario")
    )
    
    messages = [SystemMessage(content=prompt)]
    if rolling_summary := state.get("rolling_summary", ""):
        messages.append(SystemMessage(content=f"Contexto de la conversación:\n{rolling_summary}"))
    messages.append(HumanMessage(content=user_message))
    
    try:
        response = llm.invoke(messages)
        result_text = (response.content or "").strip()
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
    except Exception as e:
        error_output = create_error_output("tutor", "LLM_ERROR", f"Error generando respuesta: {str(e)}")
        return {"worker_outputs": [error_output.model_dump()], "tutor_result": error_output.model_dump_json(), "events": events}
    
    output = WorkerOutputBuilder.tutor(
        content=result_text,
        learning_objectives=["Comprender el concepto", "Aplicar en práctica"],
        summary=f"Explicación educativa generada ({len(result_text)} chars)",
        confidence=0.85 if has_evidence else 0.75,
    )
    if evidence_items:
        output.evidence = evidence_items
    output.metadata.completed_at = datetime.utcnow().isoformat()
    output.metadata.processing_time_ms = processing_time
    output.metadata.model_used = model_name
    
    logger.node_end("tutor_node", {"content_length": len(result_text)})
    events.append(event_report("tutor", "✅ Explicación lista"))
    
    return {"worker_outputs": [output.model_dump()], "tutor_result": output.model_dump_json(), "events": events}
