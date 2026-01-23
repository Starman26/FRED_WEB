"""
research_node.py - Worker especializado en investigación con RAG

CAMBIOS PRINCIPALES vs versión anterior:
1. Usa WorkerOutput contract para output estructurado
2. NO retorna done=True (el orchestrator decide)
3. Usa pending_context para recibir contexto de workers anteriores
4. Soporta next_actions para sugerir siguientes pasos
5. Usa services.py para obtener supabase/embeddings (no del state)
"""
import os
import json
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

try:
    from langchain_core.messages import ToolMessage
except ImportError:
    ToolMessage = None

from src.agent.state import AgentState
from src.agent.services import get_supabase, get_embeddings
from src.agent.contracts.worker_contract import (
    WorkerOutput, 
    WorkerOutputBuilder,
    EvidenceItem, 
    ActionItem,
    create_error_output,
)
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_execute, event_report, event_error
from src.agent.tools.rag_tools import make_retrieve_tool


RESEARCH_SYNTHESIS_PROMPT = """Eres un **Investigador Técnico Riguroso**.

Tu única fuente de verdad es la EVIDENCIA proporcionada abajo.
NO inventes información que no esté en la evidencia.

REGLAS OBLIGATORIAS:
1. **Citas obligatorias**: Cada afirmación técnica debe incluir cita en formato [Título, Pág. X-Y]
2. **Si falta evidencia**: Dilo explícitamente, no inventes
3. **Estructura clara**: Usa encabezados y listas cuando ayuden
4. **Resumen al inicio**: Comienza con un resumen de 1-2 oraciones

EVIDENCIA RECUPERADA:
{evidence_text}

CONSULTA DEL USUARIO:
{user_query}

CONTEXTO ADICIONAL (de workers anteriores):
{prior_context}

Responde en JSON:
{{
  "answer": "Tu respuesta completa con citas [Título, Pág. X-Y]",
  "sources_used": ["Título 1 (Pág. X-Y)", "Título 2 (Pág. X-Y)"],
  "confidence_score": 0.0-1.0,
  "gaps": ["Lista de información que falta si la hay"]
}}
"""


def get_tools_from_services() -> Tuple[Any, Any]:
    """Obtiene Supabase y Embeddings desde services (singletons)"""
    supabase = get_supabase()
    embeddings = get_embeddings()
    return supabase, embeddings


def get_last_user_message(state: AgentState) -> str:
    """Extrae el último mensaje del usuario"""
    messages = state.get("messages", []) or []
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
        if isinstance(m, dict):
            role = m.get("role") or m.get("type")
            if role in ("human", "user"):
                return (m.get("content") or "").strip()
    return ""


def unpack_retrieve_output(result: Any) -> Tuple[str, List[Document]]:
    """Desempaqueta el output del retrieve tool."""
    if ToolMessage is not None and isinstance(result, ToolMessage):
        content = result.content or ""
        artifact = getattr(result, "artifact", None)
        docs = artifact if isinstance(artifact, list) else []
        return str(content), docs
    
    if isinstance(result, dict):
        content = result.get("content", "") or ""
        docs = result.get("artifact") or result.get("documents") or []
        return str(content), docs if isinstance(docs, list) else []
    
    if isinstance(result, tuple) and len(result) == 2:
        content, docs = result
        return str(content or ""), docs if isinstance(docs, list) else []
    
    if hasattr(result, "content"):
        content = getattr(result, "content", "") or ""
        docs = getattr(result, "artifact", None) or []
        return str(content), docs if isinstance(docs, list) else []
    
    return str(result), []


def build_evidence_items(docs: List[Document]) -> List[EvidenceItem]:
    """Convierte Documents a EvidenceItems"""
    evidence_items = []
    for doc in docs:
        meta = doc.metadata or {}
        title = meta.get("title_original") or meta.get("doc_title") or meta.get("title") or "Documento"
        page_start = meta.get("page_start", "?")
        page_end = meta.get("page_end", "?")
        page_str = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
        
        evidence_items.append(EvidenceItem(
            source_id=meta.get("chunk_id") or meta.get("doc_id"),
            title=str(title),
            chunk=doc.page_content[:500],
            page=page_str,
            score=meta.get("relevance_score", 0.0),
            metadata={"doc_source": meta.get("doc_source"), "doc_type": meta.get("doc_type")}
        ))
    return evidence_items


def get_prior_context(state: AgentState) -> str:
    """Obtiene contexto de workers anteriores"""
    pending_context = state.get("pending_context", {})
    prior_summaries = pending_context.get("prior_summaries", [])
    if not prior_summaries:
        return "Ninguno"
    context_parts = [f"- {ps.get('worker', 'unknown')}: {ps.get('summary', '')}" for ps in prior_summaries if ps.get('summary')]
    return "\n".join(context_parts) if context_parts else "Ninguno"


def research_node(state: AgentState) -> Dict[str, Any]:
    """Worker de investigación que usa RAG para buscar en documentos."""
    start_time = datetime.utcnow()
    logger.node_start("research_node", {"task": "investigación técnica"})
    events = [event_execute("research", "Iniciando búsqueda en documentos...")]
    
    supabase, embeddings = get_tools_from_services()
    if not supabase or not embeddings:
        error_output = create_error_output("research", "TOOLS_UNAVAILABLE", "No hay conexión con la base de conocimientos")
        return {"worker_outputs": [error_output.model_dump()], "research_result": error_output.model_dump_json(), "events": events}
    
    user_query = get_last_user_message(state)
    if not user_query:
        error_output = create_error_output("research", "NO_QUERY", "No se detectó una consulta del usuario")
        return {"worker_outputs": [error_output.model_dump()], "research_result": error_output.model_dump_json(), "events": events}
    
    logger.info("research_node", f"Query: {user_query[:100]}...")
    retrieve_tool = make_retrieve_tool(supabase, embeddings)
    
    try:
        result = retrieve_tool.invoke({"query": user_query})
        serialized_content, docs = unpack_retrieve_output(result)
        evidence_items = build_evidence_items(docs) if docs else []
        logger.info("research_node", f"RAG retornó {len(docs)} documentos")
    except Exception as e:
        logger.error("research_node", f"Error en RAG: {e}")
        error_output = create_error_output("research", "RAG_ERROR", f"Error ejecutando búsqueda: {str(e)}")
        return {"worker_outputs": [error_output.model_dump()], "research_result": error_output.model_dump_json(), "events": events}
    
    # Sintetizar evidencia
    prior_context = get_prior_context(state)
    evidence_text = "\n\n".join([f"**{ev.title}** (Pág. {ev.page})\n{ev.chunk}" for ev in evidence_items]) if evidence_items else "NO HAY EVIDENCIA"
    
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        llm = ChatOpenAI(model=model_name, temperature=0)
        synthesis_prompt = RESEARCH_SYNTHESIS_PROMPT.format(evidence_text=evidence_text, user_query=user_query, prior_context=prior_context)
        response = llm.invoke([SystemMessage(content=synthesis_prompt)])
        response_text = response.content.strip()
        
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        
        parsed = json.loads(response_text)
        answer = parsed.get("answer", "")
        confidence_score = parsed.get("confidence_score", 0.5)
        gaps = parsed.get("gaps", [])
    except Exception as e:
        logger.warning("research_node", f"Error en síntesis: {e}")
        answer = serialized_content
        confidence_score = 0.5 if evidence_items else 0.1
        gaps = []
    
    status = "ok" if evidence_items and confidence_score >= 0.5 else "partial"
    # NOTA: Ya no sugerimos automáticamente llamar a tutor.
    # El orchestrator_plan decide el plan completo desde el inicio.
    next_actions = []
    
    output = WorkerOutputBuilder.research(
        content=answer,
        evidence=[ev.model_dump() for ev in evidence_items],
        summary=f"Encontré {len(evidence_items)} documentos relevantes" if evidence_items else "No encontré documentos relevantes",
        confidence=confidence_score,
        status=status,
        next_actions=next_actions,
    )
    output.metadata.completed_at = datetime.utcnow().isoformat()
    output.metadata.processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
    output.metadata.model_used = model_name
    output.extra = {"gaps": gaps, "query": user_query[:200]}
    
    logger.node_end("research_node", {"documents_found": len(evidence_items), "confidence": confidence_score})
    events.append(event_report("research", f"✅ Investigación completada ({len(evidence_items)} fuentes)"))
    
    return {"worker_outputs": [output.model_dump()], "research_result": output.model_dump_json(), "events": events}
