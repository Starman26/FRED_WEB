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
from src.agent.utils.llm_factory import get_llm, invoke_and_track


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
from src.agent.tools.rag_tools import make_retrieve_tool, make_web_search_tool


RESEARCH_SYNTHESIS_PROMPT = """Eres un **Investigador Técnico Riguroso**.

Tu única fuente de verdad es la EVIDENCIA proporcionada abajo.
NO inventes información que no esté en la evidencia.

REGLAS OBLIGATORIAS:
1. **Citas obligatorias**: Cada afirmacion tecnica debe incluir cita en formato [Titulo, Pag. X-Y]
2. **Si falta evidencia**: Dilo explicitamente, no inventes
3. **Estructura clara**: Usa ## para titulo principal, ### para subtemas, --- entre secciones tematicas distintas
4. **Resumen al inicio**: Comienza con un resumen de 1-2 oraciones
5. **Formato Markdown**: Usa listas numeradas para pasos, bullet points para items, tablas para comparaciones, **negrita** para terminos clave, `codigo inline` para valores tecnicos
6. **NO uses emojis** — nunca

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

WEB_ENHANCED_SYNTHESIS_PROMPT = """Eres un **Investigador Técnico Riguroso**.

Tienes DOS fuentes de evidencia:
1. **DOCUMENTOS INTERNOS (RAG)**: Base de conocimientos técnicos del sistema
2. **RESULTADOS WEB**: Búsqueda en internet como complemento

PRIORIDAD: Los documentos internos tienen MAYOR peso que los resultados web.
Los resultados web complementan donde la base interna no tiene cobertura.

REGLAS OBLIGATORIAS:
1. **Citas obligatorias**: Cada afirmacion tecnica debe incluir cita
   - Documentos internos: [Titulo, Pag. X-Y]
   - Fuentes web: [Titulo - Web]
2. **Si falta evidencia**: Dilo explicitamente, no inventes
3. **Estructura clara**: Usa ## para titulo principal, ### para subtemas, --- entre secciones tematicas distintas
4. **Resumen al inicio**: Comienza con un resumen de 1-2 oraciones
5. **Formato Markdown**: Usa listas numeradas para pasos, bullet points para items, tablas para comparaciones, **negrita** para terminos clave, `codigo inline` para valores tecnicos
6. **NO uses emojis** — nunca
7. **Transparencia**: Si toda la respuesta viene de web, indica claramente que no se encontraron documentos internos relevantes

EVIDENCIA DE DOCUMENTOS INTERNOS:
{rag_evidence_text}

EVIDENCIA DE BÚSQUEDA WEB:
{web_evidence_text}

CONSULTA DEL USUARIO:
{user_query}

CONTEXTO ADICIONAL (de workers anteriores):
{prior_context}

Responde en JSON:
{{
  "answer": "Tu respuesta completa con citas",
  "sources_used": ["Título 1 (Pág. X-Y)", "Título 2 - Web"],
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
    rag_evidence_text = "\n\n".join([f"**{ev.title}** (Pág. {ev.page})\n{ev.chunk}" for ev in evidence_items]) if evidence_items else "NO HAY EVIDENCIA"

    total_tokens = 0
    try:
        llm = get_llm(state, temperature=0)
        synthesis_prompt = RESEARCH_SYNTHESIS_PROMPT.format(evidence_text=rag_evidence_text, user_query=user_query, prior_context=prior_context)
        response, tokens = invoke_and_track(llm, [SystemMessage(content=synthesis_prompt)], "research_synthesis")
        total_tokens += tokens
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

    # ── Web Search Fallback ──────────────────────────────────────
    # Si RAG no encontró nada o la confianza es baja, buscar en web
    web_used = False
    WEB_SEARCH_CONFIDENCE_THRESHOLD = 0.4

    if confidence_score < WEB_SEARCH_CONFIDENCE_THRESHOLD or not evidence_items:
        logger.info("research_node", f"Confianza baja ({confidence_score:.2f}) o sin evidencia RAG. Activando búsqueda web...")
        events.append(event_execute("research", "Buscando información complementaria en internet..."))

        try:
            web_search_tool = make_web_search_tool(max_results=5)
            web_result = web_search_tool.invoke({"query": user_query})
            web_summary, web_docs = unpack_retrieve_output(web_result)
            web_evidence_items = build_evidence_items(web_docs) if web_docs else []

            # Marcar evidencia web con source_type
            for ev in web_evidence_items:
                ev.metadata["source_type"] = "web"

            if web_evidence_items:
                web_used = True
                logger.info("research_node", f"Web search retornó {len(web_evidence_items)} resultados")

                # Re-sintetizar con evidencia combinada (RAG + Web)
                web_evidence_text = "\n\n".join(
                    [f"**{ev.title}** [Web]\n{ev.chunk}" for ev in web_evidence_items]
                )

                try:
                    combined_prompt = WEB_ENHANCED_SYNTHESIS_PROMPT.format(
                        rag_evidence_text=rag_evidence_text,
                        web_evidence_text=web_evidence_text,
                        user_query=user_query,
                        prior_context=prior_context,
                    )
                    response, tokens = invoke_and_track(llm, [SystemMessage(content=combined_prompt)], "research_web_synthesis")
                    total_tokens += tokens
                    response_text = response.content.strip()

                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0].strip()

                    parsed = json.loads(response_text)
                    answer = parsed.get("answer", answer)
                    confidence_score = parsed.get("confidence_score", confidence_score)
                    gaps = parsed.get("gaps", gaps)

                    # Agregar evidencia web a la lista total
                    evidence_items.extend(web_evidence_items)
                except Exception as e:
                    logger.warning("research_node", f"Error en síntesis combinada: {e}")
                    # Mantener la síntesis original RAG
            else:
                logger.info("research_node", "Web search no retornó resultados útiles")

        except Exception as e:
            logger.warning("research_node", f"Error en búsqueda web (fallback): {e}")
            # Continuar con solo RAG, no bloquear por error de web
    # ── Fin Web Search Fallback ──────────────────────────────────

    status = "ok" if evidence_items and confidence_score >= 0.5 else "partial"
    # NOTA: Ya no sugerimos automáticamente llamar a tutor.
    # El orchestrator_plan decide el plan completo desde el inicio.
    next_actions = []
    
    # Construir resumen descriptivo
    rag_count = sum(1 for ev in evidence_items if ev.metadata.get("source_type") != "web")
    web_count = sum(1 for ev in evidence_items if ev.metadata.get("source_type") == "web")
    if evidence_items:
        summary_parts = []
        if rag_count:
            summary_parts.append(f"{rag_count} documentos internos")
        if web_count:
            summary_parts.append(f"{web_count} fuentes web")
        summary_msg = f"Encontré {' + '.join(summary_parts)}"
    else:
        summary_msg = "No encontré documentos relevantes"

    output = WorkerOutputBuilder.research(
        content=answer,
        evidence=[ev.model_dump() for ev in evidence_items],
        summary=summary_msg,
        confidence=confidence_score,
        status=status,
        next_actions=next_actions,
    )
    output.metadata.completed_at = datetime.utcnow().isoformat()
    output.metadata.processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
    output.metadata.model_used = state.get("llm_model") or os.getenv("DEFAULT_MODEL", "gemini-2.0-flash")
    output.metadata.tokens_used = total_tokens
    output.extra = {"gaps": gaps, "query": user_query[:200], "web_search_used": web_used, "web_results_count": web_count}

    logger.node_end("research_node", {"documents_found": len(evidence_items), "confidence": confidence_score, "web_used": web_used, "tokens": total_tokens})
    web_note = f" + {web_count} web" if web_used else ""
    events.append(event_report("research", f" Investigación completada ({rag_count} docs{web_note})"))

    return {"worker_outputs": [output.model_dump()], "research_result": output.model_dump_json(), "events": events, "token_usage": total_tokens}
