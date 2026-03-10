"""
research_node.py - Worker especializado en investigación con RAG

Responsabilidad: recuperar evidencia de documentos internos (y web como fallback),
sintetizar una respuesta fundamentada, y reportar confianza y gaps.

NO es un answer-writer pulido — eso es trabajo de synthesize_node.
Este worker prioriza: evidencia verificable > respuesta bonita.
"""
import os
import json
import re
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

try:
    from langchain_core.messages import ToolMessage
except ImportError:
    ToolMessage = None

from src.agent.state import AgentState
from src.agent.services import get_supabase, get_embeddings
from src.agent.contracts.worker_contract import (
    WorkerOutputBuilder,
    EvidenceItem,
    create_error_output,
)
from src.agent.utils.llm_factory import get_llm, invoke_and_track
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_execute, event_report, event_error
from src.agent.tools.db_tools.rag_tools import make_retrieve_tool, make_web_search_tool


# ============================================
# PROMPTS — simplified, evidence-first
# ============================================

_RESEARCH_SYSTEM = """You are a rigorous technical research worker.
Your job is to answer a user's question using ONLY the provided evidence.
Do not invent information. Do not fill gaps with general knowledge.
If evidence is insufficient, say so explicitly.
Return valid JSON only — no markdown fences, no preamble."""

_RESEARCH_SYNTHESIS = """## RULES
1. Every technical claim must include an inline citation: [Title, Pag. X-Y]
2. If the evidence does not support an answer, say so clearly
3. Do not repeat the same point in different words
4. Do not use emojis
5. Be useful but contained — this is an internal research output, not a final user-facing response

## EVIDENCE
{evidence_text}

## USER QUERY
{user_query}

## PRIOR CONTEXT
{prior_context}

Respond with this exact JSON structure:
{{
  "answer": "your evidence-grounded response with inline citations",
  "confidence_score": 0.0,
  "gaps": ["list of missing information if any"]
}}"""

_WEB_ENHANCED_SYSTEM = """You are a rigorous technical research worker.
You have two evidence sources: internal documents (higher priority) and web results (complementary).
If they conflict, trust internal documents and note the discrepancy.
Do not invent information. Return valid JSON only."""

_WEB_ENHANCED_SYNTHESIS = """## TRUTH HIERARCHY
- Internal documents have priority over web results
- Web only complements or covers gaps
- If there is a conflict, prioritize internal docs and mention the discrepancy

## RULES
1. Every technical claim must include an inline citation:
   - Internal: [Title, Pag. X-Y]
   - Web: [Title - Web]
2. If evidence is insufficient, say so clearly
3. Do not mix uncertain findings with confirmed facts
4. Do not use emojis

## INTERNAL EVIDENCE
{rag_evidence_text}

## WEB EVIDENCE
{web_evidence_text}

## USER QUERY
{user_query}

## PRIOR CONTEXT
{prior_context}

Respond with this exact JSON structure:
{{
  "answer": "evidence-grounded response with inline citations",
  "confidence_score": 0.0,
  "gaps": ["list of missing information if any"]
}}"""


# ============================================
# JSON PARSING
# ============================================

def _safe_parse_json(text: str) -> Optional[dict]:
    """Robust JSON extraction from LLM output."""
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

    # Extract first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


# ============================================
# EVIDENCE HELPERS
# ============================================

def _unpack_retrieve_output(result: Any) -> Tuple[str, List[Document]]:
    """Unpack output from retrieve/search tools into (text, docs)."""
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


def _build_evidence_items(docs: List[Document], source_type: str = "internal") -> List[EvidenceItem]:
    """Convert Documents to EvidenceItems with explicit source_type from the start."""
    items = []
    for doc in docs:
        meta = doc.metadata or {}
        title = (
            meta.get("title_original")
            or meta.get("doc_title")
            or meta.get("title")
            or "Documento"
        )
        page_start = meta.get("page_start", "?")
        page_end = meta.get("page_end", "?")
        page_str = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"

        items.append(EvidenceItem(
            source_id=meta.get("chunk_id") or meta.get("doc_id"),
            title=str(title),
            chunk=doc.page_content[:500],
            page=page_str,
            score=meta.get("relevance_score", 0.0),
            metadata={
                "doc_source": meta.get("doc_source"),
                "doc_type": meta.get("doc_type"),
                "source_type": source_type,
            },
        ))
    return items


def _format_evidence_text(items: List[EvidenceItem], label: str = "") -> str:
    """Format evidence items into text for the LLM prompt."""
    if not items:
        return "NO EVIDENCE AVAILABLE"
    parts = []
    for ev in items:
        suffix = f" [{label}]" if label else ""
        parts.append(f"**{ev.title}** (Pág. {ev.page}){suffix}\n{ev.chunk}")
    return "\n\n".join(parts)


def _get_prior_context(state: AgentState) -> str:
    """Extract prior context from previous workers."""
    pending = state.get("pending_context", {}) or {}
    summaries = pending.get("prior_summaries", [])
    if not summaries:
        return "None"
    parts = [
        f"- {s.get('worker', 'unknown')}: {s.get('summary', '')}"
        for s in summaries if s.get("summary")
    ]
    return "\n".join(parts) if parts else "None"


def _get_last_user_message(state: AgentState) -> str:
    for m in reversed(state.get("messages", []) or []):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
        if isinstance(m, dict) and m.get("role") in ("human", "user"):
            return (m.get("content") or "").strip()
    return ""


# ============================================
# CONFIDENCE — hybrid (code + LLM signal)
# ============================================

def _compute_confidence(
    rag_count: int,
    web_count: int,
    gaps: list,
    llm_score: float,
) -> float:
    """Hybrid confidence: structural heuristics + LLM signal as secondary input."""
    base = 0.15

    # Evidence quantity
    if rag_count >= 3:
        base += 0.35
    elif rag_count >= 1:
        base += 0.20

    if web_count > 0:
        base += 0.10

    # Penalties
    if gaps:
        base -= 0.10
    if rag_count == 0 and web_count == 0:
        return 0.05

    # Blend with LLM score (30% weight)
    blended = base * 0.7 + llm_score * 0.3
    return round(min(max(blended, 0.0), 1.0), 2)


def _compute_status(rag_count: int, web_count: int, confidence: float) -> str:
    """Determine research status from evidence and confidence."""
    if rag_count == 0 and web_count == 0:
        return "no_evidence"
    if confidence >= 0.5 and rag_count >= 1:
        return "ok"
    return "partial"


# ============================================
# RETRIEVAL STAGES
# ============================================

def _run_rag_retrieval(
    user_query: str,
    supabase,
    embeddings,
    stream,
) -> Tuple[List[EvidenceItem], str]:
    """Run RAG retrieval. Returns (evidence_items, serialized_content)."""
    stream.tool("rag_search", f"Buscando documentos sobre: {user_query[:80]}...")
    retrieve_tool = make_retrieve_tool(supabase, embeddings)

    result = retrieve_tool.invoke({"query": user_query})
    content, docs = _unpack_retrieve_output(result)
    items = _build_evidence_items(docs, source_type="internal")

    logger.info("research_node", f"RAG returned {len(docs)} documents")
    if items:
        stream.found(f"Encontré {len(items)} documentos relevantes")
    else:
        stream.status("No encontré documentos directamente relevantes, ampliando búsqueda...")

    return items, content


def _run_web_fallback(
    user_query: str,
    stream,
) -> List[EvidenceItem]:
    """Run web search fallback. Returns web evidence items."""
    stream.tool("web_search", "Buscando información complementaria en internet...")

    web_tool = make_web_search_tool(max_results=5)
    result = web_tool.invoke({"query": user_query})
    _, docs = _unpack_retrieve_output(result)
    items = _build_evidence_items(docs, source_type="web")

    if items:
        logger.info("research_node", f"Web search returned {len(items)} results")
        stream.found(f"Encontré {len(items)} resultados web complementarios")
    else:
        logger.info("research_node", "Web search returned no useful results")

    return items


# ============================================
# SYNTHESIS
# ============================================

def _synthesize(
    llm,
    user_query: str,
    rag_items: List[EvidenceItem],
    web_items: List[EvidenceItem],
    prior_context: str,
) -> Tuple[str, float, list, int]:
    """
    Synthesize evidence into an answer.
    Returns (answer, llm_confidence, gaps, tokens_used).
    """
    has_web = len(web_items) > 0
    rag_text = _format_evidence_text(rag_items)

    if has_web:
        web_text = _format_evidence_text(web_items, label="Web")
        system_msg = _WEB_ENHANCED_SYSTEM
        user_msg = _WEB_ENHANCED_SYNTHESIS.format(
            rag_evidence_text=rag_text,
            web_evidence_text=web_text,
            user_query=user_query,
            prior_context=prior_context,
        )
    else:
        system_msg = _RESEARCH_SYSTEM
        user_msg = _RESEARCH_SYNTHESIS.format(
            evidence_text=rag_text,
            user_query=user_query,
            prior_context=prior_context,
        )

    response, tokens = invoke_and_track(llm, [
        SystemMessage(content=system_msg),
        HumanMessage(content=user_msg),
    ], "research_synthesis")

    raw = response.content.strip()
    parsed = _safe_parse_json(raw)

    if parsed:
        answer = parsed.get("answer", raw)
        llm_confidence = parsed.get("confidence_score", 0.5)
        gaps = parsed.get("gaps", [])
    else:
        logger.warning("research_node", f"Failed to parse JSON, using raw output (len={len(raw)})")
        answer = raw
        llm_confidence = 0.5
        gaps = []

    return answer, llm_confidence, gaps, tokens


# ============================================
# MAIN NODE
# ============================================

WEB_SEARCH_CONFIDENCE_THRESHOLD = 0.4


def research_node(state: AgentState) -> Dict[str, Any]:
    """Research worker: RAG retrieval → synthesis → optional web fallback → output."""
    start_time = datetime.utcnow()
    logger.node_start("research_node", {"task": "research"})
    events = [event_execute("research", "Iniciando búsqueda en documentos...")]

    # - Setup -
    supabase, embeddings = get_supabase(), get_embeddings()
    if not supabase or not embeddings:
        err = create_error_output("research", "TOOLS_UNAVAILABLE", "No hay conexión con la base de conocimientos")
        return {"worker_outputs": [err.model_dump()], "research_result": err.model_dump_json(), "events": events}

    user_query = _get_last_user_message(state)
    if not user_query:
        err = create_error_output("research", "NO_QUERY", "No se detectó una consulta del usuario")
        return {"worker_outputs": [err.model_dump()], "research_result": err.model_dump_json(), "events": events}

    from src.agent.utils.stream_utils import get_worker_stream
    stream = get_worker_stream(state, "research")

    logger.info("research_node", f"Query: {user_query[:100]}...")

    # - RAG Retrieval -
    try:
        rag_items, serialized_content = _run_rag_retrieval(user_query, supabase, embeddings, stream)
    except Exception as e:
        logger.error("research_node", f"RAG error: {e}")
        err = create_error_output("research", "RAG_ERROR", f"Error en búsqueda: {str(e)}")
        return {"worker_outputs": [err.model_dump()], "research_result": err.model_dump_json(), "events": events}

    # - First Synthesis (RAG only) -
    total_tokens = 0
    prior_context = _get_prior_context(state)
    web_items: List[EvidenceItem] = []

    try:
        llm = get_llm(state, temperature=0)
        stream.status("Sintetizando evidencia encontrada...")
        answer, llm_confidence, gaps, tokens = _synthesize(llm, user_query, rag_items, [], prior_context)
        total_tokens += tokens
    except Exception as e:
        logger.warning("research_node", f"Synthesis error: {e}")
        answer = serialized_content
        llm_confidence = 0.3
        gaps = []

    # - Web Fallback (if low confidence or no evidence) -
    rag_count = len(rag_items)
    initial_confidence = _compute_confidence(rag_count, 0, gaps, llm_confidence)

    if initial_confidence < WEB_SEARCH_CONFIDENCE_THRESHOLD or rag_count == 0:
        logger.info("research_node", f"Low confidence ({initial_confidence:.2f}), activating web fallback...")
        events.append(event_execute("research", "Buscando información complementaria en internet..."))

        try:
            web_items = _run_web_fallback(user_query, stream)

            if web_items:
                # Re-synthesize with combined evidence
                try:
                    stream.status("Re-sintetizando con evidencia combinada...")
                    answer, llm_confidence, gaps, tokens = _synthesize(
                        llm, user_query, rag_items, web_items, prior_context
                    )
                    total_tokens += tokens
                except Exception as e:
                    logger.warning("research_node", f"Combined synthesis error: {e}")
                    # Keep RAG-only synthesis
        except Exception as e:
            logger.warning("research_node", f"Web fallback error: {e}")

    # - Build Output -
    web_count = len(web_items)
    all_evidence = rag_items + web_items
    confidence = _compute_confidence(rag_count, web_count, gaps, llm_confidence)
    status = _compute_status(rag_count, web_count, confidence)

    # Summary message
    if all_evidence:
        parts = []
        if rag_count:
            parts.append(f"{rag_count} documentos internos")
        if web_count:
            parts.append(f"{web_count} fuentes web")
        summary = f"Encontré {' + '.join(parts)}"
    else:
        summary = "No encontré documentos relevantes"

    output = WorkerOutputBuilder.research(
        content=answer,
        evidence=[ev.model_dump() for ev in all_evidence],
        summary=summary,
        confidence=confidence,
        status=status,
        next_actions=[],
    )

    processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
    output.metadata.completed_at = datetime.utcnow().isoformat()
    output.metadata.processing_time_ms = processing_time
    output.metadata.model_used = state.get("llm_model") or os.getenv("DEFAULT_MODEL", "gemini-2.0-flash")
    output.metadata.tokens_used = total_tokens
    output.extra = {
        "gaps": gaps,
        "query": user_query[:200],
        "web_search_used": web_count > 0,
        "web_results_count": web_count,
    }

    # - Final logging -
    web_note = f" + {web_count} web" if web_count else ""
    logger.node_end("research_node", {
        "documents_found": len(all_evidence),
        "confidence": confidence,
        "status": status,
        "web_used": web_count > 0,
        "tokens": total_tokens,
    })
    events.append(event_report("research", f"Investigación completada ({rag_count} docs{web_note}, conf={confidence:.2f})"))

    return {
        "worker_outputs": [output.model_dump()],
        "research_result": output.model_dump_json(),
        "events": events,
        "token_usage": total_tokens,
    }