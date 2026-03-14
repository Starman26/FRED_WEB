"""
rag_tools.py - Unified RAG tools for document search.

Factories:
- make_retrieve_tool()          - general search across all documents
- make_equipment_manual_tool()  - scoped search within equipment manuals
- make_web_search_tool()        - web fallback via Tavily
"""

import os
import logging
from typing import Any, List, Tuple, Optional
from langchain_core.documents import Document
from langchain_core.tools import tool

logger = logging.getLogger("rag_tools")


_TERM_MAP = {
    "diagnóstico": "diagnostic troubleshooting",
    "diagnostico": "diagnostic troubleshooting",
    "comunicación": "communication PROFINET",
    "comunicacion": "communication PROFINET",
    "error": "error fault code",
    "conexión": "connection online going-online",
    "conexion": "connection online going-online",
    "descargar": "download upload",
    "subir código": "download upload program",
    "subir codigo": "download upload program",
    "cargar": "download upload load program",
    "código": "program code",
    "codigo": "program code",
    "estado": "status state LED indicator",
    "configuración": "configuration setup parameters",
    "configuracion": "configuration setup parameters",
    "red": "network Ethernet PROFINET IP",
    "puerta": "door safety interlock",
    "cobot": "robot cobot collaborative",
    "sensor": "sensor input signal",
    "motor": "motor drive actuator",
    "parada": "stop emergency halt",
    "arranque": "start startup power-on",
    "memoria": "memory load capacity",
    "firmware": "firmware update version",
    "módulo": "module expansion",
    "modulo": "module expansion",
    "alimentación": "power supply voltage",
    "alimentacion": "power supply voltage",
    "dirección ip": "IP address assignment",
    "direccion ip": "IP address assignment",
}


def _expand_query(query: str) -> str:
    """Expand a query with English terms for cross-language matching."""
    query_lower = query.lower()
    expansions = []
    for es_term, en_terms in _TERM_MAP.items():
        if es_term in query_lower:
            expansions.append(en_terms)
    if expansions:
        expanded = f"{query} {' '.join(expansions)}"
        logger.debug(f"Query expanded: '{query}' -> '{expanded[:120]}...'")
        return expanded
    return query


def _format_page_ref(page_start, page_end) -> str:
    if page_start and page_end:
        return str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
    if page_start:
        return str(page_start)
    return "?"


def _chunk_to_document(row: dict, doc_info: Optional[dict] = None) -> Document:
    """Convert an RPC result row to a LangChain Document."""
    chunk_metadata = row.get("metadata") or {}

    if doc_info:
        title = doc_info.get("title") or "Documento"
        source = doc_info.get("source") or ""
        doc_type = doc_info.get("doc_type") or ""
    else:
        title = (
            row.get("title")
            or row.get("doc_title")
            or chunk_metadata.get("title")
            or "Documento"
        )
        source = row.get("source") or chunk_metadata.get("source") or ""
        doc_type = row.get("doc_type") or chunk_metadata.get("doc_type") or ""

    return Document(
        page_content=row.get("content", ""),
        metadata={
            "chunk_id": str(row.get("id", "")),
            "doc_id": str(row.get("doc_id", "")),
            "chunk_index": row.get("chunk_index"),
            "title": title,
            "title_original": title,
            "doc_title": title,
            "source": source,
            "doc_type": doc_type,
            "page_start": row.get("page_start"),
            "page_end": row.get("page_end"),
            "relevance_score": row.get("similarity", 0.0),
            "extra": chunk_metadata,
        },
    )


def make_retrieve_tool(supabase_client: Any, embeddings_model: Any):
    """Factory: retrieve tool that searches all documents."""

    @tool
    def retrieve_documents(
        query: str,
        match_count: int = 5,
        doc_type_filter: Optional[str] = None,
    ) -> Tuple[str, List[Document]]:
        """Search for relevant documents in the knowledge base."""
        try:
            query_embedding = embeddings_model.embed_query(query)

            rpc_params = {
                "query_embedding": query_embedding,
                "match_count": match_count,
                "doc_type_filter": doc_type_filter,
            }

            response = supabase_client.rpc(
                "match_document_chunks", rpc_params
            ).execute()

            if not response.data:
                return "No se encontraron documentos relevantes para tu consulta.", []

            documents = []
            summaries = []

            for row in response.data:
                doc = _chunk_to_document(row)
                documents.append(doc)

                page_str = _format_page_ref(row.get("page_start"), row.get("page_end"))
                similarity = row.get("similarity", 0.0)
                summaries.append(
                    f"- {doc.metadata['title']} (Pág. {page_str}) [score: {similarity:.2f}]"
                )

            summary_text = (
                f"Encontré {len(documents)} fragmentos relevantes:\n"
                + "\n".join(summaries)
            )
            return summary_text, documents

        except Exception as e:
            error_msg = f"Error en búsqueda RAG: {str(e)}"
            logger.error(error_msg)
            return error_msg, []

    return retrieve_documents


def make_retrieve_tool_with_join(supabase_client: Any, embeddings_model: Any):
    """Factory: retrieve tool with manual JOIN to documents table."""

    @tool
    def retrieve_documents(
        query: str,
        match_count: int = 5,
        doc_type_filter: Optional[str] = None,
    ) -> Tuple[str, List[Document]]:
        """Search for relevant documents with JOIN to documents table."""
        try:
            query_embedding = embeddings_model.embed_query(query)

            response = supabase_client.rpc("match_document_chunks", {
                "query_embedding": query_embedding,
                "match_count": match_count,
                "doc_type_filter": doc_type_filter,
            }).execute()

            if not response.data:
                return "No se encontraron documentos relevantes.", []

            doc_ids = list(set(
                row.get("doc_id") for row in response.data if row.get("doc_id")
            ))
            docs_data = {}
            if doc_ids:
                docs_response = supabase_client.table("documents").select(
                    "id, title, source, doc_type, metadata"
                ).in_("id", doc_ids).execute()
                for d in docs_response.data or []:
                    docs_data[str(d["id"])] = d

            documents = []
            summaries = []

            for row in response.data:
                doc_info = docs_data.get(str(row.get("doc_id", "")))
                doc = _chunk_to_document(row, doc_info=doc_info)
                documents.append(doc)

                page_str = _format_page_ref(row.get("page_start"), row.get("page_end"))
                similarity = row.get("similarity", 0.0)
                summaries.append(
                    f"- {doc.metadata['title']} (Pág. {page_str}) [score: {similarity:.2f}]"
                )

            summary_text = (
                f"Encontré {len(documents)} fragmentos relevantes:\n"
                + "\n".join(summaries)
            )
            return summary_text, documents

        except Exception as e:
            return f"Error en búsqueda RAG: {str(e)}", []

    return retrieve_documents


def make_equipment_manual_tool(
    supabase_client: Any,
    embeddings_model: Any,
    equipment_id: str,
    doc_ids: List[str],
):
    """Factory: search tool scoped to an equipment's manuals.

    Uses ES->EN query expansion and low threshold for cross-language matching.
    """

    if not doc_ids:

        @tool
        def search_equipment_manual(query: str) -> str:
            """Search equipment manual for troubleshooting information.
            No manuals are linked to this equipment."""
            return (
                "No manuals linked to this equipment. "
                "Provide general troubleshooting guidance."
            )

        return search_equipment_manual

    @tool
    def search_equipment_manual(query: str, num_results: int = 5) -> str:
        """Search the equipment's technical manual for troubleshooting information.

        Use this to find specific error codes, diagnostic procedures, wiring diagrams,
        configuration steps, and troubleshooting guides from the official manual.

        IMPORTANT: Queries should be in English (manuals are in English).

        Args:
            query: What to search for in English (e.g. "communication error LED indicators",
                   "download program CPU")
            num_results: Number of results to return (default 5)
        """
        try:
            has_non_ascii = any(ord(c) > 127 for c in query)
            has_spanish_patterns = any(w in query.lower() for w in [
                "como", "cómo", "qué", "que", "por qué", "donde", "dónde",
                "hay", "puedo", "tiene", "está", "esta", "los", "las", "del",
                "para", "sobre", "cuando", "cuándo",
            ])
            if has_non_ascii or has_spanish_patterns:
                try:
                    from src.agent.utils.llm_factory import get_llm_from_name
                    translator = get_llm_from_name("gpt-4o-mini", temperature=0, max_tokens=100)
                    from langchain_core.messages import HumanMessage as _HM
                    translated = translator.invoke([_HM(
                        content=(
                            "Translate this search query to English for a technical equipment manual. "
                            "Return ONLY the translated query, nothing else. "
                            "Use precise technical terms.\n\n"
                            f"Query: {query}"
                        )
                    )])
                    translated_query = translated.content.strip().strip('"\'')
                    if translated_query and len(translated_query) > 3:
                        logger.info(f"Query translated: '{query}' -> '{translated_query}'")
                        query = translated_query
                except Exception as te:
                    logger.warning(f"Translation failed, using original + expansion: {te}")

            expanded_query = _expand_query(query)
            query_embedding = embeddings_model.embed_query(expanded_query)

            result = supabase_client.rpc("match_document_chunks", {
                "query_embedding": query_embedding,
                "match_count": num_results + 3,
                "doc_ids": doc_ids,
            }).execute()

            if not result.data:
                return f"No relevant sections found in the manual for: '{query}'"

            sections = []
            for chunk in result.data:
                similarity = chunk.get("similarity", 0)
                if similarity < 0.15:
                    continue

                title = chunk.get("doc_title", "Manual")
                page_start = chunk.get("page_start", "?")
                page_end = chunk.get("page_end", "?")
                content = chunk.get("content", "")

                if len(content) > 1500:
                    content = content[:1500] + "..."

                page_ref = (
                    f"p.{page_start}"
                    if page_start == page_end
                    else f"pp.{page_start}-{page_end}"
                )
                sections.append(
                    f"### {title} ({page_ref}) [relevance: {similarity:.2f}]\n{content}"
                )

                if len(sections) >= num_results:
                    break

            if not sections:
                return (
                    f"No relevant sections found for: '{query}'. "
                    "Try searching with different English keywords."
                )

            logger.info(f"Manual search: query='{query}', results={len(sections)}")
            return "\n\n---\n\n".join(sections)

        except Exception as e:
            return f"Error searching manual: {str(e)}"

    return search_equipment_manual


def make_web_search_tool(max_results: int = 5):
    """Factory: web search tool via Tavily, used as fallback when RAG confidence is low."""
    from langchain_tavily import TavilySearch

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY no está configurada en las variables de entorno")

    tavily = TavilySearch(
        max_results=max_results,
        topic="general",
        include_raw_content=False,
    )

    @tool
    def web_search(query: str) -> Tuple[str, List[Document]]:
        """Search the web via Tavily as a fallback for the internal knowledge base."""
        try:
            results = tavily.invoke(query)

            if isinstance(results, str):
                doc = Document(
                    page_content=results,
                    metadata={
                        "source": "web_search",
                        "title": "Resultado web",
                        "source_type": "web",
                    },
                )
                return results, [doc]

            documents = []
            summaries = []

            items = (
                results if isinstance(results, list) else results.get("results", [])
            )
            for item in items:
                title = (
                    item.get("title", "Resultado web")
                    if isinstance(item, dict)
                    else "Resultado web"
                )
                url = item.get("url", "") if isinstance(item, dict) else ""
                content = (
                    item.get("content", str(item))
                    if isinstance(item, dict)
                    else str(item)
                )
                score = item.get("score", 0.0) if isinstance(item, dict) else 0.0

                doc = Document(
                    page_content=content,
                    metadata={
                        "title": title,
                        "title_original": title,
                        "doc_title": title,
                        "source": url,
                        "source_type": "web",
                        "relevance_score": score,
                    },
                )
                documents.append(doc)
                summaries.append(f"- {title} [web] [score: {score:.2f}]")

            summary_text = (
                f"Encontré {len(documents)} resultados web:\n" + "\n".join(summaries)
                if documents
                else "No se encontraron resultados web relevantes."
            )
            return summary_text, documents

        except Exception as e:
            error_msg = f"Error en búsqueda web: {str(e)}"
            logger.error(error_msg)
            return error_msg, []

    return web_search


def verify_rpc_function(supabase_client: Any) -> dict:
    """Verify that the match_document_chunks RPC exists and works."""
    try:
        test_embedding = [0.0] * 1536  # text-embedding-3-small dimension
        response = supabase_client.rpc(
            "match_document_chunks",
            {
                "query_embedding": test_embedding,
                "match_count": 1,
                "doc_type_filter": None,
            },
        ).execute()
        return {
            "status": "ok",
            "message": f"RPC ok. Retornó {len(response.data or [])} resultados.",
            "sample": response.data[0] if response.data else None,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error verificando RPC: {str(e)}",
            "sample": None,
        }