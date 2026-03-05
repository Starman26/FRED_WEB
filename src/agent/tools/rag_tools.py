"""
rag_tools.py - Herramientas RAG para búsqueda en documentos + Web Search (Tavily)

Adaptado al schema de Supabase:
- Tabla: document_chunks (id, doc_id, chunk_index, content, embedding, page_start, page_end, metadata)
- Tabla: documents (id, source, title, doc_type, pages_total, status, metadata)
- Función RPC: match_document_chunks(query_embedding, match_count, doc_type_filter)

Web Search:
- Usa Tavily API como fallback cuando RAG no encuentra evidencia suficiente
"""
import os
from typing import Any, List, Tuple, Optional
from langchain_core.documents import Document
from langchain_core.tools import tool


def make_retrieve_tool(supabase_client: Any, embeddings_model: Any):
    """
    Factory para crear el retrieve tool.
    
    Args:
        supabase_client: Cliente de Supabase
        embeddings_model: Modelo de embeddings (ej: OpenAIEmbeddings)
    
    Returns:
        Tool configurado para búsqueda RAG
    """
    
    @tool
    def retrieve_documents(
        query: str,
        match_count: int = 5,
        doc_type_filter: Optional[str] = None
    ) -> Tuple[str, List[Document]]:
        """
        Busca documentos relevantes en la base de conocimientos.
        
        Args:
            query: Consulta del usuario
            match_count: Número de chunks a retornar (default: 5)
            doc_type_filter: Filtro opcional por tipo de documento (ej: "paper", "manual")
            
        Returns:
            Tuple de (resumen textual, lista de Documents)
        """
        try:
            # 1. Generar embedding de la query
            query_embedding = embeddings_model.embed_query(query)
            
            # 2. Llamar a la función RPC match_document_chunks
            rpc_params = {
                "query_embedding": query_embedding,
                "match_count": match_count,
            }
            
            # Añadir filtro de doc_type si se especifica
            if doc_type_filter:
                rpc_params["doc_type_filter"] = doc_type_filter
            else:
                # Si la función requiere el parámetro, pasar None o string vacío
                rpc_params["doc_type_filter"] = None
            
            response = supabase_client.rpc(
                "match_document_chunks",
                rpc_params
            ).execute()
            
            if not response.data:
                return "No se encontraron documentos relevantes para tu consulta.", []
            
            # 3. Convertir resultados a Documents
            documents = []
            summaries = []
            
            for row in response.data:
                # Extraer campos del chunk
                chunk_id = row.get("id")
                doc_id = row.get("doc_id")
                content = row.get("content", "")
                page_start = row.get("page_start")
                page_end = row.get("page_end")
                chunk_index = row.get("chunk_index")
                chunk_metadata = row.get("metadata") or {}
                similarity = row.get("similarity", 0.0)
                
                # Extraer campos del documento padre (si vienen en el JOIN)
                # Nota: Esto depende de cómo esté configurada tu función RPC
                title = row.get("title") or row.get("doc_title") or chunk_metadata.get("title") or "Documento"
                source = row.get("source") or chunk_metadata.get("source") or ""
                doc_type = row.get("doc_type") or chunk_metadata.get("doc_type") or ""
                
                # Formatear páginas
                if page_start and page_end:
                    page_str = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
                elif page_start:
                    page_str = str(page_start)
                else:
                    page_str = "?"
                
                # Crear Document de LangChain
                doc = Document(
                    page_content=content,
                    metadata={
                        "chunk_id": str(chunk_id) if chunk_id else None,
                        "doc_id": str(doc_id) if doc_id else None,
                        "chunk_index": chunk_index,
                        "title": title,
                        "title_original": title,  # Para compatibilidad con research_node
                        "doc_title": title,
                        "source": source,
                        "doc_type": doc_type,
                        "page_start": page_start,
                        "page_end": page_end,
                        "relevance_score": similarity,
                        "extra": chunk_metadata,
                    }
                )
                documents.append(doc)
                
                # Crear línea de resumen
                summaries.append(f"- {title} (Pág. {page_str}) [score: {similarity:.2f}]")
            
            summary_text = f"Encontré {len(documents)} fragmentos relevantes:\n" + "\n".join(summaries)
            
            return summary_text, documents
            
        except Exception as e:
            error_msg = f"Error en búsqueda RAG: {str(e)}"
            print(f"[RAG_TOOL] {error_msg}")
            return error_msg, []
    
    return retrieve_documents


def make_retrieve_tool_with_join(supabase_client: Any, embeddings_model: Any):
    """
    Versión alternativa que hace JOIN manual con la tabla documents
    si tu función RPC no retorna los datos del documento padre.
    """
    
    @tool
    def retrieve_documents(
        query: str,
        match_count: int = 5,
        doc_type_filter: Optional[str] = None
    ) -> Tuple[str, List[Document]]:
        """Busca documentos relevantes con JOIN a tabla documents."""
        try:
            # 1. Generar embedding
            query_embedding = embeddings_model.embed_query(query)
            
            # 2. Llamar RPC
            rpc_params = {
                "query_embedding": query_embedding,
                "match_count": match_count,
                "doc_type_filter": doc_type_filter
            }
            
            response = supabase_client.rpc("match_document_chunks", rpc_params).execute()
            
            if not response.data:
                return "No se encontraron documentos relevantes.", []
            
            # 3. Obtener doc_ids únicos para hacer JOIN
            doc_ids = list(set(row.get("doc_id") for row in response.data if row.get("doc_id")))
            
            # 4. Obtener datos de documentos
            docs_data = {}
            if doc_ids:
                docs_response = supabase_client.table("documents").select(
                    "id, title, source, doc_type, metadata"
                ).in_("id", doc_ids).execute()
                
                for doc in docs_response.data or []:
                    docs_data[str(doc["id"])] = doc
            
            # 5. Combinar chunks con datos de documentos
            documents = []
            summaries = []
            
            for row in response.data:
                doc_id = str(row.get("doc_id", ""))
                doc_info = docs_data.get(doc_id, {})
                
                title = doc_info.get("title") or "Documento"
                source = doc_info.get("source") or ""
                doc_type = doc_info.get("doc_type") or ""
                
                page_start = row.get("page_start")
                page_end = row.get("page_end")
                page_str = str(page_start) if page_start == page_end else f"{page_start}-{page_end}" if page_start and page_end else "?"
                
                similarity = row.get("similarity", 0.0)
                
                doc = Document(
                    page_content=row.get("content", ""),
                    metadata={
                        "chunk_id": str(row.get("id", "")),
                        "doc_id": doc_id,
                        "chunk_index": row.get("chunk_index"),
                        "title": title,
                        "title_original": title,
                        "doc_title": title,
                        "source": source,
                        "doc_type": doc_type,
                        "page_start": page_start,
                        "page_end": page_end,
                        "relevance_score": similarity,
                    }
                )
                documents.append(doc)
                summaries.append(f"- {title} (Pág. {page_str}) [score: {similarity:.2f}]")
            
            summary_text = f"Encontré {len(documents)} fragmentos relevantes:\n" + "\n".join(summaries)
            return summary_text, documents
            
        except Exception as e:
            return f"Error en búsqueda RAG: {str(e)}", []
    
    return retrieve_documents


# ============================================
# HELPER: Verificar que la función RPC existe
# ============================================

def make_web_search_tool(max_results: int = 5):
    """
    Factory para crear el web search tool usando Tavily API.

    Se usa como fallback cuando RAG no encuentra evidencia suficiente
    o la confianza del resultado es baja.

    Args:
        max_results: Número máximo de resultados web (default: 5)

    Returns:
        Tool configurado para búsqueda web con Tavily
    """
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
        """
        Busca información en internet usando Tavily.
        Usar como respaldo cuando la base de conocimientos interna
        no tiene evidencia suficiente.

        Args:
            query: Consulta de búsqueda

        Returns:
            Tuple de (resumen textual, lista de Documents)
        """
        try:
            results = tavily.invoke(query)

            # tavily devuelve una lista de dicts o un string
            if isinstance(results, str):
                doc = Document(
                    page_content=results,
                    metadata={"source": "web_search", "title": "Resultado web", "source_type": "web"},
                )
                return results, [doc]

            documents = []
            summaries = []

            items = results if isinstance(results, list) else results.get("results", [])
            for item in items:
                title = item.get("title", "Resultado web") if isinstance(item, dict) else "Resultado web"
                url = item.get("url", "") if isinstance(item, dict) else ""
                content = item.get("content", str(item)) if isinstance(item, dict) else str(item)
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
            print(f"[WEB_SEARCH] {error_msg}")
            return error_msg, []

    return web_search


# ============================================
# HELPER: Verificar que la función RPC existe
# ============================================

def verify_rpc_function(supabase_client: Any) -> dict:
    """
    Verifica que la función match_document_chunks existe y funciona.
    
    Returns:
        Dict con status y mensaje
    """
    try:
        # Crear un embedding de prueba (vector de ceros)
        test_embedding = [0.0] * 1536  # Dimensión de text-embedding-3-small
        
        response = supabase_client.rpc(
            "match_document_chunks",
            {
                "query_embedding": test_embedding,
                "match_count": 1,
                "doc_type_filter": None
            }
        ).execute()
        
        return {
            "status": "ok",
            "message": f"Función RPC funciona. Retornó {len(response.data or [])} resultados.",
            "sample": response.data[0] if response.data else None
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error verificando RPC: {str(e)}",
            "sample": None
        }
