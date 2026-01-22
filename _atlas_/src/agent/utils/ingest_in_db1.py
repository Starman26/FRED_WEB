"""
ingest_in_db1.py - Script de ingesta de PDFs a Supabase

CAMBIOS PRINCIPALES:
1. Normalización de títulos consistente (documentada)
2. Guarda title_original en metadata para citas legibles
3. Mejor manejo de errores
4. Logging detallado
"""

import os
import re
from dotenv import load_dotenv
from pathlib import Path
from collections import defaultdict
import unicodedata
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from supabase.client import Client, create_client


# ----------------------------
# Utils
# ----------------------------
def safe_strip(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def normalize_title(raw: str) -> str:
    """
    Normaliza el título para lookup estable:
    - lowercase
    - sin acentos (NFKD)
    - solo a-z0-9 (sin espacios, guiones, símbolos)
    
    IMPORTANTE: Esta función debe ser IDÉNTICA a la de rag_tools.py
    
    Ejemplos:
      "Agent_Cora" -> "agentcora"
      "FrEDie-Paper" -> "frediepaper"
      "Data Driven Learning Factories" -> "datadrivenlearningfactories"
    """
    s = (raw or "").strip().lower()

    # Quitar acentos usando NFKD
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # Dejar solo letras y números
    s = re.sub(r"[^a-z0-9]+", "", s)

    return s or "untitled"


def format_title_for_display(raw: str) -> str:
    """
    Formatea el título para mostrar al usuario (citas).
    Mantiene espacios y capitalización razonable.
    
    Ejemplos:
      "agent_cora" -> "Agent Cora"
      "Data_Driven_Learning_Factories" -> "Data Driven Learning Factories"
    """
    s = (raw or "").strip()
    # Reemplazar guiones bajos y guiones por espacios
    s = re.sub(r"[_\-]+", " ", s)
    # Capitalizar cada palabra
    s = " ".join(word.capitalize() for word in s.split())
    return s or "Documento"


def batched(seq, n=200):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ----------------------------
# Main
# ----------------------------
def main():
    # Configuración de rutas
    # Este archivo está en src/agent/utils/ingest_in_db1.py
    # Subimos 3 niveles para llegar a la raíz del proyecto
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    ENV_PATH = PROJECT_ROOT / ".env"
    PDF_DIR = PROJECT_ROOT / "documents"

    print(f"📁 Directorio del proyecto: {PROJECT_ROOT}")
    print(f"📁 Directorio de PDFs: {PDF_DIR}")

    load_dotenv(ENV_PATH)

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url or not supabase_key:
        raise RuntimeError("Faltan SUPABASE_URL o SUPABASE_SERVICE_KEY en .env")

    print(f"🔗 Conectando a Supabase: {supabase_url[:30]}...")
    supabase: Client = create_client(supabase_url, supabase_key)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("✅ Embeddings inicializados (text-embedding-3-small)")

    if not PDF_DIR.exists():
        raise RuntimeError(f"No existe la carpeta PDFs: {PDF_DIR}")

    # 1) Carga PDFs desde /documents
    print(f"\n📄 Cargando PDFs desde {PDF_DIR}...")
    loader = PyPDFDirectoryLoader(str(PDF_DIR))
    pages = loader.load()
    print(f"   Páginas cargadas: {len(pages)}")

    # 2) Agrupa páginas por archivo
    by_source = defaultdict(list)
    for p in pages:
        raw_source = p.metadata.get("source", "unknown")
        filename = Path(raw_source).name
        source_key = f"documents/{filename}"
        by_source[source_key].append(p)

    print(f"   Archivos únicos: {len(by_source)}")

    # 3) Procesa cada PDF
    for source, pdf_pages in by_source.items():
        print(f"\n{'='*60}")
        print(f"📖 Procesando: {source}")
        
        pdf_pages = sorted(pdf_pages, key=lambda d: d.metadata.get("page", 0))
        pages_total = len(pdf_pages)
        print(f"   Páginas: {pages_total}")

        # Chunking según tamaño
        if pages_total <= 10:
            chunk_size, chunk_overlap = 700, 120
        else:
            chunk_size, chunk_overlap = 1000, 100

        # Extraer título original del nombre del archivo
        original_title = Path(source).stem
        display_title = format_title_for_display(original_title)
        normalized_title = normalize_title(original_title)
        
        print(f"   Título original: {original_title}")
        print(f"   Título display: {display_title}")
        print(f"   Título normalizado: {normalized_title}")

        # 4) Upsert documents
        doc_payload = {
            "source": source,
            "title": normalized_title,  # Para búsquedas
            "doc_type": "paper",
            "pages_total": pages_total,
            "status": "ready",
            "metadata": {
                "title_original": display_title,  # Para mostrar al usuario
                "title_raw": original_title,  # El nombre exacto del archivo
                "chunking": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
            },
        }

        try:
            upsert_res = (
                supabase.table("documents")
                .upsert(doc_payload, on_conflict="doc_type,title")
                .execute()
            )
            if not upsert_res.data:
                raise RuntimeError(f"No se pudo upsert documents para: {source}")

            doc_id = upsert_res.data[0]["id"]
            print(f"   ✅ Documento guardado: {doc_id}")
            
        except Exception as e:
            print(f"   ❌ Error en upsert: {e}")
            continue

        # 5) Limpia chunks previos
        try:
            supabase.table("document_chunks").delete().eq("doc_id", doc_id).execute()
            print(f"   🗑️ Chunks previos eliminados")
        except Exception as e:
            print(f"   ⚠️ Error eliminando chunks previos: {e}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        rows = []
        chunk_global_idx = 0

        # 6) Chunkear POR PÁGINA
        for page_doc in pdf_pages:
            page_num = int(page_doc.metadata.get("page", 0)) + 1
            page_text = safe_strip(page_doc.page_content)

            if not page_text:
                continue

            page_chunks = splitter.split_text(page_text)
            
            if not page_chunks:
                continue
                
            try:
                page_embeddings = embeddings.embed_documents(page_chunks)
            except Exception as e:
                print(f"   ⚠️ Error generando embeddings para página {page_num}: {e}")
                continue

            for i, chunk_text in enumerate(page_chunks):
                content = f"[PAGE {page_num}]\n{chunk_text}"

                rows.append({
                    "doc_id": doc_id,
                    "chunk_index": chunk_global_idx,
                    "content": content,
                    "embedding": page_embeddings[i],
                    "page_start": page_num,
                    "page_end": page_num,
                    "metadata": {
                        "source": source,
                        "page": page_num,
                        "pages_total": pages_total,
                        "title_original": display_title,  # Para citas
                    },
                })
                chunk_global_idx += 1

        # 7) Insert chunks
        if rows:
            for batch in batched(rows, n=200):
                try:
                    supabase.table("document_chunks").insert(batch).execute()
                except Exception as e:
                    print(f"   ❌ Error insertando batch: {e}")

            print(f"   ✅ Chunks insertados: {len(rows)}")
        else:
            print(f"   ⚠️ No se generaron chunks")

    print(f"\n{'='*60}")
    print("🎉 Ingesta completada")


if __name__ == "__main__":
    main()