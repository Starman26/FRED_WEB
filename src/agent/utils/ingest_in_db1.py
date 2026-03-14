"""
ingest_in_db1.py

PDF ingestion script: loads PDFs from /documents, chunks per page,
generates embeddings, and upserts into Supabase.
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


def safe_strip(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def normalize_title(raw: str) -> str:
    """Normalize title for stable lookup: lowercase, no accents, alphanumeric only.
    Must stay in sync with rag_tools.py."""
    s = (raw or "").strip().lower()

    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    s = re.sub(r"[^a-z0-9]+", "", s)

    return s or "untitled"


def format_title_for_display(raw: str) -> str:
    """Format title for user-facing display (citations)."""
    s = (raw or "").strip()
    s = re.sub(r"[_\-]+", " ", s)
    s = " ".join(word.capitalize() for word in s.split())
    return s or "Documento"


def batched(seq, n=200):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def main():
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

    print(f"\n📄 Cargando PDFs desde {PDF_DIR}...")
    loader = PyPDFDirectoryLoader(str(PDF_DIR))
    pages = loader.load()
    print(f"   Páginas cargadas: {len(pages)}")

    by_source = defaultdict(list)
    for p in pages:
        raw_source = p.metadata.get("source", "unknown")
        filename = Path(raw_source).name
        source_key = f"documents/{filename}"
        by_source[source_key].append(p)

    print(f"   Archivos únicos: {len(by_source)}")

    for source, pdf_pages in by_source.items():
        print(f"\n{'='*60}")
        print(f"📖 Procesando: {source}")
        
        pdf_pages = sorted(pdf_pages, key=lambda d: d.metadata.get("page", 0))
        pages_total = len(pdf_pages)
        print(f"   Páginas: {pages_total}")

        if pages_total <= 10:
            chunk_size, chunk_overlap = 700, 120
        else:
            chunk_size, chunk_overlap = 1000, 100

        original_title = Path(source).stem
        display_title = format_title_for_display(original_title)
        normalized_title = normalize_title(original_title)
        
        print(f"   Título original: {original_title}")
        print(f"   Título display: {display_title}")
        print(f"   Título normalizado: {normalized_title}")

        doc_payload = {
            "source": source,
            "title": normalized_title,
            "doc_type": "manual",
            "pages_total": pages_total,
            "status": "ready",
            "metadata": {
                "title_original": display_title,
                "title_raw": original_title,
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

        try:
            supabase.table("document_chunks").delete().eq("doc_id", doc_id).execute()
            print(f"   🗑️ Chunks previos eliminados")
        except Exception as e:
            print(f"   ⚠️ Error eliminando chunks previos: {e}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        rows = []
        chunk_global_idx = 0

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
                        "title_original": display_title,
                    },
                })
                chunk_global_idx += 1

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