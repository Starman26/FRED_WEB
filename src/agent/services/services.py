"""
services.py - Singleton clients (not serializable, must not be stored in state).
"""
import os
from typing import Optional, Any

_supabase_client: Optional[Any] = None
_embeddings_model: Optional[Any] = None
_initialized: bool = False


def get_supabase() -> Optional[Any]:
    """Get or initialize the Supabase client singleton."""
    global _supabase_client
    
    if _supabase_client is None:
        url = os.getenv("SUPABASE_URL")
        key = (
            os.getenv("SUPABASE_SERVICE_ROLE_KEY") 
            or os.getenv("SUPABASE_SERVICE_KEY")
            or os.getenv("SUPABASE_KEY")
        )
        
        if url and key:
            try:
                from supabase import create_client
                _supabase_client = create_client(url, key)
                print(f"[services] Supabase conectado: {url[:40]}...")
            except Exception as e:
                print(f"[services] Error conectando Supabase: {e}")
                return None
        else:
            print("[services] Faltan variables SUPABASE_URL o SUPABASE_KEY")
            return None
    
    return _supabase_client


def get_embeddings() -> Optional[Any]:
    """Get or initialize the embeddings model singleton."""
    global _embeddings_model
    
    if _embeddings_model is None:
        if os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_openai import OpenAIEmbeddings
                _embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
                print("[services] Embeddings inicializados (text-embedding-3-small)")
            except Exception as e:
                print(f"[services] Error inicializando embeddings: {e}")
                return None
        else:
            print("[services] Falta OPENAI_API_KEY para embeddings")
            return None
    
    return _embeddings_model


def init_services() -> dict:
    """Initialize all services and return their status."""
    global _initialized
    
    supabase = get_supabase()
    embeddings = get_embeddings()
    
    _initialized = True
    
    return {
        "supabase_connected": supabase is not None,
        "embeddings_ready": embeddings is not None,
    }


def get_services_status() -> dict:
    """Return current service status."""
    return {
        "supabase_connected": _supabase_client is not None,
        "embeddings_ready": _embeddings_model is not None,
        "initialized": _initialized,
    }


def reset_services():
    """Reset all service singletons (for testing)."""
    global _supabase_client, _embeddings_model, _initialized
    _supabase_client = None
    _embeddings_model = None
    _initialized = False
