"""
learning_profile.py - Consulta el perfil de aprendizaje del usuario desde Supabase.

El tutor_node llama a get_learning_prompt_section() SOLO cuando necesita
adaptar una explicación. No se inyecta en cada request.
"""
from typing import Dict, Optional
from src.agent.services import get_supabase
from src.agent.utils.logger import logger

# Cache en memoria para no consultar en cada llamada
_profile_cache: Dict[str, str] = {}


def get_user_learning_profile(user_id: Optional[str] = None) -> str:
    """
    Obtiene el perfil de aprendizaje como texto descriptivo.
    
    Returns:
        String descriptivo o vacío si no hay perfil.
    """
    if not user_id:
        return ""
    
    if user_id in _profile_cache:
        return _profile_cache[user_id]
    
    try:
        supabase = get_supabase()
        if not supabase:
            return ""
        
        resp = supabase.rpc("get_learning_profile", {"p_user_id": user_id}).execute()
        data = resp.data
        
        if not data:
            return ""
        
        if isinstance(data, list):
            data = data[0] if data else {}
        
        profile_text = data.get("learning_profile_text", "") or ""
        
        if profile_text:
            _profile_cache[user_id] = profile_text
        
        return profile_text
        
    except Exception as e:
        logger.error("learning_profile", f"Error obteniendo perfil: {e}")
        return ""


def get_learning_prompt_section(user_id: Optional[str] = None) -> str:
    """
    Devuelve sección de prompt lista para inyectar en el tutor.
    """
    profile_text = get_user_learning_profile(user_id)
    
    if not profile_text:
        return ""
    
    return f"""
PERFIL DE APRENDIZAJE DEL USUARIO:
{profile_text}

Adapta tu explicación a estas preferencias. No menciones que conoces su perfil."""


def clear_cache(user_id: Optional[str] = None):
    """Limpia cache. Llamar si el usuario cambia en el sidebar."""
    if user_id:
        _profile_cache.pop(user_id, None)
    else:
        _profile_cache.clear()
