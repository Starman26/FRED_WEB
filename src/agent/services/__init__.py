"""
services/ - Registry extensible de servicios

Reemplaza el antiguo services.py (singletons hardcodeados) con un registry
que permite agregar nuevos servicios sin modificar código existente.

Uso:
    from src.agent.services import get_supabase, get_embeddings, get_xarm, ServiceRegistry

Agregar un nuevo servicio:
    ServiceRegistry.register("mi_servicio", lambda: MiServicio(...))
    mi_srv = ServiceRegistry.get("mi_servicio")
"""
import os
from typing import Dict, Any, Optional, Callable


class ServiceRegistry:
    """
    Registry central de servicios con lazy initialization.
    
    Los servicios se registran con una factory function y se instancian
    solo cuando se necesitan por primera vez.
    """
    _services: Dict[str, Any] = {}
    _factories: Dict[str, Callable] = {}
    _errors: Dict[str, str] = {}
    
    @classmethod
    def register(cls, name: str, factory: Callable):
        """Registra una factory para un servicio."""
        cls._factories[name] = factory
    
    @classmethod
    def get(cls, name: str) -> Optional[Any]:
        """Obtiene un servicio (lazy init en primera llamada)."""
        if name in cls._services:
            return cls._services[name]
        
        factory = cls._factories.get(name)
        if not factory:
            return None
        
        try:
            instance = factory()
            cls._services[name] = instance
            cls._errors.pop(name, None)
            return instance
        except Exception as e:
            cls._errors[name] = str(e)
            print(f"[ServiceRegistry] Error inicializando '{name}': {e}")
            return None
    
    @classmethod
    def is_available(cls, name: str) -> bool:
        """Verifica si un servicio está registrado (no necesariamente inicializado)."""
        return name in cls._factories
    
    @classmethod
    def is_ready(cls, name: str) -> bool:
        """Verifica si un servicio está inicializado y listo."""
        return name in cls._services
    
    @classmethod
    def status(cls) -> Dict[str, Dict[str, Any]]:
        """Retorna estado de todos los servicios registrados."""
        result = {}
        for name in cls._factories:
            result[name] = {
                "registered": True,
                "initialized": name in cls._services,
                "error": cls._errors.get(name),
            }
        return result
    
    @classmethod
    def status_summary(cls) -> Dict[str, bool]:
        """Versión simplificada: {nombre: está_listo}."""
        return {name: name in cls._services for name in cls._factories}
    
    @classmethod
    def init_all(cls) -> Dict[str, bool]:
        """Inicializa todos los servicios registrados. Retorna resultado."""
        results = {}
        for name in cls._factories:
            results[name] = cls.get(name) is not None
        return results
    
    @classmethod
    def reset(cls, name: Optional[str] = None):
        """Resetea un servicio específico o todos."""
        if name:
            cls._services.pop(name, None)
            cls._errors.pop(name, None)
        else:
            cls._services.clear()
            cls._errors.clear()


# ============================================
# FACTORY FUNCTIONS
# ============================================

def _create_supabase():
    from supabase import create_client
    url = os.getenv("SUPABASE_URL")
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_SERVICE_KEY")
        or os.getenv("SUPABASE_KEY")
    )
    if not url or not key:
        raise ValueError("Faltan variables SUPABASE_URL o SUPABASE_*_KEY")
    client = create_client(url, key)
    print(f"[services] Supabase conectado: {url[:40]}...")
    return client


def _create_embeddings():
    from langchain_openai import OpenAIEmbeddings
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Falta OPENAI_API_KEY para embeddings")
    model = OpenAIEmbeddings(model="text-embedding-3-small")
    print("[services] Embeddings inicializados (text-embedding-3-small)")
    return model


def _create_xarm():
    from src.agent.tools.robot_tools.xarm_client import XArmClient
    ip = os.getenv("XARM_IP", "192.168.1.203")
    if os.getenv("XARM_ENABLED", "false").lower() != "true":
        raise ValueError("XARM_ENABLED no está activo")
    client = XArmClient(ip)
    print(f"[services] xArm client creado para {ip}")
    return client


def _create_elevenlabs():
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY not set")
    from src.agent.services.voice_service import VoiceService
    voice_id = os.getenv("ELEVENLABS_VOICE_ID") or None
    model_id = os.getenv("ELEVENLABS_MODEL_ID") or None
    return VoiceService(api_key=api_key, voice_id=voice_id, model_id=model_id)


# ============================================
# REGISTRAR SERVICIOS
# ============================================
ServiceRegistry.register("supabase", _create_supabase)
ServiceRegistry.register("embeddings", _create_embeddings)
ServiceRegistry.register("xarm", _create_xarm)
ServiceRegistry.register("elevenlabs", _create_elevenlabs)


# ============================================
# HELPERS DE CONVENIENCIA (backward-compatible)
# ============================================

def get_supabase() -> Optional[Any]:
    """Obtiene el cliente de Supabase."""
    return ServiceRegistry.get("supabase")

def get_embeddings() -> Optional[Any]:
    """Obtiene el modelo de embeddings."""
    return ServiceRegistry.get("embeddings")

def get_xarm():
    """Obtiene el cliente del xArm."""
    return ServiceRegistry.get("xarm")

def get_elevenlabs():
    """Obtiene el servicio de ElevenLabs TTS (lazy init, None si no configurado)."""
    return ServiceRegistry.get("elevenlabs")

def init_services() -> dict:
    """Inicializa servicios core y retorna estado (backward-compatible)."""
    supabase = get_supabase()
    embeddings = get_embeddings()
    # xArm se inicializa on-demand, no al arrancar
    return {
        "supabase_connected": supabase is not None,
        "embeddings_ready": embeddings is not None,
    }

def get_services_status() -> dict:
    """Retorna el estado actual de los servicios (backward-compatible)."""
    status = ServiceRegistry.status_summary()
    return {
        "supabase_connected": status.get("supabase", False),
        "embeddings_ready": status.get("embeddings", False),
        "xarm_ready": status.get("xarm", False),
        "initialized": any(status.values()),
    }

def reset_services():
    """Resetea todos los servicios."""
    ServiceRegistry.reset()
