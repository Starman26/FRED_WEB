"""
bootstrap.py - Inicialización del estado y verificación de herramientas

CAMBIOS PRINCIPALES:
1. Usa STATE_DEFAULTS de state.py para consistencia
2. Inicializa campos de orchestration multi-step
3. NO inyecta supabase/embeddings al state (no son serializables)
4. Usa services.py para inicializar clientes
"""
import os
from typing import Dict, Any, List

from src.agent.state import AgentState, STATE_DEFAULTS
from src.agent.services import init_services, get_services_status
from src.agent.utils.run_events import event_read, event_error
from src.agent.utils.logger import logger


def _as_event_list(evt) -> List[Dict[str, Any]]:
    """Helper para normalizar eventos a lista"""
    if evt is None:
        return []
    if isinstance(evt, list):
        return evt
    return [evt]


def bootstrap_node(state: AgentState) -> Dict[str, Any]:
    """
    Bootstrap node: inicializa campos faltantes y verifica herramientas.
    
    Este nodo se ejecuta PRIMERO en el grafo y asegura que:
    1. Todos los campos del state tengan valores válidos
    2. Supabase y embeddings estén disponibles (via services)
    3. Los campos de orchestration estén listos
    
    NOTA: supabase y embeddings NO se guardan en el state porque no son
    serializables. Se acceden via src.agent.services.
    
    Returns:
        Dict con patches al state
    """
    logger.node_start("bootstrap", {"action": "initializing_state"})
    
    patches: Dict[str, Any] = {}
    events: List[Dict[str, Any]] = []
    
    # ==========================================
    # 1. Asegurar canal de eventos
    # ==========================================
    if "events" not in state or state.get("events") is None:
        patches["events"] = []
    
    # ==========================================
    # 2. Inicializar campos faltantes con DEFAULTS
    # ==========================================
    for key, default_value in STATE_DEFAULTS.items():
        current_value = state.get(key)
        
        # Inicializar si falta o es None
        if current_value is None:
            patches[key] = default_value
            logger.debug("bootstrap", f"Inicializando {key} con valor por defecto")
    
    # ==========================================
    # 3. INICIALIZAR SERVICIOS (no se guardan en state)
    # ==========================================
    services_status = init_services()
    
    if services_status["supabase_connected"]:
        events.append(event_read("bootstrap", "✅ Supabase conectado"))
    else:
        events.append(event_error("bootstrap", "⚠️ Supabase no disponible"))
    
    if services_status["embeddings_ready"]:
        events.append(event_read("bootstrap", "✅ Embeddings listos"))
    else:
        events.append(event_error("bootstrap", "⚠️ Embeddings no disponibles"))
    
    # ==========================================
    # 4. Inicializar orchestration si no existe
    # ==========================================
    if state.get("orchestration_plan") is None:
        patches["orchestration_plan"] = []
    
    if state.get("current_step") is None:
        patches["current_step"] = 0
    
    if state.get("worker_outputs") is None:
        patches["worker_outputs"] = []
    
    if state.get("pending_context") is None:
        patches["pending_context"] = {}
    
    # ==========================================
    # 5. Registrar evento de inicio
    # ==========================================
    bootstrap_event = event_read("bootstrap", "🚀 Estado inicializado correctamente")
    events.append(bootstrap_event)
    
    # Añadir eventos al patch
    if "events" in patches:
        patches["events"].extend(events)
    else:
        patches["events"] = events
    
    logger.node_end("bootstrap", {
        "status": "success",
        "patches_count": len(patches),
        "supabase_connected": services_status["supabase_connected"],
        "embeddings_ready": services_status["embeddings_ready"]
    })
    
    return patches


def get_bootstrap_status(state: AgentState) -> Dict[str, Any]:
    """
    Helper para verificar el estado del bootstrap.
    Útil para debugging.
    """
    services = get_services_status()
    
    return {
        "supabase_connected": services["supabase_connected"],
        "embeddings_ready": services["embeddings_ready"],
        "orchestration_plan": state.get("orchestration_plan", []),
        "current_step": state.get("current_step", 0),
        "worker_outputs_count": len(state.get("worker_outputs", [])),
        "has_pending_context": bool(state.get("pending_context")),
        "user_name": state.get("user_name", "Unknown"),
        "customer_id": state.get("customer_id"),
        "window_count": state.get("window_count", 0),
    }
