"""
bootstrap.py - Inicialización del estado y verificación de herramientas

CAMBIOS PRINCIPALES:
1. Usa STATE_DEFAULTS de state.py para consistencia
2. Inicializa campos de orchestration multi-step
3. NO inyecta supabase/embeddings al state (no son serializables)
4. Usa services.py para inicializar clientes
"""
import json
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
    # FAST-PATH: practice mode on subsequent turns
    # Skip expensive Supabase fetches; only refresh robot_state
    # ==========================================
    interaction_mode = state.get("interaction_mode", "chat")

    if interaction_mode == "practice" and state.get("automation_md_content"):
        robot_state = {}
        try:
            from src.agent.services import get_supabase
            sb = get_supabase()
            if sb:
                r = sb.schema("lab").from_("robot_state") \
                    .select("robot_ip, robot_name, state, mode, error_code, warning_code, tcp_x, tcp_y, tcp_z, joints, is_connected, last_seen") \
                    .eq("is_connected", True) \
                    .execute()
                if r and r.data:
                    robots = {}
                    for row in r.data:
                        ip = row["robot_ip"]
                        robots[ip] = {
                            "name": row.get("robot_name", ip),
                            "ip": ip,
                            "state": row.get("state"),
                            "mode": row.get("mode"),
                            "error_code": row.get("error_code", 0),
                            "warning_code": row.get("warning_code", 0),
                            "tcp": {
                                "x": row.get("tcp_x"), "y": row.get("tcp_y"), "z": row.get("tcp_z"),
                            },
                            "joints": row.get("joints", []),
                            "is_connected": True,
                            "last_seen": row.get("last_seen"),
                        }
                    robot_state = robots
        except Exception:
            pass

        logger.node_end("bootstrap", {"status": "fast_path_practice"})
        return {
            "robot_state": robot_state,
            "events": [event_read("bootstrap", "Practice fast-path: state reused")],
        }

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
        events.append(event_read("bootstrap", " Conectado a base de datos"))
    else:
        events.append(event_error("bootstrap", " Database no disponible"))
    
    if services_status["embeddings_ready"]:
        events.append(event_read("bootstrap", " Embeddings listos"))
    else:
        events.append(event_error("bootstrap", " Embeddings no disponibles"))
    
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
    # 5. Cargar datos de Supabase (automation, robot_state, user_profile)
    # ==========================================
    if services_status["supabase_connected"]:
        from src.agent.services import get_supabase
        sb = get_supabase()

        # --- Cargar automation ---
        if state.get("automation_id"):
            try:
                result = sb.schema("lab").from_("automations") \
                    .select("md_content, type") \
                    .eq("id", state["automation_id"]) \
                    .single() \
                    .execute()
                if result.data:
                    patches["automation_md_content"] = result.data["md_content"] or ""
                    patches["automation_type"] = result.data.get("type", "automation")

                progress = sb.schema("lab").from_("user_automation_progress") \
                    .select("current_step, agent_observations, status") \
                    .eq("automation_id", state["automation_id"]) \
                    .eq("auth_user_id", state.get("auth_user_id")) \
                    .maybe_single() \
                    .execute()
                if progress.data:
                    patches["automation_step"] = progress.data.get("current_step", 1) or 1
                    patches["practice_status"] = progress.data.get("status", "in_progress")
                    obs = progress.data.get("agent_observations", [])
                    if obs:
                        if isinstance(obs, str):
                            try:
                                obs = json.loads(obs)
                            except (json.JSONDecodeError, TypeError):
                                obs = []
                        patches["automation_context"] = json.dumps(obs) if isinstance(obs, list) else str(obs)
            except Exception as e:
                logger.warning("bootstrap", f"Failed to load automation: {e}")

        # --- Cargar robot state desde lab.robot_state ---
        try:
            robot_result = sb.schema("lab").from_("robot_state") \
                .select("robot_ip, robot_name, state, mode, error_code, warning_code, tcp_x, tcp_y, tcp_z, tcp_roll, tcp_pitch, tcp_yaw, joints, velocities, efforts, temperatures, currents, safety_zone, is_connected, last_seen") \
                .eq("is_connected", True) \
                .execute()

            if robot_result.data:
                robots = {}
                for r in robot_result.data:
                    ip = r["robot_ip"]
                    robots[ip] = {
                        "name": r.get("robot_name", ip),
                        "ip": ip,
                        "state": r.get("state"),
                        "mode": r.get("mode"),
                        "error_code": r.get("error_code", 0),
                        "warning_code": r.get("warning_code", 0),
                        "tcp": {
                            "x": r.get("tcp_x"), "y": r.get("tcp_y"), "z": r.get("tcp_z"),
                            "roll": r.get("tcp_roll"), "pitch": r.get("tcp_pitch"), "yaw": r.get("tcp_yaw"),
                        },
                        "joints": r.get("joints", []),
                        "velocities": r.get("velocities", []),
                        "efforts": r.get("efforts", []),
                        "temperatures": r.get("temperatures", []),
                        "currents": r.get("currents", []),
                        "safety_zone": r.get("safety_zone"),
                        "is_connected": True,
                        "last_seen": r.get("last_seen"),
                    }
                patches["robot_state"] = robots
            else:
                patches["robot_state"] = {}
        except Exception as e:
            logger.warning("bootstrap", f"Failed to load robot state: {e}")
            patches["robot_state"] = {}

        # --- Cargar user agent profile (siempre) ---
        if state.get("auth_user_id"):
            try:
                profile_result = sb.from_("user_agent_profiles") \
                    .select("profile_md") \
                    .eq("auth_user_id", state["auth_user_id"]) \
                    .maybe_single() \
                    .execute()
                if profile_result.data and profile_result.data.get("profile_md"):
                    patches["user_profile_md"] = profile_result.data["profile_md"]
                else:
                    user_name = state.get("user_name", "Usuario")
                    template = f"# ORION Profile -- {user_name}\n\n## Experience Level\n- Overall: beginner\n\n## Agent Observations\n\n## Interaction Rules\n\n## Interests\n"
                    sb.from_("user_agent_profiles").insert({
                        "auth_user_id": state["auth_user_id"],
                        "profile_md": template,
                    }).execute()
                    patches["user_profile_md"] = template
            except Exception as e:
                logger.warning("bootstrap", f"Failed to load user profile: {e}")

    # ==========================================
    # 6. Registrar evento de inicio
    # ==========================================
    bootstrap_event = event_read("bootstrap", " Estado inicializado correctamente")
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
