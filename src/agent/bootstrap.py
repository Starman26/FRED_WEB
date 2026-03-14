"""
bootstrap.py
State initialization and service verification.
"""
import json
import os
from typing import Dict, Any, List

from src.agent.state import AgentState, STATE_DEFAULTS
from src.agent.services import init_services, get_services_status
from src.agent.utils.run_events import event_read, event_error
from src.agent.utils.logger import logger


def _as_event_list(evt) -> List[Dict[str, Any]]:
    """Normalize events to a list."""
    if evt is None:
        return []
    if isinstance(evt, list):
        return evt
    return [evt]


def _recall_similar_diagnostics(user_message: str, team_id: str = None, limit: int = 3) -> List[Dict]:
    """Retrieve similar past diagnostics using embedding similarity search."""
    try:
        from src.agent.services import get_supabase, get_embeddings
        sb = get_supabase()
        emb = get_embeddings()
        if not sb or not emb:
            return []

        query_embedding = emb.embed_query(user_message)

        result = sb.schema("lab").rpc("match_diagnostics", {
            "query_embedding": query_embedding,
            "match_threshold": 0.7,
            "match_count": limit,
            "filter_team_id": team_id,
        }).execute()

        if result and result.data:
            return result.data
        return []
    except Exception as e:
        logger.warning("bootstrap", f"Diagnostic recall failed: {e}")
        return []


def bootstrap_node(state: AgentState) -> Dict[str, Any]:
    """First node in the graph. Initializes missing state fields and verifies services.

    Supabase/embeddings are not stored in state (not serializable);
    access them via src.agent.services.
    """
    logger.node_start("bootstrap", {"action": "initializing_state"})
    
    patches: Dict[str, Any] = {}
    events: List[Dict[str, Any]] = []

    # Fast-path: practice mode on subsequent turns (skip Supabase fetches)
    interaction_mode = state.get("interaction_mode", "chat")

    if interaction_mode == "practice" and state["automation_id"]:
        # Real robot state comes from the connected bridge (WebSocket), not Supabase
        from src.agent.shared_state import get_connected_robots
        connected = get_connected_robots()

        logger.node_end("bootstrap", {"status": "fast_path_practice", "connected_devices": len(connected)})
        return {
            "robot_state": connected,
            "active_devices": connected,
            "practice_session_active": bool(connected),
            "events": [event_read("bootstrap", f"Practice fast-path: {len(connected)} devices connected")],
        }


    # Initialize missing fields with defaults
    for key, default_value in STATE_DEFAULTS.items():
        current_value = state.get(key)
        if current_value is None:
            patches[key] = default_value
            logger.debug("bootstrap", f"Inicializando {key} con valor por defecto")

    # Initialize services (not stored in state)
    services_status = init_services()
    
    if services_status["supabase_connected"]:
        events.append(event_read("bootstrap", " Conectado a base de datos"))
    else:
        events.append(event_error("bootstrap", " Database no disponible"))
    
    if services_status["embeddings_ready"]:
        events.append(event_read("bootstrap", " Embeddings listos"))
    else:
        events.append(event_error("bootstrap", " Embeddings no disponibles"))

    # Initialize orchestration fields
    if state.get("orchestration_plan") is None:
        patches["orchestration_plan"] = []
    
    if state.get("current_step") is None:
        patches["current_step"] = 0
    
    if state.get("worker_outputs") is None:
        patches["worker_outputs"] = []
    
    if state.get("pending_context") is None:
        patches["pending_context"] = {}

    # Load data from Supabase
    if services_status["supabase_connected"]:
        from src.agent.services import get_supabase
        sb = get_supabase()

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

        try:
            robot_result = sb.schema("lab").from_("robot_state") \
                .select("id, state, mode, error_code, warning_code, tcp_x, tcp_y, tcp_z, tcp_roll, tcp_pitch, tcp_yaw, joints, velocities, efforts, temperatures, currents, is_connected, space_id") \
                .eq("is_connected", True) \
                .execute()

            if robot_result.data:
                robots = {}
                for r in robot_result.data:
                    robot_id = str(r["id"])
                    robots[robot_id] = {
                        "id": robot_id,
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
                        "is_connected": True,
                        "space_id": r.get("space_id"),
                    }
                patches["robot_state"] = robots
            else:
                patches["robot_state"] = {}
        except Exception as e:
            logger.warning("bootstrap", f"Failed to load robot state: {e}")
            patches["robot_state"] = {}

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

        # Load equipment spec + agent skills (Skills System)
        equipment_id = (state.get("pending_context") or {}).get("equipment_id")
        if equipment_id:
            try:
                eq_result = sb.schema("lab").from_("equipment_profiles") \
                    .select("type, spec_md") \
                    .eq("id", equipment_id) \
                    .single() \
                    .execute()

                if eq_result.data:
                    equipment_type = eq_result.data.get("type", "generic")
                    spec_md = eq_result.data.get("spec_md") or ""
                    patches["equipment_spec"] = spec_md

                    # Load skills matching this equipment type + generic
                    skills_result = sb.schema("lab").from_("agent_skills") \
                        .select("slug, title, category, content_md") \
                        .in_("equipment_type", [equipment_type, "generic"]) \
                        .execute()

                    skills = skills_result.data or []
                    patches["loaded_skills"] = [s["content_md"] for s in skills]
                    patches["loaded_skills_meta"] = [
                        {"slug": s["slug"], "title": s["title"], "category": s["category"]}
                        for s in skills
                    ]

                    logger.info("bootstrap",
                                f"Equipment context loaded: spec_md={len(spec_md)} chars, "
                                f"skills={[s['slug'] for s in skills]}")
                    events.append(event_read("bootstrap",
                                             f"Loaded equipment context: {len(skills)} skills"))
            except Exception as e:
                logger.warning("bootstrap", f"Failed to load equipment context: {e}")

    # Recall similar diagnostics (cross-session memory)
    if interaction_mode in ("chat", "agent", "code"):
        user_msg = ""
        for m in reversed(state.get("messages", [])):
            if hasattr(m, "type") and m.type == "human":
                user_msg = (m.content or "").strip()
                break
            if isinstance(m, dict) and m.get("role", m.get("type")) in ("user", "human"):
                user_msg = (m.get("content") or "").strip()
                break

        if user_msg and len(user_msg) > 20:
            team_id = state.get("team_id")
            similar = _recall_similar_diagnostics(user_msg, team_id=team_id)
            if similar:
                recall_text = "\n".join([
                    f"- [{d.get('severity', '?')}] {d.get('user_query', '')[:100]} → {d.get('diagnosis', '')[:200]}"
                    + (f" (Lesson: {d['lesson_learned'][:100]})" if d.get("lesson_learned") else "")
                    for d in similar
                ])
                pending = patches.get("pending_context") or state.get("pending_context") or {}
                pending["diagnostic_recall"] = {
                    "count": len(similar),
                    "entries": similar,
                    "summary": recall_text,
                }
                patches["pending_context"] = pending
                events.append(event_read("bootstrap", f"Recalled {len(similar)} similar diagnostics"))

    # Populate active_devices from bridge connections
    try:
        from src.agent.shared_state import get_connected_robots
        active = get_connected_robots()
    except ImportError:
        active = {}
    patches["active_devices"] = active

    events.append(event_read("bootstrap", " Estado inicializado correctamente"))

    # Only return new events; operator.add in the reducer handles concatenation
    patches["events"] = events
    
    logger.node_end("bootstrap", {
        "status": "success",
        "patches_count": len(patches),
        "supabase_connected": services_status["supabase_connected"],
        "embeddings_ready": services_status["embeddings_ready"]
    })
    
    return patches


def get_bootstrap_status(state: AgentState) -> Dict[str, Any]:
    """Return bootstrap status summary for debugging."""
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
