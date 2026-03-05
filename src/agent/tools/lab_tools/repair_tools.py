"""
repair_tools.py - Acciones de reparación y diagnóstico del laboratorio

Uses schema('lab') for direct table operations.
Read operations are mostly delegated to other tools that use RPC.
"""
from typing import Dict, Any, Optional
from datetime import datetime

from src.agent.services import get_supabase
from src.agent.utils.logger import logger

from .station_tools import get_station_details, get_lab_overview
from .equipment_tools import (
    get_all_plcs, check_door_sensors, close_all_doors, reconnect_plc
)
from .error_tools import get_active_errors


def _lab(supabase):
    """Helper: returns supabase client scoped to 'lab' schema."""
    return supabase.schema("lab")


def reset_lab_to_safe_state() -> Dict[str, Any]:
    """
    Reinicia todo el laboratorio a un estado seguro y operativo.
    
    Acciones: cierra puertas, reconecta PLCs, para cobots, resuelve errores.
    """
    try:
        supabase = get_supabase()
        lab = _lab(supabase)
        results = {
            "doors_closed": 0,
            "plcs_reconnected": 0,
            "cobots_stopped": 0,
            "errors_resolved": 0
        }
        
        # 1. Cerrar todas las puertas
        door_result = close_all_doors()
        if door_result.get("success"):
            results["doors_closed"] = door_result.get("doors_closed", 0)
        
        # 2. Reconectar todas las PLCs
        plcs = lab.table("equipment").select("id").eq("equipment_type", "plc").execute()
        for plc in plcs.data or []:
            lab.table("equipment_status").update({
                "status": "online",
                "is_connected": True,
                "plc_run_mode": True,
                "plc_error_code": None,
                "updated_at": datetime.utcnow().isoformat()
            }).eq("equipment_id", plc["id"]).execute()
            results["plcs_reconnected"] += 1
        
        # 3. Poner cobots en idle
        cobots = lab.table("equipment").select("id").eq("equipment_type", "cobot").execute()
        for cobot in cobots.data or []:
            lab.table("equipment_status").update({
                "status": "online",
                "is_connected": True,
                "cobot_mode": 0,
                "cobot_routine_name": "idle",
                "updated_at": datetime.utcnow().isoformat()
            }).eq("equipment_id", cobot["id"]).execute()
            results["cobots_stopped"] += 1
        
        # 4. Resolver todos los errores
        errors = lab.table("errors").select("id").eq("status", "active").execute()
        if errors.data:
            lab.table("errors").update({
                "status": "resolved",
                "resolved_by": "system_reset",
                "resolved_at": datetime.utcnow().isoformat(),
                "resolution_notes": "Reset completo del laboratorio"
            }).eq("status", "active").execute()
            results["errors_resolved"] = len(errors.data)
        
        return {
            "success": True,
            "results": results,
            "message": f"✅ Laboratorio reiniciado a estado seguro"
        }
        
    except Exception as e:
        logger.error("lab_tools", f"Error en reset_lab_to_safe_state: {e}")
        return {"success": False, "error": str(e)}


def diagnose_and_suggest_fixes(station_number: Optional[int] = None) -> Dict[str, Any]:
    """Diagnostica problemas y sugiere acciones de reparación."""
    try:
        problems = []
        suggested_actions = []
        
        if station_number:
            details = get_station_details(station_number)
            if not details.get("success"):
                return details
            
            ready_details = details.get("ready_details", {})
            
            if not ready_details.get("doors_closed"):
                problems.append(f"Puerta de estación {station_number} abierta")
                suggested_actions.append({"action": "close_doors", "description": "Cerrar puertas de seguridad"})
            
            if not ready_details.get("plc_connected"):
                problems.append(f"PLC de estación {station_number} desconectada")
                suggested_actions.append({"action": "reconnect_plc", "station": station_number, "description": "Reconectar PLC"})
            
            if not ready_details.get("no_active_errors"):
                error_count = len(details.get("active_errors", []))
                problems.append(f"{error_count} error(es) activo(s) en estación {station_number}")
                suggested_actions.append({"action": "resolve_errors", "station": station_number, "description": "Resolver errores"})
        
        else:
            overview = get_lab_overview()
            if not overview.get("success"):
                return overview
            
            doors = check_door_sensors()
            if doors.get("success") and not doors.get("all_doors_closed"):
                open_count = doors.get("open_doors_count", 0)
                problems.append(f"{open_count} puerta(s) abierta(s)")
                suggested_actions.append({"action": "close_all_doors", "description": "Cerrar todas las puertas"})
            
            plcs = get_all_plcs()
            if plcs.get("success"):
                disconnected = [p for p in plcs.get("plcs", []) if not p.get("is_connected")]
                for plc in disconnected:
                    problems.append(f"PLC de estación {plc['station_number']} desconectada")
                    suggested_actions.append({
                        "action": "reconnect_plc",
                        "station": plc["station_number"],
                        "description": f"Reconectar PLC estación {plc['station_number']}"
                    })
            
            errors = get_active_errors()
            if errors.get("success") and errors.get("total_errors", 0) > 0:
                problems.append(f"{errors['total_errors']} error(es) registrado(s)")
                suggested_actions.append({"action": "resolve_all_errors", "description": "Resolver todos los errores"})
        
        return {
            "success": True,
            "problems_count": len(problems),
            "problems": problems,
            "suggested_actions": suggested_actions,
            "can_auto_fix": len(suggested_actions) > 0
        }
        
    except Exception as e:
        logger.error("lab_tools", f"Error en diagnose_and_suggest_fixes: {e}")
        return {"success": False, "error": str(e)}
