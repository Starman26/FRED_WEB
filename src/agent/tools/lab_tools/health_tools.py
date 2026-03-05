"""
health_tools.py - PLC ping-pong probe y health check de estaciones

Simula un ping activo al PLC para verificar conectividad real.
TODO: Reemplazar simulación con ping TCP/PROFINET real cuando el hardware esté listo.
"""
import random
from typing import Dict, Any
from datetime import datetime

from src.agent.services import get_supabase
from src.agent.utils.logger import logger


def _lab(supabase):
    """Helper: returns supabase client scoped to 'lab' schema."""
    return supabase.schema("lab")


def ping_plc(station_number: int) -> Dict[str, Any]:
    """
    Simulated PLC ping-pong probe. Reads DB state and simulates a network ping.

    TODO: Replace with real PROFINET/TCP ping when hardware ready.

    Args:
        station_number: Station number (1-6)

    Returns:
        Dict with ping results, health status, and DB state.
    """
    try:
        supabase = get_supabase()
        lab = _lab(supabase)

        # Get station
        station_resp = lab.table("stations").select("id, name").eq(
            "station_number", station_number
        ).single().execute()

        if not station_resp.data:
            return {"success": False, "error": f"Estacion {station_number} no encontrada"}

        station_id = station_resp.data["id"]
        station_name = station_resp.data["name"]

        # Get PLC equipment + status
        plc_resp = lab.table("equipment").select(
            "id, name, model, ip_address"
        ).eq("station_id", station_id).eq("equipment_type", "plc").single().execute()

        if not plc_resp.data:
            return {"success": False, "error": f"No hay PLC en estacion {station_number}"}

        plc = plc_resp.data
        plc_id = plc["id"]

        # Get current status
        status_resp = lab.table("equipment_status").select(
            "is_connected, plc_run_mode, status, plc_error_code"
        ).eq("equipment_id", plc_id).single().execute()

        status = status_resp.data or {}
        db_connected = status.get("is_connected", False)
        plc_run_mode = status.get("plc_run_mode", False)
        error_code = status.get("plc_error_code")

        # ── Simulate ping ──
        # Base success on DB state + small random failure (~5%)
        random_fail = random.random() < 0.05
        ping_ok = db_connected and not random_fail
        response_time_ms = round(random.uniform(80, 250), 1) if ping_ok else 0.0

        # Determine health
        if not ping_ok:
            health = "unreachable"
        elif not plc_run_mode:
            health = "degraded"
        else:
            health = "healthy"

        return {
            "success": True,
            "station_number": station_number,
            "station_name": station_name,
            "plc_name": plc.get("name", "Unknown"),
            "ip_address": plc.get("ip_address", "N/A"),
            "ping_ok": ping_ok,
            "response_time_ms": response_time_ms,
            "plc_run_mode": plc_run_mode,
            "db_connected": db_connected,
            "error_code": error_code,
            "health": health,
        }

    except Exception as e:
        logger.error("lab_tools", f"Error en ping_plc: {e}")
        return {"success": False, "error": str(e)}


def health_check_station(station_number: int) -> Dict[str, Any]:
    """
    Full health check for a station: PLC ping + door status + errors + cobot.

    Combines multiple data sources into one comprehensive health report.

    Args:
        station_number: Station number (1-6)

    Returns:
        Dict with overall health, individual checks, issues, and recommendations.
    """
    try:
        # 1. PLC ping probe
        plc_result = ping_plc(station_number)

        # 2. Station details (doors, cobot, errors)
        from src.agent.tools.lab_tools.station_tools import get_station_details
        station_details = get_station_details(station_number)

        if not station_details.get("success"):
            return station_details

        # Extract sub-checks
        plc_check = {}
        if plc_result.get("success"):
            plc_check = {
                "status": plc_result["health"],
                "ping_ok": plc_result["ping_ok"],
                "response_time_ms": plc_result["response_time_ms"],
                "run_mode": "RUN" if plc_result["plc_run_mode"] else "STOP",
                "ip_address": plc_result.get("ip_address", "N/A"),
                "error_code": plc_result.get("error_code"),
            }
        else:
            plc_check = {"status": "error", "ping_ok": False, "response_time_ms": 0,
                         "error": plc_result.get("error", "Unknown")}

        # Doors
        sensors = station_details.get("sensors", [])
        door_sensors = [s for s in sensors if s.get("type") == "door"]
        open_count = sum(1 for d in door_sensors if not d.get("triggered", False))
        doors_check = {
            "all_closed": open_count == 0,
            "open_count": open_count,
            "total": len(door_sensors),
        }

        # Errors
        active_errors = station_details.get("active_errors", [])
        critical_count = sum(1 for e in active_errors if e.get("severity") == "critical")
        errors_check = {
            "active_count": len(active_errors),
            "critical_count": critical_count,
        }

        # Cobot
        cobot_info = station_details.get("cobot")
        cobot_check = {
            "connected": cobot_info.get("is_connected", False) if cobot_info else False,
            "mode": cobot_info.get("routine", "idle") if cobot_info else "N/A",
        }

        # ── Score overall health ──
        issues = []
        recommendations = []

        # PLC issues
        if not plc_check.get("ping_ok"):
            issues.append("PLC no responde al ping")
            recommendations.append("Reconectar PLC o verificar cableado de red")
        elif plc_check.get("response_time_ms", 0) > 200:
            issues.append(f"PLC responde lento ({plc_check['response_time_ms']}ms)")
            recommendations.append("Verificar carga de red o estado del switch")
        if plc_check.get("run_mode") == "STOP":
            issues.append("PLC en modo STOP")
            recommendations.append("Poner PLC en modo RUN desde el panel")
        if plc_check.get("error_code"):
            issues.append(f"PLC reporta error: {plc_check['error_code']}")
            recommendations.append("Revisar log de errores del PLC")

        # Door issues
        if open_count > 0:
            issues.append(f"{open_count} puerta(s) abierta(s)")
            recommendations.append("Cerrar puertas de seguridad antes de operar")

        # Error issues
        if critical_count > 0:
            issues.append(f"{critical_count} error(es) critico(s) activo(s)")
            recommendations.append("Resolver errores criticos inmediatamente")
        elif len(active_errors) > 0:
            issues.append(f"{len(active_errors)} error(es) activo(s)")
            recommendations.append("Revisar y resolver errores pendientes")

        # Cobot issues
        if cobot_info and not cobot_check["connected"]:
            issues.append("Cobot desconectado")
            recommendations.append("Verificar conexion del cobot")

        # Overall health
        if not plc_check.get("ping_ok") or critical_count > 0:
            overall_health = "critical"
        elif issues:
            overall_health = "degraded"
        else:
            overall_health = "healthy"

        station_info = station_details.get("station", {})

        return {
            "success": True,
            "station_number": station_number,
            "station_name": station_info.get("name", f"Estacion {station_number}"),
            "overall_health": overall_health,
            "checks": {
                "plc": plc_check,
                "doors": doors_check,
                "errors": errors_check,
                "cobot": cobot_check,
            },
            "issues": issues,
            "recommendations": recommendations,
        }

    except Exception as e:
        logger.error("lab_tools", f"Error en health_check_station: {e}")
        return {"success": False, "error": str(e)}
