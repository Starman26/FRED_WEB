"""
equipment_tools.py - Consultas y acciones sobre PLCs, cobots y sensores

Reads usan RPC lab.get_all_equipment(p_type), writes usan schema('lab').
"""
from typing import Dict, Any, Optional
from datetime import datetime

from src.agent.services import get_supabase
from src.agent.utils.logger import logger


def _lab(supabase):
    """Helper: returns supabase client scoped to 'lab' schema."""
    return supabase.schema("lab")


# ============================================
# PLCs
# ============================================

def get_all_plcs() -> Dict[str, Any]:
    """Lista todas las PLCs del laboratorio con su estado actual. 1 RPC call."""
    try:
        supabase = get_supabase()
        
        resp = _lab(supabase).rpc("get_all_equipment", {"p_type": "plc"}).execute()
        equipment_raw = resp.data or []
        
        if isinstance(equipment_raw, list) and len(equipment_raw) == 1 and isinstance(equipment_raw[0], list):
            equipment_raw = equipment_raw[0]
        
        if not equipment_raw:
            return {"success": False, "error": "No se encontraron PLCs"}
        
        plcs = []
        for equip in equipment_raw:
            status = equip.get("status") or {}
            plcs.append({
                "id": equip.get("id"),
                "station_number": equip.get("station_number"),
                "station_name": equip.get("station_name"),
                "name": equip.get("name", "Unknown"),
                "model": equip.get("model"),
                "ip_address": equip.get("ip_address"),
                "is_connected": status.get("is_connected", False),
                "run_mode": "RUN" if status.get("plc_run_mode") else "STOP",
                "status": status.get("status", "unknown"),
                "has_error": status.get("status") == "error" or status.get("plc_error_code") is not None
            })
        
        return {
            "success": True,
            "total": len(plcs),
            "connected": sum(1 for p in plcs if p["is_connected"]),
            "with_errors": sum(1 for p in plcs if p["has_error"]),
            "plcs": plcs
        }
        
    except Exception as e:
        logger.error("lab_tools", f"Error en get_all_plcs: {e}")
        return {"success": False, "error": str(e)}


def reconnect_plc(station_number: int) -> Dict[str, Any]:
    """Intenta reconectar una PLC (simula reinicio de comunicación)."""
    try:
        supabase = get_supabase()
        lab = _lab(supabase)
        
        station = lab.table("stations").select("id").eq("station_number", station_number).single().execute()
        if not station.data:
            return {"success": False, "error": f"Estación {station_number} no encontrada"}
        
        plc = lab.table("equipment").select("id, name").eq("station_id", station.data["id"]).eq("equipment_type", "plc").single().execute()
        if not plc.data:
            return {"success": False, "error": f"No hay PLC en estación {station_number}"}
        
        lab.table("equipment_status").update({
            "status": "online",
            "is_connected": True,
            "plc_run_mode": True,
            "plc_error_code": None,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("equipment_id", plc.data["id"]).execute()
        
        return {
            "success": True,
            "station_number": station_number,
            "plc_name": plc.data["name"],
            "message": f"✅ PLC {plc.data['name']} de estación {station_number} reconectada exitosamente"
        }
        
    except Exception as e:
        logger.error("lab_tools", f"Error en reconnect_plc: {e}")
        return {"success": False, "error": str(e)}


# ============================================
# COBOTS
# ============================================

def get_all_cobots() -> Dict[str, Any]:
    """Lista todos los cobots del laboratorio con su estado actual. 1 RPC call."""
    try:
        supabase = get_supabase()
        
        resp = _lab(supabase).rpc("get_all_equipment", {"p_type": "cobot"}).execute()
        equipment_raw = resp.data or []
        
        if isinstance(equipment_raw, list) and len(equipment_raw) == 1 and isinstance(equipment_raw[0], list):
            equipment_raw = equipment_raw[0]
        
        if not equipment_raw:
            return {"success": False, "error": "No se encontraron cobots"}
        
        cobots = []
        for equip in equipment_raw:
            status = equip.get("status") or {}
            cobots.append({
                "id": equip.get("id"),
                "station_number": equip.get("station_number"),
                "station_name": equip.get("station_name"),
                "name": equip.get("name", "Unknown"),
                "model": equip.get("model"),
                "ip_address": equip.get("ip_address"),
                "is_connected": status.get("is_connected", False),
                "mode": status.get("cobot_mode", 0),
                "routine": status.get("cobot_routine_name", "idle"),
                "status": status.get("status", "unknown")
            })
        
        return {
            "success": True,
            "total": len(cobots),
            "running": sum(1 for c in cobots if c.get("mode", 0) > 0),
            "cobots": cobots
        }
        
    except Exception as e:
        logger.error("lab_tools", f"Error en get_all_cobots: {e}")
        return {"success": False, "error": str(e)}


def set_cobot_mode(station_number: int, mode: int, requested_by: str = "agent") -> Dict[str, Any]:
    """
    Cambia el modo de operación de un cobot.
    
    Args:
        station_number: Número de estación (1-6)
        mode: 0=stop/idle, 1=rutina 1, 2=rutina 2, etc.
        requested_by: Quién solicita el cambio
    """
    try:
        supabase = get_supabase()
        lab = _lab(supabase)
        
        from src.agent.tools.lab_tools.station_tools import get_station_details
        
        station_details = get_station_details(station_number)
        if not station_details.get("success"):
            return station_details
        
        cobot_id = station_details.get("cobot_id")
        if not cobot_id:
            return {"success": False, "error": f"No se encontró cobot en estación {station_number}"}
        
        if mode > 0:
            if not station_details.get("ready_to_operate"):
                reasons = []
                ready_details = station_details.get("ready_details", {})
                if not ready_details.get("doors_closed"):
                    reasons.append("Puerta(s) abierta(s)")
                if not ready_details.get("plc_connected"):
                    reasons.append("PLC no conectada o en STOP")
                if not ready_details.get("no_active_errors"):
                    reasons.append("Errores activos sin resolver")
                
                return {
                    "success": False,
                    "error": "No se puede iniciar el cobot por condiciones de seguridad",
                    "reasons": reasons,
                    "station_status": ready_details
                }
        
        routine_names = {
            0: "idle", 1: "pick_and_place", 2: "assembly",
            3: "inspection", 4: "packaging", 5: "custom_routine"
        }
        routine_name = routine_names.get(mode, f"routine_{mode}")
        
        lab.table("equipment_status").update({
            "cobot_mode": mode,
            "cobot_routine_name": routine_name,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("equipment_id", cobot_id).execute()
        
        try:
            lab.table("cobot_commands").insert({
                "equipment_id": cobot_id,
                "command_type": "set_mode",
                "command_params": {"mode": mode},
                "status": "completed",
                "door_check_passed": station_details.get("ready_details", {}).get("doors_closed", False),
                "requested_by": requested_by,
                "completed_at": datetime.utcnow().isoformat()
            }).execute()
        except Exception as cmd_err:
            logger.warning("lab_tools", f"No se pudo registrar comando: {cmd_err}")
        
        return {
            "success": True,
            "station_number": station_number,
            "cobot_id": cobot_id,
            "new_mode": mode,
            "routine_name": routine_name,
            "message": f"✅ Cobot de estación {station_number} configurado en modo {mode} ({routine_name})"
        }
        
    except Exception as e:
        logger.error("lab_tools", f"Error en set_cobot_mode: {e}")
        return {"success": False, "error": str(e)}


# ============================================
# SENSORS / DOORS
# ============================================

def check_door_sensors(station_number: Optional[int] = None) -> Dict[str, Any]:
    """
    Verifica el estado de los sensores de puerta. 1 RPC call.
    triggered=true significa CERRADA.
    """
    try:
        supabase = get_supabase()
        
        resp = _lab(supabase).rpc("get_all_equipment", {"p_type": "sensor"}).execute()
        equipment_raw = resp.data or []
        
        if isinstance(equipment_raw, list) and len(equipment_raw) == 1 and isinstance(equipment_raw[0], list):
            equipment_raw = equipment_raw[0]
        
        doors = []
        for equip in equipment_raw:
            if equip.get("sensor_type") != "door":
                continue
            if station_number and equip.get("station_number") != station_number:
                continue
            
            status = equip.get("status") or {}
            is_closed = status.get("sensor_triggered", False)
            doors.append({
                "station_number": equip.get("station_number"),
                "sensor_name": equip.get("name"),
                "location": equip.get("sensor_location"),
                "is_closed": is_closed,
                "status": "CERRADA ✅" if is_closed else "ABIERTA ⚠️"
            })
        
        open_doors = [d for d in doors if not d["is_closed"]]
        
        return {
            "success": True,
            "all_doors_closed": len(open_doors) == 0,
            "total_doors": len(doors),
            "open_doors_count": len(open_doors),
            "doors": doors,
            "warning": f"⚠️ {len(open_doors)} puerta(s) abierta(s)" if open_doors else None
        }
        
    except Exception as e:
        logger.error("lab_tools", f"Error en check_door_sensors: {e}")
        return {"success": False, "error": str(e)}


def close_all_doors() -> Dict[str, Any]:
    """Cierra todas las puertas de seguridad del laboratorio."""
    try:
        supabase = get_supabase()
        lab = _lab(supabase)
        
        sensors = lab.table("equipment").select("id").eq("equipment_type", "sensor").eq("sensor_type", "door").execute()
        if not sensors.data:
            return {"success": False, "error": "No se encontraron sensores de puerta"}
        
        sensor_ids = [s["id"] for s in sensors.data]
        
        for sensor_id in sensor_ids:
            lab.table("equipment_status").update({
                "sensor_triggered": True,
                "updated_at": datetime.utcnow().isoformat()
            }).eq("equipment_id", sensor_id).execute()
        
        return {
            "success": True,
            "doors_closed": len(sensor_ids),
            "message": f"✅ Se cerraron {len(sensor_ids)} puertas de seguridad"
        }
        
    except Exception as e:
        logger.error("lab_tools", f"Error en close_all_doors: {e}")
        return {"success": False, "error": str(e)}
