"""
station_tools.py - Consultas de estado de estaciones y overview del lab

Usa funciones RPC de Supabase para minimizar requests HTTP.
Requiere: lab.get_lab_overview() y lab.get_station_details(p_station_number) en Supabase.
"""
from typing import Dict, Any, Optional
from datetime import datetime

from src.agent.services import get_supabase
from src.agent.utils.logger import logger


def get_lab_overview() -> Dict[str, Any]:
    """
    Obtiene un resumen general del laboratorio en 1 sola llamada RPC.
    
    Returns:
        {total_stations, stations_online, active_errors_count, stations: [...]}
    """
    try:
        supabase = get_supabase()
        
        resp = supabase.schema("lab").rpc("get_lab_overview", {}).execute()
        data = resp.data
        
        if not data:
            return {"success": False, "error": "No se pudo obtener el overview del laboratorio"}
        
        # Si data es una lista, tomar el primer elemento
        if isinstance(data, list):
            data = data[0] if data else {}
        
        stations = data.get("stations", [])
        
        # Calcular estadísticas derivadas
        stations_with_errors = sum(1 for s in stations if s.get("active_errors", 0) > 0)
        
        # Formatear para compatibilidad con el resto del código
        station_summaries = []
        for s in stations:
            station_summaries.append({
                "station_number": s.get("station_number"),
                "name": s.get("name"),
                "status": s.get("status", "unknown"),
                "equipment_count": s.get("equipment_count", 0),
                "active_errors": s.get("active_errors", 0),
                "is_operational": s.get("status") == "online" and s.get("active_errors", 0) == 0,
            })
        
        return {
            "success": True,
            "total_stations": data.get("total_stations", len(stations)),
            "stations_online": data.get("stations_online", 0),
            "stations_with_errors": stations_with_errors,
            "active_errors_count": data.get("active_errors_count", 0),
            "stations": station_summaries,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("lab_tools", f"Error en get_lab_overview: {e}")
        return {"success": False, "error": str(e)}


def get_station_details(station_number: int) -> Dict[str, Any]:
    """
    Obtiene detalles completos de una estación en 1 sola llamada RPC.
    
    Args:
        station_number: Número de estación (1-6)
    """
    try:
        supabase = get_supabase()
        
        resp = supabase.schema("lab").rpc(
            "get_station_details", 
            {"p_station_number": station_number}
        ).execute()
        
        data = resp.data
        
        if not data:
            return {"success": False, "error": f"Estación {station_number} no encontrada"}
        
        # Si data es una lista, tomar el primer elemento
        if isinstance(data, list):
            data = data[0] if data else {}
        
        station = data.get("station")
        equipment_list = data.get("equipment", [])
        active_errors = data.get("active_errors", [])
        
        if not station:
            return {"success": False, "error": f"Estación {station_number} no encontrada"}
        
        # Procesar equipos
        plc_info = None
        cobot_info = None
        cobot_id = None
        sensors = []
        
        for equip in equipment_list:
            status = equip.get("status") or {}
            
            if equip.get("equipment_type") == "plc":
                plc_info = {
                    "name": equip.get("name"),
                    "model": station.get("model"),
                    "ip_address": station.get("ip_address"),
                    "is_connected": status.get("is_connected", False),
                    "run_mode": "RUN" if status.get("plc_run_mode") else "STOP",
                    "status": status.get("status", "unknown"),
                    "error_code": status.get("plc_error_code")
                }
            elif equip.get("equipment_type") == "cobot":
                cobot_id = equip.get("id")
                cobot_info = {
                    "id": equip.get("id"),
                    "name": equip.get("name"),
                    "model": station.get("model"),
                    "ip_address": station.get("ip_address"),
                    "is_connected": status.get("is_connected", False),
                    "mode": status.get("cobot_mode", 0),
                    "routine": status.get("cobot_routine_name", "idle"),
                    "status": status.get("status", "unknown")
                }
            elif equip.get("equipment_type") == "sensor":
                sensors.append({
                    "name": equip.get("name"),
                    "type": equip.get("sensor_type"),
                    "location": equip.get("sensor_location"),
                    "triggered": status.get("sensor_triggered", False),
                    "value": status.get("sensor_value"),
                    "status": status.get("status", "unknown")
                })
        
        doors_ok = all(s["triggered"] for s in sensors if s.get("type") == "door") if sensors else True
        plc_ok = plc_info is not None and plc_info.get("is_connected") and plc_info.get("run_mode") == "RUN"
        ready_to_operate = doors_ok and plc_ok and len(active_errors) == 0
        
        return {
            "success": True,
            "station": {
                "number": station.get("station_number"),
                "name": station.get("name"),
                "description": station.get("description"),
                "location": station.get("location"),
                "is_active": station.get("is_active", True)
            },
            "plc": plc_info,
            "cobot": cobot_info,
            "cobot_id": cobot_id,
            "sensors": sensors,
            "active_errors": active_errors,
            "ready_to_operate": ready_to_operate,
            "ready_details": {
                "doors_closed": doors_ok,
                "plc_connected": plc_ok,
                "no_active_errors": len(active_errors) == 0
            }
        }
        
    except Exception as e:
        logger.error("lab_tools", f"Error en get_station_details: {e}")
        return {"success": False, "error": str(e)}
