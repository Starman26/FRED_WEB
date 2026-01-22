"""
lab_tools.py - Herramientas para interactuar con el laboratorio físico

Proporciona funciones para:
- Consultar estado de estaciones, PLCs, cobots, sensores
- Verificar errores activos
- Enviar comandos a cobots (con validación de seguridad)
- Obtener resumen del laboratorio
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from src.agent.services import get_supabase
from src.agent.utils.logger import logger


# ============================================
# CONSULTAS DE ESTADO
# ============================================

def get_lab_overview() -> Dict[str, Any]:
    """
    Obtiene un resumen general del estado del laboratorio.
    
    Returns:
        {
            "total_stations": 6,
            "stations_online": 5,
            "stations_with_errors": 1,
            "active_errors_count": 2,
            "stations": [...]
        }
    """
    try:
        supabase = get_supabase()
        
        # Obtener estaciones
        stations_resp = supabase.table("lab_stations").select("*").order("station_number").execute()
        stations = stations_resp.data or []
        
        if not stations:
            return {"success": False, "error": "No se encontraron estaciones"}
        
        # Contar errores activos
        try:
            errors_resp = supabase.table("lab_errors").select("id").eq("status", "active").execute()
            active_errors = len(errors_resp.data) if errors_resp.data else 0
        except:
            active_errors = 0
        
        # Procesar cada estación
        stations_online = 0
        stations_with_errors = 0
        station_summaries = []
        
        for station in stations:
            station_id = station["id"]
            
            # Obtener equipos de esta estación
            try:
                equip_resp = supabase.table("lab_equipment").select("id, equipment_type, sensor_type").eq("station_id", station_id).execute()
                equipment = equip_resp.data or []
            except:
                equipment = []
            
            plc_status = "unknown"
            cobot_status = "unknown"
            doors_closed = True
            
            for equip in equipment:
                # Obtener status de este equipo
                try:
                    status_resp = supabase.table("lab_equipment_status").select("*").eq("equipment_id", equip["id"]).single().execute()
                    status_data = status_resp.data or {}
                except:
                    status_data = {}
                
                if equip["equipment_type"] == "plc":
                    plc_status = "online" if status_data.get("is_connected") else "offline"
                elif equip["equipment_type"] == "cobot":
                    cobot_status = status_data.get("status", "unknown")
                elif equip["equipment_type"] == "sensor" and equip.get("sensor_type") == "door":
                    if not status_data.get("sensor_triggered", True):
                        doors_closed = False
            
            is_online = plc_status == "online"
            has_error = plc_status == "offline" or cobot_status == "error" or not doors_closed
            
            if is_online:
                stations_online += 1
            if has_error:
                stations_with_errors += 1
            
            station_summaries.append({
                "station_number": station["station_number"],
                "name": station["name"],
                "plc_status": plc_status,
                "cobot_status": cobot_status,
                "doors_closed": doors_closed,
                "is_operational": is_online and not has_error
            })
        
        return {
            "success": True,
            "total_stations": len(stations),
            "stations_online": stations_online,
            "stations_with_errors": stations_with_errors,
            "active_errors_count": active_errors,
            "stations": station_summaries,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("lab_tools", f"Error en get_lab_overview: {e}")
        return {"success": False, "error": str(e)}


def get_station_details(station_number: int) -> Dict[str, Any]:
    """
    Obtiene detalles completos de una estación específica.
    
    Args:
        station_number: Número de estación (1-6)
        
    Returns:
        Información detallada de la estación, equipos, sensores y errores
    """
    try:
        supabase = get_supabase()
        
        # 1. Obtener estación
        station_resp = supabase.table("lab_stations").select("*").eq("station_number", station_number).single().execute()
        station = station_resp.data
        
        if not station:
            return {"success": False, "error": f"Estación {station_number} no encontrada"}
        
        station_id = station["id"]
        
        # 2. Obtener equipos de esta estación
        equip_resp = supabase.table("lab_equipment").select("*").eq("station_id", station_id).execute()
        equipment_list = equip_resp.data or []
        
        # 3. Obtener errores activos
        try:
            errors_resp = supabase.table("lab_errors").select("*").eq("station_id", station_id).eq("status", "active").execute()
            active_errors = errors_resp.data or []
        except:
            active_errors = []
        
        # 4. Procesar equipamiento
        plc_info = None
        cobot_info = None
        cobot_id = None
        sensors = []
        
        for equip in equipment_list:
            # Obtener status de este equipo
            try:
                status_resp = supabase.table("lab_equipment_status").select("*").eq("equipment_id", equip["id"]).single().execute()
                status = status_resp.data or {}
            except:
                status = {}
            
            if equip["equipment_type"] == "plc":
                plc_info = {
                    "name": equip.get("name"),
                    "model": equip.get("model"),
                    "ip_address": equip.get("ip_address"),
                    "is_connected": status.get("is_connected", False),
                    "run_mode": "RUN" if status.get("plc_run_mode") else "STOP",
                    "status": status.get("status", "unknown"),
                    "error_code": status.get("plc_error_code"),
                    "last_heartbeat": status.get("last_heartbeat")
                }
            elif equip["equipment_type"] == "cobot":
                cobot_id = equip["id"]
                cobot_info = {
                    "id": equip["id"],
                    "name": equip.get("name"),
                    "model": equip.get("model"),
                    "ip_address": equip.get("ip_address"),
                    "is_connected": status.get("is_connected", False),
                    "mode": status.get("cobot_mode", 0),
                    "routine": status.get("cobot_routine_name", "idle"),
                    "status": status.get("status", "unknown")
                }
            elif equip["equipment_type"] == "sensor":
                sensors.append({
                    "name": equip.get("name"),
                    "type": equip.get("sensor_type"),
                    "location": equip.get("sensor_location"),
                    "triggered": status.get("sensor_triggered", False),
                    "value": status.get("sensor_value"),
                    "status": status.get("status", "unknown")
                })
        
        # 5. Verificar si está lista para operar
        doors_ok = all(s["triggered"] for s in sensors if s.get("type") == "door") if sensors else True
        plc_ok = plc_info is not None and plc_info.get("is_connected") and plc_info.get("run_mode") == "RUN"
        ready_to_operate = doors_ok and plc_ok and len(active_errors) == 0
        
        logger.info("lab_tools", f"Station {station_number}: doors_ok={doors_ok}, plc_ok={plc_ok}, errors={len(active_errors)}")
        
        return {
            "success": True,
            "station": {
                "number": station["station_number"],
                "name": station["name"],
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


def get_all_plcs() -> Dict[str, Any]:
    """
    Lista todas las PLCs del laboratorio con su estado actual.
    
    Returns:
        Lista de PLCs con nombre, modelo, IP, estado de conexión
    """
    try:
        supabase = get_supabase()
        
        # Query simplificado - sin joins anidados que pueden fallar
        response = supabase.table("lab_equipment").select("*").eq("equipment_type", "plc").execute()
        
        logger.info("lab_tools", f"PLCs encontrados: {len(response.data) if response.data else 0}")
        
        if not response.data:
            return {
                "success": False,
                "error": "No se encontraron PLCs",
                "tables_missing": True
            }
        
        plcs = []
        for equip in response.data:
            station_id = equip.get("station_id")
            
            # Obtener info de estación
            station_name = f"Estación {station_id}"
            if station_id:
                try:
                    st_resp = supabase.table("lab_stations").select("station_number, name").eq("id", station_id).single().execute()
                    if st_resp.data:
                        station_name = st_resp.data.get("name", station_name)
                except:
                    pass
            
            # Obtener status
            status = {}
            try:
                status_resp = supabase.table("lab_equipment_status").select("*").eq("equipment_id", equip["id"]).single().execute()
                if status_resp.data:
                    status = status_resp.data
            except:
                pass
            
            plcs.append({
                "id": equip.get("id"),
                "station_number": station_id,
                "station_name": station_name,
                "name": equip.get("name", "Unknown"),
                "model": equip.get("model", "Unknown"),
                "ip_address": equip.get("ip_address"),
                "firmware": equip.get("firmware_version"),
                "is_connected": status.get("is_connected", False),
                "run_mode": "RUN" if status.get("plc_run_mode") else "STOP",
                "status": status.get("status", "unknown"),
                "has_error": status.get("status") == "error" or status.get("plc_error_code") is not None
            })
        
        connected_count = sum(1 for p in plcs if p["is_connected"])
        error_count = sum(1 for p in plcs if p["has_error"])
        
        logger.info("lab_tools", f"PLCs procesados: {len(plcs)}, conectados: {connected_count}, con error: {error_count}")
        
        return {
            "success": True,
            "total": len(plcs),
            "connected": connected_count,
            "with_errors": error_count,
            "plcs": plcs
        }
        
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error("lab_tools", f"Error en get_all_plcs: {error_msg}")
        logger.error("lab_tools", traceback.format_exc())
        return {"success": False, "error": error_msg}


def get_all_cobots() -> Dict[str, Any]:
    """
    Lista todos los cobots del laboratorio con su estado actual.
    """
    try:
        supabase = get_supabase()
        
        # Query simplificado
        response = supabase.table("lab_equipment").select("*").eq("equipment_type", "cobot").execute()
        
        if not response.data:
            return {"success": False, "error": "No se encontraron cobots"}
        
        cobots = []
        for equip in response.data:
            station_id = equip.get("station_id")
            
            # Obtener info de estación
            station_name = f"Estación {station_id}"
            if station_id:
                try:
                    st_resp = supabase.table("lab_stations").select("station_number, name").eq("id", station_id).single().execute()
                    if st_resp.data:
                        station_name = st_resp.data.get("name", station_name)
                except:
                    pass
            
            # Obtener status
            status = {}
            try:
                status_resp = supabase.table("lab_equipment_status").select("*").eq("equipment_id", equip["id"]).single().execute()
                if status_resp.data:
                    status = status_resp.data
            except:
                pass
            
            cobots.append({
                "id": equip.get("id"),
                "station_number": station_id,
                "station_name": station_name,
                "name": equip.get("name", "Unknown"),
                "model": equip.get("model", "Unknown"),
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
        import traceback
        logger.error("lab_tools", f"Error en get_all_cobots: {type(e).__name__}: {e}")
        return {"success": False, "error": str(e)}


def get_active_errors() -> Dict[str, Any]:
    """
    Obtiene todos los errores activos del laboratorio.
    
    Returns:
        Lista de errores ordenados por severidad
    """
    try:
        supabase = get_supabase()
        
        # Query simple sin joins
        response = supabase.table("lab_errors").select("*").eq("status", "active").order("created_at", desc=True).execute()
        
        errors = []
        for err in response.data or []:
            # Obtener info de estación si existe
            station_name = None
            station_number = None
            if err.get("station_id"):
                try:
                    st_resp = supabase.table("lab_stations").select("station_number, name").eq("id", err["station_id"]).single().execute()
                    if st_resp.data:
                        station_name = st_resp.data.get("name")
                        station_number = st_resp.data.get("station_number")
                except:
                    pass
            
            # Obtener info de equipo si existe
            equipment_name = None
            equipment_type = None
            if err.get("equipment_id"):
                try:
                    eq_resp = supabase.table("lab_equipment").select("name, equipment_type").eq("id", err["equipment_id"]).single().execute()
                    if eq_resp.data:
                        equipment_name = eq_resp.data.get("name")
                        equipment_type = eq_resp.data.get("equipment_type")
                except:
                    pass
            
            errors.append({
                "error_id": err["id"],
                "station_number": station_number,
                "station_name": station_name,
                "equipment_name": equipment_name,
                "equipment_type": equipment_type,
                "error_code": err.get("error_code"),
                "message": err.get("error_message"),
                "severity": err.get("severity", "warning"),
                "triggered_by": err.get("triggered_by"),
                "created_at": err.get("created_at")
            })
        
        # Ordenar por severidad
        severity_order = {"critical": 0, "error": 1, "warning": 2, "info": 3}
        errors.sort(key=lambda x: severity_order.get(x.get("severity", "info"), 4))
        
        return {
            "success": True,
            "total_errors": len(errors),
            "critical_count": sum(1 for e in errors if e.get("severity") == "critical"),
            "errors": errors
        }
        
    except Exception as e:
        logger.error("lab_tools", f"Error en get_active_errors: {e}")
        return {"success": False, "error": str(e)}


def check_door_sensors(station_number: Optional[int] = None) -> Dict[str, Any]:
    """
    Verifica el estado de los sensores de puerta.
    
    Args:
        station_number: Si se especifica, solo esa estación. Si no, todas.
        
    Returns:
        Estado de sensores de puerta (triggered=true significa CERRADA)
    """
    try:
        supabase = get_supabase()
        
        # Obtener sensores de puerta
        query = supabase.table("lab_equipment").select("*").eq("equipment_type", "sensor").eq("sensor_type", "door")
        
        if station_number:
            # Primero obtener station_id
            st_resp = supabase.table("lab_stations").select("id").eq("station_number", station_number).single().execute()
            if st_resp.data:
                query = query.eq("station_id", st_resp.data["id"])
        
        response = query.execute()
        
        doors = []
        for equip in response.data or []:
            # Obtener status
            try:
                status_resp = supabase.table("lab_equipment_status").select("*").eq("equipment_id", equip["id"]).single().execute()
                status = status_resp.data or {}
            except:
                status = {}
            
            is_closed = status.get("sensor_triggered", False)
            doors.append({
                "station_number": equip.get("station_id"),
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


# ============================================
# COMANDOS Y ACCIONES
# ============================================

def set_cobot_mode(station_number: int, mode: int, requested_by: str = "agent") -> Dict[str, Any]:
    """
    Cambia el modo de operación de un cobot.
    
    Args:
        station_number: Número de estación (1-6)
        mode: 0=stop/idle, 1=rutina 1, 2=rutina 2, etc.
        requested_by: Quién solicita el cambio
        
    Returns:
        Resultado de la operación con validaciones de seguridad
    """
    try:
        supabase = get_supabase()
        
        # Primero verificar que la estación esté lista
        station_details = get_station_details(station_number)
        if not station_details.get("success"):
            return station_details
        
        # Obtener cobot_id de station_details
        cobot_id = station_details.get("cobot_id")
        if not cobot_id:
            return {"success": False, "error": f"No se encontró cobot en estación {station_number}"}
        
        # Si el modo > 0 (ejecutar), verificar seguridad
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
        
        # Mapeo de modos a nombres de rutina
        routine_names = {
            0: "idle",
            1: "pick_and_place",
            2: "assembly",
            3: "inspection",
            4: "packaging",
            5: "custom_routine"
        }
        
        routine_name = routine_names.get(mode, f"routine_{mode}")
        
        # Actualizar estado del cobot
        update_response = supabase.table("lab_equipment_status").update({
            "cobot_mode": mode,
            "cobot_routine_name": routine_name,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("equipment_id", cobot_id).execute()
        
        logger.info("lab_tools", f"Cobot {cobot_id} actualizado a modo {mode} ({routine_name})")
        
        # Registrar comando
        try:
            supabase.table("lab_cobot_commands").insert({
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
        import traceback
        logger.error("lab_tools", f"Error en set_cobot_mode: {type(e).__name__}: {e}")
        logger.error("lab_tools", traceback.format_exc())
        return {"success": False, "error": str(e)}


def acknowledge_error(error_id: int, acknowledged_by: str = "agent") -> Dict[str, Any]:
    """
    Marca un error como reconocido.
    """
    try:
        supabase = get_supabase()
        
        response = supabase.table("lab_errors").update({
            "status": "acknowledged",
            "acknowledged_by": acknowledged_by,
            "acknowledged_at": datetime.utcnow().isoformat()
        }).eq("id", error_id).execute()
        
        return {
            "success": True,
            "error_id": error_id,
            "message": f"Error {error_id} reconocido"
        }
        
    except Exception as e:
        logger.error("lab_tools", f"Error en acknowledge_error: {e}")
        return {"success": False, "error": str(e)}


def resolve_error(error_id: int, resolved_by: str = "agent", resolution_notes: str = "") -> Dict[str, Any]:
    """
    Marca un error como resuelto.
    """
    try:
        supabase = get_supabase()
        
        response = supabase.table("lab_errors").update({
            "status": "resolved",
            "resolved_by": resolved_by,
            "resolved_at": datetime.utcnow().isoformat(),
            "resolution_notes": resolution_notes
        }).eq("id", error_id).execute()
        
        return {
            "success": True,
            "error_id": error_id,
            "message": f"Error {error_id} marcado como resuelto"
        }
        
    except Exception as e:
        logger.error("lab_tools", f"Error en resolve_error: {e}")
        return {"success": False, "error": str(e)}


# ============================================
# ACCIONES DE REPARACIÓN / MANTENIMIENTO
# ============================================

def close_all_doors() -> Dict[str, Any]:
    """
    Cierra todas las puertas de seguridad del laboratorio.
    (Simula enviar señal a los actuadores de las puertas)
    """
    try:
        supabase = get_supabase()
        
        # Obtener IDs de sensores de puerta
        sensors = supabase.table("lab_equipment").select("id").eq("equipment_type", "sensor").eq("sensor_type", "door").execute()
        
        if not sensors.data:
            return {"success": False, "error": "No se encontraron sensores de puerta"}
        
        sensor_ids = [s["id"] for s in sensors.data]
        
        # Actualizar todos a triggered=true (puerta cerrada)
        for sensor_id in sensor_ids:
            supabase.table("lab_equipment_status").update({
                "sensor_triggered": True,
                "updated_at": datetime.utcnow().isoformat()
            }).eq("equipment_id", sensor_id).execute()
        
        logger.info("lab_tools", f"Cerradas {len(sensor_ids)} puertas")
        
        return {
            "success": True,
            "doors_closed": len(sensor_ids),
            "message": f"✅ Se cerraron {len(sensor_ids)} puertas de seguridad"
        }
        
    except Exception as e:
        logger.error("lab_tools", f"Error en close_all_doors: {e}")
        return {"success": False, "error": str(e)}


def reconnect_plc(station_number: int) -> Dict[str, Any]:
    """
    Intenta reconectar una PLC (simula reinicio de comunicación).
    """
    try:
        supabase = get_supabase()
        
        # Obtener station_id
        station = supabase.table("lab_stations").select("id").eq("station_number", station_number).single().execute()
        if not station.data:
            return {"success": False, "error": f"Estación {station_number} no encontrada"}
        
        # Obtener PLC de esta estación
        plc = supabase.table("lab_equipment").select("id, name").eq("station_id", station.data["id"]).eq("equipment_type", "plc").single().execute()
        if not plc.data:
            return {"success": False, "error": f"No hay PLC en estación {station_number}"}
        
        # Simular reconexión exitosa
        supabase.table("lab_equipment_status").update({
            "status": "online",
            "is_connected": True,
            "plc_run_mode": True,
            "plc_error_code": None,
            "last_heartbeat": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }).eq("equipment_id", plc.data["id"]).execute()
        
        logger.info("lab_tools", f"PLC {plc.data['name']} reconectada")
        
        return {
            "success": True,
            "station_number": station_number,
            "plc_name": plc.data["name"],
            "message": f"✅ PLC {plc.data['name']} de estación {station_number} reconectada exitosamente"
        }
        
    except Exception as e:
        logger.error("lab_tools", f"Error en reconnect_plc: {e}")
        return {"success": False, "error": str(e)}


def resolve_station_errors(station_number: int, resolved_by: str = "agent") -> Dict[str, Any]:
    """
    Resuelve todos los errores activos de una estación.
    """
    try:
        supabase = get_supabase()
        
        # Obtener station_id
        station = supabase.table("lab_stations").select("id").eq("station_number", station_number).single().execute()
        if not station.data:
            return {"success": False, "error": f"Estación {station_number} no encontrada"}
        
        # Contar errores activos
        errors = supabase.table("lab_errors").select("id").eq("station_id", station.data["id"]).eq("status", "active").execute()
        error_count = len(errors.data) if errors.data else 0
        
        if error_count == 0:
            return {"success": True, "resolved_count": 0, "message": f"No hay errores activos en estación {station_number}"}
        
        # Resolver todos
        supabase.table("lab_errors").update({
            "status": "resolved",
            "resolved_by": resolved_by,
            "resolved_at": datetime.utcnow().isoformat(),
            "resolution_notes": "Resuelto automáticamente por agente"
        }).eq("station_id", station.data["id"]).eq("status", "active").execute()
        
        logger.info("lab_tools", f"Resueltos {error_count} errores en estación {station_number}")
        
        return {
            "success": True,
            "station_number": station_number,
            "resolved_count": error_count,
            "message": f"✅ Se resolvieron {error_count} error(es) en estación {station_number}"
        }
        
    except Exception as e:
        logger.error("lab_tools", f"Error en resolve_station_errors: {e}")
        return {"success": False, "error": str(e)}


def resolve_all_errors(resolved_by: str = "agent") -> Dict[str, Any]:
    """
    Resuelve todos los errores activos del laboratorio.
    """
    try:
        supabase = get_supabase()
        
        # Contar errores activos
        errors = supabase.table("lab_errors").select("id").eq("status", "active").execute()
        error_count = len(errors.data) if errors.data else 0
        
        if error_count == 0:
            return {"success": True, "resolved_count": 0, "message": "No hay errores activos en el laboratorio"}
        
        # Resolver todos
        supabase.table("lab_errors").update({
            "status": "resolved",
            "resolved_by": resolved_by,
            "resolved_at": datetime.utcnow().isoformat(),
            "resolution_notes": "Resuelto automáticamente por agente"
        }).eq("status", "active").execute()
        
        logger.info("lab_tools", f"Resueltos {error_count} errores en todo el lab")
        
        return {
            "success": True,
            "resolved_count": error_count,
            "message": f"✅ Se resolvieron {error_count} error(es) en todo el laboratorio"
        }
        
    except Exception as e:
        logger.error("lab_tools", f"Error en resolve_all_errors: {e}")
        return {"success": False, "error": str(e)}


def reset_lab_to_safe_state() -> Dict[str, Any]:
    """
    Reinicia todo el laboratorio a un estado seguro y operativo.
    
    Acciones:
    1. Cierra todas las puertas
    2. Reconecta todas las PLCs
    3. Pone todos los cobots en idle
    4. Resuelve todos los errores activos
    """
    try:
        supabase = get_supabase()
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
        plcs = supabase.table("lab_equipment").select("id").eq("equipment_type", "plc").execute()
        for plc in plcs.data or []:
            supabase.table("lab_equipment_status").update({
                "status": "online",
                "is_connected": True,
                "plc_run_mode": True,
                "plc_error_code": None,
                "last_heartbeat": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }).eq("equipment_id", plc["id"]).execute()
            results["plcs_reconnected"] += 1
        
        # 3. Poner cobots en idle
        cobots = supabase.table("lab_equipment").select("id").eq("equipment_type", "cobot").execute()
        for cobot in cobots.data or []:
            supabase.table("lab_equipment_status").update({
                "status": "online",
                "is_connected": True,
                "cobot_mode": 0,
                "cobot_routine_name": "idle",
                "updated_at": datetime.utcnow().isoformat()
            }).eq("equipment_id", cobot["id"]).execute()
            results["cobots_stopped"] += 1
        
        # 4. Resolver todos los errores
        errors = supabase.table("lab_errors").select("id").eq("status", "active").execute()
        if errors.data:
            supabase.table("lab_errors").update({
                "status": "resolved",
                "resolved_by": "system_reset",
                "resolved_at": datetime.utcnow().isoformat(),
                "resolution_notes": "Reset completo del laboratorio"
            }).eq("status", "active").execute()
            results["errors_resolved"] = len(errors.data)
        
        logger.info("lab_tools", f"Lab reset completado: {results}")
        
        return {
            "success": True,
            "results": results,
            "message": f"✅ Laboratorio reiniciado a estado seguro"
        }
        
    except Exception as e:
        logger.error("lab_tools", f"Error en reset_lab_to_safe_state: {e}")
        return {"success": False, "error": str(e)}


def diagnose_and_suggest_fixes(station_number: Optional[int] = None) -> Dict[str, Any]:
    """
    Diagnostica problemas y sugiere acciones de reparación.
    """
    try:
        problems = []
        suggested_actions = []
        
        if station_number:
            # Diagnóstico de estación específica
            details = get_station_details(station_number)
            if not details.get("success"):
                return details
            
            ready_details = details.get("ready_details", {})
            
            if not ready_details.get("doors_closed"):
                problems.append(f"Puerta de estación {station_number} abierta")
                suggested_actions.append({"action": "close_doors", "description": "Cerrar puertas de seguridad"})
            
            if not ready_details.get("plc_connected"):
                problems.append(f"PLC de estación {station_number} desconectada")
                suggested_actions.append({"action": "reconnect_plc", "station": station_number, "description": f"Reconectar PLC"})
            
            if not ready_details.get("no_active_errors"):
                error_count = len(details.get("active_errors", []))
                problems.append(f"{error_count} error(es) activo(s) en estación {station_number}")
                suggested_actions.append({"action": "resolve_errors", "station": station_number, "description": f"Resolver errores"})
        
        else:
            # Diagnóstico de todo el lab
            overview = get_lab_overview()
            if not overview.get("success"):
                return overview
            
            # Verificar puertas
            doors = check_door_sensors()
            if doors.get("success") and not doors.get("all_doors_closed"):
                open_count = doors.get("open_doors_count", 0)
                problems.append(f"{open_count} puerta(s) abierta(s)")
                suggested_actions.append({"action": "close_all_doors", "description": "Cerrar todas las puertas"})
            
            # Verificar PLCs
            plcs = get_all_plcs()
            if plcs.get("success"):
                disconnected = [p for p in plcs.get("plcs", []) if not p.get("is_connected")]
                for plc in disconnected:
                    problems.append(f"PLC de estación {plc['station_number']} desconectada")
                    suggested_actions.append({"action": "reconnect_plc", "station": plc['station_number'], "description": f"Reconectar PLC estación {plc['station_number']}"})
            
            # Verificar errores
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


# ============================================
# FORMATTERS PARA DISPLAY
# ============================================

def format_lab_overview_for_display(overview: Dict) -> str:
    """Formatea el resumen del lab para mostrar al usuario"""
    if not overview.get("success"):
        return f"❌ Error al obtener estado del laboratorio: {overview.get('error')}"
    
    lines = ["## 🏭 Estado del Laboratorio ATLAS\n"]
    
    # Resumen general
    lines.append(f"**Estaciones activas:** {overview['stations_online']}/{overview['total_stations']}")
    
    if overview['stations_with_errors'] > 0:
        lines.append(f"⚠️ **Estaciones con problemas:** {overview['stations_with_errors']}")
    
    if overview['active_errors_count'] > 0:
        lines.append(f"🔴 **Errores activos:** {overview['active_errors_count']}")
    
    lines.append("\n### Detalle por Estación\n")
    lines.append("| # | Nombre | PLC | Cobot | Puertas | Estado |")
    lines.append("|---|--------|-----|-------|---------|--------|")
    
    for station in overview['stations']:
        status_icon = "✅" if station['is_operational'] else "⚠️"
        doors_icon = "🔒" if station['doors_closed'] else "🚪"
        plc_icon = "🟢" if station['plc_status'] == "online" else "🔴"
        cobot_icon = "🟢" if station['cobot_status'] == "online" else "🔴"
        
        lines.append(
            f"| {station['station_number']} | {station['name'][:20]} | "
            f"{plc_icon} | {cobot_icon} | {doors_icon} | {status_icon} |"
        )
    
    return "\n".join(lines)


def format_station_details_for_display(details: Dict) -> str:
    """Formatea los detalles de una estación para mostrar al usuario"""
    if not details.get("success"):
        return f"❌ Error: {details.get('error')}"
    
    station = details['station']
    lines = [f"## 🏭 Estación {station['number']} - {station['name']}\n"]
    lines.append(f"📍 Ubicación: {station['location']}")
    lines.append(f"📝 {station['description']}\n")
    
    # PLC
    plc = details.get('plc')
    if plc:
        status_icon = "🟢" if plc['is_connected'] else "🔴"
        lines.append(f"### 🖥️ PLC: {plc['name']}")
        lines.append(f"- Modelo: {plc['model']}")
        lines.append(f"- IP: {plc['ip_address']}")
        lines.append(f"- Estado: {status_icon} {plc['status']} ({plc['run_mode']})")
        if plc.get('error_code'):
            lines.append(f"- ⚠️ Error: {plc['error_code']}")
        lines.append("")
    
    # Cobot
    cobot = details.get('cobot')
    if cobot:
        status_icon = "🟢" if cobot['is_connected'] else "🔴"
        mode_icon = "▶️" if cobot['mode'] > 0 else "⏹️"
        lines.append(f"### 🤖 Cobot: {cobot['name']}")
        lines.append(f"- Modelo: {cobot['model']}")
        lines.append(f"- IP: {cobot['ip_address']}")
        lines.append(f"- Estado: {status_icon} {cobot['status']}")
        lines.append(f"- Modo: {mode_icon} {cobot['mode']} ({cobot['routine']})")
        lines.append("")
    
    # Sensores
    sensors = details.get('sensors', [])
    if sensors:
        lines.append("### 📡 Sensores")
        for sensor in sensors:
            if sensor['type'] == 'door':
                icon = "🔒" if sensor['triggered'] else "🚪"
                status = "Cerrada" if sensor['triggered'] else "ABIERTA"
            else:
                icon = "✓" if sensor['triggered'] else "○"
                status = "Activo" if sensor['triggered'] else "Inactivo"
            lines.append(f"- {icon} {sensor['name']}: {status} ({sensor['location']})")
        lines.append("")
    
    # Errores
    errors = details.get('active_errors', [])
    if errors:
        lines.append("### ⚠️ Errores Activos")
        for err in errors:
            lines.append(f"- 🔴 [{err['severity']}] {err['error_code']}: {err['error_message']}")
        lines.append("")
    
    # Estado operativo
    ready = details['ready_to_operate']
    ready_icon = "✅" if ready else "❌"
    lines.append(f"### {ready_icon} Estado Operativo: {'LISTA' if ready else 'NO LISTA'}")
    if not ready:
        rd = details['ready_details']
        if not rd['doors_closed']:
            lines.append("- ⚠️ Puertas no cerradas")
        if not rd['plc_connected']:
            lines.append("- ⚠️ PLC no conectada")
        if not rd['no_active_errors']:
            lines.append("- ⚠️ Hay errores activos")
    
    return "\n".join(lines)


def format_errors_for_display(errors_data: Dict) -> str:
    """Formatea los errores para mostrar al usuario"""
    if not errors_data.get("success"):
        return f"❌ Error: {errors_data.get('error')}"
    
    if errors_data['total_errors'] == 0:
        return "✅ No hay errores activos en el laboratorio"
    
    lines = [f"## ⚠️ Errores Activos ({errors_data['total_errors']})\n"]
    
    if errors_data['critical_count'] > 0:
        lines.append(f"🔴 **CRÍTICOS:** {errors_data['critical_count']}\n")
    
    for err in errors_data['errors']:
        severity_icon = {
            "critical": "🔴",
            "error": "🟠",
            "warning": "🟡",
            "info": "🔵"
        }.get(err['severity'], "⚪")
        
        lines.append(f"### {severity_icon} Estación {err['station_number']} - {err['equipment_name']}")
        lines.append(f"- **Código:** {err['error_code']}")
        lines.append(f"- **Mensaje:** {err['message']}")
        lines.append(f"- **Creado:** {err['created_at']}")
        lines.append("")
    
    return "\n".join(lines)
