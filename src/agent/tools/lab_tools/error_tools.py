"""
error_tools.py - Gestión de errores del laboratorio

Reads usan RPC (1 request), writes usan queries directas con schema('lab').
"""
from typing import Dict, Any, Optional
from datetime import datetime

from src.agent.services import get_supabase
from src.agent.utils.logger import logger


def _lab(supabase):
    """Helper: returns supabase client scoped to 'lab' schema."""
    return supabase.schema("lab")


def get_active_errors() -> Dict[str, Any]:
    """Obtiene todos los errores activos en 1 sola llamada RPC."""
    try:
        supabase = get_supabase()
        
        resp = _lab(supabase).rpc("get_active_errors", {}).execute()
        errors_raw = resp.data or []
        
        # Si viene como lista envuelta, desempacar
        if isinstance(errors_raw, list) and len(errors_raw) == 1 and isinstance(errors_raw[0], list):
            errors_raw = errors_raw[0]
        
        errors = []
        for err in errors_raw:
            errors.append({
                "error_id": err.get("id"),
                "station_number": err.get("station_number"),
                "station_name": err.get("station_name"),
                "equipment_name": err.get("equipment_name"),
                "equipment_type": err.get("equipment_type"),
                "error_code": err.get("error_code"),
                "message": err.get("description"),
                "severity": err.get("severity", "warning"),
                "triggered_by": err.get("triggered_by"),
                "created_at": err.get("created_at")
            })
        
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


def acknowledge_error(error_id: int, acknowledged_by: str = "agent") -> Dict[str, Any]:
    """Marca un error como reconocido."""
    try:
        supabase = get_supabase()
        _lab(supabase).table("errors").update({
            "status": "acknowledged",
            "acknowledged_by": acknowledged_by,
            "acknowledged_at": datetime.utcnow().isoformat()
        }).eq("id", error_id).execute()
        
        return {"success": True, "error_id": error_id, "message": f"Error {error_id} reconocido"}
    except Exception as e:
        logger.error("lab_tools", f"Error en acknowledge_error: {e}")
        return {"success": False, "error": str(e)}


def resolve_error(error_id: int, resolved_by: str = "agent", resolution_notes: str = "") -> Dict[str, Any]:
    """Marca un error como resuelto."""
    try:
        supabase = get_supabase()
        _lab(supabase).table("errors").update({
            "status": "resolved",
            "resolved_by": resolved_by,
            "resolved_at": datetime.utcnow().isoformat(),
            "resolution_notes": resolution_notes
        }).eq("id", error_id).execute()
        
        return {"success": True, "error_id": error_id, "message": f"Error {error_id} marcado como resuelto"}
    except Exception as e:
        logger.error("lab_tools", f"Error en resolve_error: {e}")
        return {"success": False, "error": str(e)}


def resolve_station_errors(station_number: int, resolved_by: str = "agent") -> Dict[str, Any]:
    """Resuelve todos los errores activos de una estación."""
    try:
        supabase = get_supabase()
        lab = _lab(supabase)
        
        station = lab.table("stations").select("id").eq("station_number", station_number).single().execute()
        if not station.data:
            return {"success": False, "error": f"Estación {station_number} no encontrada"}
        
        errors = lab.table("errors").select("id").eq("station_id", station.data["id"]).eq("status", "active").execute()
        error_count = len(errors.data) if errors.data else 0
        
        if error_count == 0:
            return {"success": True, "resolved_count": 0, "message": f"No hay errores activos en estación {station_number}"}
        
        lab.table("errors").update({
            "status": "resolved",
            "resolved_by": resolved_by,
            "resolved_at": datetime.utcnow().isoformat(),
            "resolution_notes": "Resuelto automáticamente por agente"
        }).eq("station_id", station.data["id"]).eq("status", "active").execute()
        
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
    """Resuelve todos los errores activos del laboratorio."""
    try:
        supabase = get_supabase()
        lab = _lab(supabase)
        
        errors = lab.table("errors").select("id").eq("status", "active").execute()
        error_count = len(errors.data) if errors.data else 0
        
        if error_count == 0:
            return {"success": True, "resolved_count": 0, "message": "No hay errores activos en el laboratorio"}
        
        lab.table("errors").update({
            "status": "resolved",
            "resolved_by": resolved_by,
            "resolved_at": datetime.utcnow().isoformat(),
            "resolution_notes": "Resuelto automáticamente por agente"
        }).eq("status", "active").execute()
        
        return {
            "success": True,
            "resolved_count": error_count,
            "message": f"✅ Se resolvieron {error_count} error(es) en todo el laboratorio"
        }
    except Exception as e:
        logger.error("lab_tools", f"Error en resolve_all_errors: {e}")
        return {"success": False, "error": str(e)}
