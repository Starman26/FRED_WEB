"""
run_events.py - Eventos estilo Manus para UI y debugging

Estos eventos se acumulan en state["events"] y pueden ser consumidos por el frontend.
"""
from typing import Dict, Any, Optional
from datetime import datetime


def _create_event(
    event_type: str,
    source: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Crea un evento estructurado"""
    return {
        "type": event_type,
        "source": source,
        "content": content,
        "timestamp": datetime.utcnow().isoformat(),
        "metadata": metadata or {}
    }


def event_read(source: str, content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """Evento de lectura/obtención de datos"""
    return _create_event("read", source, content, metadata)


def event_execute(source: str, content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """Evento de ejecución de acción"""
    return _create_event("execute", source, content, metadata)


def event_report(source: str, content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """Evento de reporte/resultado"""
    return _create_event("report", source, content, metadata)


def event_error(source: str, content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """Evento de error"""
    return _create_event("error", source, content, metadata)


def event_plan(source: str, content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """Evento de planificación"""
    return _create_event("plan", source, content, metadata)


def event_route(source: str, content: str, route: str = "", metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """Evento de routing"""
    meta = metadata or {}
    meta["route"] = route
    return _create_event("route", source, content, meta)


def event_narration(source: str, content: str, phase: str = "thinking") -> Dict[str, Any]:
    """Evento de narración chain-of-thought para mostrar en el chat del usuario."""
    return {
        "type": "narration",
        "source": source,
        "content": content,
        "phase": phase,
        "timestamp": datetime.utcnow().isoformat(),
    }
