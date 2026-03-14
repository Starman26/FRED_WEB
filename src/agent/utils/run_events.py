"""
run_events.py - Structured event constructors for state["events"].
"""
from typing import Dict, Any, Optional
from datetime import datetime


def _create_event(
    event_type: str,
    source: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a structured event dict."""
    return {
        "type": event_type,
        "source": source,
        "content": content,
        "timestamp": datetime.utcnow().isoformat(),
        "metadata": metadata or {}
    }


def event_read(source: str, content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """Data read/fetch event."""
    return _create_event("read", source, content, metadata)


def event_execute(source: str, content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """Action execution event."""
    return _create_event("execute", source, content, metadata)


def event_report(source: str, content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """Report/result event."""
    return _create_event("report", source, content, metadata)


def event_error(source: str, content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """Error event."""
    return _create_event("error", source, content, metadata)


def event_plan(source: str, content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """Planning event."""
    return _create_event("plan", source, content, metadata)


def event_route(source: str, content: str, route: str = "", metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """Routing event."""
    meta = metadata or {}
    meta["route"] = route
    return _create_event("route", source, content, meta)


def event_narration(source: str, content: str, phase: str = "thinking") -> Dict[str, Any]:
    """Chain-of-thought narration event for the chat UI."""
    return {
        "type": "narration",
        "source": source,
        "content": content,
        "phase": phase,
        "timestamp": datetime.utcnow().isoformat(),
    }
