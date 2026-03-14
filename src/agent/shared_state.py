"""
shared_state.py

Shared registries between api_server and agent workers.
Breaks circular imports: api_server writes here, workers read from here.
"""
from typing import Dict, Any
import threading

ROBOT_CONNECTIONS: Dict[str, Any] = {}  # robot_id -> WebSocket
ROBOT_METADATA: Dict[str, dict] = {}    # robot_id -> metadata dict

_lock = threading.Lock()


def register_robot(robot_id: str, ws: Any, metadata: dict = None):
    """Register a robot bridge connection (called from api_server)."""
    with _lock:
        ROBOT_CONNECTIONS[robot_id] = ws
        if metadata:
            ROBOT_METADATA[robot_id] = metadata


def unregister_robot(robot_id: str):
    """Unregister a robot bridge connection (called from api_server)."""
    with _lock:
        ROBOT_CONNECTIONS.pop(robot_id, None)
        ROBOT_METADATA.pop(robot_id, None)


def get_connected_robots() -> Dict[str, dict]:
    """Return snapshot of connected robots with metadata (safe for workers)."""
    with _lock:
        connected = {}
        for rid in ROBOT_CONNECTIONS:
            meta = ROBOT_METADATA.get(rid, {})
            connected[rid] = {
                "robot_id": rid,
                "type": meta.get("type", "unknown"),
                "model": meta.get("model", ""),
                "capabilities": meta.get("capabilities", []),
                "is_connected": True,
                "last_heartbeat": meta.get("last_heartbeat"),
            }
        return connected
