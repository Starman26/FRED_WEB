"""
shared_state.py - Registros compartidos entre api_server y agent workers.

Evita imports circulares: api_server escribe aquí, los workers leen de aquí.
NO importar api_server desde modules dentro de src/agent/.
"""
from typing import Dict, Any
import threading

# Robot bridge WebSocket connections: robot_id → WebSocket
# Written by api_server on bridge connect/disconnect
ROBOT_CONNECTIONS: Dict[str, Any] = {}

# Robot metadata: robot_id → {"type": "xarm", "model": "xArm6", "capabilities": [...], "last_heartbeat": ...}
ROBOT_METADATA: Dict[str, dict] = {}

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
