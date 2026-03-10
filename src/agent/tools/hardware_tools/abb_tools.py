"""
abb_tools.py — Agent tools for ABB IRB robots.

Communicates via edge_router → lab bridge → RobotController (socket TCP:1025).
ABB uses TCP cartesian coordinates + quaternion/euler orientation.
"""

import json
import math
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from .edge_router import send_command, register_mock_handler


# ═══════════════════════════════════════════════════════════════
# Euler ↔ Quaternion (ABB ZYX convention) — for mock display
# ═══════════════════════════════════════════════════════════════

def _euler_to_quat(ex: float, ey: float, ez: float):
    """ABB Euler (deg, ZYX extrinsic) → quaternion (q1=w, q2=x, q3=y, q4=z)."""
    hx, hy, hz = math.radians(ex / 2), math.radians(ey / 2), math.radians(ez / 2)
    cx, sx = math.cos(hx), math.sin(hx)
    cy, sy = math.cos(hy), math.sin(hy)
    cz, sz = math.cos(hz), math.sin(hz)
    return (
        cz * cy * cx + sz * sy * sx,
        cz * cy * sx - sz * sy * cx,
        cz * sy * cx + sz * cy * sx,
        sz * cy * cx - cz * sy * sx,
    )


def _quat_to_euler(q1, q2, q3, q4):
    """Quaternion → ABB Euler (deg)."""
    sinp = max(-1.0, min(1.0, 2.0 * (q1 * q3 - q4 * q2)))
    ey = math.degrees(math.asin(sinp))
    if abs(sinp) > 0.9999:
        ex = math.degrees(math.atan2(2.0 * (q2 * q3 + q1 * q4), 1.0 - 2.0 * (q3**2 + q4**2)))
        ez = 0.0
    else:
        ex = math.degrees(math.atan2(2.0 * (q1 * q2 + q3 * q4), 1.0 - 2.0 * (q2**2 + q3**2)))
        ez = math.degrees(math.atan2(2.0 * (q1 * q4 + q2 * q3), 1.0 - 2.0 * (q3**2 + q4**2)))
    return round(ex, 2), round(ey, 2), round(ez, 2)


# ═══════════════════════════════════════════════════════════════
# Mock state
# ═══════════════════════════════════════════════════════════════

_DEFAULT_EULER = (180.0, 0.0, 0.0)
_DEFAULT_QUAT = _euler_to_quat(*_DEFAULT_EULER)

_mock_state: Dict[str, Any] = {
    "x": 400.0, "y": 0.0, "z": 500.0,
    "q1": _DEFAULT_QUAT[0], "q2": _DEFAULT_QUAT[1],
    "q3": _DEFAULT_QUAT[2], "q4": _DEFAULT_QUAT[3],
    "speed": 100,
    "connected": True,
}


def _mock_pos_dict() -> dict:
    s = _mock_state
    ex, ey, ez = _quat_to_euler(s["q1"], s["q2"], s["q3"], s["q4"])
    return {
        "x": s["x"], "y": s["y"], "z": s["z"],
        "q1": round(s["q1"], 6), "q2": round(s["q2"], 6),
        "q3": round(s["q3"], 6), "q4": round(s["q4"], 6),
        "euler": {"ex": ex, "ey": ey, "ez": ez},
    }


# ═══════════════════════════════════════════════════════════════
# Mock handlers
# ═══════════════════════════════════════════════════════════════

def _mock_get_position(params: dict, device_id: str) -> dict:
    return {"position": _mock_pos_dict(), "state": "static"}


def _mock_move_linear(params: dict, device_id: str) -> dict:
    x, y, z = params["x"], params["y"], params["z"]
    # Resolve orientation
    if "q1" in params:
        q = (params["q1"], params.get("q2", 0), params.get("q3", 0), params.get("q4", 0))
    elif "ex" in params:
        q = _euler_to_quat(
            params.get("ex", _DEFAULT_EULER[0]),
            params.get("ey", _DEFAULT_EULER[1]),
            params.get("ez", _DEFAULT_EULER[2]),
        )
    else:
        q = (_mock_state["q1"], _mock_state["q2"], _mock_state["q3"], _mock_state["q4"])

    _mock_state.update({"x": x, "y": y, "z": z, "q1": q[0], "q2": q[1], "q3": q[2], "q4": q[3]})
    return {"command": "MOVEL", "position": _mock_pos_dict(), "state": "static"}


def _mock_move_joint(params: dict, device_id: str) -> dict:
    x, y, z = params["x"], params["y"], params["z"]
    if "q1" in params:
        q = (params["q1"], params.get("q2", 0), params.get("q3", 0), params.get("q4", 0))
    elif "ex" in params:
        q = _euler_to_quat(
            params.get("ex", _DEFAULT_EULER[0]),
            params.get("ey", _DEFAULT_EULER[1]),
            params.get("ez", _DEFAULT_EULER[2]),
        )
    else:
        q = (_mock_state["q1"], _mock_state["q2"], _mock_state["q3"], _mock_state["q4"])

    _mock_state.update({"x": x, "y": y, "z": z, "q1": q[0], "q2": q[1], "q3": q[2], "q4": q[3]})
    return {"command": "MOVEJ", "position": _mock_pos_dict(), "state": "static"}


def _mock_go_home(params: dict, device_id: str) -> dict:
    _mock_state.update({
        "x": 400.0, "y": 0.0, "z": 500.0,
        "q1": _DEFAULT_QUAT[0], "q2": _DEFAULT_QUAT[1],
        "q3": _DEFAULT_QUAT[2], "q4": _DEFAULT_QUAT[3],
    })
    return {"command": "HOME", "position": _mock_pos_dict(), "state": "static"}


def _mock_set_speed(params: dict, device_id: str) -> dict:
    speed = params.get("speed", 100)
    _mock_state["speed"] = speed
    return {"speed_set": speed}


# Register mocks
for action, handler in {
    "get_position": _mock_get_position,
    "move_linear": _mock_move_linear,
    "move_joint": _mock_move_joint,
    "home": _mock_go_home,
    "set_speed": _mock_set_speed,
}.items():
    register_mock_handler("abb", action, handler)


# ═══════════════════════════════════════════════════════════════
# Agent tools
# ═══════════════════════════════════════════════════════════════

def _send(action: str, params: dict = None, device_id: str = "") -> str:
    result = send_command("abb", action, params or {}, device_id)
    return json.dumps(result, ensure_ascii=False)


@tool
def abb_get_position(device_id: str = "") -> str:
    """Read the current TCP position of an ABB robot.

    Args:
        device_id: ABB device ID (e.g. "abb-01").

    Returns:
        JSON with position {x, y, z, q1-q4, euler {ex, ey, ez}}.
    """
    return _send("get_position", device_id=device_id)


@tool
def abb_move_linear(
    x: float, y: float, z: float,
    ex: float = 180.0, ey: float = 0.0, ez: float = 0.0,
    device_id: str = "",
) -> str:
    """Move ABB robot TCP in a straight line (MoveL) to XYZ with Euler orientation.

    Uses ABB ZYX Euler convention (same as FlexPendant display).

    Args:
        x: Target X in mm.
        y: Target Y in mm.
        z: Target Z in mm.
        ex: Euler X rotation in degrees (default 180).
        ey: Euler Y rotation in degrees (default 0).
        ez: Euler Z rotation in degrees (default 0).
        device_id: ABB device ID.

    Returns:
        JSON with confirmed position after move.
    """
    return _send("move_linear", {"x": x, "y": y, "z": z, "ex": ex, "ey": ey, "ez": ez}, device_id)


@tool
def abb_move_joint(
    x: float, y: float, z: float,
    ex: float = 180.0, ey: float = 0.0, ez: float = 0.0,
    device_id: str = "",
) -> str:
    """Move ABB robot via joint interpolation (MoveJ) to XYZ with Euler orientation.

    Faster than linear for large movements. Uses ABB ZYX Euler convention.

    Args:
        x: Target X in mm.
        y: Target Y in mm.
        z: Target Z in mm.
        ex: Euler X rotation in degrees (default 180).
        ey: Euler Y rotation in degrees (default 0).
        ez: Euler Z rotation in degrees (default 0).
        device_id: ABB device ID.

    Returns:
        JSON with confirmed position after move.
    """
    return _send("move_joint", {"x": x, "y": y, "z": z, "ex": ex, "ey": ey, "ez": ez}, device_id)


@tool
def abb_go_home(device_id: str = "") -> str:
    """Send ABB robot to HOME_TARGET defined in RAPID module.

    Args:
        device_id: ABB device ID.

    Returns:
        JSON with confirmed home position.
    """
    return _send("home", device_id=device_id)


@tool
def abb_set_speed(speed: int = 100, device_id: str = "") -> str:
    """Set ABB robot TCP speed in mm/s.

    RAPID maps to nearest speeddata (v10, v20, v50, v100, v200, v500, v1000).

    Args:
        speed: Speed in mm/s (default 100).
        device_id: ABB device ID.

    Returns:
        JSON confirming speed set.
    """
    return _send("set_speed", {"speed": speed}, device_id)


# ═══════════════════════════════════════════════════════════════
# Exports
# ═══════════════════════════════════════════════════════════════

ABB_READ_TOOLS = [abb_get_position]
ABB_ACTUATE_TOOLS = [abb_move_linear, abb_move_joint, abb_go_home]
ABB_WRITE_TOOLS = [abb_set_speed]

ABB_TOOLS = ABB_READ_TOOLS + ABB_ACTUATE_TOOLS + ABB_WRITE_TOOLS
