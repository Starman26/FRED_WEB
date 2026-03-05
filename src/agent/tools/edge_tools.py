"""
edge_tools.py — Edge computing bridge for practice mode.

Sends commands to the edge layer (robot controller) and returns
feedback + robot state.  Currently mocked; flip MOCK_MODE to False
and implement _http_send() when a real edge device is available.
"""
import json
import os
import asyncio
import concurrent.futures
from typing import Dict, Any
from langchain_core.tools import tool

# ── Feature flag ──────────────────────────────────────────────
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
LAB_BRIDGE_URL = os.getenv("LAB_BRIDGE_URL", "")

# ── Command catalogue (documentation only) ────────────────────
EDGE_COMMANDS: Dict[str, str] = {
    "highlight_joint": "Mueve ligeramente un joint para mostrarlo",
    "demo_movement":   "Ejecuta movimiento de demostración",
    "go_to_pose":      "Mueve a una pose predefinida (home, pick, place, etc)",
    "show_workspace":  "Muestra rango de movimiento de un joint/eje",
    "gripper_demo":    "Demuestra apertura/cierre del gripper",
    "home":            "Vuelve a posición home",
    "move_joint":      "Mueve un joint específico a un ángulo",
    "move_linear":     "Movimiento lineal en X/Y/Z",
    "pause":           "Pausa movimiento actual",
    "freeze":          "Emergencia suave — congela posición",
}

# ── Joint metadata ────────────────────────────────────────────
_JOINT_NAMES = {
    1: "Base",
    2: "Shoulder",
    3: "Elbow",
    4: "Wrist Roll",
    5: "Wrist Pitch",
    6: "Wrist Yaw",
}

_JOINT_LIMITS = {
    1: (-360, 360),
    2: (-118, 120),
    3: (-225, 11),
    4: (-360, 360),
    5: (-97, 180),
    6: (-360, 360),
}

_POSES = {
    "home":    [0, 0, 0, 0, 0, 0],
    "pick":    [0, -30, 60, 0, 90, 0],
    "place":   [90, -30, 60, 0, 90, 0],
    "ready":   [0, -45, 30, 0, 45, 0],
    "inspect": [0, -60, 90, 0, 30, 0],
}

DEFAULT_ROBOT_STATE: Dict[str, Any] = {
    "joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "mode": "teaching",
    "speed_limit": 25,
    "gripper_open": True,
    "connected": True,
}

# Mutable state for mock session
_mock_state = dict(DEFAULT_ROBOT_STATE)


# ── Mock handlers ─────────────────────────────────────────────

def _ok(feedback: str, state_updates: dict = None, safety: str = None) -> dict:
    if state_updates:
        _mock_state.update(state_updates)
    return {
        "status": "executed",
        "feedback": feedback,
        "robot_state": dict(_mock_state),
        "safety_note": safety,
    }


def _mock_home(params: dict) -> dict:
    joints = [0.0] * 6
    return _ok(
        f"Robot en posición home {joints}",
        {"joints": joints, "mode": "teaching"},
    )


def _mock_highlight_joint(params: dict) -> dict:
    jid = params.get("joint_id", 1)
    name = _JOINT_NAMES.get(jid, f"Joint {jid}")
    return _ok(
        f"Joint {jid} ({name}) oscilando ±10° a velocidad segura",
        safety="Velocidad limitada a 10% durante demostración",
    )


def _mock_demo_movement(params: dict) -> dict:
    pattern = params.get("pattern", "pick_and_place")
    return _ok(
        f"Ejecutando demostración '{pattern}': home → pick → place → home",
        {"mode": "demo"},
        safety="Movimiento a 15% velocidad, zona segura verificada",
    )


def _mock_go_to_pose(params: dict) -> dict:
    pose_name = params.get("pose", "home")
    joints = _POSES.get(pose_name, _POSES["home"])
    return _ok(
        f"Robot en pose '{pose_name}': {joints}",
        {"joints": list(joints)},
    )


def _mock_show_workspace(params: dict) -> dict:
    jid = params.get("joint_id", 1)
    name = _JOINT_NAMES.get(jid, f"Joint {jid}")
    lo, hi = _JOINT_LIMITS.get(jid, (-180, 180))
    return _ok(
        f"Joint {jid} ({name}): rango [{lo}°, {hi}°]. Moviendo lentamente de {lo}° a {hi}°.",
        safety="Movimiento lento para demostración de rango",
    )


def _mock_gripper_demo(params: dict) -> dict:
    return _ok(
        "Gripper: apertura 100% → cierre 0% → apertura 100%",
        {"gripper_open": True},
    )


def _mock_move_joint(params: dict) -> dict:
    jid = params.get("joint_id", 1)
    angle = float(params.get("angle", 0))
    name = _JOINT_NAMES.get(jid, f"Joint {jid}")
    joints = list(_mock_state["joints"])
    if 1 <= jid <= 6:
        joints[jid - 1] = angle
    return _ok(
        f"Joint {jid} ({name}) movido a {angle:.1f}°",
        {"joints": joints},
    )


def _mock_move_linear(params: dict) -> dict:
    x = params.get("x", 0)
    y = params.get("y", 0)
    z = params.get("z", 0)
    return _ok(
        f"Movimiento lineal a X={x}, Y={y}, Z={z} mm",
        safety="Verificar espacio libre antes de movimiento lineal",
    )


def _mock_pause(params: dict) -> dict:
    return _ok("Movimiento pausado", {"mode": "paused"})


def _mock_freeze(params: dict) -> dict:
    return _ok(
        "FREEZE activado — robot congelado en posición actual",
        {"mode": "frozen", "speed_limit": 0},
        safety="Emergencia suave. Usar 'home' para retomar.",
    )


_MOCK_DISPATCH = {
    "home": _mock_home,
    "highlight_joint": _mock_highlight_joint,
    "demo_movement": _mock_demo_movement,
    "go_to_pose": _mock_go_to_pose,
    "show_workspace": _mock_show_workspace,
    "gripper_demo": _mock_gripper_demo,
    "move_joint": _mock_move_joint,
    "move_linear": _mock_move_linear,
    "pause": _mock_pause,
    "freeze": _mock_freeze,
}


# ── Tool ──────────────────────────────────────────────────────

@tool
def edge_command(command: str, params: str = "{}", robot_ids: str = "") -> str:
    """Send a command to the robot via the edge computing layer.

    Args:
        command: One of: highlight_joint, demo_movement, go_to_pose,
                 show_workspace, gripper_demo, home, move_joint,
                 move_linear, pause, freeze.
        params: JSON string with command-specific parameters.
                Examples:
                  highlight_joint: {"joint_id": 1}
                  move_joint: {"joint_id": 3, "angle": 45}
                  go_to_pose: {"pose": "pick"}
                  move_linear: {"x": 100, "y": 0, "z": 200}
        robot_ids: Comma-separated robot IDs (e.g. "xarm-201" or "xarm-201,xarm-202").

    Returns:
        JSON string with status, feedback, robot_state, and safety_note.
    """
    try:
        parsed_params = json.loads(params) if isinstance(params, str) else params
    except json.JSONDecodeError:
        parsed_params = {}

    if command not in EDGE_COMMANDS:
        result = {
            "status": "rejected",
            "feedback": f"Comando desconocido: '{command}'. Disponibles: {', '.join(EDGE_COMMANDS)}",
            "robot_state": dict(_mock_state),
            "safety_note": None,
        }
        return json.dumps(result, ensure_ascii=False)

    if MOCK_MODE:
        handler = _MOCK_DISPATCH.get(command)
        result = handler(parsed_params) if handler else {
            "status": "rejected",
            "feedback": f"Handler no implementado para '{command}'",
            "robot_state": dict(_mock_state),
            "safety_note": None,
        }
    else:
        # Future: HTTP POST to edge
        # import httpx
        # resp = httpx.post(f"{LAB_BRIDGE_URL}/api/edge/command", json={...})
        result = {"status": "rejected", "feedback": "Edge HTTP not implemented", "robot_state": {}, "safety_note": None}

    return json.dumps(result, ensure_ascii=False)


@tool
def simulate_robot_position(robot_name: str = "xArm-Lab1") -> str:
    """Simula la posición actual de un robot. Devuelve posición TCP (x,y,z), joints y estado."""
    import random
    from datetime import datetime
    position = {
        "robot_name": robot_name,
        "timestamp": datetime.now().isoformat(),
        "tcp": {
            "x": round(random.uniform(200, 600), 2),
            "y": round(random.uniform(-300, 300), 2),
            "z": round(random.uniform(50, 400), 2),
        },
        "joints": [round(random.uniform(-170, 170), 2) for _ in range(6)],
        "state": random.choice(["idle", "moving", "paused"]),
        "gripper_open": random.choice([True, False]),
    }
    return json.dumps(position)


# ── Robot bridge (via WebSocket in api_server) ───────────────

def _run_robot_command(robot_id: str, command: str, params: dict = None) -> dict:
    """Sync wrapper: schedules send_robot_command on the main event loop from a worker thread."""
    try:
        from api_server import send_robot_command, get_main_loop
        loop = get_main_loop()
        if loop is None:
            return {"status": "error", "error": "Server event loop not available"}
        future = asyncio.run_coroutine_threadsafe(
            send_robot_command(robot_id, command, params or {}),
            loop,
        )
        return future.result(timeout=15)
    except ImportError:
        return {"status": "error", "error": "api_server not available (running standalone?)"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _run_multi_robot_command(robot_ids_list: list, command: str, params: dict) -> str:
    """Send the same command to multiple robots in parallel, return combined JSON results."""
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(robot_ids_list)) as executor:
        futures = {
            executor.submit(_run_robot_command, rid, command, params): rid
            for rid in robot_ids_list
        }
        for future in concurrent.futures.as_completed(futures):
            rid = futures[future]
            try:
                results[rid] = future.result()
            except Exception as e:
                results[rid] = json.dumps({"status": "error", "error": str(e)})
    return json.dumps(results, indent=2, ensure_ascii=False)


@tool
def robot_get_position(robot_ids: str = "") -> str:
    """Lee la posición actual del robot (TCP + joints) via bridge WebSocket.

    Args:
        robot_ids: Comma-separated robot IDs to query (e.g. "xarm-201" or "xarm-201,xarm-202").

    Returns:
        JSON with robot_name, tcp {x,y,z}, joints[], state, gripper_open.
    """
    ids = [r.strip() for r in robot_ids.split(",") if r.strip()]
    if MOCK_MODE:
        return simulate_robot_position.invoke({"robot_name": ids[0] if ids else ""})
    if len(ids) > 1:
        return _run_multi_robot_command(ids, "get_position", {})
    result = _run_robot_command(ids[0] if ids else "", "get_position")
    return json.dumps(result, ensure_ascii=False)


@tool
def robot_get_full_status(robot_ids: str = "") -> str:
    """Lee estado completo del robot: posición, modo, velocidad, gripper, errores.

    Args:
        robot_ids: Comma-separated robot IDs to query (e.g. "xarm-201" or "xarm-201,xarm-202").

    Returns:
        JSON with full robot status including mode, speed_limit, errors.
    """
    ids = [r.strip() for r in robot_ids.split(",") if r.strip()]
    if MOCK_MODE:
        pos = json.loads(simulate_robot_position.invoke({"robot_name": ids[0] if ids else ""}))
        pos.update({"mode": "teaching", "speed_limit": 25, "errors": []})
        return json.dumps(pos, ensure_ascii=False)
    if len(ids) > 1:
        return _run_multi_robot_command(ids, "get_full_status", {})
    result = _run_robot_command(ids[0] if ids else "", "get_full_status")
    return json.dumps(result, ensure_ascii=False)


@tool
def robot_move_joint(joint_id: int, angle: float, robot_ids: str = "") -> str:
    """Mueve un joint específico a un ángulo dado.

    Args:
        joint_id: Joint number (1-6).
        angle: Target angle in degrees.
        robot_ids: Comma-separated robot IDs (e.g. "xarm-201" or "xarm-201,xarm-202").

    Returns:
        JSON with status, feedback, and updated robot_state.
    """
    ids = [r.strip() for r in robot_ids.split(",") if r.strip()]
    if MOCK_MODE:
        return edge_command.invoke({
            "command": "move_joint",
            "params": json.dumps({"joint_id": joint_id, "angle": angle}),
            "robot_ids": ids[0] if ids else "",
        })
    params = {"joint": joint_id, "angle": angle}
    if len(ids) > 1:
        return _run_multi_robot_command(ids, "move_joint", params)
    result = _run_robot_command(ids[0] if ids else "", "move_joint", params)
    return json.dumps(result, ensure_ascii=False)


@tool
def robot_move_linear(x: float, y: float, z: float, robot_ids: str = "") -> str:
    """Mueve el robot en línea recta a las coordenadas X, Y, Z (mm).

    Args:
        x: Target X coordinate in mm.
        y: Target Y coordinate in mm.
        z: Target Z coordinate in mm.
        robot_ids: Comma-separated robot IDs (e.g. "xarm-201" or "xarm-201,xarm-202").

    Returns:
        JSON with status, feedback, and safety_note.
    """
    ids = [r.strip() for r in robot_ids.split(",") if r.strip()]
    if MOCK_MODE:
        return edge_command.invoke({
            "command": "move_linear",
            "params": json.dumps({"x": x, "y": y, "z": z}),
            "robot_ids": ids[0] if ids else "",
        })
    params = {"x": x, "y": y, "z": z}
    if len(ids) > 1:
        return _run_multi_robot_command(ids, "move_linear", params)
    result = _run_robot_command(ids[0] if ids else "", "move_linear", params)
    return json.dumps(result, ensure_ascii=False)


@tool
def robot_go_home(robot_ids: str = "") -> str:
    """Envía el robot a posición home (todos los joints a 0°).

    Args:
        robot_ids: Comma-separated robot IDs (e.g. "xarm-201" or "xarm-201,xarm-202").

    Returns:
        JSON with status and feedback.
    """
    ids = [r.strip() for r in robot_ids.split(",") if r.strip()]
    if MOCK_MODE:
        return edge_command.invoke({"command": "home", "params": "{}", "robot_ids": ids[0] if ids else ""})
    if len(ids) > 1:
        return _run_multi_robot_command(ids, "home", {})
    result = _run_robot_command(ids[0] if ids else "", "home")
    return json.dumps(result, ensure_ascii=False)


@tool
def robot_gripper(action: str = "toggle", robot_ids: str = "") -> str:
    """Controla el gripper del robot.

    Args:
        action: 'open', 'close', or 'toggle'.
        robot_ids: Comma-separated robot IDs (e.g. "xarm-201" or "xarm-201,xarm-202").

    Returns:
        JSON with status and feedback.
    """
    ids = [r.strip() for r in robot_ids.split(",") if r.strip()]
    if MOCK_MODE:
        return edge_command.invoke({"command": "gripper_demo", "params": "{}", "robot_ids": ids[0] if ids else ""})
    params = {"action": action}
    if len(ids) > 1:
        return _run_multi_robot_command(ids, "set_gripper", params)
    result = _run_robot_command(ids[0] if ids else "", "set_gripper", params)
    return json.dumps(result, ensure_ascii=False)


@tool
def robot_emergency_stop(robot_ids: str = "") -> str:
    """EMERGENCIA: congela el robot en su posición actual inmediatamente.

    Args:
        robot_ids: Comma-separated robot IDs (e.g. "xarm-201" or "xarm-201,xarm-202").

    Returns:
        JSON with status and safety_note.
    """
    ids = [r.strip() for r in robot_ids.split(",") if r.strip()]
    if MOCK_MODE:
        return edge_command.invoke({"command": "freeze", "params": "{}", "robot_ids": ids[0] if ids else ""})
    if len(ids) > 1:
        return _run_multi_robot_command(ids, "emergency_stop", {})
    result = _run_robot_command(ids[0] if ids else "", "emergency_stop")
    return json.dumps(result, ensure_ascii=False)


# ── Export ────────────────────────────────────────────────────
EDGE_TOOLS = [edge_command, simulate_robot_position]
ROBOT_BRIDGE_TOOLS = [
    robot_get_position, robot_get_full_status,
    robot_move_joint, robot_move_linear,
    robot_go_home, robot_gripper, robot_emergency_stop,
]
