"""
Agent tools for xArm 6 Lite robots.

All tools use edge_router.send_command(device_type="xarm", ...).
Mock handlers simulate xArm SDK responses for testing without hardware.
"""

import json
from typing import Dict, Any
from langchain_core.tools import tool
from .edge_router import send_command, send_command_multi, parse_device_ids, register_mock_handler


JOINT_NAMES = {
    1: "Base", 2: "Shoulder", 3: "Elbow",
    4: "Wrist Roll", 5: "Wrist Pitch", 6: "Wrist Yaw",
}

JOINT_LIMITS = {
    1: (-360, 360), 2: (-118, 120), 3: (-225, 11),
    4: (-360, 360), 5: (-97, 180),  6: (-360, 360),
}

POSES = {
    "home": [0, -45, -25, 0, 70, 0],
    "pick": [0, -30, 60, 0, 90, 0],
    "place": [90, -30, 60, 0, 90, 0],
    "ready": [0, -45, 30, 0, 45, 0],
    "inspect": [0, -60, 90, 0, 30, 0],
}

# Safe demo ranges per joint (degrees, conservative)
JOINT_DEMO_RANGES = {
    1: 20,   # Base: ±20° — wide rotation, very visible
    2: 10,   # Shoulder: ±10° — heavy joint, keep small
    3: 15,   # Elbow: ±15° — visible bend
    4: 20,   # Wrist Roll: ±20° — light joint, easy to see
    5: 10,   # Wrist Pitch: ±10° — small but visible
    6: 20,   # Wrist Yaw: ±20° — light, spins easily
}

# Safe starting position for demo (slightly raised, clear of table)
DEMO_START_POSITION = {"x": 250.0, "y": 0.0, "z": 350.0}

_mock_state: Dict[str, Any] = {
    "joints": [0.0] * 6,
    "tcp": {"x": 207.0, "y": 0.0, "z": 112.0, "roll": 180.0, "pitch": 0.0, "yaw": 0.0},
    "mode": "position control",
    "speed_limit": 25,
    "gripper_open": True,
    "gripper_position": 850,
    "connected": True,
    "error_code": 0,
    "warning_code": 0,
    "temperatures": [25.0] * 6,
    "currents": [0.1] * 6,
}


def _mock_get_position(params: dict, device_id: str) -> dict:
    return {
        "tcp": dict(_mock_state["tcp"]),
        "joints": list(_mock_state["joints"]),
        "state": "static",
        "mode": _mock_state["mode"],
    }


def _mock_get_full_status(params: dict, device_id: str) -> dict:
    return {
        "tcp": dict(_mock_state["tcp"]),
        "joints": list(_mock_state["joints"]),
        "joint_velocities": [0.0] * 6,
        "joint_efforts": [0.0] * 6,
        "temperatures": list(_mock_state["temperatures"]),
        "currents": list(_mock_state["currents"]),
        "state": "static",
        "mode": _mock_state["mode"],
        "error_code": _mock_state["error_code"],
        "warning_code": _mock_state["warning_code"],
        "gripper_position": _mock_state["gripper_position"],
        "safety_zone": {"is_set": True},
    }


def _mock_move_joint(params: dict, device_id: str) -> dict:
    jid = params.get("joint_id", params.get("joint", 1))
    angle = float(params.get("angle", 0))
    speed = params.get("speed", 30)

    if jid < 1 or jid > 6:
        raise ValueError(f"Invalid joint {jid}. Must be 1-6.")

    lo, hi = JOINT_LIMITS.get(jid, (-360, 360))
    if angle < lo or angle > hi:
        name = JOINT_NAMES.get(jid, f"Joint {jid}")
        raise ValueError(f"Angle {angle}° out of range for J{jid} ({name}). Valid: [{lo}°, {hi}°].")

    prev = _mock_state["joints"][jid - 1]
    _mock_state["joints"][jid - 1] = angle
    return {
        "code": 0,
        "target_joint": jid,
        "joint_name": JOINT_NAMES.get(jid, f"Joint {jid}"),
        "target_angle": angle,
        "previous_angle": prev,
        "final_angles": list(_mock_state["joints"]),
        "state": "static",
    }


def _mock_move_linear(params: dict, device_id: str) -> dict:
    x = params.get("x", _mock_state["tcp"]["x"])
    y = params.get("y", _mock_state["tcp"]["y"])
    z = params.get("z", _mock_state["tcp"]["z"])
    _mock_state["tcp"].update({"x": x, "y": y, "z": z})
    return {
        "code": 0,
        "target": {"x": x, "y": y, "z": z},
        "final_tcp": {"x": x, "y": y, "z": z},
        "state": "static",
    }


def _mock_go_home(params: dict, device_id: str) -> dict:
    _mock_state["joints"] = [0.0, -45.0, -25.0, 0.0, 70.0, 0.0]
    _mock_state["tcp"] = {"x": 255.0, "y": 0.0, "z": 370.0, "roll": 180.0, "pitch": 0.0, "yaw": 0.0}
    return {
        "code": 0,
        "final_tcp": dict(_mock_state["tcp"]),
        "final_angles": list(_mock_state["joints"]),
        "state": "static",
    }


def _mock_go_to_pose(params: dict, device_id: str) -> dict:
    pose_name = params.get("pose", "home")
    joints = POSES.get(pose_name, POSES["home"])
    _mock_state["joints"] = list(joints)
    return {
        "code": 0,
        "pose": pose_name,
        "final_angles": list(_mock_state["joints"]),
        "state": "static",
    }


def _mock_gripper(params: dict, device_id: str) -> dict:
    action = params.get("action", "toggle")
    if action == "open":
        pos = 850
    elif action == "close":
        pos = 0
    else:
        pos = 0 if _mock_state["gripper_open"] else 850
    _mock_state["gripper_open"] = pos > 400
    _mock_state["gripper_position"] = pos
    return {
        "code": 0,
        "target_position": pos,
        "current_position": pos,
        "is_open": _mock_state["gripper_open"],
    }


def _mock_emergency_stop(params: dict, device_id: str) -> dict:
    _mock_state["mode"] = "frozen"
    _mock_state["speed_limit"] = 0
    return {"stopped": True, "state": "emergency_stop"}


def _mock_highlight_joint(params: dict, device_id: str) -> dict:
    jid = params.get("joint_id", 1)
    name = JOINT_NAMES.get(jid, f"Joint {jid}")
    return {
        "feedback": f"Joint {jid} ({name}) oscilando ±10° a velocidad segura",
        "safety_note": "Velocidad limitada a 10% durante demostración",
    }


def _mock_show_workspace(params: dict, device_id: str) -> dict:
    jid = params.get("joint_id", 1)
    name = JOINT_NAMES.get(jid, f"Joint {jid}")
    lo, hi = JOINT_LIMITS.get(jid, (-180, 180))
    return {
        "feedback": f"Joint {jid} ({name}): rango [{lo}°, {hi}°]",
        "safety_note": "Movimiento lento para demostración de rango",
    }


def _mock_demo_movement(params: dict, device_id: str) -> dict:
    pattern = params.get("pattern", "pick_and_place")
    return {
        "feedback": f"Ejecutando demostración '{pattern}': home → pick → place → home",
        "safety_note": "Movimiento a 15% velocidad",
    }


def _mock_clear_error(params: dict, device_id: str) -> dict:
    _mock_state["error_code"] = 0
    _mock_state["warning_code"] = 0
    _mock_state["mode"] = "position control"
    return {
        "code": 0,
        "error_cleared": True,
        "previous_error": _mock_state.get("error_code", 0),
        "state": "ready",
        "mode": "position control",
    }


def _mock_set_collision_sensitivity(params: dict, device_id: str) -> dict:
    level = params.get("level", 3)
    if level < 1 or level > 5:
        raise ValueError(f"Sensitivity level must be 1-5, got {level}")
    return {
        "code": 0,
        "sensitivity_level": level,
        "description": {
            1: "Very low — ignores most collisions",
            2: "Low — tolerates moderate forces",
            3: "Medium — balanced (default)",
            4: "High — sensitive to light contact",
            5: "Very high — stops on minimal force",
        }.get(level, "Unknown"),
    }


def _mock_show_all_joints(params: dict, device_id: str) -> dict:
    speed = params.get("speed", 15)
    sequence = []

    for jid in range(1, 7):
        name = JOINT_NAMES[jid]
        demo_range = JOINT_DEMO_RANGES[jid]
        sequence.append({
            "joint": jid,
            "name": name,
            "movement": f"0° → +{demo_range}° → -{demo_range}° → 0°",
            "demo_range_deg": demo_range,
        })

    return {
        "code": 0,
        "start_position": DEMO_START_POSITION,
        "speed_deg_s": speed,
        "sequence": sequence,
        "total_joints": 6,
        "feedback": "Demostración completa: el robot movió cada joint individualmente",
        "safety_note": f"Velocidad limitada a {speed} deg/s durante demostración",
    }


def _mock_say_hi(params: dict, device_id: str) -> dict:
    speed = params.get("speed", 25)

    wave_sequence = [
        {"action": "go_home", "description": "Moving to safe position"},
        {"action": "wave_1", "joints": {"J1": 20, "J6": 30}, "description": "Wave right + wrist twist"},
        {"action": "wave_2", "joints": {"J1": -20, "J6": -30}, "description": "Wave left + wrist twist back"},
        {"action": "wave_3", "joints": {"J1": 20, "J6": 30}, "description": "Wave right again"},
        {"action": "wave_4", "joints": {"J1": -20, "J6": -30}, "description": "Wave left again"},
        {"action": "wave_5", "joints": {"J1": 0, "J6": 0}, "description": "Return to center"},
    ]

    return {
        "code": 0,
        "sequence": wave_sequence,
        "speed_deg_s": speed,
        "feedback": "Robot saludó moviendo la base y la muñeca",
        "safety_note": f"Animación a {speed} deg/s desde posición segura",
    }


_MOCK_MAP = {
    "get_position": _mock_get_position,
    "get_full_status": _mock_get_full_status,
    "move_joint": _mock_move_joint,
    "move_linear": _mock_move_linear,
    "home": _mock_go_home,
    "go_to_pose": _mock_go_to_pose,
    "set_gripper": _mock_gripper,
    "emergency_stop": _mock_emergency_stop,
    "highlight_joint": _mock_highlight_joint,
    "show_workspace": _mock_show_workspace,
    "demo_movement": _mock_demo_movement,
    "clear_error": _mock_clear_error,
    "set_collision_sensitivity": _mock_set_collision_sensitivity,
    "show_all_joints": _mock_show_all_joints,
    "say_hi": _mock_say_hi,
}

for action, handler in _MOCK_MAP.items():
    register_mock_handler("xarm", action, handler)


def _send(action: str, params: dict = None, device_id: str = "") -> str:
    result = send_command("xarm", action, params or {}, device_id)
    return json.dumps(result, ensure_ascii=False)


@tool
def xarm_get_position(device_id: str = "") -> str:
    """Read the current TCP position and joint angles of an xArm robot.

    Args:
        device_id: Robot device ID (e.g. "xarm-185"). Leave empty for default.

    Returns:
        JSON with tcp {x,y,z,roll,pitch,yaw}, joints[], state, mode.
    """
    return _send("get_position", device_id=device_id)


@tool
def xarm_get_full_status(device_id: str = "") -> str:
    """Read complete status of an xArm: position, temperatures, currents, errors, gripper.

    Args:
        device_id: Robot device ID (e.g. "xarm-185").

    Returns:
        JSON with tcp, joints, temperatures, currents, error_code, warning_code,
        gripper_position, safety_zone status.
    """
    return _send("get_full_status", device_id=device_id)


@tool
def xarm_move_joint(joint_id: int, angle: float, speed: int = 50, device_id: str = "") -> str:
    """Move a specific xArm joint to a target angle.

    Args:
        joint_id: Joint number (1-6). 1=Base, 2=Shoulder, 3=Elbow,
                  4=Wrist Roll, 5=Wrist Pitch, 6=Wrist Yaw.
        angle: Target angle in degrees.
        speed: Movement speed in deg/s (default 50).
        device_id: Robot device ID.

    Returns:
        JSON with code, joint_name, target/previous angle, final_angles.
    """
    return _send("move_joint", {"joint_id": joint_id, "angle": angle, "speed": speed}, device_id)


@tool
def xarm_move_linear(x: float, y: float, z: float, speed: int = 100, device_id: str = "") -> str:
    """Move xArm TCP in a straight line to XYZ coordinates (mm).

    Args:
        x: Target X in mm.
        y: Target Y in mm.
        z: Target Z in mm.
        speed: Movement speed in mm/s (default 100).
        device_id: Robot device ID.

    Returns:
        JSON with target, final_tcp, state.
    """
    return _send("move_linear", {"x": x, "y": y, "z": z, "speed": speed}, device_id)


@tool
def xarm_go_home(device_id: str = "") -> str:
    """Send xArm to safe home position (brazo levantado, centrado).

    Joints: J1=0, J2=-45, J3=-25, J4=0, J5=70, J6=0
    This position keeps the gripper clear of the table/floor.

    Args:
        device_id: Robot device ID.

    Returns:
        JSON with final_tcp and state.
    """
    return _send("home", device_id=device_id)


@tool
def xarm_go_to_pose(pose: str = "home", device_id: str = "") -> str:
    """Move xArm to a predefined pose.

    Args:
        pose: Pose name, one of: home, pick, place, ready, inspect.
        device_id: Robot device ID.

    Returns:
        JSON with pose name and final_angles.
    """
    return _send("go_to_pose", {"pose": pose}, device_id)


@tool
def xarm_gripper(action: str = "toggle", device_id: str = "") -> str:
    """Control the xArm gripper.

    Args:
        action: "open", "close", or "toggle".
        device_id: Robot device ID.

    Returns:
        JSON with target_position, current_position, is_open.
    """
    return _send("set_gripper", {"action": action}, device_id)


@tool
def xarm_emergency_stop(device_id: str = "") -> str:
    """EMERGENCY: freeze the xArm in its current position immediately.

    Args:
        device_id: Robot device ID.

    Returns:
        JSON with stopped status and safety note.
    """
    return _send("emergency_stop", device_id=device_id)


@tool
def xarm_highlight_joint(joint_id: int = 1, device_id: str = "") -> str:
    """Oscillate a joint slightly to visually identify it (teaching mode).

    Args:
        joint_id: Joint to highlight (1-6).
        device_id: Robot device ID.

    Returns:
        JSON with feedback and safety note.
    """
    return _send("highlight_joint", {"joint_id": joint_id}, device_id)


@tool
def xarm_show_workspace(joint_id: int = 1, device_id: str = "") -> str:
    """Slowly move a joint through its full range to show workspace limits.

    Args:
        joint_id: Joint to demonstrate (1-6).
        device_id: Robot device ID.

    Returns:
        JSON with range info and safety note.
    """
    return _send("show_workspace", {"joint_id": joint_id}, device_id)


@tool
def xarm_demo_movement(pattern: str = "pick_and_place", device_id: str = "") -> str:
    """Execute a demonstration movement pattern on the xArm.

    Args:
        pattern: Demo pattern name (e.g. "pick_and_place").
        device_id: Robot device ID.

    Returns:
        JSON with feedback and safety note.
    """
    return _send("demo_movement", {"pattern": pattern}, device_id)


@tool
def xarm_clear_error(device_id: str = "") -> str:
    """Clear error state on xArm and return to ready mode.

    Use this after a collision, e-stop, or any error_code != 0.
    The robot must be physically safe before clearing — verify no obstructions first.

    Args:
        device_id: Robot device ID.

    Returns:
        JSON with error_cleared status, previous error code, and new state.
    """
    return _send("clear_error", device_id=device_id)


@tool
def xarm_set_collision_sensitivity(level: int = 3, device_id: str = "") -> str:
    """Set the collision detection sensitivity level on xArm.

    Level 1 = very low (ignores most collisions, for heavy payloads).
    Level 3 = medium (default, balanced).
    Level 5 = very high (stops on minimal force, safest for teaching).

    Args:
        level: Sensitivity level 1-5 (default 3).
        device_id: Robot device ID.

    Returns:
        JSON with sensitivity_level set and description.
    """
    return _send("set_collision_sensitivity", {"level": level}, device_id)


@tool
def xarm_show_all_joints(speed: int = 15, device_id: str = "") -> str:
    """Demonstrate all 6 joints one by one for teaching purposes.

    The robot first moves to a safe position (X=250, Y=0, Z=350mm),
    then moves each joint individually through a safe range:
    - J1 Base: ±20°
    - J2 Shoulder: ±10°
    - J3 Elbow: ±15°
    - J4 Wrist Roll: ±20°
    - J5 Wrist Pitch: ±10°
    - J6 Wrist Yaw: ±20°

    Each joint returns to 0° before the next one moves.

    Args:
        speed: Demo speed in deg/s (default 15, slow for visibility).
        device_id: Robot device ID.

    Returns:
        JSON with start_position, sequence of joint demos, and safety note.
    """
    return _send("show_all_joints", {"speed": speed}, device_id)


@tool
def xarm_say_hi(speed: int = 25, device_id: str = "") -> str:
    """Make the xArm wave hello! The robot goes to safe home position first,
    then waves by rotating the base (J1) left and right while twisting
    the wrist (J6) back and forth simultaneously.

    Fun tool for demos and greeting students.

    Args:
        speed: Wave speed in deg/s (default 25).
        device_id: Robot device ID.

    Returns:
        JSON with wave sequence and safety note.
    """
    return _send("say_hi", {"speed": speed}, device_id)


XARM_READ_TOOLS = [xarm_get_position, xarm_get_full_status]

XARM_ACTUATE_TOOLS = [
    xarm_move_joint, xarm_move_linear, xarm_go_home,
    xarm_go_to_pose, xarm_gripper, xarm_emergency_stop,
    xarm_clear_error,
]

XARM_CONFIG_TOOLS = [xarm_set_collision_sensitivity]

XARM_DEMO_TOOLS = [xarm_highlight_joint, xarm_show_workspace, xarm_demo_movement, xarm_show_all_joints, xarm_say_hi]

XARM_TOOLS = XARM_READ_TOOLS + XARM_ACTUATE_TOOLS + XARM_CONFIG_TOOLS + XARM_DEMO_TOOLS
