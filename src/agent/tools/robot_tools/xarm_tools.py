"""
xarm_tools.py - Robot tools DISABLED for safety

All robot commands are currently disabled.
To re-enable, restore the original xarm_tools.py
"""
from langchain_core.tools import tool
from typing import Optional

_DISABLED_MSG = "⚠️ Control del robot está deshabilitado por seguridad. Contacta al administrador para habilitarlo."


@tool
def robot_connect() -> str:
    """Conectar al robot xArm."""
    return _DISABLED_MSG

@tool
def robot_get_position() -> str:
    """Obtener la posición actual del robot xArm."""
    return _DISABLED_MSG

@tool
def robot_move_to(x: float, y: float, z: float, speed: Optional[float] = None) -> str:
    """Mover el robot xArm a una posición absoluta."""
    return _DISABLED_MSG

@tool
def robot_step(axis: str, distance: float) -> str:
    """Mover el robot xArm incrementalmente en un eje."""
    return _DISABLED_MSG

@tool
def robot_home() -> str:
    """Enviar el robot xArm a su posición Home."""
    return _DISABLED_MSG

@tool
def robot_emergency_stop() -> str:
    """PARO DE EMERGENCIA del robot xArm."""
    return _DISABLED_MSG

@tool
def robot_gripper(action: str) -> str:
    """Controlar el gripper del robot xArm."""
    return _DISABLED_MSG

@tool
def robot_status() -> str:
    """Obtener el estado del robot xArm."""
    return _DISABLED_MSG

@tool
def robot_clear_errors() -> str:
    """Limpiar errores del robot xArm."""
    return _DISABLED_MSG

XARM_TOOLS = [
    robot_connect, robot_get_position, robot_move_to, robot_step,
    robot_home, robot_emergency_stop, robot_gripper, robot_status,
    robot_clear_errors,
]
