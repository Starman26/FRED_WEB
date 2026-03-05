"""
robot_tools/ - Tools para control del robot xArm

Módulos:
- xarm_client: Cliente desacoplado del robot (thread-safe)
- xarm_tools: LangChain @tool wrappers para el agente
"""
from .xarm_tools import XARM_TOOLS

try:
    from .xarm_client import XArmClient, RobotState, RobotPosition
    ROBOT_TOOLS_AVAILABLE = True
except ImportError:
    ROBOT_TOOLS_AVAILABLE = False

__all__ = ["XARM_TOOLS", "ROBOT_TOOLS_AVAILABLE"]

if ROBOT_TOOLS_AVAILABLE:
    __all__.extend(["XArmClient", "RobotState", "RobotPosition"])
