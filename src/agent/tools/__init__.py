"""
Tools del sistema multi-agente SENTINEL/ORION

Subpaquetes:
- hardware_tools/: Device tools (xArm, ABB, PLC, network) via edge_router
- db_tools/:       Knowledge tools (RAG, analyst SQL)
"""

# ── Knowledge tools (db_tools) ──
from .db_tools import make_retrieve_tool, make_web_search_tool, ANALYST_TOOLS

# ── Hardware tools ──
try:
    from .hardware_tools import XARM_TOOLS, ALL_DEVICE_TOOLS, ALL_READ_TOOLS
    ROBOT_TOOLS_AVAILABLE = True
except ImportError:
    XARM_TOOLS = []
    ALL_DEVICE_TOOLS = []
    ALL_READ_TOOLS = []
    ROBOT_TOOLS_AVAILABLE = False


__all__ = [
    "make_retrieve_tool",
    "make_web_search_tool",
    "ANALYST_TOOLS",
    "XARM_TOOLS",
    "ALL_DEVICE_TOOLS",
    "ALL_READ_TOOLS",
    "ROBOT_TOOLS_AVAILABLE",
]
