"""
Tools del sistema multi-agente ATLAS

Módulos:
- lab_tools/: Herramientas del laboratorio (split modular)
- robot_tools/: Control del robot xArm  
- rag_tools: Búsqueda RAG en documentos
"""
from .rag_tools import make_retrieve_tool, make_web_search_tool

# Lab tools
try:
    from .lab_tools import (
        get_lab_overview,
        get_station_details,
        get_all_plcs,
        get_all_cobots,
        get_active_errors,
        check_door_sensors,
        set_cobot_mode,
        acknowledge_error,
        resolve_error,
        close_all_doors,
        reconnect_plc,
        resolve_station_errors,
        resolve_all_errors,
        reset_lab_to_safe_state,
        diagnose_and_suggest_fixes,
        format_lab_overview_for_display,
        format_station_details_for_display,
        format_errors_for_display,
    )
    LAB_TOOLS_AVAILABLE = True
except ImportError:
    LAB_TOOLS_AVAILABLE = False

# Robot tools
try:
    from .robot_tools import XARM_TOOLS, ROBOT_TOOLS_AVAILABLE
except ImportError:
    XARM_TOOLS = []
    ROBOT_TOOLS_AVAILABLE = False


__all__ = [
    "make_retrieve_tool",
    "make_web_search_tool",
    "LAB_TOOLS_AVAILABLE",
    "XARM_TOOLS",
    "ROBOT_TOOLS_AVAILABLE",
]

if LAB_TOOLS_AVAILABLE:
    __all__.extend([
        "get_lab_overview", "get_station_details",
        "get_all_plcs", "get_all_cobots", "get_active_errors",
        "check_door_sensors", "set_cobot_mode",
        "acknowledge_error", "resolve_error",
        "close_all_doors", "reconnect_plc",
        "resolve_station_errors", "resolve_all_errors",
        "reset_lab_to_safe_state", "diagnose_and_suggest_fixes",
        "format_lab_overview_for_display",
        "format_station_details_for_display",
        "format_errors_for_display",
    ])
