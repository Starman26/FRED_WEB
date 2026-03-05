"""
lab_tools/ - Herramientas del laboratorio (split modular)

Antes era un único archivo de 1087 líneas.
Ahora está dividido en módulos por dominio:
  - station_tools: overview del lab, detalles de estación
  - equipment_tools: PLCs, cobots, sensores de puerta
  - error_tools: gestión de errores
  - repair_tools: diagnóstico, reset, auto-fix
  - formatters: formateo para display
  
Los imports se mantienen idénticos al archivo original para backward-compatibility.
"""

from .station_tools import (
    get_lab_overview,
    get_station_details,
)

from .equipment_tools import (
    get_all_plcs,
    reconnect_plc,
    get_all_cobots,
    set_cobot_mode,
    check_door_sensors,
    close_all_doors,
)

from .error_tools import (
    get_active_errors,
    acknowledge_error,
    resolve_error,
    resolve_station_errors,
    resolve_all_errors,
)

from .repair_tools import (
    reset_lab_to_safe_state,
    diagnose_and_suggest_fixes,
)

from .health_tools import (
    ping_plc,
    health_check_station,
)

from .formatters import (
    format_lab_overview_for_display,
    format_station_details_for_display,
    format_errors_for_display,
)

__all__ = [
    # Station
    "get_lab_overview",
    "get_station_details",
    # Equipment
    "get_all_plcs",
    "reconnect_plc",
    "get_all_cobots",
    "set_cobot_mode",
    "check_door_sensors",
    "close_all_doors",
    # Errors
    "get_active_errors",
    "acknowledge_error",
    "resolve_error",
    "resolve_station_errors",
    "resolve_all_errors",
    # Repair
    "reset_lab_to_safe_state",
    "diagnose_and_suggest_fixes",
    # Health
    "ping_plc",
    "health_check_station",
    # Formatters
    "format_lab_overview_for_display",
    "format_station_details_for_display",
    "format_errors_for_display",
    # Diagnostic tools (for bind_tools / ReAct)
    "DIAGNOSTIC_TOOLS",
]


# ============================================
# DIAGNOSTIC TOOLS — LangChain @tool wrappers
# for ReAct tool-calling in troubleshooter
# ============================================
from langchain_core.tools import tool


@tool
def lab_overview_tool() -> dict:
    """Get a general overview of the entire laboratory.
    Returns: total stations, stations online, active error count, and a
    per-station summary with status and error counts.
    Use this FIRST to understand the overall lab state before diving deeper."""
    return get_lab_overview()


@tool
def station_details_tool(station_number: int) -> dict:
    """Get full details for a specific station (1-6).
    Returns: PLC status, cobot status, door sensors, active errors,
    and whether the station is ready to operate.
    Args:
        station_number: Station number (1-6)."""
    return get_station_details(station_number)


@tool
def plc_status_tool() -> dict:
    """Get the status of ALL PLCs across the laboratory.
    Returns: list of PLCs with connection status, run mode (RUN/STOP),
    IP address, model, and error flags."""
    return get_all_plcs()


@tool
def cobot_status_tool() -> dict:
    """Get the status of ALL cobots across the laboratory.
    Returns: list of cobots with connection status, current routine/mode,
    model, and station assignment."""
    return get_all_cobots()


@tool
def active_errors_tool() -> dict:
    """Get all currently active (unresolved) errors in the laboratory.
    Returns: list of errors with station, equipment, severity, error code,
    and description."""
    return get_active_errors()


@tool
def door_sensors_tool() -> dict:
    """Check the status of all safety door sensors in the laboratory.
    Returns: per-station door status (open/closed) and count of open doors.
    Important: cobots CANNOT operate if their station door is open."""
    return check_door_sensors()


@tool
def ping_plc_tool(station_number: int) -> dict:
    """Ping a specific station's PLC to check connectivity and health.
    Returns: ping result, response time in ms, PLC run mode, DB connection status.
    Args:
        station_number: Station number (1-6)."""
    return ping_plc(station_number)


@tool
def health_check_tool(station_number: int) -> dict:
    """Run a comprehensive health check on a specific station.
    Checks PLC, cobot, doors, and active errors. Returns overall health
    assessment, list of issues, and recommendations.
    Args:
        station_number: Station number (1-6)."""
    return health_check_station(station_number)


@tool
def diagnose_station_tool(station_number: int = None) -> dict:
    """Run automatic diagnosis on a station (or entire lab if no station specified).
    Identifies problems and suggests specific fix actions.
    Args:
        station_number: Station number (1-6), or None for entire lab."""
    return diagnose_and_suggest_fixes(station_number)


DIAGNOSTIC_TOOLS = [
    lab_overview_tool,
    station_details_tool,
    plc_status_tool,
    cobot_status_tool,
    active_errors_tool,
    door_sensors_tool,
    ping_plc_tool,
    health_check_tool,
    diagnose_station_tool,
]
