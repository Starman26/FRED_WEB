"""
hardware_tools/ — Device tools (comunicación con hardware via edge_router).

- xarm_tools:    Control de robots xArm 6 Lite
- abb_tools:     Control de robots ABB IRB
- plc_tools:     Lectura/escritura de PLCs Siemens S7
- network_tools: Diagnóstico de red y comandos shell

Infrastructure:
- edge_router:   Transport layer unificado (WebSocket + mock dispatch)
"""

# ── Edge router (must import first — device tools register mocks on import) ──
from .edge_router import (
    send_command,
    send_command_multi,
    is_mock_mode,
    set_mock_mode,
)

# ── Device tools ──
from .xarm_tools import XARM_TOOLS, XARM_READ_TOOLS, XARM_ACTUATE_TOOLS, XARM_CONFIG_TOOLS, XARM_DEMO_TOOLS
from .abb_tools import ABB_TOOLS, ABB_READ_TOOLS, ABB_ACTUATE_TOOLS
from .plc_tools import PLC_TOOLS, PLC_READ_TOOLS, PLC_WRITE_TOOLS
from .network_tools import NETWORK_TOOLS, NETWORK_READ_TOOLS

# ── Convenience groups ──
ALL_DEVICE_TOOLS = XARM_TOOLS + ABB_TOOLS + PLC_TOOLS + NETWORK_TOOLS
ALL_READ_TOOLS = XARM_READ_TOOLS + ABB_READ_TOOLS + PLC_READ_TOOLS + NETWORK_READ_TOOLS

__all__ = [
    # Router
    "send_command",
    "send_command_multi",
    "is_mock_mode",
    "set_mock_mode",
    # Device tool lists
    "XARM_TOOLS",
    "XARM_READ_TOOLS",
    "XARM_ACTUATE_TOOLS",
    "XARM_CONFIG_TOOLS",
    "XARM_DEMO_TOOLS",
    "ABB_TOOLS",
    "ABB_READ_TOOLS",
    "ABB_ACTUATE_TOOLS",
    "PLC_TOOLS",
    "PLC_READ_TOOLS",
    "PLC_WRITE_TOOLS",
    "NETWORK_TOOLS",
    "NETWORK_READ_TOOLS",
    "ALL_DEVICE_TOOLS",
    "ALL_READ_TOOLS",
]
