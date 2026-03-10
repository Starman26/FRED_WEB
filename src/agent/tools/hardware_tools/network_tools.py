"""
network_tools.py — Agent tools for network diagnostics and shell commands.

Communicates via edge_router → lab bridge → subprocess on the lab PC.
Commands are validated against a whitelist on the bridge side.
"""

import json
import random
from typing import Dict, Any
from langchain_core.tools import tool
from .edge_router import send_command, register_mock_handler


# ═══════════════════════════════════════════════════════════════
# Mock handlers
# ═══════════════════════════════════════════════════════════════

def _mock_ping(params: dict, device_id: str) -> dict:
    ip = params.get("ip", "192.168.1.1")
    count = params.get("count", 4)
    # Simulate realistic ping results
    reachable = not ip.startswith("10.99")  # 10.99.x.x = unreachable in mock
    if reachable:
        times = [round(random.uniform(0.5, 15.0), 1) for _ in range(count)]
        return {
            "ip": ip,
            "reachable": True,
            "packets_sent": count,
            "packets_received": count,
            "packet_loss": 0.0,
            "times_ms": times,
            "avg_ms": round(sum(times) / len(times), 1),
            "min_ms": min(times),
            "max_ms": max(times),
        }
    else:
        return {
            "ip": ip,
            "reachable": False,
            "packets_sent": count,
            "packets_received": 0,
            "packet_loss": 100.0,
            "times_ms": [],
            "error": "Request timed out",
        }


def _mock_exec_command(params: dict, device_id: str) -> dict:
    command = params.get("command", "")
    return {
        "command": command,
        "exit_code": 0,
        "stdout": f"[MOCK] Command executed: {command}\nSimulated output for testing.",
        "stderr": "",
        "duration_ms": round(random.uniform(50, 2000), 1),
        "platform": "Windows",
    }


for action, handler in {
    "ping": _mock_ping,
    "exec_command": _mock_exec_command,
}.items():
    register_mock_handler("shell", action, handler)


# ═══════════════════════════════════════════════════════════════
# Agent tools
# ═══════════════════════════════════════════════════════════════

def _send(action: str, params: dict = None, device_id: str = "") -> str:
    result = send_command("shell", action, params or {}, device_id)
    return json.dumps(result, ensure_ascii=False)


@tool
def net_ping(ip: str, count: int = 4) -> str:
    """Ping an IP address from the lab PC to check network connectivity.

    Useful for diagnosing communication issues with robots, PLCs, or other devices.

    Args:
        ip: IP address to ping (e.g. "192.168.1.185").
        count: Number of ping packets (default 4).

    Returns:
        JSON with reachable, packet_loss, avg/min/max times in ms.
    """
    return _send("ping", {"ip": ip, "count": count})


@tool
def net_exec_command(command: str, timeout: int = 30) -> str:
    """Execute a whitelisted shell command on the lab PC.

    Only commands in the bridge whitelist are allowed (ping, ipconfig, systeminfo,
    dir, type, python, etc.). Dangerous commands are always blocked.

    Args:
        command: Shell command to execute (e.g. "ipconfig", "systeminfo").
        timeout: Max execution time in seconds (default 30, max 120).

    Returns:
        JSON with exit_code, stdout, stderr, duration_ms.
    """
    return _send("exec_command", {"command": command, "timeout": min(timeout, 120)})


# ═══════════════════════════════════════════════════════════════
# Exports
# ═══════════════════════════════════════════════════════════════

NETWORK_READ_TOOLS = [net_ping]
NETWORK_WRITE_TOOLS = [net_exec_command]

NETWORK_TOOLS = NETWORK_READ_TOOLS + NETWORK_WRITE_TOOLS
