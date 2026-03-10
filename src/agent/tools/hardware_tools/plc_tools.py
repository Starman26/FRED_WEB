"""
plc_tools.py — Agent tools for Siemens S7-1200/1500 PLCs.

Communicates via edge_router → lab bridge → snap7 (S7comm protocol).
Supports reading/writing Inputs (I), Outputs (Q), and Memory (M) areas.
"""

import json
import random
from typing import Dict, Any
from langchain_core.tools import tool
from .edge_router import send_command, register_mock_handler


# ═══════════════════════════════════════════════════════════════
# Mock state
# ═══════════════════════════════════════════════════════════════

_mock_plcs: Dict[str, Dict[str, Any]] = {
    "192.168.1.101": {"connected": True, "name": "PLC Station 1"},
    "192.168.1.102": {"connected": True, "name": "PLC Station 2"},
    "192.168.1.103": {"connected": False, "name": "PLC Station 3"},
    "192.168.1.104": {"connected": True, "name": "PLC Station 4"},
    "192.168.1.105": {"connected": True, "name": "PLC Station 5"},
}

# Mock memory: {plc_ip: {area: {byte_addr: int_value}}}
_mock_memory: Dict[str, Dict[str, Dict[int, int]]] = {}


def _get_mock_byte(plc_ip: str, area: str, byte_addr: int) -> int:
    """Get a mock byte value, generating random on first access."""
    if plc_ip not in _mock_memory:
        _mock_memory[plc_ip] = {}
    if area not in _mock_memory[plc_ip]:
        _mock_memory[plc_ip][area] = {}
    if byte_addr not in _mock_memory[plc_ip][area]:
        _mock_memory[plc_ip][area][byte_addr] = random.randint(0, 255)
    return _mock_memory[plc_ip][area][byte_addr]


def _byte_to_bits(value: int) -> dict:
    return {f"bit_{i}": bool(value & (1 << i)) for i in range(8)}


def _set_bit(value: int, bit: int, on: bool) -> int:
    if on:
        return value | (1 << bit)
    return value & ~(1 << bit)


# ═══════════════════════════════════════════════════════════════
# Mock handlers
# ═══════════════════════════════════════════════════════════════

def _mock_read_area(params: dict, device_id: str) -> dict:
    plc_ip = params.get("plc_ip", device_id)
    area = params.get("area", "input")
    byte_addr = params.get("byte_address", 0)

    if plc_ip not in _mock_plcs:
        raise ValueError(f"PLC {plc_ip} not found")
    if not _mock_plcs[plc_ip]["connected"]:
        raise ConnectionError(f"PLC {plc_ip} not connected")

    value = _get_mock_byte(plc_ip, area, byte_addr)
    return {
        "ip": plc_ip,
        "area": area,
        "byte": byte_addr,
        "bits": _byte_to_bits(value),
        "raw_value": value,
    }


def _mock_write_bit(params: dict, device_id: str) -> dict:
    plc_ip = params.get("plc_ip", device_id)
    area = params.get("area", "output")
    byte_addr = params.get("byte_address", 0)
    bit_num = params.get("bit", 0)
    value = params.get("value", False)

    if plc_ip not in _mock_plcs:
        raise ValueError(f"PLC {plc_ip} not found")
    if not _mock_plcs[plc_ip]["connected"]:
        raise ConnectionError(f"PLC {plc_ip} not connected")
    if not 0 <= bit_num <= 7:
        raise ValueError(f"Bit must be 0-7, got {bit_num}")

    current = _get_mock_byte(plc_ip, area, byte_addr)
    new_value = _set_bit(current, bit_num, bool(value))
    _mock_memory[plc_ip][area][byte_addr] = new_value

    return {
        "ip": plc_ip,
        "area": area,
        "byte": byte_addr,
        "bit": bit_num,
        "success": True,
        "value_written": bool(value),
        "byte_before": current,
        "byte_after": new_value,
    }


def _mock_list_connections(params: dict, device_id: str) -> dict:
    return {
        "plcs": {
            ip: {"connected": info["connected"], "name": info["name"]}
            for ip, info in _mock_plcs.items()
        },
        "connected_count": sum(1 for p in _mock_plcs.values() if p["connected"]),
        "total_count": len(_mock_plcs),
    }


# Register mocks
for action, handler in {
    "read_area": _mock_read_area,
    "write_bit": _mock_write_bit,
    "list_connections": _mock_list_connections,
}.items():
    register_mock_handler("plc", action, handler)


# ═══════════════════════════════════════════════════════════════
# Agent tools
# ═══════════════════════════════════════════════════════════════

def _send(action: str, params: dict = None, device_id: str = "") -> str:
    result = send_command("plc", action, params or {}, device_id)
    return json.dumps(result, ensure_ascii=False)


@tool
def plc_read_input(plc_ip: str, byte_address: int, device_id: str = "") -> str:
    """Read an Input byte (I area / process inputs) from a Siemens PLC.

    Returns all 8 bits of the specified byte.

    Args:
        plc_ip: PLC IP address (e.g. "192.168.1.101").
        byte_address: Byte number to read (e.g. 0 for IB0).
        device_id: Optional device ID override.

    Returns:
        JSON with ip, area, byte, bits {bit_0..bit_7}, raw_value.
    """
    return _send("read_area", {
        "plc_ip": plc_ip, "area": "input", "byte_address": byte_address,
    }, device_id or plc_ip)


@tool
def plc_read_output(plc_ip: str, byte_address: int, device_id: str = "") -> str:
    """Read an Output byte (Q area / process outputs) from a Siemens PLC.

    Args:
        plc_ip: PLC IP address.
        byte_address: Byte number to read (e.g. 0 for QB0).
        device_id: Optional device ID override.

    Returns:
        JSON with ip, area, byte, bits {bit_0..bit_7}, raw_value.
    """
    return _send("read_area", {
        "plc_ip": plc_ip, "area": "output", "byte_address": byte_address,
    }, device_id or plc_ip)


@tool
def plc_read_memory(plc_ip: str, byte_address: int, device_id: str = "") -> str:
    """Read a Memory byte (M area / markers) from a Siemens PLC.

    Args:
        plc_ip: PLC IP address.
        byte_address: Byte number to read (e.g. 0 for MB0).
        device_id: Optional device ID override.

    Returns:
        JSON with ip, area, byte, bits {bit_0..bit_7}, raw_value.
    """
    return _send("read_area", {
        "plc_ip": plc_ip, "area": "memory", "byte_address": byte_address,
    }, device_id or plc_ip)


@tool
def plc_write_output(plc_ip: str, byte_address: int, bit: int, value: bool, device_id: str = "") -> str:
    """Write a single bit in an Output byte (Q area) of a Siemens PLC.

    Args:
        plc_ip: PLC IP address.
        byte_address: Output byte number (e.g. 0 for QB0).
        bit: Bit position within the byte (0-7).
        value: True to set, False to clear.
        device_id: Optional device ID override.

    Returns:
        JSON with success status, bit written, byte before/after.
    """
    return _send("write_bit", {
        "plc_ip": plc_ip, "area": "output",
        "byte_address": byte_address, "bit": bit, "value": value,
    }, device_id or plc_ip)


@tool
def plc_write_memory(plc_ip: str, byte_address: int, bit: int, value: bool, device_id: str = "") -> str:
    """Write a single bit in a Memory byte (M area / marker) of a Siemens PLC.

    Args:
        plc_ip: PLC IP address.
        byte_address: Memory byte number (e.g. 0 for MB0).
        bit: Bit position within the byte (0-7).
        value: True to set, False to clear.
        device_id: Optional device ID override.

    Returns:
        JSON with success status, bit written, byte before/after.
    """
    return _send("write_bit", {
        "plc_ip": plc_ip, "area": "memory",
        "byte_address": byte_address, "bit": bit, "value": value,
    }, device_id or plc_ip)


@tool
def plc_list_connections(device_id: str = "") -> str:
    """List all PLC connections and their status.

    Returns:
        JSON with plcs dict {ip: {connected, name}}, connected_count, total_count.
    """
    return _send("list_connections", device_id=device_id)


# ═══════════════════════════════════════════════════════════════
# Exports
# ═══════════════════════════════════════════════════════════════

PLC_READ_TOOLS = [plc_read_input, plc_read_output, plc_read_memory, plc_list_connections]
PLC_WRITE_TOOLS = [plc_write_output, plc_write_memory]

PLC_TOOLS = PLC_READ_TOOLS + PLC_WRITE_TOOLS
