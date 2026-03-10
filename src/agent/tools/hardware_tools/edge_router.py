"""
edge_router.py — Transport layer unificado para comunicación con el lab bridge.

Todas las device tools (xarm, abb, plc, network) usan send_command() de este módulo.
En MOCK_MODE, despacha a mock handlers locales. En modo real, envía por WebSocket.

Configuración vía env vars:
  MOCK_MODE=true|false       (default: true)
  LAB_BRIDGE_URL=wss://...   (WebSocket URL del bridge en Cloud Run)

El mock mode es switcheable en runtime vía set_mock_mode(bool).
"""

import json
import os
import asyncio
import logging
import time
import uuid
from typing import Any, Callable, Dict, Optional
from datetime import datetime, timezone

logger = logging.getLogger("edge_router")

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

_mock_mode: bool = os.getenv("MOCK_MODE", "false").lower() == "true"
LAB_BRIDGE_URL: str = os.getenv("LAB_BRIDGE_URL", "")

# Mock handler registry: {device_type: {action: handler_fn}}
_mock_handlers: Dict[str, Dict[str, Callable]] = {}


def is_mock_mode() -> bool:
    return _mock_mode


def set_mock_mode(enabled: bool) -> None:
    """Switch mock mode on/off at runtime."""
    global _mock_mode
    _mock_mode = enabled
    logger.info(f"Mock mode {'enabled' if enabled else 'disabled'}")


def register_mock_handler(device_type: str, action: str, handler: Callable) -> None:
    """Register a mock handler for a device_type + action pair."""
    if device_type not in _mock_handlers:
        _mock_handlers[device_type] = {}
    _mock_handlers[device_type][action] = handler


# ═══════════════════════════════════════════════════════════════
# Core: send_command
# ═══════════════════════════════════════════════════════════════

def send_command(
    device_type: str,
    action: str,
    params: dict = None,
    device_id: str = "",
    timeout_s: float = 15.0,
) -> dict:
    """Send a command to a device via the lab bridge (or mock).

    This is the SINGLE entry point all device tools use.

    Args:
        device_type: "xarm" | "abb" | "plc" | "shell"
        action: Handler-specific action (e.g. "move_joint", "read_input")
        params: Action parameters dict
        device_id: Target device ID (e.g. "xarm-185", "plc-101")
        timeout_s: Response timeout in seconds

    Returns:
        dict with "status" ("ok"|"error"), "data" or "error" message
    """
    params = params or {}
    request_id = str(uuid.uuid4())[:8]

    if _mock_mode:
        return _dispatch_mock(device_type, action, params, device_id, request_id)
    else:
        return _dispatch_ws(device_type, action, params, device_id, request_id, timeout_s)


def _dispatch_mock(
    device_type: str, action: str, params: dict, device_id: str, request_id: str
) -> dict:
    """Dispatch to registered mock handler."""
    handlers = _mock_handlers.get(device_type, {})
    handler = handlers.get(action)

    if handler is None:
        return {
            "id": request_id,
            "status": "error",
            "device_id": device_id,
            "error": f"No mock handler for {device_type}.{action}",
            "timestamp": _now(),
        }

    try:
        data = handler(params, device_id)
        return {
            "id": request_id,
            "status": "ok",
            "device_id": device_id,
            "data": data,
            "timestamp": _now(),
        }
    except Exception as e:
        logger.error(f"Mock handler error {device_type}.{action}: {e}")
        return {
            "id": request_id,
            "status": "error",
            "device_id": device_id,
            "error": str(e),
            "timestamp": _now(),
        }


def _dispatch_ws(
    device_type: str,
    action: str,
    params: dict,
    device_id: str,
    request_id: str,
    timeout_s: float,
) -> dict:
    """Send command via WebSocket to the lab bridge."""
    try:
        from api_server import send_robot_command, get_main_loop

        loop = get_main_loop()
        if loop is None:
            return _error_response(request_id, device_id, "Server event loop not available")

        # Pack device_type and device_id into params so they reach the lab_bridge.
        # send_robot_command sends: {"id": cmd_id, "command": action, "params": enriched_params}
        # The lab_bridge dispatcher reads device_type from params.
        enriched_params = dict(params)
        enriched_params["_device_type"] = device_type
        enriched_params["_device_id"] = device_id

        future = asyncio.run_coroutine_threadsafe(
            send_robot_command(device_id, action, enriched_params, timeout=timeout_s),
            loop,
        )
        result = future.result(timeout=timeout_s)

        # Normalize response
        if isinstance(result, dict):
            if "status" not in result:
                result["status"] = "ok" if result.get("success", True) else "error"
            result["id"] = request_id
            result["device_id"] = device_id
            result["timestamp"] = result.get("timestamp", _now())
            return result

        return {
            "id": request_id,
            "status": "ok",
            "device_id": device_id,
            "data": result,
            "timestamp": _now(),
        }

    except ImportError:
        return _error_response(request_id, device_id, "api_server not available (standalone mode?)")
    except Exception as e:
        return _error_response(request_id, device_id, str(e))


# ═══════════════════════════════════════════════════════════════
# Multi-device support
# ═══════════════════════════════════════════════════════════════

def send_command_multi(
    device_type: str,
    action: str,
    params: dict,
    device_ids: list,
    timeout_s: float = 15.0,
) -> dict:
    """Send the same command to multiple devices in parallel.

    Returns:
        dict mapping device_id → response
    """
    import concurrent.futures

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(device_ids)) as executor:
        futures = {
            executor.submit(send_command, device_type, action, params, did, timeout_s): did
            for did in device_ids
        }
        for future in concurrent.futures.as_completed(futures):
            did = futures[future]
            try:
                results[did] = future.result()
            except Exception as e:
                results[did] = _error_response("", did, str(e))

    return results


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _error_response(request_id: str, device_id: str, error: str) -> dict:
    return {
        "id": request_id,
        "status": "error",
        "device_id": device_id,
        "error": error,
        "timestamp": _now(),
    }


def parse_device_ids(robot_ids: str) -> list:
    """Parse comma-separated device IDs string to list."""
    return [r.strip() for r in robot_ids.split(",") if r.strip()]
