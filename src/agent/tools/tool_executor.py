"""
tool_executor.py

Executes registered tools with full lifecycle:
PLANNED -> VALIDATING -> EXECUTING -> VERIFYING -> COMPLETED/FAILED/TIMEOUT/BLOCKED
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone

from src.agent.tools.tool_registry import ToolRegistry, ToolSpec, ToolType, SafetyLevel


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    result: Any = None
    verified: bool = False
    verification_result: Any = None
    duration_ms: float = 0
    error: Optional[str] = None
    retries_used: int = 0
    phase: str = "completed"  # completed | failed | timeout | verification_failed | blocked

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result,
            "verified": self.verified,
            "verification_result": self.verification_result,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "retries_used": self.retries_used,
            "phase": self.phase,
        }

    def to_log_entry(self) -> Dict[str, Any]:
        """Convert to AgentState.tool_execution_log format."""
        return {
            "tool": self.tool_name,
            "phase": self.phase,
            "success": self.success,
            "verified": self.verified,
            "duration_ms": round(self.duration_ms, 1),
            "error": self.error,
            "retries_used": self.retries_used,
            "timestamp": time.time(),
        }


_INTERNAL_KWARGS = frozenset({
    "_stream_callback",
    "_session_id",
    "_worker",
    "station",  # used by safety gate, not by tool functions
})


def _filter_kwargs(kwargs: dict) -> dict:
    """Strip internal flags before passing to tool/verify functions."""
    return {k: v for k, v in kwargs.items() if k not in _INTERNAL_KWARGS}


class ToolExecutor:
    """Runs tools with timeout, retry, verification, and lifecycle streaming."""

    def __init__(self, stream_callback: Optional[Callable] = None):
        self._callback = stream_callback

    def _emit(self, tool_name: str, phase: str, detail: Optional[Dict] = None) -> None:
        if not self._callback:
            return
        payload = {
            "type": "tool_lifecycle",
            "tool": tool_name,
            "phase": phase,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if detail:
            payload.update(detail)
        try:
            self._callback(payload)
        except Exception:
            pass

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        spec = ToolRegistry.get(tool_name)
        if spec is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                phase="failed",
                error=f"Tool '{tool_name}' no encontrado en ToolRegistry",
            )

        self._emit(tool_name, "planned", {
            "tool_type": spec.tool_type.value,
            "safety_level": spec.safety_level.value,
            "timeout_ms": spec.timeout_ms,
        })

        self._emit(tool_name, "validating")

        clean_kwargs = _filter_kwargs(kwargs)
        max_attempts = 1 + spec.retries
        last_error = None

        for attempt in range(max_attempts):
            self._emit(tool_name, "executing", {
                "attempt": attempt + 1,
                "max_attempts": max_attempts,
            })

            t0 = time.perf_counter()
            try:
                result = self._run_with_timeout(spec.fn, clean_kwargs, spec.timeout_ms)
                duration_ms = (time.perf_counter() - t0) * 1000
            except FuturesTimeoutError:
                duration_ms = (time.perf_counter() - t0) * 1000
                last_error = f"Timeout después de {spec.timeout_ms}ms"
                self._emit(tool_name, "timeout", {
                    "attempt": attempt + 1,
                    "duration_ms": round(duration_ms, 1),
                })
                continue
            except Exception as e:
                duration_ms = (time.perf_counter() - t0) * 1000
                last_error = str(e)
                self._emit(tool_name, "failed", {
                    "attempt": attempt + 1,
                    "error": last_error,
                    "duration_ms": round(duration_ms, 1),
                })
                continue

            tool_result = ToolResult(
                tool_name=tool_name,
                success=True,
                result=result,
                duration_ms=round(duration_ms, 1),
                retries_used=attempt,
                phase="completed",
            )

            if spec.verify_fn is not None:
                self._emit(tool_name, "verifying")
                time.sleep(0.3)  # let hardware/bridge settle
                try:
                    v_result = spec.verify_fn(**clean_kwargs)
                    tool_result.verification_result = v_result
                    verified = v_result.get("verified", False) if isinstance(v_result, dict) else bool(v_result)
                    tool_result.verified = verified

                    if not verified:
                        tool_result.success = False
                        tool_result.phase = "verification_failed"
                        reason = v_result.get("reason", "Verificación falló") if isinstance(v_result, dict) else "Verificación falló"
                        tool_result.error = reason
                        self._emit(tool_name, "verification_failed", {
                            "reason": reason,
                            "duration_ms": round(duration_ms, 1),
                        })
                        return tool_result
                except Exception as e:
                    tool_result.verification_result = {"error": str(e)}
                    tool_result.verified = False
                    tool_result.phase = "verification_failed"
                    tool_result.success = False
                    tool_result.error = f"Error en verificación: {e}"
                    self._emit(tool_name, "verification_failed", {
                        "reason": str(e),
                        "duration_ms": round(duration_ms, 1),
                    })
                    return tool_result

            self._emit(tool_name, "completed", {
                "duration_ms": round(duration_ms, 1),
                "verified": tool_result.verified,
            })
            return tool_result

        # All attempts exhausted
        return ToolResult(
            tool_name=tool_name,
            success=False,
            phase="timeout" if "Timeout" in (last_error or "") else "failed",
            error=last_error,
            retries_used=max_attempts - 1,
        )

    @staticmethod
    def _run_with_timeout(fn: Callable, kwargs: dict, timeout_ms: int) -> Any:
        timeout_s = timeout_ms / 1000
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn, **kwargs)
            return future.result(timeout=timeout_s)
