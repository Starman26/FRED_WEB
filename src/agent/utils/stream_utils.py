"""
stream_utils.py - Real-time streaming helper for workers.
"""

from typing import Optional, Callable
from src.agent.utils.logger import logger


class WorkerStream:
    """Emits streaming chunks from a worker."""

    def __init__(self, callback: Optional[Callable], worker_name: str):
        self._cb = callback
        self._worker = worker_name

    @property
    def is_active(self) -> bool:
        return self._cb is not None

    def _emit(self, chunk_type: str, content: str, **extra):
        if self._cb:
            try:
                self._cb({
                    "type": chunk_type,
                    "source": self._worker,
                    "content": content,
                    **extra,
                })
            except Exception as e:
                logger.warning(self._worker, f"Stream emit failed: {e}")

    def tool(self, tool_name: str, description: str):
        """Emit tool execution start."""
        self._emit("tool_status", description, tool=tool_name, status="executing")

    def tool_done(self, tool_name: str, description: str):
        """Emit tool completion."""
        self._emit("tool_status", description, tool=tool_name, status="completed")

    def status(self, message: str):
        """Emit a general status update."""
        self._emit("tool_status", message, tool=self._worker, status="info")

    def found(self, message: str):
        """Emit a discovery notification."""
        self._emit("tool_status", message, tool=self._worker, status="found")

    def thinking(self, message: str):
        """Emit a brief thinking message (truncated to 150 chars)."""
        if len(message) > 150:
            message = message[:150] + "..."
        self._emit("partial", message)


def get_worker_stream(state: dict, worker_name: str) -> WorkerStream:
    """Get a WorkerStream for the current worker. Returns an inactive no-op stream if no callback is registered."""
    callback = None
    session_id = state.get("_stream_session_id")
    if session_id:
        try:
            from api_server import get_stream_callback
            callback = get_stream_callback(session_id)
        except ImportError:
            pass
    return WorkerStream(callback, worker_name)
