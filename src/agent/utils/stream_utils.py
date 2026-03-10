"""
stream_utils.py - Helper para streaming en tiempo real desde workers.

Uso en cualquier worker:
    from src.agent.utils.stream_utils import get_worker_stream

    stream = get_worker_stream(state, "research")
    stream.tool("rag_search", "Buscando documentos sobre PID...")
    stream.found("Encontre 3 documentos relevantes")
    stream.status("Analizando resultados...")
"""

from typing import Optional, Callable
from src.agent.utils.logger import logger


class WorkerStream:
    """Wrapper para emitir chunks de streaming desde un worker."""

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
        """Narrar que se esta ejecutando un tool."""
        self._emit("tool_status", description, tool=tool_name, status="executing")

    def tool_done(self, tool_name: str, description: str):
        """Narrar que un tool termino."""
        self._emit("tool_status", description, tool=tool_name, status="completed")

    def status(self, message: str):
        """Narrar un status general."""
        self._emit("tool_status", message, tool=self._worker, status="info")

    def found(self, message: str):
        """Narrar que se encontro algo."""
        self._emit("tool_status", message, tool=self._worker, status="found")

    def thinking(self, message: str):
        """Narrar pensamiento breve (1 linea max)."""
        if len(message) > 150:
            message = message[:150] + "..."
        self._emit("partial", message)


def get_worker_stream(state: dict, worker_name: str) -> WorkerStream:
    """
    Obtener un WorkerStream para el worker actual.

    Si no hay callback registrado (modo local, tests, voice), retorna
    un WorkerStream inactivo que ignora todas las emisiones.
    """
    callback = None
    session_id = state.get("_stream_session_id")
    if session_id:
        try:
            from api_server import get_stream_callback
            callback = get_stream_callback(session_id)
        except ImportError:
            pass  # Fuera del contexto de api_server (Streamlit, tests)
    return WorkerStream(callback, worker_name)
