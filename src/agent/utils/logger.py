"""
logger.py - Logger estructurado para el sistema multi-agente
"""
import os
from datetime import datetime
from typing import Dict, Optional


class AgentLogger:
    """Logger estructurado para nodos del grafo"""

    def __init__(self, level: str = "INFO"):
        self.level = (level or "INFO").upper()
        self.levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        self.current_level = self.levels.get(self.level, 1)
        self.enabled = os.getenv("AGENT_LOG_ENABLED", "true").lower() == "true"

    def _should_log(self, level: str) -> bool:
        return self.enabled and self.levels.get(level.upper(), 1) >= self.current_level

    def _format_message(self, level: str, source: str, message: str, data: Optional[Dict] = None) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        base = f"[{timestamp}] [{level}] [{source}] {message}"
        if data:
            data_str = " | ".join([f"{k}={v}" for k, v in data.items()])
            base += f" | {data_str}"
        return base

    def debug(self, source: str, message: str, data: Optional[Dict] = None):
        if self._should_log("DEBUG"):
            print(self._format_message("DEBUG", source, message, data), flush=True)

    def info(self, source: str, message: str, data: Optional[Dict] = None):
        if self._should_log("INFO"):
            print(self._format_message("INFO", source, message, data), flush=True)

    def warning(self, source: str, message: str, data: Optional[Dict] = None):
        if self._should_log("WARNING"):
            print(self._format_message("WARNING", source, message, data), flush=True)

    def error(self, source: str, message: str, data: Optional[Dict] = None):
        if self._should_log("ERROR"):
            print(self._format_message("ERROR", source, message, data), flush=True)

    def node_start(self, node_name: str, data: Optional[Dict] = None):
        self.info(node_name, "▶ Iniciando nodo", data)

    def node_end(self, node_name: str, data: Optional[Dict] = None):
        self.info(node_name, "✓ Nodo completado", data)


logger = AgentLogger(level=os.getenv("AGENT_LOG_LEVEL", "INFO"))
