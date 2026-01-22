"""
utils.py - Utilidades generales
"""
from typing import Any, Optional


def safe_strip(value: Any) -> str:
    """Safely strip a value, returning empty string if not a string"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def truncate(text: str, max_len: int = 100, suffix: str = "...") -> str:
    """Trunca texto a un máximo de caracteres"""
    if len(text) <= max_len:
        return text
    return text[:max_len - len(suffix)] + suffix


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON, returning default on failure"""
    import json
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default
