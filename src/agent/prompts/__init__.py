"""Prompts del sistema multi-agente"""
from .tutor_prompt import TUTOR_SYSTEM_PROMPT
from .troubleshooter_prompt import TROUBLESHOOTER_SYSTEM_PROMPT
from .summarizer_prompt import SUMMARIZER_SYSTEM_PROMPT

__all__ = [
    "TUTOR_SYSTEM_PROMPT",
    "TROUBLESHOOTER_SYSTEM_PROMPT", 
    "SUMMARIZER_SYSTEM_PROMPT",
]
