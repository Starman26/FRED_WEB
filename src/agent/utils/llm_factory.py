"""
llm_factory.py - Centralized LLM instantiation

Place in: src/agent/utils/llm_factory.py

Usage:
    from src.agent.utils.llm_factory import get_llm
    
    llm = get_llm(state)  # reads state["llm_model"] or DEFAULT_MODEL
    llm = get_llm(state, temperature=0.3, max_tokens=500)
    llm = get_llm_from_name("gemini-2.0-flash", temperature=0.7)
"""

import os
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Model name → provider mapping
_PROVIDER_MAP = {
    "claude": "anthropic",
    "anthropic": "anthropic",
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "gemini": "google",
    "gemma": "google",
}


def _detect_provider(model_name: str) -> str:
    """Detect provider from model name."""
    name_lower = model_name.lower()
    for keyword, provider in _PROVIDER_MAP.items():
        if keyword in name_lower:
            return provider
    # Default to openai if unknown
    return "openai"


def get_llm_from_name(
    model_name: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> Any:
    """
    Create an LLM instance from a model name string.
    Supports: Anthropic (Claude), OpenAI (GPT), Google (Gemini)
    """
    provider = _detect_provider(model_name)
    
    kwargs: Dict[str, Any] = {"temperature": temperature}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    
    try:
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model_name, **kwargs)
        
        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            # Gemini uses max_output_tokens instead of max_tokens
            if "max_tokens" in kwargs:
                kwargs["max_output_tokens"] = kwargs.pop("max_tokens")
            return ChatGoogleGenerativeAI(model=model_name, **kwargs)
        
        else:  # openai (default)
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model_name, **kwargs)
    
    except Exception as e:
        logger.error(f"Failed to create LLM ({provider}/{model_name}): {e}")
        # Fallback chain: try openai → gemini → raise
        try:
            from langchain_openai import ChatOpenAI
            fallback = os.getenv("FALLBACK_MODEL", "gpt-4o-mini")
            logger.warning(f"Falling back to {fallback}")
            return ChatOpenAI(model=fallback, temperature=temperature)
        except Exception:
            raise RuntimeError(f"Cannot create any LLM. Original error: {e}")


def get_llm(
    state: Dict[str, Any],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> Any:
    """
    Create LLM from agent state.
    Reads state["llm_model"], falls back to DEFAULT_MODEL env var.
    """
    model_name = (
        state.get("llm_model")
        or os.getenv("DEFAULT_MODEL", "gemini-2.0-flash")
    )
    return get_llm_from_name(model_name, temperature=temperature, max_tokens=max_tokens)


def invoke_and_track(llm, messages, label: str = "llm_call") -> tuple:
    """
    Invoca el LLM y extrae los tokens usados de la respuesta.

    Args:
        llm: Instancia del LLM (ChatOpenAI, ChatAnthropic, etc.)
        messages: Mensajes a enviar al LLM
        label: Etiqueta para logging

    Returns:
        (response, tokens_used) - la respuesta del LLM y el total de tokens
    """
    from src.agent.utils.token_manager import get_usage_from_response

    response = llm.invoke(messages)
    tokens = get_usage_from_response(response)
    if tokens > 0:
        logger.debug(f"[{label}] Tokens used: {tokens}")
    return response, tokens
