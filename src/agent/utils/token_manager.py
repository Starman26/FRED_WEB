"""
token_manager.py - Token tracking and balance management

Usage:
    from src.agent.utils.token_manager import check_balance, deduct_tokens, get_usage_from_response
"""

import os
import re
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Schema de Supabase donde viven las funciones RPC de tokens.
# Usar "public" si las funciones están en el schema por defecto.
_TOKEN_SCHEMA = os.getenv("TOKEN_SCHEMA", "public")

# UUID validation regex (compiled once at module level)
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _get_supabase():
    """Lazy import to avoid circular deps"""
    try:
        from src.agent.services import get_supabase
        return get_supabase()
    except Exception as e:
        logger.warning(f"Could not get supabase client: {e}")
        return None


def _rpc_call(supabase, fn_name: str, params: dict):
    """
    Ejecuta una llamada RPC en el schema configurado.
    Si el schema es 'public' llama directo (sin .schema()),
    de lo contrario usa .schema(TOKEN_SCHEMA).
    """
    if _TOKEN_SCHEMA and _TOKEN_SCHEMA != "public":
        return supabase.schema(_TOKEN_SCHEMA).rpc(fn_name, params).execute()
    return supabase.rpc(fn_name, params).execute()


def check_balance(user_id: str) -> Dict[str, Any]:
    """
    Check token balance for a user.

    Returns:
        {"total": 10000, "used": 500, "available": 9500, "has_credits": True}
        or {"has_credits": True, "error": "..."} on failure (fail-open)
    """
    if not user_id:
        return {"has_credits": True, "available": 0, "error": "No user_id"}

    try:
        supabase = _get_supabase()
        if not supabase:
            return {"has_credits": True, "available": 0, "error": "No supabase"}

        resp = _rpc_call(supabase, "check_token_balance", {"p_user_id": user_id})
        if resp.data:
            return resp.data
        return {"has_credits": True, "available": 0, "error": "No balance record"}
    except Exception as e:
        logger.warning(f"Token balance check failed: {e}")
        # Fail open - don't block users if token system has issues
        return {"has_credits": True, "available": 0, "error": str(e)}


def deduct_tokens(
    user_id: str,
    amount: int,
    description: str = "API usage",
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Deduct tokens after a successful API call.

    Returns:
        {"success": True, "deducted": 150, "remaining": 9850}
        or {"success": False, "error": "..."}
    """
    if not user_id or amount <= 0:
        return {"success": False, "error": "Invalid user_id or amount"}

    try:
        supabase = _get_supabase()
        if not supabase:
            return {"success": False, "error": "No supabase"}

        params = {
            "p_user_id": user_id,
            "p_amount": amount,
            "p_description": description,
        }
        # Only pass session_id if it's a valid UUID (Supabase expects uuid type)
        if session_id:
            is_uuid = bool(_UUID_RE.match(session_id))
            print(f"[TOKEN_MGR] session_id='{session_id}' is_uuid={is_uuid}", flush=True)
            if is_uuid:
                params["p_session_id"] = session_id

        resp = _rpc_call(supabase, "deduct_tokens", params)
        result = resp.data if resp.data else {"success": False, "error": "No response"}

        if isinstance(result, dict) and result.get("success"):
            logger.info(f"Tokens deducted: {amount} for user {user_id[:8]}... | remaining: {result.get('remaining')}")
        else:
            logger.warning(f"Token deduction failed: {result}")

        return result
    except Exception as e:
        logger.warning(f"Token deduction error: {e}")
        return {"success": False, "error": str(e)}


def get_usage_from_response(response) -> int:
    """
    Extract total token usage (input + output) from an LLM response.

    Works with:
    - ChatAnthropic responses (response.usage_metadata or response.response_metadata)
    - ChatOpenAI responses (response.usage_metadata)
    - ChatGoogleGenerativeAI responses
    """
    total = 0

    # Method 1: usage_metadata (works for both Anthropic and OpenAI via LangChain)
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        meta = response.usage_metadata
        if isinstance(meta, dict):
            total = meta.get("input_tokens", 0) + meta.get("output_tokens", 0)
        elif hasattr(meta, "input_tokens"):
            total = (meta.input_tokens or 0) + (meta.output_tokens or 0)
        if total > 0:
            return total

    # Method 2: response_metadata (Anthropic specific)
    if hasattr(response, "response_metadata") and response.response_metadata:
        rm = response.response_metadata
        usage = rm.get("usage", {})
        if usage:
            total = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        if total > 0:
            return total

    # Method 3: Estimate from content length (fallback)
    if hasattr(response, "content") and response.content:
        # Rough estimate: ~4 chars per token for output, assume input = 2x output
        output_estimate = len(response.content) // 4
        total = output_estimate * 3  # rough input+output estimate

    return total
