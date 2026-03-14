"""
token_manager.py - Token tracking and balance management.
"""

import os
import re
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

_TOKEN_SCHEMA = os.getenv("TOKEN_SCHEMA", "public")

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
    """Execute RPC call using the configured schema."""
    if _TOKEN_SCHEMA and _TOKEN_SCHEMA != "public":
        return supabase.schema(_TOKEN_SCHEMA).rpc(fn_name, params).execute()
    return supabase.rpc(fn_name, params).execute()


def check_balance(user_id: str) -> Dict[str, Any]:
    """Check token balance for a user. Fails open on errors."""
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
        return {"has_credits": True, "available": 0, "error": str(e)}


def deduct_tokens(
    user_id: str,
    amount: int,
    description: str = "API usage",
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Deduct tokens after a successful API call."""
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
        # Supabase expects uuid type for session_id
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
    """Extract total token usage (input + output) from an LLM response."""
    total = 0

    # usage_metadata (Anthropic, OpenAI via LangChain)
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        meta = response.usage_metadata
        if isinstance(meta, dict):
            total = meta.get("input_tokens", 0) + meta.get("output_tokens", 0)
        elif hasattr(meta, "input_tokens"):
            total = (meta.input_tokens or 0) + (meta.output_tokens or 0)
        if total > 0:
            return total

    # response_metadata (Anthropic specific)
    if hasattr(response, "response_metadata") and response.response_metadata:
        rm = response.response_metadata
        usage = rm.get("usage", {})
        if usage:
            total = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        if total > 0:
            return total

    # Rough estimate fallback (~4 chars per token)
    if hasattr(response, "content") and response.content:
        output_estimate = len(response.content) // 4
        total = output_estimate * 3

    return total
