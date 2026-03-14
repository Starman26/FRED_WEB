"""analyst_tools.py - SQL analysis tools for the analysis worker."""
import json
from typing import Optional
from langchain_core.tools import tool


def _get_supabase():
    """Get Supabase client from service registry."""
    from src.agent.services import get_supabase
    client = get_supabase()
    if not client:
        raise RuntimeError("Supabase not available")
    return client


@tool
def query_sql(query: str, description: str = "") -> dict:
    """Execute a read-only SQL query (SELECT/CTE only, max 1000 rows, 10s timeout).

    Args:
        query: SQL SELECT query to execute.
        description: Brief description of what this query does (for logging).
    """
    try:
        sb = _get_supabase()
        result = sb.rpc("execute_readonly_query", {"p_query": query}).execute()

        rows = result.data if result.data else []
        # RPC sometimes wraps result in a JSON string
        if isinstance(rows, str):
            rows = json.loads(rows)
        if rows is None:
            rows = []

        return {
            "success": True,
            "data": rows[:1000],
            "row_count": len(rows),
            "description": description,
        }
    except Exception as e:
        error_msg = str(e)
        if hasattr(e, 'message'):
            error_msg = e.message
        elif hasattr(e, 'args') and len(e.args) > 0:
            error_msg = str(e.args[0])
        return {
            "success": False,
            "error": error_msg,
            "description": description,
        }


@tool
def list_tables(schema_name: str = "public") -> dict:
    """List all tables in a database schema.

    Args:
        schema_name: Schema name (default: 'public'). Common schemas: public, chat, lab.
    """
    query = f"""
        SELECT table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = '{schema_name}'
        ORDER BY table_name
    """
    try:
        sb = _get_supabase()
        result = sb.rpc("execute_readonly_query", {"p_query": query}).execute()
        rows = result.data if result.data else []
        if isinstance(rows, str):
            rows = json.loads(rows)
        if rows is None:
            rows = []
        return {
            "success": True,
            "schema": schema_name,
            "tables": rows,
            "count": len(rows),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def describe_table(table_name: str, schema_name: str = "public") -> dict:
    """Describe the columns of a database table.

    Args:
        table_name: Name of the table to describe.
        schema_name: Schema name (default: 'public').
    """
    query = f"""
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = '{schema_name}' AND table_name = '{table_name}'
        ORDER BY ordinal_position
    """
    try:
        sb = _get_supabase()
        result = sb.rpc("execute_readonly_query", {"p_query": query}).execute()
        rows = result.data if result.data else []
        if isinstance(rows, str):
            rows = json.loads(rows)
        if rows is None:
            rows = []
        return {
            "success": True,
            "schema": schema_name,
            "table": table_name,
            "columns": rows,
            "count": len(rows),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def prepare_chart(
    chart_type: str,
    title: str,
    x_axis: str,
    y_axis: str,
    data: list,
) -> dict:
    """Prepare structured chart data for frontend visualization (sent as SSE 'chart' event).

    Args:
        chart_type: 'bar', 'line', 'pie', 'scatter', or 'table'.
        title: Chart title displayed to the user.
        x_axis: Label for the X axis (or category field for pie charts).
        y_axis: Label for the Y axis (or value field for pie charts).
        data: List of data points, e.g. [{"x": "Jan", "y": 10}, ...].
    """
    chart_data = {
        "chart_type": chart_type,
        "title": title,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "data": data[:500],
        "data_count": len(data),
    }
    return {
        "success": True,
        "chart_data": chart_data,
    }


@tool
def list_rpc_functions(schema_name: str = "public") -> dict:
    """List available RPC (stored) functions in a database schema.

    Args:
        schema_name: Schema name (default: 'public'). Common schemas: public, chat, lab.
    """
    query = f"""
        SELECT
            p.proname AS function_name,
            pg_get_function_arguments(p.oid) AS arguments,
            t.typname AS return_type,
            d.description AS description
        FROM pg_proc p
        JOIN pg_namespace n ON p.pronamespace = n.oid
        JOIN pg_type t ON p.prorettype = t.oid
        LEFT JOIN pg_description d ON p.oid = d.objoid
        WHERE n.nspname = '{schema_name}'
        AND p.prokind = 'f'
        ORDER BY p.proname
    """
    try:
        sb = _get_supabase()
        result = sb.rpc("execute_readonly_query", {"p_query": query}).execute()
        rows = result.data if result.data else []
        if isinstance(rows, str):
            rows = json.loads(rows)
        if rows is None:
            rows = []
        return {
            "success": True,
            "schema": schema_name,
            "functions": rows,
            "count": len(rows),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


ANALYST_TOOLS = [query_sql, list_tables, describe_table, list_rpc_functions]
