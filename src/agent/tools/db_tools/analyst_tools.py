"""
analyst_tools.py - SQL analysis tools for the analysis worker

Tools:
- query_sql: Execute read-only SQL via Supabase RPC
- list_tables: List tables in a schema
- describe_table: Describe columns of a table
- list_rpc_functions: List stored functions in a schema
"""
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
    """Execute a read-only SQL query on the database.

    The query MUST be a SELECT or WITH (CTE) statement. INSERT, UPDATE, DELETE,
    DROP, and other write operations are blocked at the database level.
    Results are limited to 1000 rows and 10 seconds timeout.

    Args:
        query: SQL SELECT query to execute.
        description: Brief description of what this query does (for logging).

    Returns:
        dict with 'success', 'data' (list of row dicts), 'row_count', and 'description'.
    """
    try:
        sb = _get_supabase()
        result = sb.rpc("execute_readonly_query", {"p_query": query}).execute()

        rows = result.data if result.data else []
        # RPC wraps result in a JSON string sometimes
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
        # Try to extract detailed error from Supabase response
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

    Returns:
        dict with 'success' and 'tables' (list of table info dicts).
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

    Returns:
        dict with 'success' and 'columns' (list of column info dicts).
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
    """Prepare structured chart data for frontend visualization.

    Use this tool AFTER querying data to format it for rendering as a chart.
    The frontend will receive this as a dedicated SSE 'chart' event.

    Args:
        chart_type: Type of chart ('bar', 'line', 'pie', 'scatter', 'table').
        title: Chart title displayed to the user.
        x_axis: Label for the X axis (or category field for pie charts).
        y_axis: Label for the Y axis (or value field for pie charts).
        data: List of data points. Each point is a dict, e.g. [{"x": "Jan", "y": 10}, ...].

    Returns:
        dict with 'success' and the structured chart_data.
    """
    chart_data = {
        "chart_type": chart_type,
        "title": title,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "data": data[:500],  # Cap at 500 data points for frontend performance
        "data_count": len(data),
    }
    return {
        "success": True,
        "chart_data": chart_data,
    }


@tool
def list_rpc_functions(schema_name: str = "public") -> dict:
    """List available RPC (stored) functions in a database schema.

    Use this to discover useful pre-built functions that can provide
    aggregated data, stats, or perform complex operations.

    Args:
        schema_name: Schema name (default: 'public'). Common schemas: public, chat, lab.

    Returns:
        dict with 'success' and 'functions' (list of function info dicts).
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


# All tools exported for bind_tools
ANALYST_TOOLS = [query_sql, list_tables, describe_table, list_rpc_functions]
