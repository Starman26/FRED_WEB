"""
analysis_node.py - SQL data analysis worker.

Uses iterative tool-calling (bind_tools) to query Supabase and generate charts.
Only accessible via interaction_mode == 'analysis'.
"""
import json
import os
import re
import traceback
from typing import Dict, Any
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from src.agent.state import AgentState
from src.agent.contracts.worker_contract import WorkerOutputBuilder
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_execute, event_report

from src.agent.tools.db_tools.analyst_tools import ANALYST_TOOLS

MAX_TOOL_ITERATIONS = 15
ANALYSIS_SYSTEM_PROMPT = """You are ORION's Data Analyst. You execute SQL queries on a PostgreSQL database (Supabase).

AVAILABLE TOOLS:
- list_tables(schema_name): List all tables in a schema
- describe_table(table_name, schema_name): Get column names and types
- query_sql(query, description): Execute a read-only SELECT query
- list_rpc_functions(schema_name): List available RPC/stored functions

AVAILABLE SCHEMAS: public, chat, lab

═══════════════════════════════════════════════════════════
YOU MUST FOLLOW THIS EXACT THINKING PROCESS — DO NOT SKIP STEPS
═══════════════════════════════════════════════════════════

PHASE 1: UNDERSTAND THE REQUEST
Before doing ANYTHING, think about what the user is asking.
- What data do they need?
- Which schemas are likely relevant? (lab for equipment/sensors, chat for messages/sessions, public for users/teams)
- What would a complete answer look like?

PHASE 2: DISCOVER — Map the entire landscape
Do NOT write any query_sql yet. This is read-only exploration.
a) Call list_tables() for EVERY relevant schema
b) Call list_rpc_functions() for EVERY relevant schema — pre-built functions often solve the problem directly
c) Look at the table names and function names. Which ones relate to the user's question?
d) Call describe_table() on EVERY table that could be relevant (at least 3)

IMPORTANT: After list_rpc_functions() returns results, READ every function name carefully.
If ANY function name matches the user's question (even partially), call it IMMEDIATELY using:
  SELECT * FROM schema.function_name(arguments)
This is FASTER and MORE RELIABLE than writing your own SQL.

PHASE 3: PLAN — Design your query strategy
Now that you know what exists, design a plan:
- Which tables have the data I need?
- Are there RPC functions that already do what I need? (e.g., check_door_status, get_equipment_summary)
- What columns will I use? (ONLY use column names confirmed by describe_table)
- What JOINs do I need?
- What filters? (auth_user_id, team_id, status, etc.)
- Plan at least 2 different approaches in case the first fails

PHASE 4: EXECUTE — Run queries, one at a time
- Start with the simplest query to validate your assumptions
- If a query fails, READ THE ERROR and fix it — do NOT repeat the same query
- If a query returns 0 rows, check: Am I filtering wrong? Try SELECT DISTINCT to check actual values
- If an RPC function exists for what you need, call it: SELECT * FROM schema.function_name(params)
- Build complexity gradually — simple counts first, then detailed analysis

PHASE 5: ANALYZE & VISUALIZE
- Look at the data you collected
- Create ==CHART== blocks for every meaningful data breakdown
- Write a brief summary (max 5-7 lines of text) — let the charts speak

PHASE 6: PRESENT RESULTS
Structure your response as:
1. One-liner key finding (bold)
2. Charts with brief captions
3. Sources section: which tables, how many records, filters applied

═══════════════════════════════════════════════════════════
HARD RULES
═══════════════════════════════════════════════════════════

NEVER skip Phase 2. NEVER write query_sql before completing discovery.
NEVER repeat a failed query — if it failed once, change the approach.
NEVER invent table or column names — only use confirmed names.
NEVER say "no data" without exploring all schemas and at least 3 tables.
NEVER include SUGGESTIONS--- blocks.
ALWAYS use schema-qualified names: lab.table_name, chat.table_name
ALWAYS include LIMIT (default 100, max 1000)
ALWAYS respond in the same language the user writes in.
NEVER use emojis.

═══════════════════════════════════════════════════════════
OUTPUT FORMAT — CHARTS FIRST
═══════════════════════════════════════════════════════════

ALWAYS prioritize presenting data as charts. Text is allowed but charts come FIRST.

Structure:
1. ONE-LINER SUMMARY (1-2 sentences, bold)
2. CHARTS — convert data into ==CHART== blocks before writing text
3. DETAILED TEXT — expand with analysis, explanations, recommendations as needed
4. SOURCES

MINIMUM 2 charts per response. Convert data to charts whenever possible:
- Lists of items with attributes → table chart
- Categories with counts → bar or pie chart
- Items ranked or scored → bar chart
- Data over time → line chart
- Comparisons → bar chart

NEVER use markdown tables (| col | col |) — ALWAYS use ==CHART== with "type": "table" instead.

Example:
==CHART==
{{"type": "table", "title": "Fortalezas", "columns": ["Fortaleza", "Evidencia", "Nivel"], "data": [{{"Fortaleza": "Comunicacion", "Evidencia": "Mensajes claros", "Nivel": "Alto"}}]}}
==END_CHART==

CHART FORMAT (embed inline):
==CHART==
{{"type": "pie|bar|line|table", "title": "...", "data": [...]}}
==END_CHART==

Chart types:
- pie: {{"type": "pie", "title": "...", "data": [{{"label": "X", "value": 10}}]}}
- bar: {{"type": "bar", "title": "...", "xKey": "cat", "yKey": "val", "data": [{{"cat": "A", "val": 10}}]}}
- line: {{"type": "line", "title": "...", "xKey": "date", "yKey": "val", "data": [{{"date": "2026-01", "val": 5}}]}}
- table: {{"type": "table", "title": "...", "columns": ["A", "B"], "data": [{{"A": 1, "B": 2}}]}}

CURRENT USER CONTEXT:
- auth_user_id: {auth_user_id}
- team_id: {team_id}
- User name: {user_name}

When querying user-specific data, ALWAYS filter by auth_user_id or team_id.
"""


def _convert_markdown_tables_to_charts(text: str) -> str:
    """Convert any markdown tables to ==CHART== blocks as a safety net."""
    table_pattern = r'(\|[^\n]+\|\n\|[-:\| ]+\|\n(?:\|[^\n]+\|\n?)+)'

    def _replace_table(match):
        table_text = match.group(0).strip()
        lines = [l.strip() for l in table_text.split('\n') if l.strip()]

        if len(lines) < 3:
            return table_text

        headers = [h.strip() for h in lines[0].split('|') if h.strip()]

        data = []
        for line in lines[2:]:
            cells = [c.strip() for c in line.split('|') if c.strip()]
            if len(cells) == len(headers):
                row = {headers[i]: cells[i] for i in range(len(headers))}
                data.append(row)

        if not data:
            return table_text

        title = headers[0] + " Data"

        chart_obj = {
            "type": "table",
            "title": title,
            "columns": headers,
            "data": data,
        }

        return f'\n==CHART==\n{json.dumps(chart_obj, ensure_ascii=False)}\n==END_CHART==\n'

    return re.sub(table_pattern, _replace_table, text)


def _build_conversation_history(state, max_turns: int = 2):
    """Build recent conversation history, excluding the current user message."""
    from langchain_core.messages import HumanMessage as HM, AIMessage as AIM

    raw_messages = state.get("messages", []) or []
    history = []

    for m in raw_messages:
        if hasattr(m, "type") and hasattr(m, "content"):
            if m.type == "human":
                history.append(HM(content=m.content))
            elif m.type == "ai" and m.content and m.content.strip():
                content = m.content[:300] + "..." if len(m.content) > 300 else m.content
                history.append(AIM(content=content))
        elif isinstance(m, dict):
            role = m.get("role", m.get("type", ""))
            content = m.get("content", "")
            if role in ("human", "user") and content:
                history.append(HM(content=content))
            elif role in ("ai", "assistant") and content and content.strip():
                content = content[:300] + "..." if len(content) > 300 else content
                history.append(AIM(content=content))

    if history and isinstance(history[-1], HM):
        history = history[:-1]

    if len(history) > max_turns * 2:
        history = history[-(max_turns * 2):]

    return history


def get_last_user_message(state: AgentState) -> str:
    for m in reversed(state.get("messages", []) or []):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
        if isinstance(m, dict) and m.get("role") in ("human", "user"):
            return (m.get("content") or "").strip()
    return ""


def analysis_node(state: AgentState) -> Dict[str, Any]:
    """SQL analysis worker using iterative tool-calling."""
    start_time = datetime.utcnow()
    logger.node_start("analysis_node", {})
    events = [event_execute("analysis", "Analyzing data...")]

    from src.agent.utils.stream_utils import get_worker_stream
    stream = get_worker_stream(state, "analysis")

    user_message = get_last_user_message(state)
    user_name = state.get("user_name", "User")
    model_name = state.get("llm_model") or os.getenv("DEFAULT_MODEL", "gemini-2.0-flash")
    logger.info("analysis_node", f"user_message='{user_message[:200]}' (len={len(user_message)})")
    logger.info("analysis_node", f"model_name='{model_name}', user_name='{user_name}'")

    if not user_message:
        logger.info("analysis_node", "No user message found, returning awaiting query")
        output = WorkerOutputBuilder.tutor(
            content="What data would you like me to analyze?",
            summary="Awaiting analysis query",
            confidence=1.0,
        )
        output.worker = "analysis"
        return {
            "worker_outputs": [output.model_dump()],
            "events": events + [event_report("analysis", "Awaiting query")],
        }

    # Upgrade to a stronger model for multi-step SQL reasoning
    if model_name in ("gpt-4o-mini", "gemini-2.0-flash"):
        model_name = "gpt-4o"
        state = {**state, "llm_model": model_name}
        logger.info("analysis_node", f"Upgraded model to {model_name} for analysis")

    try:
        from src.agent.utils.llm_factory import get_llm
        logger.info("analysis_node", "Calling get_llm...")
        llm = get_llm(state, temperature=0)
        logger.info("analysis_node", f"LLM initialized: {type(llm).__name__}, model={model_name}")
        llm_with_tools = llm.bind_tools(ANALYST_TOOLS)
        logger.info("analysis_node", f"bind_tools OK, tools={[t.name for t in ANALYST_TOOLS]}")
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("analysis_node", f"Error initializing LLM or bind_tools: {e}\n{tb}")
        output = WorkerOutputBuilder.tutor(
            content=f"Error initializing analysis: {e}",
            summary="LLM init error",
            confidence=0.0,
        )
        output.worker = "analysis"
        return {
            "worker_outputs": [output.model_dump()],
            "events": events + [event_report("analysis", "Error")],
        }

    # Check for previous worker content (hybrid enrichment mode)
    worker_outputs = state.get("worker_outputs", [])
    previous_content = ""
    for wo in worker_outputs:
        if isinstance(wo, dict) and wo.get("content"):
            previous_content += wo["content"] + "\n"
    has_previous_context = len(previous_content.strip()) > 50
    if has_previous_context:
        logger.info("analysis_node", f"Enrichment mode: {len(previous_content)} chars from previous workers")
    else:
        logger.info("analysis_node", "Full exploration mode: no previous worker context")

    auth_user_id = state.get("auth_user_id", "unknown")
    team_id = state.get("team_id", "unknown")
    logger.info("analysis_node", f"auth_user_id={auth_user_id}, team_id={team_id}")

    context_section = ""
    if has_previous_context:
        context_section = f"""

PREVIOUS WORKER OUTPUT (use this as your starting point):
\"\"\"
{previous_content[:3000]}
\"\"\"

You already have a base answer above from another worker. Your job is to:
1. ENRICH it with additional SQL queries if the data can be improved
2. FORMAT it with charts and statistics
3. Keep the SQL tools for any additional data you need
4. Do NOT repeat work already done — build on top of it
"""

    try:
        system_prompt = ANALYSIS_SYSTEM_PROMPT.format(
            user_name=user_name,
            auth_user_id=auth_user_id,
            team_id=team_id,
        ) + context_section
        logger.info("analysis_node", f"System prompt built, length={len(system_prompt)}, has_context={has_previous_context}")
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("analysis_node", f"System prompt format failed: {e}\n{tb}")
        output = WorkerOutputBuilder.tutor(
            content=f"Error building analysis prompt: {e}",
            summary="Prompt build error",
            confidence=0.0,
        )
        output.worker = "analysis"
        return {
            "worker_outputs": [output.model_dump()],
            "events": events + [event_report("analysis", "Error")],
        }

    try:
        messages = [SystemMessage(content=system_prompt)]
        history = _build_conversation_history(state, max_turns=2)
        logger.info("analysis_node", f"History built: {len(history)} messages")
        messages.extend(history)
        messages.append(HumanMessage(content=user_message))
        logger.info("analysis_node", f"Total messages for LLM: {len(messages)}")
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("analysis_node", f"Message building failed: {e}\n{tb}")
        output = WorkerOutputBuilder.tutor(
            content=f"Error building messages: {e}",
            summary="Message build error",
            confidence=0.0,
        )
        output.worker = "analysis"
        return {
            "worker_outputs": [output.model_dump()],
            "events": events + [event_report("analysis", "Error")],
        }

    total_tokens = 0
    tool_map = {t.name: t for t in ANALYST_TOOLS}
    executed_queries = set()
    consecutive_failures = 0
    force_stop = False
    stream.tool("sql_query", "Explorando estructura de la base de datos...")
    logger.info("analysis_node", f"Starting tool loop, max_iterations={MAX_TOOL_ITERATIONS}, tool_map keys={list(tool_map.keys())}")

    iteration = 0
    for iteration in range(MAX_TOOL_ITERATIONS):
        if force_stop:
            logger.info("analysis_node", f"Iteration {iteration}: force_stop=True, breaking loop")
            break

        logger.info("analysis_node", f"Iteration {iteration}: calling LLM...")
        try:
            from src.agent.utils.llm_factory import invoke_and_track
            response, tokens = invoke_and_track(llm_with_tools, messages, "analysis")
            total_tokens += tokens
            logger.info("analysis_node", f"Iteration {iteration}: LLM responded, tool_calls={len(response.tool_calls) if response.tool_calls else 0}, content_len={len(response.content) if response.content else 0}, type={type(response).__name__}")
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("analysis_node", f"LLM call failed at iteration {iteration}: {type(e).__name__}: {e}\n{tb}")
            break

        messages.append(response)

        if not response.tool_calls:
            logger.info("analysis_node", f"Iteration {iteration}: no tool calls, LLM finished. content preview: '{(response.content or '')[:200]}'")
            break

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc.get("id", f"call_{iteration}_{tool_name}")

            events.append(event_execute("analysis", f"Running {tool_name}..."))
            logger.info("analysis_node", f"Tool call: {tool_name}(args={tool_args})")
            if tool_name == "query_sql":
                stream.tool("sql_execute", "Ejecutando consulta en la base de datos...")
            elif tool_name in ("list_tables", "describe_table", "list_rpc_functions"):
                stream.tool(tool_name, f"Explorando esquema: {tool_name}...")

            if tool_name == "query_sql":
                query_key = tool_args.get("query", "").strip()
                if query_key in executed_queries:
                    logger.info("analysis_node", f"Duplicate query detected, consecutive_failures={consecutive_failures}")
                    messages.append(ToolMessage(
                        content="ERROR: You already tried this exact query. You MUST try a DIFFERENT approach. Use describe_table() to check column names, or try a completely different table.",
                        tool_call_id=tool_id,
                    ))
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        messages.append(ToolMessage(
                            content="STOP: You have repeated failed queries 3 times. Summarize what you found so far and present your results. Do not make any more queries.",
                            tool_call_id=f"{tool_id}_stop",
                        ))
                        force_stop = True
                    continue
                executed_queries.add(query_key)

            if tool_name in tool_map:
                try:
                    result = tool_map[tool_name].invoke(tool_args)
                    logger.info("analysis_node", f"Tool {tool_name} returned: success={result.get('success') if isinstance(result, dict) else 'N/A'}")
                    if tool_name == "query_sql" and isinstance(result, dict):
                        if not result.get("success"):
                            consecutive_failures += 1
                            logger.info("analysis_node", f"query_sql failed, consecutive_failures={consecutive_failures}")
                        else:
                            consecutive_failures = 0
                            row_count = result.get("row_count", 0)
                            if row_count:
                                stream.found(f"Obtuve {row_count} registros, procesando...")
                except Exception as e:
                    tb_tool = traceback.format_exc()
                    logger.error("analysis_node", f"Tool {tool_name} failed: {type(e).__name__}: {e}\n{tb_tool}")
                    result = {"success": False, "error": str(e)}
                    if tool_name == "query_sql":
                        consecutive_failures += 1
            else:
                logger.error("analysis_node", f"Unknown tool: {tool_name}, available: {list(tool_map.keys())}")
                result = {"success": False, "error": f"Unknown tool: {tool_name}"}

            messages.append(ToolMessage(
                content=str(result) if not isinstance(result, str) else result,
                tool_call_id=tool_id,
            ))

        if consecutive_failures >= 5:
            logger.info("analysis_node", f"5 consecutive failures, forcing stop")
            force_stop = True

    logger.info("analysis_node", f"Tool loop ended. iteration={iteration}, total_tokens={total_tokens}, total_messages={len(messages)}, consecutive_failures={consecutive_failures}")

    result_text = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            result_text = msg.content.strip()
            break

    if not result_text:
        msg_summary = []
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            content_preview = ""
            if hasattr(msg, "content") and msg.content:
                content_preview = f" content='{str(msg.content)[:100]}'"
            has_tools = bool(msg.tool_calls) if hasattr(msg, "tool_calls") else False
            msg_summary.append(f"  [{i}] {msg_type}: has_content={bool(content_preview)}, tool_calls={has_tools}{content_preview}")
        logger.error("analysis_node", f"No final response text found after {iteration + 1} iterations. Messages:\n" + "\n".join(msg_summary))
        result_text = "Analysis complete but no summary was generated. Please try again."

    result_text = re.sub(
        r'---?SUGGESTIONS---?.*?---?END_SUGGESTIONS---?',
        '',
        result_text,
        flags=re.DOTALL,
    ).strip()

    stream.status("Preparando visualizacion de datos...")
    result_text = _convert_markdown_tables_to_charts(result_text)

    processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

    output = WorkerOutputBuilder.tutor(
        content=result_text,
        summary="Data analysis",
        confidence=0.9,
    )
    output.worker = "analysis"
    output.metadata.completed_at = datetime.utcnow().isoformat()
    output.metadata.processing_time_ms = processing_time
    output.metadata.model_used = model_name

    logger.node_end("analysis_node", {"content_length": len(result_text), "iterations": iteration + 1})
    events.append(event_report("analysis", "Analysis complete"))

    return {
        "worker_outputs": [output.model_dump()],
        "events": events,
        "follow_up_suggestions": [],
        "token_usage": total_tokens,
    }
