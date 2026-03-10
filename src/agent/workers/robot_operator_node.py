"""
robot_operator_node.py - Worker para control de dispositivos (xArm, ABB, PLC, network)

Usa tool-calling nativo de LangChain: el LLM decide qué herramientas invocar.
La comunicación va por edge_router → lab_bridge.
"""
import os
from typing import Dict, Any, Tuple
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

from src.agent.utils.llm_factory import get_llm

from src.agent.state import AgentState
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_execute, event_report, event_error


# ============================================
# TOOLS
# ============================================
try:
    from src.agent.tools.hardware_tools import (
        XARM_TOOLS,
        ABB_TOOLS,
        PLC_WRITE_TOOLS,
        NETWORK_TOOLS,
    )
    TOOLS_AVAILABLE = True
except ImportError:
    XARM_TOOLS = []
    ABB_TOOLS = []
    PLC_WRITE_TOOLS = []
    NETWORK_TOOLS = []
    TOOLS_AVAILABLE = False


# ============================================
# TOOL SELECTOR
# ============================================

def _select_operator_tools(equipment_type: str = "", equipment_brand: str = "") -> Tuple[list, str]:
    """Select actuate + read tools based on equipment type.
    Returns (tools_list, device_label) for the prompt.
    """
    hint = f"{equipment_type} {equipment_brand}".lower()

    if any(kw in hint for kw in ["xarm", "ufactory", "lite"]):
        return XARM_TOOLS, "xArm Lite 6"
    elif any(kw in hint for kw in ["abb", "irb", "omnicore", "irc5"]):
        return ABB_TOOLS, "ABB IRB"
    elif any(kw in hint for kw in ["plc", "siemens", "s7", "1200", "1500"]):
        return PLC_WRITE_TOOLS + NETWORK_TOOLS, "Siemens PLC"
    elif any(kw in hint for kw in ["robot", "cobot"]):
        return XARM_TOOLS + ABB_TOOLS, "Robot"
    else:
        return XARM_TOOLS + ABB_TOOLS + PLC_WRITE_TOOLS + NETWORK_TOOLS, "Device"


# ============================================
# PROMPT
# ============================================

ROBOT_OPERATOR_PROMPT = """You are the device operator for the FrED Factory lab.
You control a {device_label} using the tools provided.

AVAILABLE TOOLS:
{tool_list}

SAFETY RULES:
1. Communication goes through the lab bridge — you don't connect directly.
2. For large movements (>100mm), verify current position first.
3. If the user says "stop", "para", "emergency" → emergency stop IMMEDIATELY, no questions.
4. After an emergency stop, inform the user they need to manually recover.
5. Report position after every movement.
6. If a tool returns an error, DO NOT retry automatically — inform the user what happened.
7. For PLC writes: confirm with the user before writing to outputs (Q area). Memory (M area) is safe to write.

RESPONSE STYLE:
- Be brief and direct, like a technician at the workbench
- After executing a command, report what happened in one sentence
- Cite actual values: positions, angles, bit states
- Same language as the user

Context: {intent_context}"""


# ============================================
# HELPERS
# ============================================

def _get_last_user_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content
        elif isinstance(msg, dict) and msg.get("role") in ("human", "user"):
            return msg.get("content", "")
    return ""


def _make_output(content: str, summary: str, status: str = "ok",
                 actions_taken: list = None, model_used: str = "",
                 processing_time_ms: float = 0, confidence: float = 0.9) -> Dict[str, Any]:
    """Construye un worker output como dict (compatible con WorkerOutput)."""
    return {
        "worker": "robot_operator",
        "status": status,
        "summary": summary,
        "content": content,
        "evidence": [],
        "next_actions": [],
        "clarification_questions": [],
        "errors": [],
        "confidence": confidence,
        "extra": {"actions_taken": actions_taken or []},
        "metadata": {
            "model_used": model_used,
            "processing_time_ms": processing_time_ms,
        }
    }


# ============================================
# MAIN NODE
# ============================================

def robot_operator_node(state: AgentState) -> Dict[str, Any]:
    """Worker para control de dispositivos via tool-calling del LLM."""
    start_time = datetime.utcnow()
    logger.node_start("robot_operator", {})
    events = [event_execute("robot_operator", "Procesando comando de dispositivo...")]

    from src.agent.utils.stream_utils import get_worker_stream
    stream = get_worker_stream(state, "robot_operator")

    user_message = _get_last_user_message(state)
    intent_analysis = state.get("intent_analysis", {})
    intent_context = (
        f"Intent: {intent_analysis.get('intent', '?')}, "
        f"Action: {intent_analysis.get('action', '?')}, "
        f"Entities: {intent_analysis.get('entities', {})}"
    )

    # ==========================================
    # SELECT TOOLS BASED ON EQUIPMENT
    # ==========================================
    pending = state.get("pending_context", {}) or {}
    eq_type = pending.get("equipment_type", "")
    eq_brand = pending.get("equipment_brand", "")

    entities = intent_analysis.get("entities", {})
    if not eq_type:
        eq_type = entities.get("equipment", "")

    selected_tools, device_label = _select_operator_tools(eq_type, eq_brand)

    if not TOOLS_AVAILABLE or not selected_tools:
        logger.error("robot_operator", "No tools available for this device type")
        output = _make_output(
            content=f"No hay herramientas disponibles para este tipo de dispositivo ({eq_type or 'unknown'}).",
            summary="Tools no disponibles", status="error", confidence=0.0
        )
        return {
            "worker_outputs": [output],
            "events": events + [event_error("robot_operator", "No tools for device type")],
        }

    logger.info("robot_operator", f"Selected {len(selected_tools)} tools for {device_label} (type={eq_type}, brand={eq_brand})")

    # ==========================================
    # CONFIGURAR LLM CON TOOLS
    # ==========================================
    try:
        llm = get_llm(state, temperature=0.1)

        llm_with_tools = llm.bind_tools(selected_tools)
    except Exception as e:
        logger.error("robot_operator", f"Error init LLM: {e}")
        output = _make_output(
            content=f"Error inicializando LLM: {e}",
            summary="Error LLM", status="error", confidence=0.0
        )
        return {
            "worker_outputs": [output],
            "events": events + [event_error("robot_operator", str(e))],
        }

    # ==========================================
    # TOOL-CALLING LOOP
    # ==========================================
    tool_list = "\n".join(f"- {t.name}: {t.description.split(chr(10))[0]}" for t in selected_tools)
    prompt = ROBOT_OPERATOR_PROMPT.format(
        device_label=device_label,
        tool_list=tool_list,
        intent_context=intent_context,
    )
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=user_message)
    ]

    actions_taken = []
    max_iterations = 6
    response = None

    for i in range(max_iterations):
        try:
            response = llm_with_tools.invoke(messages)

            if not response.tool_calls:
                break

            messages.append(response)

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]

                logger.info("robot_operator", f"Tool: {tool_name}({tool_args})")
                stream.tool("robot_command", f"Enviando comando: {tool_name}...")

                tool_fn = next((t for t in selected_tools if t.name == tool_name), None)

                if tool_fn:
                    try:
                        result = tool_fn.invoke(tool_args)
                    except Exception as tool_err:
                        result = f"Error ejecutando {tool_name}: {tool_err}"

                    actions_taken.append(f"{tool_name}({tool_args}) -> {result}")
                    logger.info("robot_operator", f"   -> {result}")
                    stream.tool_done("robot_command", f"{tool_name}: {str(result)[:100]}")
                else:
                    result = f"Tool '{tool_name}' no encontrada"
                    logger.warning("robot_operator", result)

                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        except Exception as e:
            logger.error("robot_operator", f"Error en iteración {i}: {e}")
            messages.append(HumanMessage(
                content=f"Error ejecutando herramienta: {e}. Informa al usuario del error."
            ))
            break

    # ==========================================
    # CONSTRUIR RESPUESTA
    # ==========================================
    final_text = ""
    if response and hasattr(response, 'content') and response.content:
        final_text = response.content
    elif actions_taken:
        final_text = "Operación completada:\n" + "\n".join(f"- {a}" for a in actions_taken)
    else:
        final_text = "No se ejecutaron acciones."

    processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

    output = _make_output(
        content=final_text,
        summary=f"{device_label}: {intent_analysis.get('action', 'operación')}",
        actions_taken=actions_taken,
        model_used=state.get("llm_model") or os.getenv("DEFAULT_MODEL", "gemini-2.0-flash"),
        processing_time_ms=processing_time,
        confidence=0.9 if actions_taken else 0.5,
    )

    events.append(event_report("robot_operator", f"{len(actions_taken)} acción(es) ejecutadas"))
    logger.info("robot_operator", f"Completado | actions={len(actions_taken)} | time={processing_time:.0f}ms")

    return {
        "worker_outputs": [output],
        "events": events,
    }
