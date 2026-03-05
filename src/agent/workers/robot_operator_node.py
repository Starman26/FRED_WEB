"""
robot_operator_node.py - Worker para control del robot xArm

Usa tool-calling nativo de LangChain: el LLM decide qué herramientas invocar.
Incluye AUTO-CONEXIÓN: si el robot no está conectado, se conecta automáticamente.
"""
import os
from typing import Dict, Any
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

from src.agent.utils.llm_factory import get_llm

from src.agent.state import AgentState
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_execute, event_report, event_error


# ============================================
# TOOLS DEL ROBOT
# ============================================
try:
    from src.agent.tools.robot_tools.xarm_tools import XARM_TOOLS
    TOOLS_AVAILABLE = True
except ImportError:
    XARM_TOOLS = []
    TOOLS_AVAILABLE = False


ROBOT_OPERATOR_PROMPT = """Eres el operador del robot xArm en el laboratorio FrED Factory.
Controlas un robot xArm Lite 6 con gripper eléctrico.

TOOLS DISPONIBLES:
- robot_connect: Conectar al robot (YA FUE HECHO automáticamente)
- robot_get_position: Ver posición actual (X, Y, Z)
- robot_move_to: Mover a posición absoluta (x, y, z en mm)
- robot_step: Movimiento incremental en un eje (axis: 'x'/'y'/'z', distance en mm)
- robot_home: Ir a posición Home segura
- robot_emergency_stop: PARO DE EMERGENCIA
- robot_gripper: Abrir/cerrar pinza (action: 'open'/'close')
- robot_status: Estado completo del robot
- robot_clear_errors: Limpiar errores y re-habilitar

REGLAS DE SEGURIDAD:
1. El robot YA ESTÁ CONECTADO. No necesitas llamar robot_connect.
2. Para movimientos grandes (>100mm), primero verifica posición actual.
3. Si el usuario dice "para", "stop", "emergencia" → robot_emergency_stop SIN preguntar.
4. Después de un paro de emergencia, informa que se debe reconectar.
5. Reporta la posición después de cada movimiento.
6. Si hay error, NO reintentes automáticamente — informa al usuario.

LÍMITES DEL WORKSPACE:
- X: -500 a 500 mm
- Y: -500 a 500 mm
- Z: 0 a 400 mm (nunca negativo)
- Velocidad máxima: 150 mm/s
- Paso máximo: ±50 mm

FORMATO: Sé breve y directo. Se directo y claro.

Contexto del usuario: {intent_context}"""


def _get_last_user_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content
        elif isinstance(msg, dict) and msg.get("role") in ("human", "user"):
            return msg.get("content", "")
    return ""


def _auto_connect_robot() -> str:
    """Intenta auto-conectar el robot si no lo está. Retorna status."""
    try:
        from src.agent.services import get_xarm
        client = get_xarm()
        if not client:
            return "no_service"
        
        if client.is_connected:
            return "already_connected"
        
        result = client.connect()
        if result.get("success"):
            return "connected"
        return f"error: {result.get('error', 'unknown')}"
    except Exception as e:
        return f"error: {e}"


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


def robot_operator_node(state: AgentState) -> Dict[str, Any]:
    """Worker para control del robot xArm via tool-calling del LLM."""
    start_time = datetime.utcnow()
    logger.node_start("robot_operator", {})
    events = [event_execute("robot_operator", "Procesando comando de robot...")]
    
    user_message = _get_last_user_message(state)
    intent_analysis = state.get("intent_analysis", {})
    intent_context = (
        f"Intent: {intent_analysis.get('intent', '?')}, "
        f"Action: {intent_analysis.get('action', '?')}, "
        f"Entities: {intent_analysis.get('entities', {})}"
    )
    
    # ==========================================
    # CHECK: Tools disponibles
    # ==========================================
    if not TOOLS_AVAILABLE or not XARM_TOOLS:
        logger.error("robot_operator", "Tools del robot no disponibles")
        output = _make_output(
            content="Las herramientas del robot no están disponibles. "
                    "Verifica que `xarm-python-sdk` esté instalado.",
            summary="Tools no disponibles", status="error", confidence=0.0
        )
        return {
            "worker_outputs": [output],
            "events": events + [event_error("robot_operator", "Tools not available")],
        }
    
    # ==========================================
    # AUTO-CONEXIÓN
    # ==========================================
    connect_status = _auto_connect_robot()
    logger.info("robot_operator", f"Auto-connect: {connect_status}")
    
    if connect_status == "no_service":
        output = _make_output(
            content="El servicio del robot xArm no está configurado. "
                    "Verifica que `XARM_ENABLED=true` y `XARM_IP` estén en tu `.env`.",
            summary="Servicio xArm no disponible", status="error", confidence=0.0
        )
        return {
            "worker_outputs": [output],
            "events": events + [event_error("robot_operator", "xArm service not configured")],
        }
    
    if connect_status.startswith("error"):
        output = _make_output(
            content=f"No se pudo conectar al robot: {connect_status}\n\n"
                    "Verifica que el robot esté encendido y accesible en la red.",
            summary="Error de conexión", status="error", confidence=0.0
        )
        return {
            "worker_outputs": [output],
            "events": events + [event_error("robot_operator", connect_status)],
        }
    
    events.append(event_report("robot_operator", f"🤖 Robot: {connect_status}"))
    
    # ==========================================
    # CONFIGURAR LLM CON TOOLS
    # ==========================================
    try:
        llm = get_llm(state, temperature=0.1)
        
        llm_with_tools = llm.bind_tools(XARM_TOOLS)
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
    prompt = ROBOT_OPERATOR_PROMPT.format(intent_context=intent_context)
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
                
                logger.info("robot_operator", f"🔧 Tool: {tool_name}({tool_args})")
                
                tool_fn = next((t for t in XARM_TOOLS if t.name == tool_name), None)
                
                if tool_fn:
                    try:
                        result = tool_fn.invoke(tool_args)
                    except Exception as tool_err:
                        result = f"Error ejecutando {tool_name}: {tool_err}"
                    
                    actions_taken.append(f"{tool_name}({tool_args}) → {result}")
                    logger.info("robot_operator", f"   → {result}")
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
        summary=f"Robot: {intent_analysis.get('action', 'operación')}",
        actions_taken=actions_taken,
        model_used=state.get("llm_model") or os.getenv("DEFAULT_MODEL", "gemini-2.0-flash"),
        processing_time_ms=processing_time,
        confidence=0.9 if actions_taken else 0.5,
    )
    
    events.append(event_report("robot_operator", f"✅ {len(actions_taken)} acción(es)"))
    logger.info("robot_operator", f"✓ Completado | actions={len(actions_taken)} | time={processing_time:.0f}ms")
    
    return {
        "worker_outputs": [output],
        "events": events,
    }
