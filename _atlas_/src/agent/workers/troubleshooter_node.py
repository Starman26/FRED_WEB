"""
troubleshooter_node.py - Worker especializado en diagnóstico y troubleshooting

CAPACIDADES:
1. Diagnóstico técnico general (PLCs, Cobots, etc.)
2. Integración con laboratorio ATLAS (consulta estado real de equipos)
3. Preguntas estructuradas con opciones del lab real
4. Ejecución de acciones (cambiar modo cobot, etc.)
"""
import os
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from src.agent.state import AgentState
from src.agent.contracts.worker_contract import WorkerOutputBuilder, create_error_output
from src.agent.contracts.question_schema import (
    QuestionSet,
    ClarificationQuestion,
    QuestionOption,
    TROUBLESHOOTING_QUESTIONS,
    create_choice_question,
    create_text_question,
)
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_execute, event_report, event_error


# ============================================
# GENERACIÓN DINÁMICA DE PREGUNTAS CON LLM
# ============================================

QUESTION_GENERATION_PROMPT = '''Eres un asistente técnico experto en sistemas industriales.

El usuario ha hecho una solicitud pero necesitas más información para ayudarlo correctamente.

**Solicitud del usuario:**
{user_message}

**Contexto actual del laboratorio:**
{lab_context}

**Tu tarea:**
Genera 1-3 preguntas de clarificación que te ayuden a entender mejor lo que el usuario necesita.

**Reglas:**
1. Las preguntas deben ser ESPECÍFICAS al contexto del usuario
2. Si el usuario pregunta sobre algo vago, pide detalles concretos
3. Si hay equipos específicos involucrados, pregunta sobre ellos
4. Máximo 3 preguntas, mínimo 1
5. Cada pregunta puede ser de tipo: "choice" (opciones), "text" (texto libre), o "boolean" (sí/no)

**Responde SOLO con JSON en este formato exacto:**
{{
  "context": "Frase breve explicando por qué necesitas esta información",
  "questions": [
    {{
      "id": "q1",
      "question": "¿Cuál es el problema específico?",
      "type": "choice",
      "options": [
        {{"id": "1", "label": "Opción 1", "description": "Descripción opcional"}},
        {{"id": "2", "label": "Opción 2"}}
      ]
    }},
    {{
      "id": "q2", 
      "question": "¿Cuál es el mensaje de error?",
      "type": "text"
    }}
  ]
}}

**Tipos de pregunta:**
- "choice": Incluye array "options" con opciones
- "text": Pregunta abierta (no incluir "options")
- "boolean": Pregunta Sí/No (no incluir "options", se generan automáticamente)

Genera las preguntas más relevantes para esta situación específica:'''


def generate_dynamic_questions(
    user_message: str,
    lab_context: str = "",
    llm = None
) -> Optional[QuestionSet]:
    """
    Usa el LLM para generar preguntas de clarificación dinámicas
    basadas en el contexto específico del usuario.
    
    Args:
        user_message: Mensaje original del usuario
        lab_context: Contexto actual del laboratorio (opcional)
        llm: Instancia del LLM a usar
        
    Returns:
        QuestionSet con preguntas generadas o None si falla
    """
    if not llm:
        # Crear LLM si no se proporciona
        model_name = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-20250514")
        try:
            if "claude" in model_name.lower():
                llm = ChatAnthropic(model=model_name, temperature=0.3)
            else:
                llm = ChatOpenAI(model=model_name, temperature=0.3)
        except Exception as e:
            logger.error("troubleshooter_node", f"Error creando LLM para preguntas: {e}")
            return None
    
    try:
        prompt = QUESTION_GENERATION_PROMPT.format(
            user_message=user_message,
            lab_context=lab_context or "No hay contexto adicional del laboratorio."
        )
        
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        
        # Extraer JSON del response
        # Buscar el primer { y el último }
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            logger.error("troubleshooter_node", f"No se encontró JSON en respuesta: {content[:200]}")
            return None
        
        json_str = content[start_idx:end_idx]
        data = json.loads(json_str)
        
        # Convertir a QuestionSet
        questions = []
        for q_data in data.get("questions", [])[:3]:
            q_type = q_data.get("type", "text")
            
            if q_type == "choice":
                options = []
                for opt in q_data.get("options", []):
                    options.append(QuestionOption(
                        id=str(opt.get("id", "")),
                        label=opt.get("label", ""),
                        description=opt.get("description")
                    ))
                
                if not options:
                    # Si no hay opciones, convertir a texto
                    q_type = "text"
                    questions.append(ClarificationQuestion(
                        id=q_data.get("id", f"q{len(questions)+1}"),
                        question=q_data.get("question", ""),
                        type="text"
                    ))
                else:
                    questions.append(ClarificationQuestion(
                        id=q_data.get("id", f"q{len(questions)+1}"),
                        question=q_data.get("question", ""),
                        type="choice",
                        options=options
                    ))
            
            elif q_type == "boolean":
                questions.append(ClarificationQuestion(
                    id=q_data.get("id", f"q{len(questions)+1}"),
                    question=q_data.get("question", ""),
                    type="boolean",
                    options=[
                        QuestionOption(id="yes", label="Sí"),
                        QuestionOption(id="no", label="No"),
                    ]
                ))
            
            else:  # text
                questions.append(ClarificationQuestion(
                    id=q_data.get("id", f"q{len(questions)+1}"),
                    question=q_data.get("question", ""),
                    type="text",
                    placeholder=q_data.get("placeholder", "Escribe tu respuesta...")
                ))
        
        if not questions:
            logger.warning("troubleshooter_node", "LLM no generó preguntas válidas")
            return None
        
        return QuestionSet(
            questions=questions,
            context=data.get("context", "Necesito más información:"),
            worker="troubleshooting"
        )
        
    except json.JSONDecodeError as e:
        logger.error("troubleshooter_node", f"Error parseando JSON de preguntas: {e}")
        return None
    except Exception as e:
        logger.error("troubleshooter_node", f"Error generando preguntas dinámicas: {e}")
        return None

# Import lab tools
try:
    from src.agent.tools.lab_tools import (
        # Consultas
        get_lab_overview,
        get_station_details,
        get_all_plcs,
        get_all_cobots,
        get_active_errors,
        check_door_sensors,
        # Acciones
        set_cobot_mode,
        close_all_doors,
        reconnect_plc,
        resolve_station_errors,
        resolve_all_errors,
        reset_lab_to_safe_state,
        diagnose_and_suggest_fixes,
        # Formatters
        format_lab_overview_for_display,
        format_station_details_for_display,
        format_errors_for_display,
    )
    LAB_TOOLS_AVAILABLE = True
except ImportError:
    LAB_TOOLS_AVAILABLE = False


# ============================================
# PROMPTS
# ============================================

TROUBLESHOOTER_PROMPT = """Eres un **Experto en Diagnóstico Técnico** del Laboratorio ATLAS.

## CONTEXTO DEL LABORATORIO
{lab_context}

## EQUIPOS DEL LABORATORIO
El laboratorio ATLAS tiene 6 estaciones de trabajo. Cada estación incluye:
- 1 PLC (Siemens S7-1200/S7-1500)
- 1 Cobot (Universal Robots / FANUC)
- Sensores de seguridad (puerta, E-stop)

**Protocolo de seguridad:** El cobot NO puede ejecutar rutinas si:
- La puerta de seguridad está ABIERTA
- La PLC no está conectada o en STOP
- Hay errores activos sin resolver

## INFORMACIÓN DEL USUARIO
{clarification_section}

## EVIDENCIA DE DOCUMENTACIÓN
{evidence_section}

## METODOLOGÍA DE DIAGNÓSTICO
1. **Identificar equipo afectado**: ¿Qué estación? ¿PLC, Cobot, Sensor?
2. **Verificar estado actual**: Consultar estado real del equipo
3. **Identificar síntoma**: ¿Qué se esperaba vs qué ocurrió?
4. **Formular hipótesis**: Causas posibles por probabilidad
5. **Plan de acción**: Pasos concretos para resolver

## FORMATO DE RESPUESTA
🔍 **Diagnóstico del Problema**

**Equipo afectado**: [Estación X - Tipo - Nombre]
**Estado actual**: [Datos reales del sistema]

**Síntoma identificado**: [Descripción]

**Causas probables**:
1. **[Causa]** (Probabilidad: Alta/Media/Baja)

**Plan de solución**:
1. [Paso específico]

⚠️ **Precauciones**: [Si aplica]
🔙 **Rollback**: [Si aplica]

Usuario: {user_name}
"""

TROUBLESHOOTER_PROMPT_SIMPLE = """Eres un **Experto en Diagnóstico Técnico** especializado en:
- PLCs (Siemens, Allen-Bradley, etc.)
- Cobots y robótica industrial
- Sistemas de control

## INFORMACIÓN DEL USUARIO
{clarification_section}

## EVIDENCIA
{evidence_section}

## FORMATO DE RESPUESTA
🔍 **Diagnóstico del Problema**
**Síntoma**: [Descripción]
**Causas Probables**:
1. **[Causa]** (Probabilidad: Alta/Media/Baja)
**Plan de Solución**:
1. [Paso específico]
⚠️ **Precauciones**: [...]

Usuario: {user_name}
"""


# ============================================
# DETECCIÓN DE CONTEXTO
# ============================================

def is_lab_related(message: str) -> bool:
    """Detecta si el mensaje está relacionado con el laboratorio ATLAS"""
    msg = message.lower()
    
    lab_keywords = [
        # Equipos del lab
        "estación", "estacion", "station",
        "laboratorio", "lab", "atlas",
        # Referencias específicas
        "plc-st", "cobot-st", "door-sensor",
        "estación 1", "estación 2", "estación 3", "estación 4", "estación 5", "estación 6",
        "est1", "est2", "est3", "est4", "est5", "est6",
        # Protocolos del lab
        "puerta", "door", "interlock",
        "rutina", "routine",
        # Acciones del lab
        "iniciar cobot", "parar cobot", "estado del lab",
        "checar", "verificar estado",
    ]
    
    return any(kw in msg for kw in lab_keywords)


def detect_station_number(message: str) -> Optional[int]:
    """Extrae número de estación del mensaje si se menciona"""
    msg = message.lower()
    
    patterns = [
        ("estación 1", 1), ("estacion 1", 1), ("est1", 1), ("station 1", 1),
        ("estación 2", 2), ("estacion 2", 2), ("est2", 2), ("station 2", 2),
        ("estación 3", 3), ("estacion 3", 3), ("est3", 3), ("station 3", 3),
        ("estación 4", 4), ("estacion 4", 4), ("est4", 4), ("station 4", 4),
        ("estación 5", 5), ("estacion 5", 5), ("est5", 5), ("station 5", 5),
        ("estación 6", 6), ("estacion 6", 6), ("est6", 6), ("station 6", 6),
    ]
    
    for pattern, num in patterns:
        if pattern in msg:
            return num
    
    return None


def detect_equipment_type(message: str) -> Optional[str]:
    """Detecta qué tipo de equipo se menciona"""
    msg = message.lower()
    
    if any(kw in msg for kw in ["plc", "s7", "siemens", "allen"]):
        return "plc"
    if any(kw in msg for kw in ["cobot", "robot", "ur5", "ur10", "fanuc", "brazo"]):
        return "cobot"
    if any(kw in msg for kw in ["sensor", "puerta", "door", "proximidad", "e-stop"]):
        return "sensor"
    
    return None


def detect_action_request(message: str) -> Optional[Dict]:
    """Detecta si el usuario quiere ejecutar una acción"""
    msg = message.lower()
    
    # Iniciar cobot / rutina
    start_phrases = [
        "iniciar cobot", "arrancar cobot", "ejecutar rutina", "start cobot", "correr rutina",
        "comienza rutina", "comenzar rutina", "inicia rutina", "iniciar rutina",
        "arranca rutina", "corre rutina", "run routine", "start routine",
        "enciende cobot", "activa cobot", "activa rutina"
    ]
    if any(phrase in msg for phrase in start_phrases):
        station = detect_station_number(message)
        # Detectar rutina
        mode = 1  # Default: rutina 1
        if "rutina 2" in msg or "routine 2" in msg or "modo 2" in msg:
            mode = 2
        elif "rutina 3" in msg or "routine 3" in msg or "modo 3" in msg:
            mode = 3
        elif "rutina 4" in msg or "routine 4" in msg or "modo 4" in msg:
            mode = 4
        return {"action": "start_cobot", "station": station, "mode": mode}
    
    # Parar cobot
    stop_phrases = [
        "parar cobot", "detener cobot", "stop cobot", "parar rutina",
        "para cobot", "para rutina", "detén cobot", "deten cobot",
        "apagar cobot", "apaga cobot", "stop routine",
        "apaga la rutina", "apagar rutina", "apaga rutina",
        "detener rutina", "deten la rutina", "detén la rutina",
        "para la rutina", "stop la rutina", "off cobot"
    ]
    if any(phrase in msg for phrase in stop_phrases):
        station = detect_station_number(message)
        return {"action": "stop_cobot", "station": station, "mode": 0}
    
    # === ACCIONES DE REPARACIÓN ===
    
    # Reset completo del lab
    reset_phrases = [
        "reset lab", "resetear lab", "reiniciar lab", "reinicia el lab",
        "reset completo", "reinicio completo", "restaurar lab",
        "pon todo en orden", "arregla todo", "fix everything"
    ]
    if any(phrase in msg for phrase in reset_phrases):
        return {"action": "reset_lab", "needs_confirmation": True}
    
    # Cerrar puertas
    door_phrases = [
        "cierra las puertas", "cerrar puertas", "close doors",
        "cierra todas las puertas", "asegura las puertas"
    ]
    if any(phrase in msg for phrase in door_phrases):
        return {"action": "close_doors"}
    
    # Reconectar PLC
    reconnect_phrases = [
        "reconectar plc", "reconecta la plc", "reconnect plc",
        "reiniciar plc", "reinicia la plc"
    ]
    if any(phrase in msg for phrase in reconnect_phrases):
        station = detect_station_number(message)
        return {"action": "reconnect_plc", "station": station}
    
    # Resolver errores
    resolve_phrases = [
        "resolver errores", "resuelve los errores", "limpia los errores",
        "clear errors", "fix errors", "arregla los errores"
    ]
    if any(phrase in msg for phrase in resolve_phrases):
        station = detect_station_number(message)
        return {"action": "resolve_errors", "station": station}
    
    # Intentar arreglar (genérico - requiere confirmación)
    fix_phrases = [
        "intenta arreglarlo", "arreglalo", "arréglalo", "fix it",
        "intenta solucionarlo", "soluciona", "repara", "repáralo",
        "puedes arreglarlo", "arregla eso", "soluciona eso",
        "hazlo", "procede", "adelante", "sí, arréglalo", "si, arreglalo",
        "dale", "ok arreglalo", "ok, arreglalo"
    ]
    if any(phrase in msg for phrase in fix_phrases):
        station = detect_station_number(message)
        return {"action": "auto_fix", "station": station, "needs_confirmation": False}  # Ya confirmó
    
    # Ver estado general del lab
    status_phrases = [
        "estado del lab", "resumen del lab", "ver laboratorio", "lab status",
        "estado laboratorio", "status lab", "como está el lab", "como esta el lab",
        "estado de las estaciones", "ver estaciones", "mostrar estaciones"
    ]
    if any(phrase in msg for phrase in status_phrases):
        return {"action": "show_lab_status"}
    
    return None


def detect_query_request(message: str) -> Optional[Dict]:
    """Detecta si el usuario está haciendo una consulta sobre el estado del lab"""
    msg = message.lower()
    
    # Consulta sobre errores
    error_queries = [
        "errores activos", "hay errores", "que errores", "cuantos errores",
        "estaciones con errores", "problemas activos", "fallas activas",
        "hay algun error", "hay algún error", "mas errores", "más errores",
        "otros errores", "lista de errores"
    ]
    if any(phrase in msg for phrase in error_queries):
        return {"query": "active_errors"}
    
    # Consulta sobre PLCs
    plc_queries = [
        "estado de las plc", "plcs conectadas", "plc desconectada",
        "que plc", "cuales plc", "lista de plc", "plcs del lab"
    ]
    if any(phrase in msg for phrase in plc_queries):
        return {"query": "plc_status"}
    
    # Consulta sobre Cobots
    cobot_queries = [
        "estado de los cobot", "cobots activos", "que cobot",
        "cobots ejecutando", "cobots en rutina", "lista de cobot"
    ]
    if any(phrase in msg for phrase in cobot_queries):
        return {"query": "cobot_status"}
    
    # Consulta sobre puertas
    door_queries = [
        "puertas abiertas", "puertas cerradas", "estado de las puertas",
        "sensores de puerta", "alguna puerta abierta", "doors"
    ]
    if any(phrase in msg for phrase in door_queries):
        return {"query": "door_status"}
    
    # Consulta sobre estación específica
    station = detect_station_number(message)
    if station and any(word in msg for word in ["estado", "status", "como está", "como esta", "info", "detalles"]):
        return {"query": "station_details", "station": station}
    
    # Consulta general si menciona el lab
    if is_lab_related(message) and any(word in msg for word in ["hay", "cuantos", "cuántos", "cuales", "cuáles", "lista", "ver", "mostrar"]):
        return {"query": "lab_overview"}
    
    return None


# ============================================
# CONSULTAS AL LABORATORIO
# ============================================

def get_lab_context() -> str:
    """Obtiene contexto actual del laboratorio para el prompt"""
    if not LAB_TOOLS_AVAILABLE:
        return "⚠️ Tools de laboratorio no disponibles."
    
    try:
        # Obtener resumen del lab
        overview = get_lab_overview()
        if not overview.get("success"):
            return "⚠️ No se pudo obtener estado del laboratorio."
        
        lines = ["### Estado Actual del Laboratorio\n"]
        lines.append(f"- **Estaciones activas:** {overview['stations_online']}/{overview['total_stations']}")
        
        if overview['stations_with_errors'] > 0:
            lines.append(f"- ⚠️ **Estaciones con problemas:** {overview['stations_with_errors']}")
        
        if overview['active_errors_count'] > 0:
            lines.append(f"- 🔴 **Errores activos:** {overview['active_errors_count']}")
            
            # Obtener detalle de errores
            errors = get_active_errors()
            if errors.get("success") and errors.get("errors"):
                lines.append("\n**Errores activos:**")
                for err in errors["errors"][:3]:  # Max 3 errores
                    lines.append(f"  - [{err['severity']}] Estación {err['station_number']}: {err['message']}")
        
        # Resumen de estaciones
        lines.append("\n**Estado por estación:**")
        for st in overview['stations']:
            status_icon = "✅" if st['is_operational'] else "⚠️"
            lines.append(f"  - Est. {st['station_number']}: {status_icon} PLC:{st['plc_status']} Cobot:{st['cobot_status']} Puerta:{'🔒' if st['doors_closed'] else '🚪'}")
        
        return "\n".join(lines)
        
    except Exception as e:
        logger.error("troubleshooter_node", f"Error obteniendo contexto del lab: {e}")
        return f"⚠️ Error consultando laboratorio: {str(e)}"


def create_lab_questions(
    message: str,
    equipment_type: Optional[str] = None,
    station_number: Optional[int] = None
) -> Optional[QuestionSet]:
    """
    Crea preguntas dinámicas basadas en los equipos reales del laboratorio.
    NO genera preguntas si el mensaje parece ser un COMANDO (no un reporte de problema).
    """
    if not LAB_TOOLS_AVAILABLE:
        return None
    
    msg_lower = message.lower()
    
    # NO generar preguntas si es un comando claro (iniciar, parar, etc.)
    command_keywords = [
        "iniciar", "arrancar", "ejecutar", "comienza", "comenzar", "inicia", 
        "arranca", "corre", "enciende", "activa", "start", "run",
        "parar", "detener", "stop", "para", "apagar", "apaga", "deten",
        "cerrar", "cierra", "abrir", "abre", "reset", "reinicia",
        "reconectar", "reconecta", "resolver", "arreglar", "arregla"
    ]
    if any(cmd in msg_lower for cmd in command_keywords):
        return None  # Es un comando, no generar preguntas de diagnóstico
    
    questions = []
    
    # Si menciona PLC pero no especifica cuál
    if equipment_type == "plc" and station_number is None:
        plcs_data = get_all_plcs()
        if plcs_data.get("success") and plcs_data.get("plcs"):
            plc_options = []
            for plc in plcs_data["plcs"]:
                status = "🔴" if plc["has_error"] else ("🟢" if plc["is_connected"] else "⚪")
                plc_options.append((
                    str(plc["station_number"]),
                    f"Est. {plc['station_number']}: {plc['name']}",
                    f"{status} {plc['model']} - {plc['ip_address']}"
                ))
            
            questions.append(create_choice_question(
                "plc_selection",
                "¿Con cuál PLC del laboratorio tienes el problema?",
                plc_options,
                include_other=True
            ))
    
    # Si menciona cobot pero no especifica
    elif equipment_type == "cobot" and station_number is None:
        cobots_data = get_all_cobots()
        if cobots_data.get("success") and cobots_data.get("cobots"):
            cobot_options = []
            for cobot in cobots_data["cobots"]:
                status = "🟢" if cobot["is_connected"] else "🔴"
                mode_desc = f"Modo {cobot['mode']}" if cobot["mode"] > 0 else "Idle"
                cobot_options.append((
                    str(cobot["station_number"]),
                    f"Est. {cobot['station_number']}: {cobot['name']}",
                    f"{status} {cobot['model']} - {mode_desc}"
                ))
            
            questions.append(create_choice_question(
                "cobot_selection",
                "¿Con cuál cobot tienes el problema?",
                cobot_options,
                include_other=True
            ))
    
    # Si no especifica estación para problema general
    elif station_number is None and is_lab_related(message):
        questions.append(create_choice_question(
            "station_selection",
            "¿En cuál estación del laboratorio ocurre el problema?",
            [
                ("1", "Estación 1 - Ensamblaje Inicial"),
                ("2", "Estación 2 - Soldadura"),
                ("3", "Estación 3 - Inspección Visual"),
                ("4", "Estación 4 - Ensamblaje Final"),
                ("5", "Estación 5 - Testing"),
                ("6", "Estación 6 - Empaque"),
            ],
            include_other=False
        ))
    
    # Agregar pregunta sobre tipo de error si no se detectó
    if len(questions) < 3 and not any(kw in message.lower() for kw in ["no conecta", "error", "stop", "falla"]):
        questions.append(create_choice_question(
            "error_type",
            "¿Qué tipo de problema observas?",
            [
                ("1", "No conecta / Timeout", "El equipo no responde"),
                ("2", "En modo STOP / Error", "El equipo muestra error"),
                ("3", "Puerta abierta / Interlock", "Problema de seguridad"),
                ("4", "No ejecuta rutina", "Cobot no inicia"),
                ("5", "Comportamiento extraño", "Funciona pero mal"),
            ],
            include_other=True
        ))
    
    if questions:
        return QuestionSet(
            questions=questions[:3],
            context="Para diagnosticar el problema en el laboratorio, necesito saber:",
            worker="troubleshooting"
        )
    
    return None


# ============================================
# EJECUCIÓN DE ACCIONES
# ============================================

def execute_lab_action(action: Dict, user_name: str = "agent") -> Dict[str, Any]:
    """Ejecuta una acción en el laboratorio"""
    if not LAB_TOOLS_AVAILABLE:
        return {"success": False, "error": "Lab tools no disponibles"}
    
    action_type = action.get("action")
    
    if action_type == "show_lab_status":
        overview = get_lab_overview()
        if overview.get("success"):
            return {
                "success": True,
                "type": "status",
                "content": format_lab_overview_for_display(overview),
                "data": overview
            }
        return overview
    
    if action_type in ["start_cobot", "stop_cobot"]:
        station = action.get("station")
        mode = action.get("mode", 0)
        
        if station is None:
            return {
                "success": False,
                "error": "No se especificó la estación",
                "needs_clarification": True
            }
        
        result = set_cobot_mode(station, mode, f"agent:{user_name}")
        
        # Construir mensaje detallado
        if result.get("success"):
            content = result.get("message", f"✅ Cobot configurado en modo {mode}")
        else:
            # Incluir razones del fallo
            error_msg = result.get("error", "Error desconocido")
            reasons = result.get("reasons", [])
            
            content = f"❌ **No se pudo iniciar el cobot**\n\n**Razón:** {error_msg}"
            if reasons:
                content += "\n\n**Problemas detectados:**\n"
                for reason in reasons:
                    content += f"- ⚠️ {reason}\n"
            
            # Mostrar estado actual
            station_status = result.get("station_status", {})
            if station_status:
                content += "\n**Estado de la estación:**\n"
                content += f"- Puertas cerradas: {'✅' if station_status.get('doors_closed') else '❌ ABIERTA'}\n"
                content += f"- PLC conectada: {'✅' if station_status.get('plc_connected') else '❌'}\n"
                content += f"- Sin errores activos: {'✅' if station_status.get('no_active_errors') else '❌'}\n"
        
        return {
            "success": result.get("success", False),
            "type": "action",
            "content": content,
            "data": result
        }
    
    # === ACCIONES DE REPARACIÓN ===
    
    if action_type == "close_doors":
        result = close_all_doors()
        return {
            "success": result.get("success", False),
            "type": "repair",
            "content": result.get("message", "Operación completada"),
            "data": result
        }
    
    if action_type == "reconnect_plc":
        station = action.get("station")
        if station is None:
            return {"success": False, "error": "No se especificó la estación", "needs_clarification": True}
        
        result = reconnect_plc(station)
        return {
            "success": result.get("success", False),
            "type": "repair",
            "content": result.get("message", "Operación completada"),
            "data": result
        }
    
    if action_type == "resolve_errors":
        station = action.get("station")
        if station:
            result = resolve_station_errors(station, f"agent:{user_name}")
        else:
            result = resolve_all_errors(f"agent:{user_name}")
        
        return {
            "success": result.get("success", False),
            "type": "repair",
            "content": result.get("message", "Operación completada"),
            "data": result
        }
    
    if action_type == "reset_lab":
        result = reset_lab_to_safe_state()
        return {
            "success": result.get("success", False),
            "type": "repair",
            "content": result.get("message", "Reset completado"),
            "data": result
        }
    
    if action_type == "auto_fix":
        # Diagnosticar y arreglar automáticamente
        station = action.get("station")
        diagnosis = diagnose_and_suggest_fixes(station)
        
        if not diagnosis.get("success"):
            return diagnosis
        
        if diagnosis.get("problems_count", 0) == 0:
            return {
                "success": True,
                "type": "info",
                "content": "✅ No encontré problemas que necesiten reparación. Todo parece estar en orden.",
                "data": diagnosis
            }
        
        # Ejecutar las acciones sugeridas
        results = []
        for suggested in diagnosis.get("suggested_actions", []):
            action_name = suggested.get("action")
            
            if action_name == "close_all_doors":
                r = close_all_doors()
                results.append(f"{'✅' if r.get('success') else '❌'} Cerrar puertas: {r.get('message', r.get('error'))}")
            
            elif action_name == "reconnect_plc":
                st = suggested.get("station")
                r = reconnect_plc(st)
                results.append(f"{'✅' if r.get('success') else '❌'} Reconectar PLC est.{st}: {r.get('message', r.get('error'))}")
            
            elif action_name == "resolve_errors" or action_name == "resolve_all_errors":
                st = suggested.get("station")
                if st:
                    r = resolve_station_errors(st, f"agent:{user_name}")
                else:
                    r = resolve_all_errors(f"agent:{user_name}")
                results.append(f"{'✅' if r.get('success') else '❌'} Resolver errores: {r.get('message', r.get('error'))}")
        
        content = f"""🔧 **Reparación Automática Completada**

**Problemas encontrados:** {diagnosis.get('problems_count')}
{chr(10).join(['- ' + p for p in diagnosis.get('problems', [])])}

**Acciones ejecutadas:**
{chr(10).join(results)}

¿Quieres que verifique el estado actual del laboratorio?"""
        
        return {
            "success": True,
            "type": "repair",
            "content": content,
            "data": {"diagnosis": diagnosis, "results": results}
        }
    
    return {"success": False, "error": f"Acción desconocida: {action_type}"}


def execute_lab_query(query: Dict) -> Dict[str, Any]:
    """Ejecuta una consulta de estado del laboratorio - respuestas conversacionales"""
    if not LAB_TOOLS_AVAILABLE:
        return {"success": False, "error": "Lab tools no disponibles"}
    
    query_type = query.get("query")
    
    if query_type == "active_errors":
        # Obtener errores registrados
        errors = get_active_errors()
        
        # Obtener PLCs desconectadas
        plcs = get_all_plcs()
        disconnected_plcs = []
        if plcs.get("success"):
            disconnected_plcs = [p for p in plcs.get("plcs", []) if not p.get("is_connected")]
        
        # Obtener puertas abiertas
        doors = check_door_sensors()
        open_doors = doors.get("open_doors_count", 0) if doors.get("success") else 0
        
        total_problems = 0
        lines = []
        
        # 1. Errores registrados
        if errors.get("success") and errors.get("total_errors", 0) > 0:
            lines.append("### 📋 Errores Registrados\n")
            for err in errors.get("errors", []):
                severity_icon = {"critical": "🔴", "error": "🟠", "warning": "🟡", "info": "🔵"}.get(err.get("severity"), "⚪")
                lines.append(f"- {severity_icon} **Est. {err.get('station_number')}** - {err.get('equipment_name', 'N/A')}: `{err.get('error_code')}` - {err.get('message')}")
                total_problems += 1
            lines.append("")
        
        # 2. PLCs desconectadas
        if disconnected_plcs:
            lines.append("### 🔌 Equipos Desconectados\n")
            for plc in disconnected_plcs:
                lines.append(f"- 🔴 **Est. {plc.get('station_number')}** - {plc.get('name')}: PLC desconectada ({plc.get('ip_address')})")
                total_problems += 1
            lines.append("")
        
        # 3. Cobots con error
        cobots = get_all_cobots()
        if cobots.get("success"):
            error_cobots = [c for c in cobots.get("cobots", []) if c.get("status") == "error" or not c.get("is_connected")]
            if error_cobots:
                lines.append("### 🤖 Cobots con Problemas\n")
                for cobot in error_cobots:
                    status = "desconectado" if not cobot.get("is_connected") else cobot.get("status")
                    lines.append(f"- 🔴 **Est. {cobot.get('station_number')}** - {cobot.get('name')}: {status}")
                    total_problems += 1
                lines.append("")
        
        # 4. Puertas abiertas
        if open_doors > 0:
            lines.append(f"### 🚪 Puertas Abiertas: {open_doors}\n")
            for door in doors.get("doors", []):
                if not door.get("is_closed"):
                    lines.append(f"- ⚠️ **Est. {door.get('station_number')}** - Puerta abierta")
                    total_problems += 1
            lines.append("")
        
        # Construir respuesta conversacional
        if total_problems == 0:
            content = "¡Buenas noticias! 🎉 **No hay problemas activos en el laboratorio.**\n\nTodas las estaciones están operando normalmente. ¿Necesitas algo más?"
            offer_help = False
        else:
            # Introducción conversacional
            if total_problems == 1:
                intro = "Encontré **1 problema** en el laboratorio:\n\n"
            else:
                intro = f"Encontré **{total_problems} problemas** en el laboratorio:\n\n"
            
            content = intro + "\n".join(lines)
            # Nota: La pregunta de confirmación se maneja via HITL, no en el content
            offer_help = True
        
        return {
            "success": True, 
            "content": content, 
            "data": {"errors": errors, "disconnected_plcs": disconnected_plcs, "total_problems": total_problems},
            "offer_help": offer_help
        }
    
    if query_type == "plc_status":
        plcs = get_all_plcs()
        if plcs.get("success"):
            disconnected = plcs['total'] - plcs['connected']
            
            # Intro conversacional
            if disconnected == 0:
                intro = f"Todas las **{plcs['total']} PLCs** están conectadas y funcionando correctamente. 👍\n\n"
            else:
                intro = f"Tenemos **{disconnected} PLC(s) desconectada(s)** de {plcs['total']} totales. ⚠️\n\n"
            
            lines = [intro]
            lines.append("| Estación | PLC | Estado | Modo |")
            lines.append("|----------|-----|--------|------|")
            
            for plc in plcs.get("plcs", []):
                status_icon = "🟢" if plc.get("is_connected") else "🔴"
                mode = plc.get("run_mode", "?")
                lines.append(f"| {plc['station_number']} | {plc['name']} | {status_icon} {plc['status']} | {mode} |")
            
            return {"success": True, "content": "\n".join(lines), "data": plcs, "offer_help": disconnected > 0, "total_problems": disconnected}
        return plcs
    
    if query_type == "cobot_status":
        cobots = get_all_cobots()
        if cobots.get("success"):
            running = cobots.get("running", 0)
            total = cobots.get("total", 0)
            
            if running == 0:
                intro = f"Los **{total} cobots** están en espera (idle). Ninguno ejecutando rutinas.\n\n"
            else:
                intro = f"**{running} de {total} cobots** están ejecutando rutinas.\n\n"
            
            lines = [intro]
            for cobot in cobots.get("cobots", []):
                status_icon = "🟢" if cobot.get("is_connected") else "🔴"
                mode = cobot.get("mode", 0)
                routine = cobot.get("routine", "idle")
                mode_icon = "▶️" if mode > 0 else "⏹️"
                lines.append(f"- **Est. {cobot['station_number']}** {cobot['name']}: {status_icon} {mode_icon} {routine}")
            
            return {"success": True, "content": "\n".join(lines), "data": cobots}
        return cobots
    
    if query_type == "door_status":
        doors = check_door_sensors()
        if doors.get("success"):
            open_count = doors.get("open_doors_count", 0)
            
            if open_count == 0:
                intro = "✅ **Todas las puertas están cerradas** - El laboratorio está seguro para operar.\n\n"
            else:
                intro = f"⚠️ **{open_count} puerta(s) abierta(s)** - Los cobots no pueden operar hasta cerrarlas.\n\n"
            
            lines = [intro]
            for door in doors.get("doors", []):
                icon = "🔒" if door.get("is_closed") else "🚪 ⚠️"
                lines.append(f"- **Est. {door['station_number']}**: {icon} {door['status']}")
            
            return {"success": True, "content": "\n".join(lines), "data": doors, "offer_help": open_count > 0, "total_problems": open_count}
        return doors
    
    if query_type == "station_details":
        station_num = query.get("station")
        if station_num:
            details = get_station_details(station_num)
            if details.get("success"):
                content = format_station_details_for_display(details)
                has_problems = not details.get("ready_to_operate")
                return {"success": True, "content": content, "data": details, "offer_help": has_problems, "total_problems": 1 if has_problems else 0}
            return details
        return {"success": False, "error": "No se especificó estación"}
    
    if query_type == "lab_overview":
        overview = get_lab_overview()
        if overview.get("success"):
            # Intro conversacional
            problems = overview.get("stations_with_errors", 0)
            if problems == 0:
                intro = "🏭 **El laboratorio está funcionando perfectamente.** Todas las estaciones operativas.\n\n"
            else:
                intro = f"🏭 **El laboratorio tiene {problems} estación(es) con problemas.**\n\n"
            
            content = intro + format_lab_overview_for_display(overview)
            
            return {"success": True, "content": content, "data": overview, "offer_help": problems > 0, "total_problems": problems}
        return overview
    
    return {"success": False, "error": f"Consulta no reconocida: {query_type}"}


# ============================================
# HELPERS
# ============================================

def get_last_user_message(state: AgentState) -> str:
    """Obtiene el último mensaje del usuario"""
    for m in reversed(state.get("messages", []) or []):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
        if isinstance(m, dict) and m.get("role") in ("human", "user"):
            return (m.get("content") or "").strip()
    return ""


def get_evidence_from_context(state: AgentState) -> str:
    """Extrae evidencia de contexto previo"""
    evidence_data = state.get("pending_context", {}).get("evidence", [])
    if not evidence_data:
        for output in state.get("worker_outputs", []):
            if output.get("worker") == "research":
                evidence_data = output.get("evidence", [])
                break
    if not evidence_data:
        return "No hay documentación de referencia."
    return "\n".join([
        f"- **{ev.get('title', 'Doc')}** (Pág. {ev.get('page', '?')}): {ev.get('chunk', '')[:200]}..."
        for ev in evidence_data
    ])


def extract_severity(content: str) -> str:
    """Extrae severidad del problema"""
    content_lower = content.lower()
    if any(kw in content_lower for kw in ["crítico", "producción parada", "urgente", "emergency"]):
        return "critical"
    elif any(kw in content_lower for kw in ["error", "no funciona", "bloqueado", "stop"]):
        return "high"
    elif any(kw in content_lower for kw in ["lento", "intermitente", "warning"]):
        return "medium"
    return "low"


# ============================================
# NODO PRINCIPAL
# ============================================

def troubleshooter_node(state: AgentState) -> Dict[str, Any]:
    """
    Worker de troubleshooting con integración al laboratorio ATLAS.
    
    Flujo:
    1. Detectar si es problema del laboratorio
    2. Consultar estado real de equipos
    3. Si falta info → preguntas basadas en equipos reales
    4. Generar diagnóstico con contexto real
    5. Ejecutar acciones si se solicitan
    """
    start_time = datetime.utcnow()
    logger.node_start("troubleshooter_node", {})
    events = [event_execute("troubleshooting", "Analizando problema...")]
    
    pending_context = state.get("pending_context", {})
    user_clarification = pending_context.get("user_clarification", "")
    original_query = pending_context.get("original_query", "")
    
    # ==========================================
    # 1. DETERMINAR MENSAJE A USAR
    # ==========================================
    if user_clarification and original_query:
        user_message = original_query
        clarification_text = user_clarification
        logger.info("troubleshooter_node", "Usando mensaje original + clarificación")
    else:
        user_message = get_last_user_message(state)
        clarification_text = ""
        
        if not user_message:
            error_output = create_error_output("troubleshooting", "NO_MESSAGE", "No hay mensaje")
            return {
                "worker_outputs": [error_output.model_dump()],
                "troubleshooting_result": error_output.model_dump_json(),
                "events": events
            }
    
    # ==========================================
    # 2. USAR INTENT ANALYSIS + DETECTAR CONTEXTO
    # ==========================================
    # Primero intentar usar intent_analysis del state (más preciso)
    intent_analysis = state.get("intent_analysis", {})
    pending_ctx = state.get("pending_context", {})
    
    # Extraer de intent_analysis si está disponible
    if intent_analysis:
        entities = intent_analysis.get("entities", {})
        station_num = entities.get("station") or detect_station_number(user_message)
        equipment = entities.get("equipment") or detect_equipment_type(user_message)
        detected_action = intent_analysis.get("action")
        intent_type = intent_analysis.get("intent")
        
        action_request = None
        query_request = None
        
        # Si es QUERY, mapear a query_request (NO a action_request)
        if intent_type == "query":
            if detected_action == "check_errors":
                query_request = {"query": "active_errors"}
            elif detected_action == "check_status":
                if station_num:
                    query_request = {"query": "station_details", "station": station_num}
                else:
                    query_request = {"query": "lab_overview"}
            # NO crear action_request para queries
        
        # Si es COMMAND, mapear a action_request
        elif intent_type == "command" and detected_action:
            action_request = {
                "action": detected_action,
                "station": station_num,
                "mode": entities.get("routine", 1) if detected_action == "start_cobot" else 0
            }
        
        # Si es TROUBLESHOOT sin acción específica, no crear ninguno (irá a diagnóstico)
        
        is_lab = True if (station_num or equipment or detected_action) else is_lab_related(user_message)
        
        logger.info("troubleshooter_node", 
            f"Usando intent_analysis: intent={intent_type} action={detected_action} station={station_num}")
    else:
        # Fallback: usar detección local
        is_lab = is_lab_related(user_message) or is_lab_related(clarification_text)
        station_num = detect_station_number(user_message) or detect_station_number(clarification_text)
        equipment = detect_equipment_type(user_message) or detect_equipment_type(clarification_text)
        action_request = detect_action_request(user_message)
        query_request = detect_query_request(user_message)
    
    logger.info("troubleshooter_node", f"Mensaje: '{user_message[:50]}...' | is_lab={is_lab} | station={station_num} | action={action_request} | query={query_request}")
    
    # ==========================================
    # 2.5. VERIFICAR SI ES CONFIRMACIÓN DE REPARACIÓN
    # ==========================================
    if pending_context.get("awaiting_repair_confirmation"):
        # Verificar respuesta del usuario
        user_response = (clarification_text or user_message).lower().strip()
        
        # Respuestas afirmativas
        affirmative = any(word in user_response for word in [
            "yes", "sí", "si", "arreglalo", "arréglalo", "dale", "ok", 
            "adelante", "procede", "hazlo", "1", "repair"
        ])
        
        if affirmative:
            logger.info("troubleshooter_node", "Usuario confirmó reparación - ejecutando auto_fix")
            events.append(event_report("troubleshooting", "🔧 Ejecutando reparación automática..."))
            
            # Ejecutar reparación con los datos guardados
            result = execute_lab_action({"action": "auto_fix", "station": station_num}, state.get("user_name", "user"))
            
            if result.get("success"):
                output = WorkerOutputBuilder.troubleshooting(
                    content=result.get("content", "Reparación completada"),
                    problem_identified="Reparación automática ejecutada",
                    severity="low",
                    summary="Reparación completada exitosamente",
                    confidence=1.0,
                )
                events.append(event_report("troubleshooting", "✅ Reparación completada"))
            else:
                output = WorkerOutputBuilder.troubleshooting(
                    content=f"❌ Error en reparación: {result.get('error')}",
                    problem_identified="Error en reparación",
                    severity="medium",
                    summary="No se pudo completar la reparación",
                    confidence=0.5,
                )
            
            # Limpiar el contexto de confirmación
            return {
                "worker_outputs": [output.model_dump()],
                "troubleshooting_result": output.model_dump_json(),
                "events": events,
                "pending_context": {},  # Limpiar contexto
            }
        else:
            # Usuario dijo que no
            logger.info("troubleshooter_node", "Usuario rechazó reparación")
            output = WorkerOutputBuilder.troubleshooting(
                content="Entendido, no realizaré cambios. ¿Necesitas algo más?",
                problem_identified="Reparación cancelada por usuario",
                severity="low",
                summary="Usuario rechazó reparación",
                confidence=1.0,
            )
            
            return {
                "worker_outputs": [output.model_dump()],
                "troubleshooting_result": output.model_dump_json(),
                "events": events,
                "pending_context": {},  # Limpiar contexto
            }
    
    # ==========================================
    # 3. EJECUTAR ACCIÓN SI SE SOLICITA
    # ==========================================
    if action_request and not action_request.get("needs_clarification"):
        events.append(event_report("troubleshooting", f"🔧 Ejecutando acción: {action_request['action']}"))
        
        result = execute_lab_action(action_request, state.get("user_name", "user"))
        
        if result.get("success"):
            output = WorkerOutputBuilder.troubleshooting(
                content=result.get("content", "Acción ejecutada"),
                problem_identified="Acción ejecutada",
                severity="low",
                summary="Acción completada exitosamente",
                confidence=1.0,
            )
            events.append(event_report("troubleshooting", "✅ Acción completada"))
        else:
            # Usar content si está disponible (tiene mensaje detallado), sino usar error
            error_content = result.get("content") or result.get("error") or "Error desconocido"
            output = WorkerOutputBuilder.troubleshooting(
                content=error_content,
                problem_identified="Error en acción",
                severity="medium",
                summary="No se pudo ejecutar la acción por condiciones de seguridad",
                confidence=0.5,
            )
            events.append(event_report("troubleshooting", "⚠️ Acción rechazada por seguridad"))
        
        return {
            "worker_outputs": [output.model_dump()],
            "troubleshooting_result": output.model_dump_json(),
            "events": events,
        }
    
    # ==========================================
    # 3.5. EJECUTAR CONSULTA SI SE SOLICITA
    # ==========================================
    if query_request and LAB_TOOLS_AVAILABLE:
        events.append(event_report("troubleshooting", f"🔍 Consultando: {query_request['query']}"))
        
        result = execute_lab_query(query_request)
        
        if result.get("success"):
            # Si hay problemas y se ofrece ayuda, usar HITL para confirmación
            offer_help = result.get("offer_help", False)
            total_problems = result.get("total_problems", 0) or result.get("data", {}).get("total_problems", 0)
            
            if offer_help and total_problems > 0:
                events.append(event_report("troubleshooting", "💡 Ofreciendo reparación automática"))
                
                # Obtener el contenido con los errores encontrados (SIN la pregunta)
                errors_content = result.get("content", "Se encontraron problemas")
                
                # Crear pregunta de confirmación
                confirmation_question = create_choice_question(
                    "repair_confirmation",
                    "¿Quieres que intente reparar estos problemas automáticamente?",
                    [
                        ("yes", "Sí, arréglalo", "Ejecutar reparación automática"),
                        ("no", "No, solo quería ver el estado", "No hacer cambios"),
                    ],
                    include_other=False
                )
                
                question_set = QuestionSet(
                    questions=[confirmation_question],
                    context="",
                    worker="troubleshooting"
                )
                
                # Guardar contexto para cuando confirme
                updated_context = pending_context.copy()
                updated_context["original_query"] = user_message
                updated_context["awaiting_repair_confirmation"] = True
                updated_context["problems_data"] = result.get("data", {})
                updated_context["query_type"] = query_request.get("query")
                
                # IMPORTANTE: Solo mostrar los errores, la pregunta se muestra via HITL widget
                output = WorkerOutputBuilder.troubleshooting(
                    content=errors_content,  # Solo errores, sin pregunta
                    problem_identified="Problemas detectados - esperando confirmación",
                    severity="medium",
                    summary=f"Encontrados {total_problems} problemas",
                    confidence=0.9,
                )
                output_dict = output.model_dump()
                output_dict["status"] = "needs_context"  # El orchestrator busca esto
                output_dict["clarification_questions"] = question_set.model_dump().get("questions", [])
                
                return {
                    "worker_outputs": [output_dict],
                    "troubleshooting_result": json.dumps(output_dict),
                    "events": events,
                    "pending_context": updated_context,
                }
            
            # Sin problemas o sin oferta de ayuda - respuesta normal
            output = WorkerOutputBuilder.troubleshooting(
                content=result.get("content", "Consulta completada"),
                problem_identified="Consulta de estado",
                severity="low",
                summary=f"Consulta: {query_request['query']}",
                confidence=1.0,
            )
            events.append(event_report("troubleshooting", "✅ Consulta completada"))
        else:
            output = WorkerOutputBuilder.troubleshooting(
                content=f"❌ Error en consulta: {result.get('error')}",
                problem_identified="Error en consulta",
                severity="low",
                summary="No se pudo completar la consulta",
                confidence=0.5,
            )
        
        return {
            "worker_outputs": [output.model_dump()],
            "troubleshooting_result": output.model_dump_json(),
            "events": events,
        }
    
    # ==========================================
    # 4. VERIFICAR SI NECESITA MÁS INFO
    # ==========================================
    # Solo pedir más info si NO es un comando claro
    command_keywords = [
        "iniciar", "arrancar", "ejecutar", "comienza", "comenzar", "inicia", 
        "arranca", "corre", "enciende", "activa", "start", "run",
        "parar", "detener", "stop", "para", "apagar", "apaga", "deten",
        "cerrar", "cierra", "abrir", "abre", "reset", "reinicia",
        "reconectar", "reconecta", "resolver", "arreglar", "arregla"
    ]
    is_command = any(cmd in user_message.lower() for cmd in command_keywords)
    
    # Si es un comando pero no se detectó, dar feedback claro
    if is_command and not action_request and not query_request:
        logger.info("troubleshooter_node", f"Comando no reconocido: {user_message[:50]}")
        
        # Intentar dar una respuesta útil
        msg_lower = user_message.lower()
        suggestions = []
        
        if "rutina" in msg_lower or "cobot" in msg_lower:
            if station_num:
                suggestions.append(f"• 'Inicia rutina 1 en estación {station_num}'")
                suggestions.append(f"• 'Para el cobot de estación {station_num}'")
            else:
                suggestions.append("• 'Inicia rutina 1 en estación 1'")
                suggestions.append("• 'Para el cobot de estación 2'")
        
        if "plc" in msg_lower:
            suggestions.append("• 'Reconecta la PLC de estación 3'")
        
        if "puerta" in msg_lower:
            suggestions.append("• 'Cierra todas las puertas'")
        
        if not suggestions:
            suggestions = [
                "• 'Inicia rutina 1 en estación 1'",
                "• 'Para el cobot de estación 2'",
                "• 'Cierra todas las puertas'",
                "• 'Reconecta la PLC de estación 3'",
                "• '¿Hay errores activos?'"
            ]
        
        content = f"""No pude identificar exactamente qué acción quieres realizar.

**Algunos comandos que puedo ejecutar:**
{chr(10).join(suggestions)}

¿Podrías reformular tu solicitud?"""
        
        output = WorkerOutputBuilder.troubleshooting(
            content=content,
            problem_identified="Comando no reconocido",
            severity="low",
            summary="Solicitud de clarificación",
            confidence=0.5,
        )
        
        return {
            "worker_outputs": [output.model_dump()],
            "troubleshooting_result": output.model_dump_json(),
            "events": events,
        }
    
    if not user_clarification and not is_command:
        question_set = None
        
        # Obtener contexto del lab para las preguntas
        lab_context_for_questions = ""
        if is_lab and LAB_TOOLS_AVAILABLE:
            try:
                lab_context_for_questions = get_lab_context()
            except:
                pass
        
        # ==========================================
        # OPCIÓN 1: Generar preguntas dinámicas con LLM
        # ==========================================
        if len(user_message.split()) < 20:  # Solo para mensajes cortos/vagos
            events.append(event_report("troubleshooting", "🤔 Generando preguntas contextuales..."))
            
            # Crear LLM para generar preguntas
            model_name = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-20250514")
            try:
                if "claude" in model_name.lower():
                    question_llm = ChatAnthropic(model=model_name, temperature=0.3)
                else:
                    question_llm = ChatOpenAI(model=model_name, temperature=0.3)
                
                question_set = generate_dynamic_questions(
                    user_message=user_message,
                    lab_context=lab_context_for_questions,
                    llm=question_llm
                )
                
                if question_set:
                    logger.info("troubleshooter_node", f"Generadas {len(question_set.questions)} preguntas dinámicas")
            except Exception as e:
                logger.warning("troubleshooter_node", f"Error generando preguntas dinámicas: {e}")
                question_set = None
        
        # ==========================================
        # OPCIÓN 2: Fallback a preguntas del lab específicas
        # ==========================================
        if not question_set and is_lab and LAB_TOOLS_AVAILABLE:
            question_set = create_lab_questions(user_message, equipment, station_num)
        
        # ==========================================
        # OPCIÓN 3: Fallback último - preguntas genéricas
        # ==========================================
        if not question_set and len(user_message.split()) < 10:
            # Solo para mensajes muy cortos, usar template básico
            question_set = QuestionSet(
                questions=[
                    create_text_question(
                        "more_details",
                        "¿Podrías darme más detalles sobre tu consulta?",
                        "Describe el problema o lo que necesitas..."
                    )
                ],
                context="Tu mensaje es un poco breve. Para ayudarte mejor:",
                worker="troubleshooting"
            )
        
        if question_set:
            updated_context = pending_context.copy()
            updated_context["original_query"] = user_message
            updated_context["is_lab_related"] = is_lab
            updated_context["detected_station"] = station_num
            updated_context["detected_equipment"] = equipment
            
            output = WorkerOutputBuilder.troubleshooting(
                content=question_set.to_display_text(),
                problem_identified="Recopilando información",
                severity="pending",
                summary="Necesito más información",
                confidence=0.0,
                status="needs_context"
            )
            
            questions_data = question_set.to_dict_list()
            events.append(event_report("troubleshooting", f"📋 {len(questions_data)} preguntas generadas"))
            
            return {
                "worker_outputs": [output.model_dump()],
                "troubleshooting_result": output.model_dump_json(),
                "needs_human_input": True,
                "clarification_questions": questions_data,
                "pending_context": updated_context,
                "events": events,
            }
    
    # ==========================================
    # 5. OBTENER CONTEXTO DEL LAB (si aplica)
    # ==========================================
    lab_context = ""
    station_details = None
    
    if is_lab and LAB_TOOLS_AVAILABLE:
        events.append(event_report("troubleshooting", "🔍 Consultando estado del laboratorio..."))
        
        # Obtener contexto general
        lab_context = get_lab_context()
        
        # Si tenemos estación específica, obtener detalles
        # Intentar extraer de clarificación también
        if not station_num and clarification_text:
            for i in range(1, 7):
                if str(i) in clarification_text:
                    station_num = i
                    break
        
        if station_num:
            details = get_station_details(station_num)
            if details.get("success"):
                station_details = details
                lab_context += f"\n\n### Detalles de Estación {station_num}\n"
                lab_context += format_station_details_for_display(details)
    
    # ==========================================
    # 6. GENERAR DIAGNÓSTICO
    # ==========================================
    evidence_text = get_evidence_from_context(state)
    clarification_section = clarification_text if clarification_text else "No hay información adicional."
    
    model_name = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-20250514")
    try:
        if "claude" in model_name.lower():
            llm = ChatAnthropic(model=model_name, temperature=0.3)
        else:
            llm = ChatOpenAI(model=model_name, temperature=0.3)
    except Exception as e:
        error_output = create_error_output("troubleshooting", "LLM_INIT_ERROR", str(e))
        return {
            "worker_outputs": [error_output.model_dump()],
            "troubleshooting_result": error_output.model_dump_json(),
            "events": events
        }
    
    # Seleccionar prompt según contexto
    if is_lab and lab_context:
        prompt = TROUBLESHOOTER_PROMPT.format(
            lab_context=lab_context,
            clarification_section=clarification_section,
            evidence_section=evidence_text,
            user_name=state.get("user_name", "Usuario")
        )
    else:
        prompt = TROUBLESHOOTER_PROMPT_SIMPLE.format(
            clarification_section=clarification_section,
            evidence_section=evidence_text,
            user_name=state.get("user_name", "Usuario")
        )
    
    # Mensaje combinado
    full_message = user_message
    if clarification_text:
        full_message = f"{user_message}\n\n**Información del usuario:**\n{clarification_text}"
    
    try:
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=full_message)
        ])
        result_text = (response.content or "").strip()
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
    except Exception as e:
        error_output = create_error_output("troubleshooting", "LLM_ERROR", str(e))
        return {
            "worker_outputs": [error_output.model_dump()],
            "troubleshooting_result": error_output.model_dump_json(),
            "events": events
        }
    
    # ==========================================
    # 7. CONSTRUIR RESPUESTA
    # ==========================================
    severity = extract_severity(user_message + " " + result_text)
    
    # Si obtuvimos datos del lab, incluir acciones sugeridas
    extra_content = ""
    if station_details and not station_details.get("ready_to_operate"):
        extra_content = "\n\n---\n💡 **Acciones disponibles:**\n"
        if not station_details["ready_details"]["doors_closed"]:
            extra_content += "- Cierra la puerta de seguridad antes de operar el cobot\n"
        if not station_details["ready_details"]["plc_connected"]:
            extra_content += "- Verifica la conexión de la PLC\n"
    
    output = WorkerOutputBuilder.troubleshooting(
        content=result_text + extra_content,
        problem_identified=f"Problema en {'Estación ' + str(station_num) if station_num else 'equipo'} ({severity})",
        severity=severity,
        summary=f"Diagnóstico completado - Severidad: {severity}",
        confidence=0.85 if is_lab and station_details else 0.75,
    )
    output.metadata.completed_at = datetime.utcnow().isoformat()
    output.metadata.processing_time_ms = processing_time
    output.metadata.model_used = model_name
    
    logger.node_end("troubleshooter_node", {"severity": severity, "is_lab": is_lab})
    events.append(event_report("troubleshooting", f"✅ Diagnóstico listo (Severidad: {severity})"))
    
    # Limpiar contexto
    clean_context = pending_context.copy()
    clean_context.pop("user_clarification", None)
    clean_context.pop("original_query", None)
    
    return {
        "worker_outputs": [output.model_dump()],
        "troubleshooting_result": output.model_dump_json(),
        "pending_context": clean_context,
        "clarification_questions": [],
        "needs_human_input": False,
        "events": events,
    }
