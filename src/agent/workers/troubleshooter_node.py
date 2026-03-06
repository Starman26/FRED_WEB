"""
troubleshooter_node.py - Worker especializado en diagnóstico y troubleshooting

CAPACIDADES:
1. Diagnóstico técnico general (PLCs, Cobots, etc.)
2. Integración con laboratorio ATLAS (consulta estado real de equipos)
3. Preguntas estructuradas con opciones del lab real
4. Ejecución de acciones (cambiar modo cobot, etc.)
5. Conocimiento base del laboratorio (robots, estaciones, terminología)
"""
import os
import json
from typing import Dict, Any, Optional
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from src.agent.utils.llm_factory import get_llm, invoke_and_track

from src.agent.state import AgentState
from src.agent.contracts.worker_contract import WorkerOutputBuilder, create_error_output
from src.agent.contracts.question_schema_v2 import (
    QuestionBuilder,
    QuestionSet,
)
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_execute, event_report, event_error
from src.agent.prompts.format_rules import MARKDOWN_FORMAT_RULES

def extract_suggestions_from_text(text: str) -> tuple[str, list[str]]:
    """
    Extrae las sugerencias del texto de respuesta del LLM.
    
    Returns:
        (content_without_suggestions, list_of_suggestions)
    """
    suggestions = []
    content = text
    
    if "---SUGGESTIONS---" in text and "---END_SUGGESTIONS---" in text:
        parts = text.split("---SUGGESTIONS---")
        content = parts[0].strip()
        
        if len(parts) > 1:
            suggestions_block = parts[1].split("---END_SUGGESTIONS---")[0]
            for line in suggestions_block.strip().split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    clean_line = line.lstrip("0123456789.-) ").strip()
                    if clean_line:
                        suggestions.append(clean_line)
    
    return content, suggestions[:3]


# Importar conocimiento del laboratorio
try:
    from src.agent.knowledge import (
        get_lab_knowledge_summary,
        get_robot_info,
        get_station_info,
        get_error_solution,
        ROBOTS,
        COMMON_ERRORS,
    )
    LAB_KNOWLEDGE_AVAILABLE = True
except ImportError:
    LAB_KNOWLEDGE_AVAILABLE = False


def get_knowledge_context(user_message: str) -> str:
    """
    Obtiene contexto de conocimiento relevante basado en el mensaje del usuario.
    Busca términos, robots, estaciones mencionadas.
    """
    if not LAB_KNOWLEDGE_AVAILABLE:
        return ""
    
    context_parts = []
    msg_lower = user_message.lower()
    
    # Buscar robots mencionados
    for robot_name in ROBOTS.keys():
        if robot_name.lower() in msg_lower:
            context_parts.append(get_robot_info(robot_name))
    
    # Buscar estaciones mencionadas
    for i in range(1, 7):
        if f"estacion {i}" in msg_lower or f"estación {i}" in msg_lower or f"est {i}" in msg_lower:
            context_parts.append(get_station_info(i))
    
    # Buscar códigos de error
    for error_code in COMMON_ERRORS.keys():
        if error_code.lower() in msg_lower:
            context_parts.append(get_error_solution(error_code))
    
    # Si no encontró nada específico, dar resumen general
    if not context_parts:
        context_parts.append(get_lab_knowledge_summary())
    
    return "\n\n".join(context_parts)


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
      "question": "What is the specific problem?",
      "type": "choice",
      "options": [
        {{"id": "1", "label": "Opción 1", "description": "Descripción opcional"}},
        {{"id": "2", "label": "Opción 2"}}
      ]
    }},
    {{
      "id": "q2", 
      "question": "What is the error message?",
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
        from src.agent.utils.llm_factory import get_llm_from_name
        import os
        model_name = os.getenv("DEFAULT_MODEL", "gemini-2.0-flash")
        try:
            llm = get_llm_from_name(model_name, temperature=0.3)
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
        
        # Convertir a QuestionSet via QuestionBuilder
        builder = QuestionBuilder("troubleshooting")
        builder.context(data.get("context", "Necesito más información:"))

        for q_data in data.get("questions", [])[:3]:
            q_type = q_data.get("type", "text")
            q_id = q_data.get("id", f"q{len(builder._questions)+1}")
            q_text = q_data.get("question", "")

            if q_type == "choice":
                raw_opts = q_data.get("options", [])
                options = [
                    (str(o.get("id", "")), o.get("label", ""), o.get("description", ""))
                    for o in raw_opts
                ]
                if options:
                    builder.choice(q_id, q_text, options, include_other=False)
                else:
                    builder.text(q_id, q_text)
            elif q_type == "boolean":
                builder.boolean(q_id, q_text)
            else:  # text
                builder.text(q_id, q_text,
                             placeholder=q_data.get("placeholder", "Escribe tu respuesta..."))

        if not builder._questions:
            logger.warning("troubleshooter_node", "LLM no generó preguntas válidas")
            return None

        return builder.build()
        
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
        # Health
        ping_plc,
        health_check_station,
        # Formatters
        format_lab_overview_for_display,
        format_station_details_for_display,
        format_errors_for_display,
    )
    LAB_TOOLS_AVAILABLE = True
except ImportError:
    LAB_TOOLS_AVAILABLE = False

# Import diagnostic tools for ReAct tool-calling
try:
    from src.agent.tools.lab_tools import DIAGNOSTIC_TOOLS
    DIAGNOSTIC_TOOLS_AVAILABLE = bool(DIAGNOSTIC_TOOLS)
except ImportError:
    DIAGNOSTIC_TOOLS = []
    DIAGNOSTIC_TOOLS_AVAILABLE = False

MAX_DIAGNOSTIC_ITERATIONS = 8


# ============================================
# PROMPTS
# ============================================

TROUBLESHOOTER_PROMPT = """Eres un **Experto en Diagnóstico Técnico** del Laboratorio ATLAS.

## CONOCIMIENTO DEL LABORATORIO
{knowledge_context}

## ESTADO ACTUAL DEL LABORATORIO
{lab_context}

## EQUIPOS DEL LABORATORIO
El laboratorio ATLAS tiene 6 estaciones de trabajo. Cada estación incluye:
- 1 PLC (Siemens S7-1200/S7-1500)
- 1 Cobot (Universal Robots / FANUC)
- Sensores de seguridad (puerta, E-stop)

**Robot principal: ALFREDO** - Cobot UR5e en Estación 1, usado para ensamblaje inicial.

**Protocolo de seguridad:** El cobot NO puede execute routines si:
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

{format_rules}

### Template de diagnostico:

## Diagnostico del Problema

**Equipo afectado**: [Estacion X - Tipo - Nombre]
**Estado actual**: [Datos reales del sistema]

---

### Sintoma identificado
[Descripcion]

### Causas probables
1. **[Causa]** — Probabilidad: Alta/Media/Baja

### Plan de solucion
1. [Paso especifico]

---

**Precauciones**: [Si aplica]
**Rollback**: [Si aplica]

Usuario: {user_name}
"""

TROUBLESHOOTER_PROMPT_SIMPLE = """Eres un **Experto en Diagnostico Tecnico** especializado en:
- PLCs (Siemens, Allen-Bradley, etc.)
- Cobots y robotica industrial
- Sistemas de control

## CONOCIMIENTO BASE
{knowledge_context}

## INFORMACION DEL USUARIO
{clarification_section}

## EVIDENCIA
{evidence_section}

{format_rules}

### Template de diagnostico:

## Diagnostico del Problema

**Sintoma**: [Descripcion]

### Causas Probables
1. **[Causa]** — Probabilidad: Alta/Media/Baja

### Plan de Solucion
1. [Paso especifico]

---

**Precauciones**: [...]

Usuario: {user_name}
"""


TROUBLESHOOTER_COT_PROMPT = """You are an **Expert Diagnostic Technician** for the ATLAS Manufacturing Laboratory.

You have access to diagnostic tools that let you query real-time equipment status.
THINK step by step — use your tools to investigate before diagnosing.

## LABORATORY KNOWLEDGE
{knowledge_context}

## USER CONTEXT
{clarification_section}

## EVIDENCE FROM DOCUMENTATION
{evidence_section}

## DIAGNOSTIC METHODOLOGY — FOLLOW THESE STEPS

**STEP 1: SYMPTOM** — What is the user reporting? Restate the problem clearly.

**STEP 2: HYPOTHESES** — List 2-3 possible causes, ordered by probability.
For each hypothesis, note what data you'd need to confirm or reject it.

**STEP 3: INVESTIGATE** — Call the appropriate tools to gather evidence.
- Start with `lab_overview_tool` for the big picture if needed
- Use `station_details_tool(station_number)` for a specific station
- Use `active_errors_tool` to check for error logs
- Use `ping_plc_tool(station_number)` or `health_check_tool(station_number)` for deeper checks
- Use `diagnose_station_tool(station_number)` for automated diagnosis

**STEP 4: ANALYZE** — After each tool result, evaluate:
- Does this confirm or reject any hypothesis?
- Do I need more data from another tool?
- If yes, call the next tool. If no, proceed to diagnosis.

**STEP 5: DIAGNOSE** — Based on the evidence collected, state:
- Root cause (confirmed or most probable)
- Supporting evidence from tool results

**STEP 6: RECOMMEND** — Provide actionable steps the user can take.

## RULES
- ALWAYS call at least one tool before diagnosing — never guess without data
- Do NOT repeat the same tool call with the same arguments
- Maximum {max_iterations} tool calls total
- After gathering enough data, produce your final diagnosis WITHOUT calling more tools
- Respond in the same language as the user's question
- NEVER use emojis
- Use professional Markdown formatting

{format_rules}

### Diagnostic template:

## Problem Diagnosis

**Affected equipment**: [Station X - Type - Name]
**Current status**: [Real data from tools]

---

### Identified symptom
[Description]

### Probable causes
1. **[Cause]** — Probability: High/Medium/Low — Evidence: [what the tools showed]

### Solution plan
1. [Specific step]

---

**Precautions**: [If applicable]

User: {user_name}
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
        "start cobot", "stop cobot", "lab status",
        "checar", "verificar estado",
        # Robots específicos
        "alfredo", "ur5", "ur10", "universal robots",
        # Estaciones por nombre
        "ensamblaje", "soldadura", "inspección", "inspeccion", "testing", "empaque",
        # Terminología técnica del lab
        "profinet", "tia portal", "polyscope", "teach pendant",
        "oee", "tiempo de ciclo", "celda",
    ]
    
    # Agregar nombres de robots del conocimiento si está disponible
    if LAB_KNOWLEDGE_AVAILABLE:
        try:
            for robot_name in ROBOTS.keys():
                if robot_name.lower() not in lab_keywords:
                    lab_keywords.append(robot_name.lower())
        except:
            pass
    
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
    if any(kw in msg for kw in ["cobot", "robot", "ur5", "ur10", "fanuc", "brazo", "alfredo"]):
        return "cobot"
    if any(kw in msg for kw in ["sensor", "puerta", "door", "proximidad", "e-stop"]):
        return "sensor"
    
    return None


def detect_action_request(message: str) -> Optional[Dict]:
    """Detecta si el usuario quiere ejecutar una acción"""
    msg = message.lower()
    
    # Iniciar cobot / rutina
    start_phrases = [
        "start cobot", "start cobot", "execute routine", "start cobot", "run routine",
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
        "stop cobot", "detener cobot", "stop cobot", "parar rutina",
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
    
    # Reconnect PLC
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
        "lab status", "resumen del lab", "ver laboratorio", "lab status",
        "estado laboratorio", "status lab", "como está el lab", "como esta el lab",
        "estado de las estaciones", "ver estaciones", "mostrar estaciones"
    ]
    if any(phrase in msg for phrase in status_phrases):
        return {"action": "show_lab_status"}
    
    return None


def detect_query_request(message: str) -> Optional[Dict]:
    """Detecta si el usuario está haciendo una consulta sobre el lab status"""
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
        return "Tools de laboratorio no disponibles."
    
    try:
        # Obtener resumen del lab
        overview = get_lab_overview()
        if not overview.get("success"):
            return "No se pudo obtener estado del laboratorio."
        
        lines = ["### Estado Actual del Laboratorio\n"]
        lines.append(f"- **Estaciones activas:** {overview['stations_online']}/{overview['total_stations']}")
        
        if overview['stations_with_errors'] > 0:
            lines.append(f"- **Estaciones con problemas:** {overview['stations_with_errors']}")
        
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
            has_errors = st.get('active_errors', 0) > 0
            is_active = st.get('is_active', True) if 'is_active' in st else st.get('is_operational', True)
            status_icon = "[WARN]" if has_errors else ("[OK]" if is_active else "[OFF]")
            
            # Nuevo formato RPC: is_active, equipment_count, active_errors
            parts = [f"Est. {st['station_number']}"]
            if st.get('name'):
                parts.append(st['name'])
            if st.get('location'):
                parts.append(f"({st['location']})")
            
            extras = []
            if 'equipment_count' in st:
                extras.append(f"Equipos:{st['equipment_count']}")
            if has_errors:
                extras.append(f"Errores:{st['active_errors']}")
            # Backward compat: si tiene el formato viejo
            if 'plc_status' in st:
                extras.append(f"PLC:{st['plc_status']}")
            if 'cobot_status' in st:
                extras.append(f"Cobot:{st['cobot_status']}")
            if 'doors_closed' in st:
                extras.append(f"Puerta:{'🔒' if st['doors_closed'] else '🚪'}")
            
            line = f"  - {status_icon} {' | '.join(parts)}"
            if extras:
                line += f" [{', '.join(extras)}]"
            lines.append(line)
        
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
    
    builder = QuestionBuilder("troubleshooting")
    builder.context("Para diagnosticar el problema en el laboratorio, necesito saber:")

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
            builder.choice("plc_selection",
                           "Which PLC in the laboratory is having the issue?",
                           plc_options, include_other=True)

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
            builder.choice("cobot_selection",
                           "Which cobot is having the issue?",
                           cobot_options, include_other=True)

    # Si no especifica estación para problema general
    elif station_number is None and is_lab_related(message):
        builder.choice("station_selection",
                       "At which laboratory station is the problem occurring?",
                       [
                           ("1", "Station 1 - Initial Assembly"),
                           ("2", "Station 2 - Welding"),
                           ("3", "Station 3 - Visual Inspection"),
                           ("4", "Station 4 - Final Assembly"),
                           ("5", "Station 5 - Testing"),
                           ("6", "Station 6 - Packaging"),
                       ], include_other=False)

    # Agregar pregunta sobre tipo de error si no se detectó
    if len(builder._questions) < 3 and not any(kw in message.lower() for kw in ["no conecta", "error", "stop", "falla"]):
        builder.choice("error_type",
                       "What type of problem are you observing?",
                       [
                           ("1", "Not connecting / Timeout", "Equipment not responding"),
                           ("2", "In STOP mode / Error", "Equipment showing error"),
                           ("3", "Door open / Interlock", "Safety issue"),
                           ("4", "Not executing routine", "Cobot not starting"),
                           ("5", "Comportamiento extraño", "Funciona pero mal"),
                       ], include_other=True)

    if builder._questions:
        return builder.build()

    return None


# ============================================
# CHEQUEO DE SEGURIDAD
# ============================================

def perform_safety_check(station_number: int) -> Dict[str, Any]:
    """
    Realiza un chequeo de seguridad completo de una estación antes de ejecutar acciones.
    
    Returns:
        Dict con resultado del chequeo, incluyendo:
        - safe: bool indicando si es seguro proceder
        - checks: lista de verificaciones realizadas
        - warnings: lista de advertencias
        - blockers: lista de problemas que impiden la operación
    """
    if not LAB_TOOLS_AVAILABLE:
        return {
            "safe": False,
            "checks": [],
            "warnings": [],
            "blockers": ["Sistema de laboratorio no disponible"]
        }
    
    checks = []
    warnings = []
    blockers = []
    
    try:
        # 1. Obtener estado de la estación
        station_details = get_station_details(station_number)
        
        if not station_details.get("success"):
            return {
                "safe": False,
                "checks": [],
                "warnings": [],
                "blockers": [f"No se pudo obtener estado de estación {station_number}"]
            }
        
        # 2. Verificar PLC
        plc_info = station_details.get("plc", {})
        if plc_info.get("is_connected"):
            checks.append(f"PLC conectada ({plc_info.get('name', 'N/A')})")
            if plc_info.get("run_mode") == "RUN":
                checks.append("PLC en modo RUN")
            else:
                warnings.append(f"PLC en modo {plc_info.get('run_mode', 'UNKNOWN')} (no RUN)")
        else:
            blockers.append(f"PLC desconectada - {plc_info.get('ip_address', 'IP desconocida')}")
        
        # 3. Verificar Cobot
        cobot_info = station_details.get("cobot", {})
        if cobot_info.get("is_connected"):
            checks.append(f"Cobot conectado ({cobot_info.get('name', 'N/A')})")
            cobot_status = cobot_info.get("status", "unknown")
            if cobot_status == "error":
                blockers.append(f"Cobot en estado de ERROR")
            elif cobot_status == "busy":
                warnings.append("Cobot ocupado ejecutando otra rutina")
        else:
            blockers.append("Cobot no conectado")
        
        # 4. Verificar sensores de seguridad (puertas)
        sensors = station_details.get("sensors", [])
        door_sensor = next((s for s in sensors if "door" in s.get("name", "").lower()), None)
        
        if door_sensor:
            # 'triggered' = True significa puerta cerrada (sensor activado)
            is_door_closed = door_sensor.get("triggered", False) or door_sensor.get("is_closed", False)
            if is_door_closed:
                checks.append("Puerta de seguridad cerrada")
            else:
                blockers.append("Puerta de seguridad ABIERTA - Cerrar antes de operar")
        else:
            # Si no hay sensor de puerta, asumimos que está OK (no es un blocker)
            checks.append("✅ Estación sin sensor de puerta (OK)")
        
        # 5. Verificar E-Stop
        estop_sensor = next((s for s in sensors if "stop" in s.get("name", "").lower()), None)
        if estop_sensor:
            # triggered=True o value=1 significa E-Stop presionado (malo)
            is_estop_pressed = estop_sensor.get("triggered", False) or estop_sensor.get("value") == 1
            if not is_estop_pressed:
                checks.append("✅ E-Stop no activado")
            else:
                blockers.append("❌ E-Stop ACTIVADO - Liberar antes de operar")
        else:
            # Sin sensor de E-Stop no es un blocker
            checks.append("✅ Estación sin E-Stop dedicado (OK)")
        
        # 6. Verificar errores activos
        active_errors = station_details.get("active_errors", [])
        if active_errors:
            for err in active_errors:
                severity = err.get("severity", "error")
                if severity in ["critical", "error"]:
                    blockers.append(f"❌ Error activo: {err.get('message', 'Error desconocido')}")
                else:
                    warnings.append(f"⚠️ Advertencia: {err.get('message', 'Advertencia')}")
        else:
            checks.append("✅ Sin errores activos")
        
        # 7. Verificar si está lista para operar
        ready = station_details.get("ready_to_operate", False)
        if ready:
            checks.append("✅ Estación lista para operar")
        
        # Determinar si es seguro
        is_safe = len(blockers) == 0
        
        return {
            "safe": is_safe,
            "checks": checks,
            "warnings": warnings,
            "blockers": blockers,
            "station_details": station_details
        }
        
    except Exception as e:
        logger.error("troubleshooter_node", f"Error en chequeo de seguridad: {e}")
        return {
            "safe": False,
            "checks": [],
            "warnings": [],
            "blockers": [f"❌ Error en chequeo de seguridad: {str(e)}"]
        }


def format_safety_check_report(station_number: int, safety_result: Dict, action_description: str) -> str:
    """Formatea el reporte de chequeo de seguridad para mostrar al usuario."""
    
    lines = [
        f"## 🔒 Chequeo de Seguridad - Estación {station_number}",
        f"**Acción solicitada:** {action_description}",
        "",
        "### Verificaciones realizadas:",
        ""
    ]
    
    # Checks exitosos
    for check in safety_result.get("checks", []):
        lines.append(check)
    
    # Warnings
    if safety_result.get("warnings"):
        lines.append("")
        lines.append("### ⚠️ Advertencias:")
        for warning in safety_result["warnings"]:
            lines.append(warning)
    
    # Blockers
    if safety_result.get("blockers"):
        lines.append("")
        lines.append("### 🚫 Problemas que impiden la operación:")
        for blocker in safety_result["blockers"]:
            lines.append(blocker)
    
    # Resultado final
    lines.append("")
    if safety_result.get("safe"):
        lines.append("---")
        lines.append("### ✅ Safety Check: PASSED")
        lines.append("")
        lines.append("The station is ready to execute the action.")
    else:
        lines.append("---")
        lines.append("### ❌ Safety Check: NOT PASSED")
        lines.append("")
        lines.append("Resuelve los problemas indicados antes de continuar.")
    
    return "\n".join(lines)


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
            content = result.get("message", f"✅ Cobot configured in mode {mode}")
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
                results.append(f"{'✅' if r.get('success') else '❌'} Close doors: {r.get('message', r.get('error'))}")
            
            elif action_name == "reconnect_plc":
                st = suggested.get("station")
                r = reconnect_plc(st)
                results.append(f"{'✅' if r.get('success') else '❌'} Reconnect PLC est.{st}: {r.get('message', r.get('error'))}")
            
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

Do you want me to check the current laboratory status?"""
        
        return {
            "success": True,
            "type": "repair",
            "content": content,
            "data": {"diagnosis": diagnosis, "results": results}
        }
    
    return {"success": False, "error": f"Acción desconocida: {action_type}"}


def execute_lab_query(query: Dict, interaction_mode: str = "chat") -> Dict[str, Any]:
    """Ejecuta una consulta de lab - en modo agent solo hace la llamada mínima."""
    if not LAB_TOOLS_AVAILABLE:
        return {"success": False, "error": "Lab tools no disponibles"}
    
    query_type = query.get("query")
    lean = interaction_mode in ("agent", "voice")  # modo ligero: sin calls extra
    
    if query_type == "active_errors":
        # Obtener errores registrados
        errors = get_active_errors()
        
        if lean:
            # Solo errores, sin enriquecer con PLCs/puertas
            total = errors.get("total_errors", 0) if errors.get("success") else 0
            if total == 0:
                return {"success": True, "content": "No hay errores activos.", "data": errors, "offer_help": False, "total_problems": 0}
            lines = []
            for err in errors.get("errors", []):
                lines.append(f"[{err.get('severity')}] Est.{err.get('station_number')} {err.get('equipment_name','')}: {err.get('message','')}")
            return {"success": True, "content": "\n".join(lines), "data": errors, "offer_help": True, "total_problems": total}
        
        # Modo completo: también checa PLCs y puertas
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
            lines.append("### Errores Registrados\n")
            lines.append("| Estacion | Equipo | Codigo | Severidad | Mensaje |")
            lines.append("|----------|--------|--------|-----------|---------|")
            for err in errors.get("errors", []):
                sev = err.get("severity", "info").upper()
                lines.append(f"| {err.get('station_number', '?')} | {err.get('equipment_name', 'N/A')} | `{err.get('error_code', '-')}` | **{sev}** | {err.get('message', '')} |")
                total_problems += 1
            lines.append("")

        # 2. PLCs desconectadas
        if disconnected_plcs:
            lines.append("### Equipos Desconectados\n")
            for plc in disconnected_plcs:
                lines.append(f"- **Est. {plc.get('station_number')}** — {plc.get('name')}: PLC desconectada (`{plc.get('ip_address')}`)")
                total_problems += 1
            lines.append("")

        # 3. Cobots con error
        cobots = get_all_cobots()
        if cobots.get("success"):
            error_cobots = [c for c in cobots.get("cobots", []) if c.get("status") == "error" or not c.get("is_connected")]
            if error_cobots:
                lines.append("### Cobots con Problemas\n")
                for cobot in error_cobots:
                    status = "desconectado" if not cobot.get("is_connected") else cobot.get("status")
                    lines.append(f"- **Est. {cobot.get('station_number')}** — {cobot.get('name')}: {status}")
                    total_problems += 1
                lines.append("")

        # 4. Puertas abiertas
        if open_doors > 0:
            lines.append(f"### Puertas Abiertas: {open_doors}\n")
            for door in doors.get("doors", []):
                if not door.get("is_closed"):
                    lines.append(f"- **Est. {door.get('station_number')}** — Puerta abierta")
                    total_problems += 1
            lines.append("")

        # Construir respuesta conversacional
        if total_problems == 0:
            content = "## Estado del Laboratorio\n\n**No hay problemas activos en el laboratorio.** Todas las estaciones estan operando normalmente."
            offer_help = False
        else:
            # Introduccion conversacional
            if total_problems == 1:
                intro = "## Estado del Laboratorio\n\nSe encontro **1 problema**:\n\n"
            else:
                intro = f"## Estado del Laboratorio\n\nSe encontraron **{total_problems} problemas**:\n\n"

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

            if disconnected == 0:
                intro = f"## Estado de PLCs\n\nTodas las **{plcs['total']} PLCs** estan conectadas y funcionando correctamente.\n\n"
            else:
                intro = f"## Estado de PLCs\n\n**{disconnected} PLC(s) desconectada(s)** de {plcs['total']} totales.\n\n"

            lines = [intro]
            lines.append("| Estacion | PLC | Conectada | Estado | Modo |")
            lines.append("|----------|-----|-----------|--------|------|")

            for plc in plcs.get("plcs", []):
                connected = "Si" if plc.get("is_connected") else "No"
                mode = plc.get("run_mode", "?")
                lines.append(f"| {plc['station_number']} | {plc['name']} | {connected} | {plc['status']} | {mode} |")

            return {"success": True, "content": "\n".join(lines), "data": plcs, "offer_help": disconnected > 0, "total_problems": disconnected}
        return plcs
    
    if query_type == "cobot_status":
        cobots = get_all_cobots()
        if cobots.get("success"):
            running = cobots.get("running", 0)
            total = cobots.get("total", 0)

            if running == 0:
                intro = f"## Estado de Cobots\n\nLos **{total} cobots** estan en espera (idle). Ninguno ejecutando rutinas.\n\n"
            else:
                intro = f"## Estado de Cobots\n\n**{running} de {total} cobots** estan ejecutando rutinas.\n\n"

            lines = [intro]
            lines.append("| Estacion | Cobot | Conectado | Rutina |")
            lines.append("|----------|-------|-----------|--------|")
            for cobot in cobots.get("cobots", []):
                connected = "Si" if cobot.get("is_connected") else "No"
                routine = cobot.get("routine", "idle")
                lines.append(f"| {cobot['station_number']} | {cobot['name']} | {connected} | {routine} |")

            return {"success": True, "content": "\n".join(lines), "data": cobots}
        return cobots
    
    if query_type == "door_status":
        doors = check_door_sensors()
        if doors.get("success"):
            open_count = doors.get("open_doors_count", 0)

            if open_count == 0:
                intro = "## Estado de Puertas\n\n**Todas las puertas estan cerradas.** El laboratorio esta seguro para operar.\n\n"
            else:
                intro = f"## Estado de Puertas\n\n**{open_count} puerta(s) abierta(s)** — los cobots no pueden operar hasta cerrarlas.\n\n"

            lines = [intro]
            lines.append("| Estacion | Estado |")
            lines.append("|----------|--------|")
            for door in doors.get("doors", []):
                status_text = "Cerrada" if door.get("is_closed") else "**ABIERTA**"
                lines.append(f"| {door['station_number']} | {status_text} |")

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
            problems = overview.get("stations_with_errors", 0)
            if problems == 0:
                intro = "## Resumen del Laboratorio\n\n**El laboratorio esta funcionando correctamente.** Todas las estaciones operativas.\n\n"
            else:
                intro = f"## Resumen del Laboratorio\n\n**{problems} estacion(es) con problemas.**\n\n"

            content = intro + format_lab_overview_for_display(overview)

            return {"success": True, "content": content, "data": overview, "offer_help": problems > 0, "total_problems": problems}
        return overview
    
    if query_type == "ping_plc":
        station_num = query.get("station")
        if not station_num:
            return {"success": False, "error": "No se especificó estación", "needs_station": True}
        result = ping_plc(station_num)
        if result.get("success"):
            health = result.get("health", "unknown")
            ip = result.get("ip_address", "?")
            ms = result.get("response_time_ms", "?")
            content = f"## Ping PLC — Estacion {station_num}\n\n"
            content += f"**Estado:** {health.upper()}\n\n"
            content += "| Metrica | Valor |\n"
            content += "|---------|-------|\n"
            content += f"| PLC | {result.get('plc_name', '?')} |\n"
            content += f"| IP | `{ip}` |\n"
            content += f"| Ping | {'OK' if result['ping_ok'] else 'FAILED'} ({ms}ms) |\n"
            content += f"| Modo | {('RUN' if result.get('plc_run_mode') else 'STOP')} |\n"
            content += f"| DB Status | {('conectado' if result.get('db_connected') else 'desconectado')} |\n"
            if result.get("error_code"):
                content += f"| Error | `{result['error_code']}` |\n"
            return {"success": True, "content": content, "data": result,
                    "offer_help": health != "healthy", "total_problems": 0 if health == "healthy" else 1}
        return result

    if query_type == "health_check":
        station_num = query.get("station")
        if not station_num:
            return {"success": False, "error": "No se especificó estación", "needs_station": True}
        result = health_check_station(station_num)
        if result.get("success"):
            health = result.get("overall_health", "unknown")
            checks = result.get("checks", {})
            issues = result.get("issues", [])
            recommendations = result.get("recommendations", [])

            lines = [f"## Health Check — Estacion {station_num}\n"]
            lines.append(f"**Estado general:** {health.upper()}\n")

            # Summary table
            plc = checks.get("plc", {})
            doors = checks.get("doors", {})
            errs = checks.get("errors", {})
            cobot = checks.get("cobot", {})
            open_n = doors.get("open_count", 0)

            lines.append("| Componente | Estado | Detalle |")
            lines.append("|------------|--------|---------|")
            lines.append(f"| **PLC** | {plc.get('status', '?')} | Ping: {plc.get('response_time_ms', '?')}ms |")
            lines.append(f"| **Puertas** | {'Cerradas' if doors.get('all_closed') else f'{open_n} abierta(s)'} | {doors.get('total', '?')} total |")
            lines.append(f"| **Errores** | {errs.get('active_count', 0)} activos | {errs.get('critical_count', 0)} criticos |")
            lines.append(f"| **Cobot** | {'Conectado' if cobot.get('connected') else 'Desconectado'} | {cobot.get('mode', '?')} |")

            if issues:
                lines.append(f"\n---\n\n### Problemas detectados\n")
                for issue in issues:
                    lines.append(f"- {issue}")

            if recommendations:
                lines.append(f"\n---\n\n### Recomendaciones\n")
                for rec in recommendations:
                    lines.append(f"- {rec}")

            return {"success": True, "content": "\n".join(lines), "data": result,
                    "offer_help": len(issues) > 0, "total_problems": len(issues)}
        return result

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
    
    pending_context = state.get("pending_context", {}) or {}
    user_clarification = pending_context.get("user_clarification", "")
    original_query = pending_context.get("original_query", "")
    already_clarified = bool(user_clarification or pending_context.get("_hitl_consumed"))

    # ==========================================
    # 1. DETERMINAR MENSAJE A USAR
    # ==========================================
    if user_clarification and original_query:
        user_message = original_query
        clarification_text = user_clarification
        logger.info("troubleshooter_node", f"Usando original_query + clarificación: '{original_query[:60]}...' + '{clarification_text[:60]}...'")
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
            elif detected_action in ("check_status", "get_station_info", "get_station_details"):
                if station_num:
                    query_request = {"query": "station_details", "station": station_num}
                else:
                    query_request = {"query": "lab_overview"}
            elif detected_action in ("check_door_status", "get_door_status"):
                if station_num:
                    query_request = {"query": "door_status", "station": station_num}
                else:
                    query_request = {"query": "door_status"}
            elif detected_action in ("check_plc_status", "get_plc_status"):
                query_request = {"query": "plc_status"}
            elif detected_action in ("check_cobot_status", "get_cobot_status",
                                     "check_running_routines", "check_cobot_routines",
                                     "check_active_routines", "get_running_cobots"):
                query_request = {"query": "cobot_status"}
            elif detected_action in ("ping_plc", "plc_health", "plc_ping"):
                if station_num:
                    query_request = {"query": "ping_plc", "station": station_num}
                else:
                    query_request = {"query": "ping_plc", "station": None}  # HITL below
            elif detected_action in ("health_check", "station_health"):
                if station_num:
                    query_request = {"query": "health_check", "station": station_num}
                else:
                    query_request = {"query": "health_check", "station": None}  # HITL below
            elif detected_action == "search_docs":
                # search_docs is a research action, not troubleshooting.
                # When troubleshooter runs in a multi-step plan after research,
                # provide station/lab data instead.
                if station_num:
                    query_request = {"query": "station_details", "station": station_num}
                elif equipment:
                    equip_query_map = {"door": "door_status", "plc": "plc_status", "cobot": "cobot_status"}
                    query_request = {"query": equip_query_map.get(equipment, "lab_overview")}
                else:
                    query_request = {"query": "lab_overview"}
            elif detected_action in ("get_station_count", "get_lab_overview", "lab_overview"):
                query_request = {"query": "lab_overview"}
            elif station_num:
                # Si el LLM devolvió acción desconocida pero hay estación → station_details
                query_request = {"query": "station_details", "station": station_num}
            else:
                # keyword-based fallback for unknown LLM-generated actions
                _action_lower = (detected_action or "").lower()
                _msg_lower = user_message.lower()
                _combined = _action_lower + " " + _msg_lower
                
                if any(kw in _combined for kw in ["routine", "cobot", "pick", "place", "robot", "running"]):
                    query_request = {"query": "cobot_status"}
                elif any(kw in _combined for kw in ["door", "puerta", "cerrad", "abiert"]):
                    query_request = {"query": "door_status"}
                elif any(kw in _combined for kw in ["plc", "controlador", "controller"]):
                    query_request = {"query": "plc_status"}
                elif any(kw in _combined for kw in ["error", "alarm", "falla", "fault"]):
                    query_request = {"query": "active_errors"}
                else:
                    # Final fallback
                    query_request = detect_query_request(user_message)
                    if not query_request and is_lab_related(user_message):
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
    # 2.5. VERIFICAR SI ES CONFIRMACIÓN DE ACCIÓN DE COBOT
    # ==========================================
    if pending_context.get("awaiting_cobot_confirmation"):
        user_response = (clarification_text or user_message).lower().strip()
        
        # Respuestas afirmativas
        affirmative = any(word in user_response for word in [
            "yes", "sí", "si", "confirmo", "confirmar", "dale", "ok", 
            "adelante", "procede", "hazlo", "ejecutar", "ejecuta",
            "1", "confirm", "iniciar", "inicia"
        ])
        
        if affirmative:
            # Recuperar la acción guardada
            saved_action = pending_context.get("saved_cobot_action", {})
            station_num = saved_action.get("station")
            mode = saved_action.get("mode", 1)
            action_type = saved_action.get("action", "start_cobot")
            
            logger.info("troubleshooter_node", f"Usuario confirmó acción de cobot: {action_type} en estación {station_num}")
            events.append(event_report("troubleshooting", f"✅ Confirmación recibida - Ejecutando acción en estación {station_num}..."))
            
            # Ejecutar la acción
            result = execute_lab_action(saved_action, state.get("user_name", "user"))
            
            if result.get("success"):
                action_desc = "Cobot started" if action_type == "start_cobot" else "Cobot detenido"
                content = f"""## ✅ Action Executed Successfully

**Station:** {station_num}
**Action:** {action_desc}
**Mode:** {mode if action_type == "start_cobot" else "Detenido"}

{result.get("content", "")}

Do you need anything else?"""
                
                output = WorkerOutputBuilder.troubleshooting(
                    content=content,
                    problem_identified=f"Acción ejecutada: {action_desc}",
                    severity="low",
                    summary=f"{action_desc} en estación {station_num}",
                    confidence=1.0,
                )
                events.append(event_report("troubleshooting", f"✅ {action_desc} correctamente"))
            else:
                output = WorkerOutputBuilder.troubleshooting(
                    content=result.get("content") or f"❌ Error: {result.get('error')}",
                    problem_identified="Error en acción de cobot",
                    severity="medium",
                    summary="No se pudo ejecutar la acción",
                    confidence=0.5,
                )
                events.append(event_error("troubleshooting", result.get("error", "Error desconocido")))
            
            # Limpiar contexto
            return {
                "worker_outputs": [output.model_dump()],
                "troubleshooting_result": output.model_dump_json(),
                "events": events,
                "pending_context": {},
            }
        else:
            # Usuario canceló
            logger.info("troubleshooter_node", "Usuario canceló acción de cobot")
            output = WorkerOutputBuilder.troubleshooting(
                content="❌ **Acción cancelada**\n\nNo se realizó ningún cambio. Do you need anything else?",
                problem_identified="Acción cancelada por usuario",
                severity="low",
                summary="Usuario canceló la acción",
                confidence=1.0,
            )
            
            return {
                "worker_outputs": [output.model_dump()],
                "troubleshooting_result": output.model_dump_json(),
                "events": events,
                "pending_context": {},
            }
    
    # ==========================================
    # 2.6. VERIFICAR SI ES CONFIRMACIÓN DE REPARACIÓN
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
                content="Entendido, no realizaré cambios. Do you need anything else?",
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
    # 2.7. VERIFICAR SI ES RESPUESTA DE HITL PARA PING/HEALTH CHECK
    # ==========================================
    if pending_context.get("pending_health_query"):
        # User answered the station-selection HITL
        user_response = (clarification_text or user_message).strip()
        selected_station = detect_station_number(user_response)
        # Also try bare digit
        if not selected_station:
            for char in user_response:
                if char.isdigit() and 1 <= int(char) <= 6:
                    selected_station = int(char)
                    break

        health_query = pending_context["pending_health_query"]
        if selected_station:
            logger.info("troubleshooter_node", f"HITL: usuario seleccionó estación {selected_station} para {health_query}")
            events.append(event_report("troubleshooting", f"Ejecutando {health_query} en estación {selected_station}..."))
            result = execute_lab_query(
                {"query": health_query, "station": selected_station},
                interaction_mode=state.get("interaction_mode", "chat").lower(),
            )
            if result.get("success"):
                output = WorkerOutputBuilder.troubleshooting(
                    content=result.get("content", "Consulta completada"),
                    problem_identified=f"{health_query} completado",
                    severity="low",
                    summary=f"{health_query} estación {selected_station}",
                    confidence=1.0,
                )
            else:
                output = WorkerOutputBuilder.troubleshooting(
                    content=f"Error: {result.get('error', 'Error desconocido')}",
                    problem_identified="Error en consulta",
                    severity="low",
                    summary="No se pudo completar la consulta",
                    confidence=0.5,
                )
            return {
                "worker_outputs": [output.model_dump()],
                "troubleshooting_result": output.model_dump_json(),
                "events": events,
                "pending_context": {},
            }

    # ==========================================
    # 3. ACCIONES DE COBOT - REQUIEREN CHEQUEO DE SEGURIDAD Y CONFIRMACIÓN
    # ==========================================
    if action_request and action_request.get("action") in ["start_cobot", "stop_cobot"]:
        action_type = action_request["action"]
        station = action_request.get("station")
        mode = action_request.get("mode", 1 if action_type == "start_cobot" else 0)
        
        # Verificar que tenemos estación
        if station is None:
            # Pedir estación via HITL
            qs = (
                QuestionBuilder("troubleshooting")
                .title("Selección de Estación")
                .context("Necesito saber en qué estación ejecutar la acción:")
                .wizard(allow_skip=False)
                .choice("station_selection",
                        "At which station do you want to execute this action?",
                        [
                            ("1", "Estación 1", "Ensamblaje Inicial"),
                            ("2", "Estación 2", "Soldadura"),
                            ("3", "Estación 3", "Inspección Visual"),
                            ("4", "Estación 4", "Ensamblaje Final"),
                            ("5", "Estación 5", "Testing"),
                            ("6", "Estación 6", "Empaque"),
                        ], include_other=False)
                .build()
            )
            payload = qs.to_interrupt_payload()

            updated_context = pending_context.copy()
            updated_context["original_query"] = user_message
            updated_context["pending_cobot_action"] = action_type
            updated_context["pending_cobot_mode"] = mode
            updated_context["question_set"] = qs.model_dump_json()
            updated_context["current_worker"] = "troubleshooting"

            output = WorkerOutputBuilder.troubleshooting(
                content=qs.to_display_text(),
                problem_identified="Selección de estación requerida",
                severity="pending",
                summary="Esperando selección de estación",
                confidence=0.0,
                status="needs_context"
            )

            output_dict = output.model_dump()
            output_dict["status"] = "needs_context"
            output_dict["clarification_questions"] = payload

            return {
                "worker_outputs": [output_dict],
                "troubleshooting_result": json.dumps(output_dict),
                "needs_human_input": True,
                "clarification_questions": payload,
                "pending_context": updated_context,
                "events": events,
            }
        
        # Tenemos estación - realizar chequeo de seguridad
        events.append(event_report("troubleshooting", f"🔒 Realizando chequeo de seguridad en estación {station}..."))
        logger.info("troubleshooter_node", f"Iniciando chequeo de seguridad para estación {station}")
        
        safety_result = perform_safety_check(station)
        
        # Action description
        if action_type == "start_cobot":
            action_description = f"Start Cobot in mode/routine {mode}"
        else:
            action_description = "Stop Cobot"
        
        # Formatear reporte de seguridad
        safety_report = format_safety_check_report(station, safety_result, action_description)
        
        events.append(event_report("troubleshooting", 
            f"{'✅' if safety_result['safe'] else '❌'} Safety check completed"))
        
        if safety_result.get("safe"):
            # Check passed - requesting confirmation
            qs = (
                QuestionBuilder("troubleshooting")
                .title("Action Confirmation")
                .context("Safety check passed. Please confirm the action:")
                .wizard(allow_skip=False)
                .on_complete("Processing request...")
                .boolean("confirm_action",
                         f"Do you confirm you want to {action_description.lower()} at station {station}?",
                         help_text="This action will start the cobot movement. Make sure the area is clear.")
                .build()
            )
            payload = qs.to_interrupt_payload()

            # Contenido completo: reporte + pregunta de confirmación
            full_content = safety_report

            updated_context = pending_context.copy()
            updated_context["awaiting_cobot_confirmation"] = True
            updated_context["saved_cobot_action"] = {
                "action": action_type,
                "station": station,
                "mode": mode
            }
            updated_context["safety_report"] = safety_report
            updated_context["question_set"] = qs.model_dump_json()
            updated_context["current_worker"] = "troubleshooting"

            output = WorkerOutputBuilder.troubleshooting(
                content=full_content,
                problem_identified=f"Safety check passed - Awaiting confirmation",
                severity="pending",
                summary=f"Confirmar: {action_description}",
                confidence=0.9,
                status="needs_context"
            )

            output_dict = output.model_dump()
            output_dict["status"] = "needs_context"
            output_dict["clarification_questions"] = payload

            return {
                "worker_outputs": [output_dict],
                "troubleshooting_result": json.dumps(output_dict),
                "needs_human_input": True,
                "clarification_questions": payload,
                "pending_context": updated_context,
                "events": events,
            }
        else:
            # Check NOT passed - showing issues
            full_content = safety_report + """

---
### 🛠️ What can I do?

I can try to resolve some issues automatically:
- Close open doors
- Reconnect disconnected PLCs
- Clear non-critical errors

Do you want me to try to fix the issues found?"""
            
            # Ofrecer reparación automática
            qs = (
                QuestionBuilder("troubleshooting")
                .title("Automatic Repair")
                .context("Safety check found issues:")
                .wizard(allow_skip=False)
                .boolean("auto_repair",
                         "Do you want me to try to repair the problems automatically?",
                         help_text="I will try to close doors, reconnect equipment, and clear errors.")
                .build()
            )
            payload = qs.to_interrupt_payload()

            updated_context = pending_context.copy()
            updated_context["awaiting_repair_confirmation"] = True
            updated_context["failed_safety_check"] = True
            updated_context["target_station"] = station
            updated_context["original_cobot_action"] = {
                "action": action_type,
                "station": station,
                "mode": mode
            }
            updated_context["question_set"] = qs.model_dump_json()
            updated_context["current_worker"] = "troubleshooting"

            output = WorkerOutputBuilder.troubleshooting(
                content=full_content,
                problem_identified=f"Safety check failed - {len(safety_result.get('blockers', []))} blockers",
                severity="high",
                summary="Problemas de seguridad detectados",
                confidence=0.9,
                status="needs_context"
            )

            output_dict = output.model_dump()
            output_dict["status"] = "needs_context"
            output_dict["clarification_questions"] = payload

            return {
                "worker_outputs": [output_dict],
                "troubleshooting_result": json.dumps(output_dict),
                "needs_human_input": True,
                "clarification_questions": payload,
                "pending_context": updated_context,
                "events": events,
            }
    
    # ==========================================
    # 3.5. OTRAS ACCIONES (no requieren confirmación de seguridad)
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
    # If ping_plc or health_check needs a station, ask via HITL
    if (query_request
        and query_request.get("query") in ("ping_plc", "health_check")
        and not query_request.get("station")):
        query_label = "Ping PLC" if query_request["query"] == "ping_plc" else "Health Check"
        qs = (
            QuestionBuilder("troubleshooting")
            .title(f"Seleccion de Estacion — {query_label}")
            .context(f"Necesito saber en que estacion ejecutar el {query_label}:")
            .wizard(allow_skip=False)
            .choice("station_selection",
                    f"A cual estacion quieres hacerle {query_label}?",
                    [
                        ("1", "Estacion 1", "Ensamblaje Inicial"),
                        ("2", "Estacion 2", "Soldadura"),
                        ("3", "Estacion 3", "Inspeccion Visual"),
                        ("4", "Estacion 4", "Ensamblaje Final"),
                        ("5", "Estacion 5", "Testing"),
                        ("6", "Estacion 6", "Empaque"),
                    ], include_other=False)
            .build()
        )
        payload = qs.to_interrupt_payload()

        updated_context = pending_context.copy()
        updated_context["original_query"] = user_message
        updated_context["pending_health_query"] = query_request["query"]
        updated_context["question_set"] = qs.model_dump_json()
        updated_context["current_worker"] = "troubleshooting"

        output = WorkerOutputBuilder.troubleshooting(
            content=qs.to_display_text(),
            problem_identified="Seleccion de estacion requerida",
            severity="pending",
            summary=f"Esperando seleccion de estacion para {query_label}",
            confidence=0.0,
            status="needs_context",
        )
        output_dict = output.model_dump()
        output_dict["status"] = "needs_context"
        output_dict["clarification_questions"] = payload

        return {
            "worker_outputs": [output_dict],
            "troubleshooting_result": json.dumps(output_dict),
            "needs_human_input": True,
            "clarification_questions": payload,
            "pending_context": updated_context,
            "events": events,
        }

    if query_request and LAB_TOOLS_AVAILABLE:
        events.append(event_report("troubleshooting", f"Consultando: {query_request['query']}"))
        
        result = execute_lab_query(query_request, interaction_mode=state.get("interaction_mode", "chat").lower())
        
        if result.get("success"):
            # Si hay problemas y se ofrece ayuda, usar HITL para confirmación
            offer_help = result.get("offer_help", False)
            total_problems = result.get("total_problems", 0) or result.get("data", {}).get("total_problems", 0)
            
            if offer_help and total_problems > 0:
                events.append(event_report("troubleshooting", "💡 Ofreciendo reparación automática"))
                
                # Obtener el contenido con los errores encontrados (SIN la pregunta)
                errors_content = result.get("content", "Se encontraron problemas")
                
                # Crear pregunta de confirmación
                qs = (
                    QuestionBuilder("troubleshooting")
                    .choice("repair_confirmation",
                            "Do you want me to try to repair these problems automatically?",
                            [
                                ("yes", "Yes, fix it", "Execute automatic repair"),
                                ("no", "No, solo quería ver el estado", "No hacer cambios"),
                            ], include_other=False)
                    .build()
                )
                repair_payload = qs.to_interrupt_payload()

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
                output_dict["clarification_questions"] = repair_payload
                
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
            
            # Generar sugerencias basadas en el tipo de query
            query_type = query_request.get("query", "")
            if query_type == "station_details":
                station = query_request.get("station", "")
                suggestions = [
                    f"Check if there are active errors at station {station}",
                    f"Start a routine on station {station}",
                    "View the complete lab overview"
                ]
            elif query_type == "lab_overview":
                suggestions = [
                    "Check details of a specific station",
                    "Show me all active errors in the lab",
                    "What cobots are currently running?"
                ]
            elif query_type == "active_errors":
                suggestions = [
                    "Try to fix these errors automatically",
                    "Show me which stations have problems",
                    "What is the status of the PLCs?"
                ]
            else:
                suggestions = [
                    "Check status of other equipment",
                    "Run a diagnostic on a specific station",
                    "Show me the complete lab overview"
                ]
        else:
            output = WorkerOutputBuilder.troubleshooting(
                content=f"❌ Error en consulta: {result.get('error')}",
                problem_identified="Error en consulta",
                severity="low",
                summary="No se pudo completar la consulta",
                confidence=0.5,
            )
            suggestions = [
                "Try checking a different station",
                "View the complete lab status",
                "What equipment is currently connected?"
            ]
        
        return {
            "worker_outputs": [output.model_dump()],
            "troubleshooting_result": output.model_dump_json(),
            "events": events,
            "follow_up_suggestions": suggestions,
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

Could you please rephrase your request?"""
        
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
    
    # Skip clarification for troubleshoot mode — equipment context already provided
    if state.get("interaction_mode", "").lower() == "troubleshoot":
        already_clarified = True

    if not already_clarified and not is_command:
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
            events.append(event_report("troubleshooting", " Generando preguntas contextuales..."))
            
            try:
                question_llm = get_llm(state, temperature=0.3)
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
            question_set = (
                QuestionBuilder("troubleshooting")
                .context("Tu mensaje es un poco breve. Para ayudarte mejor:")
                .text("more_details",
                      "Could you give me more details about your query?",
                      placeholder="Describe el problema o lo que necesitas...")
                .build()
            )
        
        if question_set:
            payload = question_set.to_interrupt_payload()

            updated_context = pending_context.copy()
            updated_context["original_query"] = user_message
            updated_context["is_lab_related"] = is_lab
            updated_context["detected_station"] = station_num
            updated_context["detected_equipment"] = equipment
            updated_context["question_set"] = question_set.model_dump_json()
            updated_context["current_worker"] = "troubleshooting"

            output = WorkerOutputBuilder.troubleshooting(
                content=question_set.to_display_text(),
                problem_identified="Recopilando información",
                severity="pending",
                summary="Necesito más información",
                confidence=0.0,
                status="needs_context"
            )

            output_dict = output.model_dump()
            output_dict["clarification_questions"] = payload

            events.append(event_report("troubleshooting", f"📋 {len(question_set.questions)} preguntas"))

            return {
                "worker_outputs": [output_dict],
                "troubleshooting_result": json.dumps(output_dict),
                "needs_human_input": True,
                "clarification_questions": payload,
                "pending_context": updated_context,
                "events": events,
            }
    
    # ==========================================
    # 5. GENERAR DIAGNÓSTICO (ReAct Chain-of-Thought)
    # ==========================================
    evidence_text = get_evidence_from_context(state)
    clarification_section = clarification_text if clarification_text else "No hay información adicional."

    try:
        llm = get_llm(state, temperature=0.3)
    except Exception as e:
        error_output = create_error_output("troubleshooting", "LLM_INIT_ERROR", str(e))
        return {
            "worker_outputs": [error_output.model_dump()],
            "troubleshooting_result": error_output.model_dump_json(),
            "events": events
        }

    # Obtener conocimiento relevante
    knowledge_context = ""
    if LAB_KNOWLEDGE_AVAILABLE:
        try:
            knowledge_context = get_knowledge_context(user_message)
        except Exception as e:
            logger.warning("troubleshooter_node", f"Error obteniendo conocimiento: {e}")

    # Mensaje combinado
    full_message = user_message
    if clarification_text:
        full_message = f"{user_message}\n\n**Información del usuario:**\n{clarification_text}"

    # ── Decide path: ReAct (with tools) or single-call (no tools) ──
    use_react = is_lab and DIAGNOSTIC_TOOLS_AVAILABLE

    if use_react:
        # ══════════════════════════════════════════
        # ReAct path: chain-of-thought with tool-calling
        # ══════════════════════════════════════════
        events.append(event_report("troubleshooting", "Investigating with diagnostic tools..."))
        logger.info("troubleshooter_node", "Using ReAct CoT with diagnostic tools")

        prompt = TROUBLESHOOTER_COT_PROMPT.format(
            knowledge_context=knowledge_context,
            clarification_section=clarification_section,
            evidence_section=evidence_text,
            format_rules=MARKDOWN_FORMAT_RULES,
            max_iterations=MAX_DIAGNOSTIC_ITERATIONS,
            user_name=state.get("user_name", "Usuario"),
        )

        llm_with_tools = llm.bind_tools(DIAGNOSTIC_TOOLS)
        tool_map = {t.name: t for t in DIAGNOSTIC_TOOLS}
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=full_message),
        ]

        tokens_used = 0
        called_tools = set()  # track (tool_name, args_key) to detect repeats
        consecutive_failures = 0

        for iteration in range(MAX_DIAGNOSTIC_ITERATIONS):
            try:
                response, tokens = invoke_and_track(llm_with_tools, messages, "troubleshooter")
                tokens_used += tokens
            except Exception as e:
                logger.error("troubleshooter_node", f"ReAct LLM error at iteration {iteration}: {e}")
                break

            messages.append(response)

            if not response.tool_calls:
                logger.info("troubleshooter_node",
                            f"ReAct done after {iteration + 1} iterations (no more tool calls)")
                break

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id = tc.get("id", f"call_{iteration}_{tool_name}")

                # Anti-repeat: same tool + same args → reject
                args_key = json.dumps(tool_args, sort_keys=True)
                call_key = (tool_name, args_key)

                if call_key in called_tools:
                    logger.info("troubleshooter_node", f"Duplicate tool call: {tool_name}({tool_args})")
                    messages.append(ToolMessage(
                        content="ERROR: You already called this tool with the same arguments. "
                                "Use a DIFFERENT tool or different arguments, or present your diagnosis.",
                        tool_call_id=tool_id,
                    ))
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        messages.append(ToolMessage(
                            content="STOP: Too many repeated calls. Present your diagnosis with the data you have.",
                            tool_call_id=f"{tool_id}_stop",
                        ))
                    continue

                called_tools.add(call_key)
                events.append(event_execute("troubleshooting", f"Calling {tool_name}..."))

                if tool_name in tool_map:
                    try:
                        result = tool_map[tool_name].invoke(tool_args)
                        consecutive_failures = 0
                    except Exception as e:
                        logger.error("troubleshooter_node", f"Tool {tool_name} failed: {e}")
                        result = {"success": False, "error": str(e)}
                        consecutive_failures += 1
                else:
                    result = {"success": False, "error": f"Unknown tool: {tool_name}"}
                    consecutive_failures += 1

                messages.append(ToolMessage(
                    content=str(result) if not isinstance(result, str) else result,
                    tool_call_id=tool_id,
                ))

            if consecutive_failures >= 3:
                logger.warning("troubleshooter_node", "3+ consecutive failures, stopping ReAct loop")
                break

        # Extract final response from last AIMessage without tool_calls
        result_text = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                result_text = msg.content.strip()
                break

        if not result_text:
            result_text = "Could not complete the diagnosis. Please provide more details."

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info("troubleshooter_node",
                     f"ReAct complete: {len(called_tools)} tool calls, {tokens_used} tokens, {processing_time:.0f}ms")

    else:
        # ══════════════════════════════════════════
        # Fallback: single LLM call (no lab tools available)
        # ══════════════════════════════════════════
        prompt = TROUBLESHOOTER_PROMPT_SIMPLE.format(
            knowledge_context=knowledge_context,
            clarification_section=clarification_section,
            evidence_section=evidence_text,
            format_rules=MARKDOWN_FORMAT_RULES,
            user_name=state.get("user_name", "Usuario"),
        )

        tokens_used = 0
        try:
            response, tokens_used = invoke_and_track(llm, [
                SystemMessage(content=prompt),
                HumanMessage(content=full_message),
            ], "troubleshooter")
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
    # 6. CONSTRUIR RESPUESTA
    # ==========================================
    severity = extract_severity(user_message + " " + result_text)

    # Extraer sugerencias del resultado
    clean_result, suggestions = extract_suggestions_from_text(result_text)

    output = WorkerOutputBuilder.troubleshooting(
        content=clean_result,
        problem_identified=f"Issue at {'Station ' + str(station_num) if station_num else 'equipment'} ({severity})",
        severity=severity,
        summary=f"Diagnosis complete - Severity: {severity}",
        confidence=0.85 if is_lab else 0.75,
    )
    output.metadata.completed_at = datetime.utcnow().isoformat()
    output.metadata.processing_time_ms = processing_time
    output.metadata.model_used = state.get("llm_model") or os.getenv("DEFAULT_MODEL", "gemini-2.0-flash")

    logger.node_end("troubleshooter_node", {"severity": severity, "is_lab": is_lab})
    events.append(event_report("troubleshooting", f"Diagnosis ready (Severity: {severity})"))

    # Limpiar contexto
    clean_context = pending_context.copy()
    clean_context.pop("user_clarification", None)
    clean_context.pop("original_query", None)

    # Sugerencias por defecto si no se extrajeron del LLM
    if not suggestions:
        suggestions = [
            "Check other stations for similar issues",
            "Review recent error logs",
            "Run a full system diagnostic"
        ]

    return {
        "worker_outputs": [output.model_dump()],
        "troubleshooting_result": output.model_dump_json(),
        "pending_context": clean_context,
        "clarification_questions": [],
        "needs_human_input": False,
        "events": events,
        "follow_up_suggestions": suggestions,
        "token_usage": tokens_used,
    }
