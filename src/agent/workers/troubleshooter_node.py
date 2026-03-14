"""Troubleshooting worker: diagnóstico técnico, integración con lab ATLAS,
preguntas estructuradas, ejecución de acciones y conocimiento base del laboratorio."""
import os
import re
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from src.agent.utils.llm_factory import get_llm, invoke_and_track
from src.agent.tools.db_tools.rag_tools import make_equipment_manual_tool
from src.agent.tools.web_search_tool import get_web_search_tool
from src.agent.tools.hardware_tools import (
    ALL_READ_TOOLS,
    XARM_READ_TOOLS,
    ABB_READ_TOOLS,
    PLC_READ_TOOLS,
    NETWORK_READ_TOOLS,
    is_mock_mode,
)

from src.agent.state import AgentState
from src.agent.helpers.skill_injector import build_equipment_context_block
from src.agent.contracts.worker_contract import WorkerOutputBuilder, create_error_output
from src.agent.contracts.question_schema_v2 import (
    QuestionBuilder,
    QuestionSet,
)
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_execute, event_report, event_error, event_narration
from src.agent.prompts.format_rules import MARKDOWN_FORMAT_RULES
from src.agent.interaction_modes import get_truth_hierarchy, get_mode_instructions

def extract_suggestions_from_text(text: str) -> tuple[str, list[str]]:
    """Extract suggestions block from LLM response text."""
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
    """Build knowledge context by matching robots, stations, and error codes in the message."""
    if not LAB_KNOWLEDGE_AVAILABLE:
        return ""
    
    context_parts = []
    msg_lower = user_message.lower()

    for robot_name in ROBOTS.keys():
        if robot_name.lower() in msg_lower:
            context_parts.append(get_robot_info(robot_name))

    for i in range(1, 7):
        if f"estacion {i}" in msg_lower or f"estación {i}" in msg_lower or f"est {i}" in msg_lower:
            context_parts.append(get_station_info(i))

    for error_code in COMMON_ERRORS.keys():
        if error_code.lower() in msg_lower:
            context_parts.append(get_error_solution(error_code))

    if not context_parts:
        context_parts.append(get_lab_knowledge_summary())
    
    return "\n\n".join(context_parts)


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
6. Generate questions in the same language as the user's message
7. Keep questions concise — no more than 2 sentences per question

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
    """Use the LLM to generate context-specific clarification questions."""
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

        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            logger.error("troubleshooter_node", f"No se encontró JSON en respuesta: {content[:200]}")
            return None
        
        json_str = content[start_idx:end_idx]
        data = json.loads(json_str)

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

LAB_TOOLS_AVAILABLE = False

MAX_DIAGNOSTIC_ITERATIONS = 8


_DIAGNOSTIC_BASE_RULES = f"""
{get_truth_hierarchy()}

## UNIVERSAL DIAGNOSTIC RULES
- Respond in the same language as the user
- Never use emojis
- Do not invent real-time status, sensor values, or tool results
- Do not fabricate sources or measurements
- If evidence is insufficient, say so explicitly — do not fill gaps with assumptions
- Cite specific values when available: error codes, IPs, response times, modes
- Prefer real observed state over interpretation
"""


TROUBLESHOOTER_PROMPT_SIMPLE = """You are a senior diagnostic engineer specializing in industrial automation.

{base_rules}

## KNOWLEDGE BASE
{{knowledge_context}}

## USER CONTEXT
{{clarification_section}}

## DOCUMENTED EVIDENCE
{{evidence_section}}

## YOUR TASK
Diagnose the problem using only the information available above and the user's message.

## RESPONSE RULES
- Start with the most likely diagnosis
- Then the reasoning briefly
- Then specific actionable next steps
- If evidence is incomplete, say what you can confirm and what you cannot
- Write like a senior engineer talking to a colleague
- Short paragraphs, bold only the most important findings
- No section headers, no formal report template
- If multiple causes are possible, rank them by probability — do not list them equally

## NEVER
- Never claim equipment status without evidence in the context above
- Never use formal report templates (### 1. UNDERSTAND, ### 2. DIAGNOSE, etc.)
- Never list every possible cause without ranking
- Never give generic advice ("restart the system") without checking specifics first

{{format_rules}}

User: {{user_name}}
""".format(base_rules=_DIAGNOSTIC_BASE_RULES)


TROUBLESHOOTER_COT_PROMPT = """You are an autonomous diagnostic agent for the ATLAS Manufacturing Laboratory.
Your job is to INVESTIGATE using real-time diagnostic tools, not to write about investigating.

{base_rules}

## LABORATORY KNOWLEDGE
{{knowledge_context}}

## USER CONTEXT
{{clarification_section}}

## DOCUMENTED EVIDENCE
{{evidence_section}}

## INVESTIGATION PROTOCOL — MANDATORY

Your FIRST action must be a tool call. Do not write explanatory text before investigating.

**Available Tools:**

*Live hardware (use FIRST):*
- `net_ping(ip)` — ping device IP, returns reachable/unreachable + latency
- `xarm_get_position(device_id)` — xArm TCP position + joint angles
- `xarm_get_full_status(device_id)` — xArm errors, temps, mode, gripper, safety zone
- `abb_get_position(device_id)` — ABB robot TCP + quaternion orientation
- `plc_list_connections()` — which PLCs are connected/disconnected
- `plc_read_input(plc_ip, byte_address)` — read PLC input byte (I area, 8 bits)
- `plc_read_output(plc_ip, byte_address)` — read PLC output byte (Q area, 8 bits)
- `plc_read_memory(plc_ip, byte_address)` — read PLC memory byte (M area, 8 bits)
- `net_exec_command(command)` — run whitelisted shell command on lab PC

*Documentation (use AFTER hardware checks):*
- `search_equipment_manual(query)` — search equipment PDF manual (ENGLISH queries only)
- `web_search_diagnostic(query)` — search internet for guides/forums (ENGLISH queries only)

Not all tools are available in every session — only use tools that are bound to you.

**Investigation strategy:**
1. FIRST: Check hardware status (ping, read position/status, check PLC I/O)
2. IF hardware shows error: search the manual for that specific error code
3. IF hardware OK but user says problem: search manual + web for the symptom
4. Combine all evidence for your diagnosis

**Execution rules:**
1. FIRST response = tool calls (hardware status checks), not text
2. If ping fails, that IS your diagnosis — don't keep searching
3. If hardware status shows a clear error, cite it and explain what it means
4. Only search manuals/web if hardware seems OK but user reports a problem
5. Maximum {{max_iterations}} tool calls total
6. NEVER repeat same tool with identical arguments

## RESPONSE FORMAT — ONLY AFTER INVESTIGATION

Once you have gathered evidence, write your diagnosis with real data from tools:
- Cite specific values from tool results (IPs, error codes, modes, response times)
- Start with the diagnosis, then supporting evidence, then recommended actions
- Short paragraphs, bold key findings
- No section headers, no templates
- Same language as user

## NEVER
- Never diagnose without calling at least one tool first
- Never claim "everything is fine" without checking
- Never repeat a tool call with the same arguments
- Never write a formal report — write like a senior technician explaining to a colleague

{{format_rules}}

User: {{user_name}}
""".format(base_rules=_DIAGNOSTIC_BASE_RULES)


TROUBLESHOOT_AUTONOMOUS_PROMPT = """You are an autonomous diagnostic agent for industrial equipment.
Your job is to CHECK REAL HARDWARE STATUS FIRST, then investigate deeper only if needed.

{base_rules}

## EQUIPMENT
{{equipment_context}}

IMPORTANT: Focus your investigation on the equipment IP shown above. If other devices appear in tool results, mention them only if they're directly relevant to the problem.

## PROBLEM
{{problem_description}}

## INVESTIGATION FLOW — FOLLOW THIS ORDER

**Phase 1: Check hardware status (ALWAYS do this first)**
1. `net_ping(ip)` — verify the device is reachable on the network
2. Read current device status:
   - xArm: `xarm_get_full_status(device_id)` → check error_code, warning_code, mode, temperatures
   - ABB: `abb_get_position(device_id)` → check if robot responds, current TCP position
   - PLC: `plc_list_connections()` then `plc_read_output(ip, byte)` / `plc_read_input(ip, byte)` → check I/O state
3. Report what you find — actual error codes, connection status, current state

**Phase 2: If a tool call FAILS or returns a SPECIFIC error**
- DO NOT skip it. The failure itself is diagnostic information.
- Log what failed and what error was returned.
- Classify the error:
  - **Timeout / unreachable** → report connectivity issue, suggest checking cables/IP config. No manual search needed.
  - **Specific device error** (e.g. "function refused by CPU", "error code 22", "safety limit triggered", "motion abnormal"):
    1. FIRST call `search_equipment_manual(query)` with the EXACT error message in ENGLISH (e.g. "S7-1200 function refused by CPU", "xArm error code 22 motion abnormal")
    2. If manual returns nothing, try `web_search_diagnostic(query)` with the error + device model
    3. THEN give your diagnosis using what the manual/web says — cite the source
  - **Unknown / generic error** → try alternative approaches (different IP, different tool, ping first)
- IMPORTANT: Do NOT guess what an error means. If a device returns a specific error string, SEARCH for it before explaining it to the user.

**Phase 3: If hardware checks PASS but user says something is wrong**
Only now search documentation:
1. `search_equipment_manual(query)` — search for the specific symptom in the manual (ENGLISH queries)
2. `web_search_diagnostic(query)` — search online for community solutions (ENGLISH queries)
3. Cross-reference manual/web findings with the live status you read in Phase 1

**Available Tools:**

*Live hardware (use FIRST):*
- `net_ping(ip)` — ping device IP, returns reachable/unreachable + latency
- `xarm_get_position(device_id)` — xArm TCP position + joint angles
- `xarm_get_full_status(device_id)` — xArm errors, temps, mode, gripper, safety zone
- `abb_get_position(device_id)` — ABB robot TCP + quaternion orientation
- `plc_list_connections()` — which PLCs are connected/disconnected
- `plc_read_input(plc_ip, byte_address)` — read PLC input byte (I area, 8 bits)
- `plc_read_output(plc_ip, byte_address)` — read PLC output byte (Q area, 8 bits)
- `plc_read_memory(plc_ip, byte_address)` — read PLC memory byte (M area, 8 bits)
- `net_exec_command(command)` — run whitelisted shell command on lab PC

*Documentation (use AFTER hardware checks):*
- `search_equipment_manual(query)` — search equipment PDF manual (ENGLISH queries only)
- `web_search_diagnostic(query)` — search internet for guides/forums (ENGLISH queries only)

Not all tools are available in every session — only use tools that are bound to you.

**Query tips for manual/web search:**
- ALL search queries MUST be in ENGLISH regardless of user language
- Extract the specific symptom + component: "S7-1200 communication timeout", "xArm error code 22"
- If first search returns nothing, try broader terms or the error code directly
- Try at least 2 different queries before giving up on the manual

**Execution rules:**
1. FIRST response = tool calls (hardware status checks), not text
2. If ping fails, that IS your diagnosis — don't keep searching
3. If hardware status shows a clear error, cite it and explain what it means
4. Only search manuals/web if hardware seems OK but user reports a problem
5. Maximum {{max_iterations}} tool calls total
6. NEVER repeat same tool with identical arguments

## RESPONSE FORMAT — AFTER INVESTIGATION

The user has already seen your investigation steps in real-time. Your final response is ONLY the diagnosis and next step.

FORMAT your response directly:
- Use **bold** for key findings (error codes, IPs, response times)
- Use ==highlighted text== for the main conclusion (one sentence)
- Write 2-4 short sentences max
- Same language as the user

Example:
==La PLC en 192.168.1.101 responde al ping (1.2ms) pero rechaza lecturas S7comm.== Probablemente el **acceso PUT/GET** no está habilitado en la configuración de protección. Abre TIA Portal, ve a **Propiedades del PLC > Protección y acceso**, y activa "Permitir acceso PUT/GET".

Do NOT repeat what the tools found — the user already saw that.
Do NOT use numbered lists, bullet points, or section headers.
Do NOT prefix your response with [troubleshooting]: or any label.
Maximum 4 sentences.

## NEVER
- Never skip Phase 1 (hardware checks) and go straight to manuals
- Never ignore a tool failure — it IS diagnostic information
- Never claim a device is OK without actually checking it
- Never give generic advice when you have real data available
- Never claim the manual says something without actually searching it
- Never report a hardware error without searching the manual for what it means (unless it's a simple timeout/unreachable)

User: {{user_name}}
""".format(base_rules=_DIAGNOSTIC_BASE_RULES)


def is_lab_related(message: str) -> bool:
    """Check if the message references ATLAS lab equipment or terminology."""
    msg = message.lower()

    lab_keywords = [
        "estación", "estacion", "station",
        "laboratorio", "lab", "atlas",
        "plc-st", "cobot-st", "door-sensor",
        "estación 1", "estación 2", "estación 3", "estación 4", "estación 5", "estación 6",
        "est1", "est2", "est3", "est4", "est5", "est6",
        "puerta", "door", "interlock",
        "rutina", "routine",
        "start cobot", "stop cobot", "lab status",
        "checar", "verificar estado",
        "alfredo", "ur5", "ur10", "universal robots",
        "ensamblaje", "soldadura", "inspección", "inspeccion", "testing", "empaque",
        "profinet", "tia portal", "polyscope", "teach pendant",
        "oee", "tiempo de ciclo", "celda",
    ]

    if LAB_KNOWLEDGE_AVAILABLE:
        try:
            for robot_name in ROBOTS.keys():
                if robot_name.lower() not in lab_keywords:
                    lab_keywords.append(robot_name.lower())
        except:
            pass
    
    return any(kw in msg for kw in lab_keywords)


def detect_station_number(message: str) -> Optional[int]:
    """Extract station number from message, if mentioned."""
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
    """Detect equipment type (plc/cobot/sensor) from message keywords."""
    msg = message.lower()
    
    if any(kw in msg for kw in ["plc", "s7", "siemens", "allen"]):
        return "plc"
    if any(kw in msg for kw in ["cobot", "robot", "ur5", "ur10", "fanuc", "brazo", "alfredo"]):
        return "cobot"
    if any(kw in msg for kw in ["sensor", "puerta", "door", "proximidad", "e-stop"]):
        return "sensor"
    
    return None


def detect_action_request(message: str, pending_context: dict = None) -> Optional[Dict]:
    """Detect if the user wants to execute a lab action (start/stop cobot, reset, etc.)."""
    msg = message.lower()

    start_phrases = [
        "start cobot", "start cobot", "execute routine", "start cobot", "run routine",
        "comienza rutina", "comenzar rutina", "inicia rutina", "iniciar rutina",
        "arranca rutina", "corre rutina", "run routine", "start routine",
        "enciende cobot", "activa cobot", "activa rutina"
    ]
    if any(phrase in msg for phrase in start_phrases):
        station = detect_station_number(message)
        mode = 1
        if "rutina 2" in msg or "routine 2" in msg or "modo 2" in msg:
            mode = 2
        elif "rutina 3" in msg or "routine 3" in msg or "modo 3" in msg:
            mode = 3
        elif "rutina 4" in msg or "routine 4" in msg or "modo 4" in msg:
            mode = 4
        return {"action": "start_cobot", "station": station, "mode": mode}

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

    reset_phrases = [
        "reset lab", "resetear lab", "reiniciar lab", "reinicia el lab",
        "reset completo", "reinicio completo", "restaurar lab",
        "pon todo en orden", "arregla todo", "fix everything"
    ]
    if any(phrase in msg for phrase in reset_phrases):
        return {"action": "reset_lab", "needs_confirmation": True}

    door_phrases = [
        "cierra las puertas", "cerrar puertas", "close doors",
        "cierra todas las puertas", "asegura las puertas"
    ]
    if any(phrase in msg for phrase in door_phrases):
        return {"action": "close_doors"}

    reconnect_phrases = [
        "reconectar plc", "reconecta la plc", "reconnect plc",
        "reiniciar plc", "reinicia la plc"
    ]
    if any(phrase in msg for phrase in reconnect_phrases):
        station = detect_station_number(message)
        return {"action": "reconnect_plc", "station": station}

    resolve_phrases = [
        "resolver errores", "resuelve los errores", "limpia los errores",
        "clear errors", "fix errors", "arregla los errores"
    ]
    if any(phrase in msg for phrase in resolve_phrases):
        station = detect_station_number(message)
        return {"action": "resolve_errors", "station": station}

    fix_phrases = [
        "intenta arreglarlo", "arreglalo", "arréglalo", "fix it",
        "intenta solucionarlo", "soluciona", "repara", "repáralo",
        "puedes arreglarlo", "arregla eso", "soluciona eso",
        "hazlo", "procede", "adelante", "sí, arréglalo", "si, arreglalo",
        "dale", "ok arreglalo", "ok, arreglalo"
    ]
    if any(phrase in msg for phrase in fix_phrases):
        pending_context = pending_context or {}
        has_repair_context = (
            pending_context.get("awaiting_repair_confirmation")
            or pending_context.get("hitl", {}).get("type") == "repair_confirmation"
        )
        station = detect_station_number(message)
        if station or has_repair_context:
            return {"action": "auto_fix", "station": station, "needs_confirmation": False}
        # Casual phrases like "dale" shouldn't trigger auto_fix without context
        return None

    status_phrases = [
        "lab status", "resumen del lab", "ver laboratorio", "lab status",
        "estado laboratorio", "status lab", "como está el lab", "como esta el lab",
        "estado de las estaciones", "ver estaciones", "mostrar estaciones"
    ]
    if any(phrase in msg for phrase in status_phrases):
        return {"action": "show_lab_status"}
    
    return None


def detect_query_request(message: str) -> Optional[Dict]:
    """Detect if the user is querying lab status (errors, PLCs, cobots, doors, etc.)."""
    msg = message.lower()

    error_queries = [
        "errores activos", "hay errores", "que errores", "cuantos errores",
        "estaciones con errores", "problemas activos", "fallas activas",
        "hay algun error", "hay algún error", "mas errores", "más errores",
        "otros errores", "lista de errores"
    ]
    if any(phrase in msg for phrase in error_queries):
        return {"query": "active_errors"}

    plc_queries = [
        "estado de las plc", "plcs conectadas", "plc desconectada",
        "que plc", "cuales plc", "lista de plc", "plcs del lab"
    ]
    if any(phrase in msg for phrase in plc_queries):
        return {"query": "plc_status"}

    cobot_queries = [
        "estado de los cobot", "cobots activos", "que cobot",
        "cobots ejecutando", "cobots en rutina", "lista de cobot"
    ]
    if any(phrase in msg for phrase in cobot_queries):
        return {"query": "cobot_status"}

    door_queries = [
        "puertas abiertas", "puertas cerradas", "estado de las puertas",
        "sensores de puerta", "alguna puerta abierta", "doors"
    ]
    if any(phrase in msg for phrase in door_queries):
        return {"query": "door_status"}

    station = detect_station_number(message)
    if station and any(word in msg for word in ["estado", "status", "como está", "como esta", "info", "detalles"]):
        return {"query": "station_details", "station": station}

    if is_lab_related(message) and any(word in msg for word in ["hay", "cuantos", "cuántos", "cuales", "cuáles", "lista", "ver", "mostrar"]):
        return {"query": "lab_overview"}
    
    return None


def get_last_user_message(state: AgentState) -> str:
    """Return the last user message from state."""
    for m in reversed(state.get("messages", []) or []):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
        if isinstance(m, dict) and m.get("role") in ("human", "user"):
            return (m.get("content") or "").strip()
    return ""


def get_evidence_from_context(state: AgentState) -> str:
    """Extract evidence from pending_context or prior research worker output."""
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
    """Classify problem severity from content keywords."""
    content_lower = content.lower()
    if any(kw in content_lower for kw in ["crítico", "producción parada", "urgente", "emergency"]):
        return "critical"
    elif any(kw in content_lower for kw in ["error", "no funciona", "bloqueado", "stop"]):
        return "high"
    elif any(kw in content_lower for kw in ["lento", "intermitente", "warning"]):
        return "medium"
    return "low"



def _get_stream_cb(state) -> object:
    """Resolve stream callback from state's session ID."""
    session_id = state.get("_stream_session_id")
    if not session_id:
        return None
    try:
        from api_server import get_stream_callback
        return get_stream_callback(session_id)
    except (ImportError, Exception):
        return None


@dataclass
class TroubleshooterContext:
    """Normalized context extracted from AgentState for all handlers."""
    state: AgentState
    user_message: str
    clarification_text: str
    already_clarified: bool
    pending_context: Dict[str, Any]
    events: list
    start_time: datetime

    is_lab: bool
    station_num: Optional[int]
    equipment: Optional[str]
    action_request: Optional[Dict]
    query_request: Optional[Dict]
    is_command: bool
    interaction_mode: str
    user_name: str
    intent_analysis: Dict[str, Any]


def _build_context(state: AgentState) -> Optional[TroubleshooterContext]:
    """Extract and normalize all context from AgentState. Returns None if no user message."""
    start_time = datetime.utcnow()
    logger.node_start("troubleshooter_node", {})
    events = [event_execute("troubleshooting", "Analizando problema...")]

    pending_context = state.get("pending_context", {}) or {}
    user_clarification = pending_context.get("user_clarification", "")
    original_query = pending_context.get("original_query", "")
    already_clarified = bool(user_clarification or pending_context.get("_hitl_consumed"))

    if user_clarification and original_query:
        user_message = original_query
        clarification_text = user_clarification
        logger.info("troubleshooter_node", f"Usando original_query + clarificación: '{original_query[:60]}...' + '{clarification_text[:60]}...'")
    else:
        user_message = get_last_user_message(state)
        clarification_text = ""

        if not user_message:
            return None

    intent_analysis = state.get("intent_analysis", {})

    if intent_analysis:
        entities = intent_analysis.get("entities", {})
        station_num = entities.get("station") or detect_station_number(user_message)
        equipment = entities.get("equipment") or detect_equipment_type(user_message)
        detected_action = intent_analysis.get("action")
        intent_type = intent_analysis.get("intent")

        action_request = None
        query_request = None

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
                # search_docs is research, not troubleshooting; provide lab data instead
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
                query_request = {"query": "station_details", "station": station_num}
            else:
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
                    query_request = detect_query_request(user_message)
                    if not query_request and is_lab_related(user_message):
                        query_request = {"query": "lab_overview"}

        elif intent_type == "command" and detected_action:
            action_request = {
                "action": detected_action,
                "station": station_num,
                "mode": entities.get("routine", 1) if detected_action == "start_cobot" else 0
            }

        is_lab = True if (station_num or equipment or detected_action) else is_lab_related(user_message)

        logger.info("troubleshooter_node",
            f"Usando intent_analysis: intent={intent_type} action={detected_action} station={station_num}")
    else:
        is_lab = is_lab_related(user_message) or is_lab_related(clarification_text)
        station_num = detect_station_number(user_message) or detect_station_number(clarification_text)
        equipment = detect_equipment_type(user_message) or detect_equipment_type(clarification_text)
        action_request = detect_action_request(user_message, pending_context)
        query_request = detect_query_request(user_message)

    logger.info("troubleshooter_node", f"Mensaje: '{user_message[:50]}...' | is_lab={is_lab} | station={station_num} | action={action_request} | query={query_request}")

    command_keywords = [
        "iniciar", "arrancar", "ejecutar", "comienza", "comenzar", "inicia",
        "arranca", "corre", "enciende", "activa", "start", "run",
        "parar", "detener", "stop", "para", "apagar", "apaga", "deten",
        "cerrar", "cierra", "abrir", "abre", "reset", "reinicia",
        "reconectar", "reconecta", "resolver", "arreglar", "arregla"
    ]
    is_command = any(cmd in user_message.lower() for cmd in command_keywords)

    return TroubleshooterContext(
        state=state,
        user_message=user_message,
        clarification_text=clarification_text,
        already_clarified=already_clarified,
        pending_context=pending_context,
        events=events,
        start_time=start_time,
        is_lab=is_lab,
        station_num=station_num,
        equipment=equipment,
        action_request=action_request,
        query_request=query_request,
        is_command=is_command,
        interaction_mode=state.get("interaction_mode", "chat").lower(),
        user_name=state.get("user_name", "Usuario"),
        intent_analysis=intent_analysis,
    )


_AFFIRMATIVE = frozenset({
    "yes", "sí", "si", "confirmo", "confirmar", "dale", "ok", "okay",
    "adelante", "procede", "hazlo", "ejecutar", "ejecuta",
    "1", "confirm", "iniciar", "inicia", "repair", "fix",
    "arreglalo", "arréglalo", "claro", "sale", "va", "venga",
})

_NEGATIVE = frozenset({
    "no", "cancelar", "cancel", "2", "nope", "nah",
    "negativo", "dejalo", "déjalo", "skip", "omitir",
})

_PUNCT_RE = re.compile(r'[,\.!?¡¿;:\-–—"\'""\'\'()]+')


def _normalize_response(text: str) -> list:
    """Normalize user response: lowercase, strip punctuation, tokenize."""
    text = text.lower().strip()
    text = _PUNCT_RE.sub(' ', text)
    return [w for w in text.split() if w]


def _is_affirmative(text: str) -> bool:
    words = _normalize_response(text)
    return any(w in _AFFIRMATIVE for w in words)


def _is_negative(text: str) -> bool:
    words = _normalize_response(text)
    return any(w in _NEGATIVE for w in words)


_ACTIONS_REQUIRE_CONFIRMATION = frozenset([
    "start_cobot",
    "stop_cobot",
    "reset_lab",
])

_ACTIONS_DIRECT = frozenset([
    "show_lab_status",
    "close_doors",
    "reconnect_plc",
    "resolve_errors",
    "auto_fix",
])

_ACTIONS_READ_ONLY = frozenset([
    "show_lab_status",
])


def _validate_action_safety(action: dict, pending_context: dict = None) -> tuple:
    """Returns (is_safe, reason)."""
    action_type = action.get("action", "")
    pending_context = pending_context or {}

    if action_type in _ACTIONS_REQUIRE_CONFIRMATION:
        if not action.get("_confirmed"):
            return False, f"Action '{action_type}' requires explicit user confirmation"

    if action_type not in _ACTIONS_DIRECT and action_type not in _ACTIONS_REQUIRE_CONFIRMATION:
        return False, f"Unknown action type: '{action_type}'"

    if action_type == "auto_fix":
        has_repair_context = (
            pending_context.get("awaiting_repair_confirmation")
            or (pending_context.get("hitl") or {}).get("type") == "repair_confirmation"
        )
        if not has_repair_context and not action.get("station"):
            return False, "auto_fix requires repair context or explicit station"

    return True, "ok"


def _read_hitl_type(pending: dict) -> str:
    """Read HITL type from structured or legacy format."""
    hitl = pending.get("hitl", {})
    if isinstance(hitl, dict) and hitl.get("type"):
        if not hitl.get("consumed"):
            return hitl["type"]
        return ""
    if pending.get("awaiting_cobot_confirmation"):
        return "cobot_confirmation"
    if pending.get("awaiting_repair_confirmation"):
        return "repair_confirmation"
    if pending.get("pending_health_query"):
        return "health_query"
    return ""


def _compute_diagnostic_confidence(
    called_tools: set,
    has_evidence: bool,
    is_lab: bool,
    has_clarification: bool,
) -> float:
    """Compute confidence based on observable signals, not LLM self-assessment."""
    base = 0.3

    if len(called_tools) >= 3:
        base += 0.30
    elif len(called_tools) >= 1:
        base += 0.15

    if has_evidence:
        base += 0.15

    if is_lab and called_tools:
        base += 0.10

    if has_clarification:
        base += 0.05

    return round(min(base, 1.0), 2)


def _return_needs_context(ctx: TroubleshooterContext, output: Any, payload: list,
                          extra_pending: Optional[Dict] = None,
                          hitl_type: str = "worker_clarification",
                          hitl_reason: str = "worker_clarification") -> Dict[str, Any]:
    """Unified HITL return with structured hitl namespace."""
    output_dict = output.model_dump()
    output_dict["status"] = "needs_context"
    output_dict["clarification_questions"] = payload

    updated_context = ctx.pending_context.copy()
    if extra_pending:
        updated_context.update(extra_pending)

    updated_context["hitl"] = {"type": hitl_type, "consumed": False}

    return {
        "worker_outputs": [output_dict],
        "troubleshooting_result": json.dumps(output_dict),
        "needs_human_input": True,
        "clarification_questions": payload,
        "pending_context": updated_context,
        "events": ctx.events,
        "human_input_reason": hitl_reason,
    }


def _return_result(ctx: TroubleshooterContext, output: Any,
                   suggestions: Optional[list] = None,
                   clean_pending: bool = False,
                   tokens_used: int = 0) -> Dict[str, Any]:
    """Unified normal return."""
    result = {
        "worker_outputs": [output.model_dump()],
        "troubleshooting_result": output.model_dump_json(),
        "events": ctx.events,
    }
    if suggestions:
        result["follow_up_suggestions"] = suggestions
    if clean_pending:
        result.update({
            "pending_context": {},
            "clarification_questions": [],
            "needs_human_input": False,
        })
    if tokens_used:
        result["token_usage"] = tokens_used
    return result


def _return_error(ctx: TroubleshooterContext, code: str, message: str) -> Dict[str, Any]:
    """Unified error return."""
    error_output = create_error_output("troubleshooting", code, message)
    return {
        "worker_outputs": [error_output.model_dump()],
        "troubleshooting_result": error_output.model_dump_json(),
        "events": ctx.events,
    }

def _handle_unrecognized_command(ctx: TroubleshooterContext) -> Dict[str, Any]:
    logger.info("troubleshooter_node", f"[HANDLER] unrecognized_command: '{ctx.user_message[:50]}'")

    # Route to autonomous diagnosis when we have enough context
    pending = ctx.pending_context or {}
    has_equipment = bool(pending.get("equipment_id"))
    has_troubleshoot_intent = (ctx.intent_analysis or {}).get("intent") == "troubleshoot"

    if has_equipment or has_troubleshoot_intent:
        logger.info(
            "troubleshooter_node",
            f"[HANDLER] unrecognized_command → routing to autonomous diagnosis "
            f"(equipment={has_equipment}, troubleshoot_intent={has_troubleshoot_intent})"
        )
        return _run_autonomous_diagnosis(ctx)

    msg_lower = ctx.user_message.lower()
    suggestions = []

    if "rutina" in msg_lower or "cobot" in msg_lower:
        if ctx.station_num:
            suggestions.append(f"• 'Inicia rutina 1 en estación {ctx.station_num}'")
            suggestions.append(f"• 'Para el cobot de estación {ctx.station_num}'")
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
        "events": ctx.events,
    }


def _try_request_clarification(ctx: TroubleshooterContext) -> Optional[Dict[str, Any]]:
    """Returns a HITL dict if clarification questions were generated, else None."""
    logger.info("troubleshooter_node", f"[HANDLER] try_clarification: msg_len={len(ctx.user_message)} is_lab={ctx.is_lab}")
    # Troubleshoot mode already has equipment context
    if ctx.interaction_mode == "troubleshoot":
        return None

    question_set = None

    # Only generate dynamic questions for short/vague messages
    if len(ctx.user_message.split()) < 20:
        ctx.events.append(event_report("troubleshooting", " Generando preguntas contextuales..."))

        try:
            question_llm = get_llm(ctx.state, temperature=0.3)
            question_set = generate_dynamic_questions(
                user_message=ctx.user_message,
                lab_context="",
                llm=question_llm
            )

            if question_set:
                logger.info("troubleshooter_node", f"Generadas {len(question_set.questions)} preguntas dinámicas")
        except Exception as e:
            logger.warning("troubleshooter_node", f"Error generando preguntas dinámicas: {e}")
            question_set = None

    # Generic fallback for very short messages
    if not question_set and len(ctx.user_message.split()) < 10:
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

        ctx.events.append(event_report("troubleshooting", f"📋 {len(question_set.questions)} preguntas"))

        return _return_needs_context(ctx, output, payload, extra_pending={
            "original_query": ctx.user_message,
            "is_lab_related": ctx.is_lab,
            "detected_station": ctx.station_num,
            "detected_equipment": ctx.equipment,
            "question_set": question_set.model_dump_json(),
            "current_worker": "troubleshooting",
        })

    return None


def _summarize_tool_result(tool_name: str, tool_args: dict, result) -> str:
    """Generate a short human-readable summary of a tool result for real-time streaming."""
    try:
        data = result
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                data = parsed.get("data", parsed)
            except (json.JSONDecodeError, TypeError):
                data = result

        if tool_name == "net_ping":
            ip = tool_args.get("ip", "?")
            if isinstance(data, dict):
                if data.get("reachable"):
                    avg = data.get("avg_ms", "?")
                    loss = data.get("packet_loss", 0)
                    return f"Ping a {ip}: responde en {avg}ms, {loss}% pérdida"
                else:
                    return f"Ping a {ip}: no responde, timeout"
            if isinstance(data, str) and "error" in data.lower():
                return f"Ping a {ip}: error: {data[:100]}"

        elif tool_name == "plc_list_connections":
            if isinstance(data, dict):
                connected = data.get("connected_count", 0)
                total = data.get("total_count", 0)
                return f"PLCs: {connected}/{total} conectadas"

        elif tool_name in ("plc_read_input", "plc_read_output", "plc_read_memory"):
            ip = tool_args.get("plc_ip", "?")
            byte_addr = tool_args.get("byte_address", "?")
            area = tool_args.get("area", tool_name.split("_")[-1])
            if isinstance(data, dict) and "bits" in data:
                active = [k for k, v in data["bits"].items() if v]
                if active:
                    return f"PLC {ip} {area} byte {byte_addr}: bits activos = {', '.join(active)}"
                else:
                    return f"PLC {ip} {area} byte {byte_addr}: todos los bits en 0"

        elif tool_name in ("xarm_get_position", "xarm_get_full_status"):
            if isinstance(data, dict):
                tcp = data.get("tcp", {})
                if tcp:
                    x = tcp.get('x', '?')
                    y = tcp.get('y', '?')
                    z = tcp.get('z', '?')
                    x_s = f"{x:.1f}" if isinstance(x, (int, float)) else str(x)
                    y_s = f"{y:.1f}" if isinstance(y, (int, float)) else str(y)
                    z_s = f"{z:.1f}" if isinstance(z, (int, float)) else str(z)
                    return f"xArm: X={x_s} Y={y_s} Z={z_s}, state={data.get('state', '?')}"
                error = data.get("error_code", 0)
                if error:
                    return f"xArm: error code {error}, warning {data.get('warning_code', 0)}"

        elif tool_name == "abb_get_position":
            if isinstance(data, dict):
                pos = data.get("position", {})
                if pos:
                    x = pos.get('x', '?')
                    y = pos.get('y', '?')
                    z = pos.get('z', '?')
                    x_s = f"{x:.1f}" if isinstance(x, (int, float)) else str(x)
                    y_s = f"{y:.1f}" if isinstance(y, (int, float)) else str(y)
                    z_s = f"{z:.1f}" if isinstance(z, (int, float)) else str(z)
                    return f"ABB: X={x_s} Y={y_s} Z={z_s}"

        elif tool_name == "search_equipment_manual":
            if isinstance(data, str):
                if "no relevant" in data.lower() or "no se encontr" in data.lower():
                    query = tool_args.get("query", "?")
                    return f"Manual: nada encontrado para '{query}'"
                else:
                    sections = data.count("###")
                    return f"Manual: {sections} secciones relevantes encontradas"

        elif tool_name == "web_search_diagnostic":
            if isinstance(data, str):
                if "error" in data.lower():
                    return f"Web search: error en búsqueda"
                results = data.count("**Result")
                if results:
                    return f"Web: {results} resultados encontrados"

        if isinstance(data, dict) and data.get("status") == "error":
            error_msg = data.get("error", "error desconocido")
            return f"{tool_name}: error: {error_msg[:80]}"

    except Exception:
        pass

    return ""


def _run_tool_agent_loop(
    llm_with_tools,
    tool_map: Dict[str, Any],
    messages_chain: list,
    max_iterations: int = MAX_DIAGNOSTIC_ITERATIONS,
    events: list = None,
    stream_cb=None,
) -> tuple:
    """ReAct tool-calling loop. Mutates messages_chain in-place. Returns (tokens_used, called_tools_set)."""
    tokens_used = 0
    called_tools = set()
    consecutive_failures = 0

    for iteration in range(max_iterations):
        try:
            response, tokens = invoke_and_track(llm_with_tools, messages_chain, "troubleshooter")
            tokens_used += tokens
        except Exception as e:
            logger.error("troubleshooter_node", f"LLM error at iteration {iteration}: {e}")
            break

        has_tools = bool(response.tool_calls)
        logger.info("troubleshooter_node",
                    f"Iteration {iteration}: has_tool_calls={has_tools}, "
                    f"content_len={len(response.content or '')}, "
                    f"tool_calls={[tc['name'] for tc in (response.tool_calls or [])]}")

        messages_chain.append(response)

        if response.content:
            thinking_text = response.content.strip()
            if thinking_text:
                # Only narrate short texts; long ones are final analysis
                if len(thinking_text) < 200:
                    if events is not None:
                        events.append(event_narration("troubleshooting", thinking_text, phase="thinking"))
                if stream_cb:
                    stream_cb({"type": "partial", "content": thinking_text})

        # Force tool usage on first iteration
        if iteration == 0 and not response.tool_calls and response.content:
            logger.warning("troubleshooter_node",
                           "First iteration: no tool calls. Re-prompting to force investigation.")
            messages_chain.append(HumanMessage(
                content="You must investigate using your tools before diagnosing. "
                        "Call a diagnostic tool now. Do not write more text, use a tool."
            ))
            continue

        if not response.tool_calls:
            logger.info("troubleshooter_node",
                        f"Investigation complete after {iteration + 1} iterations")
            break

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc.get("id", f"call_{iteration}_{tool_name}")

            args_key = json.dumps(tool_args, sort_keys=True)
            call_key = (tool_name, args_key)

            if call_key in called_tools:
                logger.info("troubleshooter_node", f"Duplicate tool call: {tool_name}({tool_args})")
                messages_chain.append(ToolMessage(
                    content="You already called this tool with the same arguments. "
                            "Use a DIFFERENT tool or different arguments, or present your diagnosis.",
                    tool_call_id=tool_id,
                ))
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    messages_chain.append(ToolMessage(
                        content="STOP: Too many repeated calls. Present your diagnosis with the data you have.",
                        tool_call_id=f"{tool_id}_stop",
                    ))
                continue

            called_tools.add(call_key)

            if tool_name == "search_equipment_manual":
                query = tool_args.get("query", "")
                if events is not None:
                    events.append(event_narration("troubleshooting", f"Searching manual: \"{query}\"", phase="tool"))
                if stream_cb:
                    stream_cb({"type": "tool_status", "tool": "search_equipment_manual", "status": "executing", "content": f"Searching manual: \"{query}\""})
            elif tool_name == "web_search_diagnostic":
                query = tool_args.get("query", "")
                if events is not None:
                    events.append(event_narration("troubleshooting", f"Searching online: \"{query}\"", phase="tool"))
                if stream_cb:
                    stream_cb({"type": "tool_status", "tool": "web_search_diagnostic", "status": "executing", "content": f"Searching online: \"{query}\""})
            elif tool_name.startswith("xarm_") or tool_name.startswith("abb_"):
                if events is not None:
                    events.append(event_narration("troubleshooting", f"Checking robot status: {tool_name}", phase="tool"))
                if stream_cb:
                    stream_cb({"type": "tool_status", "tool": tool_name, "status": "executing", "content": f"Reading robot: {tool_name}"})
            elif tool_name.startswith("plc_"):
                detail = tool_args.get("plc_ip", "")
                if events is not None:
                    events.append(event_narration("troubleshooting", f"Reading PLC {detail}", phase="tool"))
                if stream_cb:
                    stream_cb({"type": "tool_status", "tool": tool_name, "status": "executing", "content": f"Reading PLC: {tool_name} {detail}"})
            elif tool_name == "net_ping":
                ip = tool_args.get("ip", "")
                if events is not None:
                    events.append(event_narration("troubleshooting", f"Pinging {ip}...", phase="tool"))
                if stream_cb:
                    stream_cb({"type": "tool_status", "tool": "net_ping", "status": "executing", "content": f"Pinging {ip}..."})
            elif tool_name == "net_exec_command":
                cmd = tool_args.get("command", "")
                if events is not None:
                    events.append(event_narration("troubleshooting", f"Running: {cmd}", phase="tool"))
                if stream_cb:
                    stream_cb({"type": "tool_status", "tool": "net_exec_command", "status": "executing", "content": f"Executing: {cmd}"})
            else:
                if events is not None:
                    events.append(event_execute("troubleshooting", f"Calling {tool_name}..."))
                if stream_cb:
                    stream_cb({"type": "tool_status", "tool": tool_name, "status": "executing", "content": f"Executing: {tool_name}"})

            if tool_name in tool_map:
                try:
                    result = tool_map[tool_name].invoke(tool_args)
                    consecutive_failures = 0

                    result_summary = _summarize_tool_result(tool_name, tool_args, result)
                    if result_summary and stream_cb:
                        stream_cb({"type": "partial", "content": result_summary})
                    if result_summary and events is not None:
                        events.append(event_narration("troubleshooting", result_summary, phase="result"))

                    if stream_cb:
                        stream_cb({"type": "tool_status", "tool": tool_name, "status": "completed", "content": "Found relevant sections" if tool_name == "search_equipment_manual" else str(result)[:200]})

                except Exception as e:
                    logger.error("troubleshooter_node", f"Tool {tool_name} failed: {e}")
                    result = f"Error: {str(e)}"
                    consecutive_failures += 1
            else:
                result = f"Unknown tool: {tool_name}"
                consecutive_failures += 1

            # Retry RAG with different keywords on empty results
            if tool_name == "search_equipment_manual":
                result_str = str(result)
                no_results = (
                    "no relevant" in result_str.lower()
                    or "no se encontr" in result_str.lower()
                    or "no results" in result_str.lower()
                    or len(result_str.strip()) < 80
                )
                if no_results:
                    prev_query = tool_args.get("query", "")
                    logger.info("troubleshooter_node",
                                f"Manual search returned empty for '{prev_query}', forcing retry with different keywords")
                    messages_chain.append(ToolMessage(
                        content=(
                            "The manual search returned no relevant results for that query. "
                            "Try again with DIFFERENT technical keywords IN ENGLISH. "
                            "Focus on the specific symptom, error code, or component name. "
                            f"Previous failed query: '{prev_query}'. "
                            "Try more specific terms like: "
                            "'PROFINET communication error', 'CPU download failure', "
                            "'IP address configuration', 'TIA Portal connection timeout', "
                            "'S7-1200 network diagnostic', 'LED status indicators', "
                            "'communication module settings', 'firmware update procedure'. "
                            "Do NOT repeat the same query."
                        ),
                        tool_call_id=tool_id,
                    ))
                    called_tools.discard(call_key)
                    continue

            messages_chain.append(ToolMessage(
                content=str(result) if not isinstance(result, str) else result,
                tool_call_id=tool_id,
            ))

        if consecutive_failures >= 3:
            logger.warning("troubleshooter_node", "3+ consecutive failures, stopping tool loop")
            break

    return tokens_used, called_tools


def _select_hardware_tools(equipment_type: str = "") -> list:
    """Select read-only hardware tools based on equipment type."""
    tools = []
    tools.extend(NETWORK_READ_TOOLS)

    type_lower = equipment_type.lower()
    if type_lower == "xarm":
        tools.extend(XARM_READ_TOOLS)
    elif type_lower == "abb":
        tools.extend(ABB_READ_TOOLS)
    elif type_lower == "plc":
        tools.extend(PLC_READ_TOOLS)
    else:
        tools.extend(ALL_READ_TOOLS)

    return tools


def _run_autonomous_diagnosis(ctx: TroubleshooterContext) -> Dict[str, Any]:
    _msg_clean = ctx.user_message.strip().lower()
    if "[equipment context]" in _msg_clean:
        _parts = ctx.user_message.split("[Problem]", 1)
        _msg_clean = _parts[-1].strip().lower() if len(_parts) > 1 else _msg_clean

    _greetings = {"hola", "hello", "hi", "hey", "buenas", "buenos dias", "buenos días",
                  "buenas tardes", "que tal", "qué tal"}
    if _msg_clean in _greetings or (len(_msg_clean) < 15 and any(_msg_clean.startswith(g) for g in _greetings)):
        eq_name = ctx.pending_context.get("equipment_name", "el equipo")
        greeting_response = f"¡Hola! Estoy listo para diagnosticar {eq_name}. ¿Qué problema estás observando?"

        output = WorkerOutputBuilder.troubleshooting(
            content=greeting_response,
            problem_identified="Greeting, awaiting problem description",
            severity="low",
            summary="User greeted, awaiting problem description",
            confidence=1.0,
        )
        return _return_result(ctx, output, suggestions=[
            "Describe the specific error or symptom",
            "Check current device status",
            "Search the manual for a topic",
        ])

    evidence_text = get_evidence_from_context(ctx.state)
    clarification_section = ctx.clarification_text if ctx.clarification_text else "No hay información adicional."

    try:
        llm = get_llm(ctx.state, temperature=0.3)
    except Exception as e:
        return _return_error(ctx, "LLM_INIT_ERROR", str(e))

    knowledge_context = ""
    if LAB_KNOWLEDGE_AVAILABLE:
        try:
            knowledge_context = get_knowledge_context(ctx.user_message)
        except Exception as e:
            logger.warning("troubleshooter_node", f"Error obteniendo conocimiento: {e}")

    full_message = ctx.user_message
    if ctx.clarification_text:
        full_message = f"{ctx.user_message}\n\n**Información del usuario:**\n{ctx.clarification_text}"

    ctx.events.append(event_report("troubleshooting", "Starting autonomous diagnosis..."))
    logger.info("troubleshooter_node", f"[HANDLER] autonomous_diagnosis: equipment_id={ctx.pending_context.get('equipment_id')}")
    from src.agent.utils.stream_utils import get_worker_stream
    stream = get_worker_stream(ctx.state, "troubleshooting")
    stream_cb = stream._cb if stream.is_active else None

    equipment_doc_ids = []
    manual_tool = None
    try:
        from src.agent.services import get_supabase, get_embeddings
        sb = get_supabase()
        emb = get_embeddings()

        equipment_id = ctx.pending_context.get("equipment_id")

        if equipment_id:
            equipment_doc_ids = ctx.pending_context.get("equipment_doc_ids", [])
            logger.info("troubleshooter_node", f"Found {len(equipment_doc_ids)} manual docs for equipment {equipment_id}")

        manual_tool = make_equipment_manual_tool(sb, emb, equipment_id or "", equipment_doc_ids)

    except Exception as e:
        logger.error("troubleshooter_node", f"Error creating manual tool: {e}")

    pending = ctx.pending_context or {}
    equipment_id = pending.get("equipment_id")
    equipment_info = ""
    problem_desc = ctx.user_message

    if equipment_id:
        if "[Equipment Context]" in ctx.user_message:
            parts = ctx.user_message.split("\n\n", 1)
            equipment_info = parts[0]
            problem_desc = parts[1] if len(parts) > 1 else ctx.user_message
        else:
            eq_name = pending.get("equipment_name", "")
            eq_brand = pending.get("equipment_brand", "")
            eq_model = pending.get("equipment_model", "")
            eq_ip = pending.get("equipment_ip", "")
            eq_desc = pending.get("equipment_description", "")
            parts = [f"Equipment: {eq_name}"] if eq_name else [f"Equipment ID: {equipment_id}"]
            if eq_brand: parts.append(f"Brand: {eq_brand}")
            if eq_model: parts.append(f"Model: {eq_model}")
            if eq_ip: parts.append(f"IP: {eq_ip}")
            if eq_desc: parts.append(f"Description: {eq_desc}")
            equipment_info = "\n".join(parts)
    else:
        equipment_info = ctx.user_message[:200]

    if ctx.clarification_text:
        problem_desc = f"{problem_desc}\n\n**User clarification:**\n{ctx.clarification_text}"

    prompt = TROUBLESHOOT_AUTONOMOUS_PROMPT.format(
        equipment_context=equipment_info,
        problem_description=problem_desc,
        max_iterations=MAX_DIAGNOSTIC_ITERATIONS,
        user_name=ctx.user_name,
    )

    mode_instr = get_mode_instructions(ctx.state)
    if mode_instr:
        prompt += mode_instr

    # Inject equipment spec + troubleshoot skills
    eq_context = build_equipment_context_block(ctx.state, categories=["troubleshoot"])
    if eq_context:
        prompt = eq_context + "\n\n" + prompt

    ts_tools = []

    eq_type = pending.get("equipment_type", "")
    hw_tools = _select_hardware_tools(eq_type)
    ts_tools.extend(hw_tools)

    if manual_tool:
        ts_tools.append(manual_tool)

    web_tool = get_web_search_tool()
    if web_tool:
        ts_tools.append(web_tool)

    logger.info("troubleshooter_node",
                f"Tools: hw={[t.name for t in hw_tools]} "
                f"manual={'yes' if manual_tool else 'no'} "
                f"web={'yes' if web_tool else 'no'} "
                f"mock={is_mock_mode()}")

    if ts_tools:
        llm_with_tools = llm.bind_tools(ts_tools)
        tool_map = {t.name: t for t in ts_tools}
        logger.info("troubleshooter_node", f"Tools bound successfully: {list(tool_map.keys())}")
    else:
        llm_with_tools = llm
        tool_map = {}
        logger.warning("troubleshooter_node", "NO TOOLS BOUND, agent will not be able to investigate")

    model_name = ctx.state.get("llm_model") or os.getenv("DEFAULT_MODEL", "")
    logger.info("troubleshooter_node", f"Using model: {model_name}")

    messages_chain = [
        SystemMessage(content=prompt),
        HumanMessage(content=full_message),
    ]

    tokens_used, called_tools = _run_tool_agent_loop(
        llm_with_tools=llm_with_tools,
        tool_map=tool_map,
        messages_chain=messages_chain,
        max_iterations=MAX_DIAGNOSTIC_ITERATIONS,
        events=ctx.events,
        stream_cb=stream_cb,
    )

    result_text = ""
    for msg in reversed(messages_chain):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            result_text = msg.content.strip()
            break

    if not result_text:
        result_text = "Could not complete the diagnosis. Please provide more details about the problem."

    processing_time = (datetime.utcnow() - ctx.start_time).total_seconds() * 1000
    logger.info("troubleshooter_node",
                 f"Troubleshoot complete: {len(called_tools)} searches, {tokens_used} tokens, {processing_time:.0f}ms")

    if stream_cb:
        stream_cb({"type": "response", "content": result_text})

    severity = extract_severity(ctx.user_message + " " + result_text)
    clean_result, suggestions = extract_suggestions_from_text(result_text)

    _evidence_tools = {
        "search_equipment_manual", "web_search_diagnostic",
        "xarm_get_full_status", "xarm_get_position",
        "abb_get_position",
        "plc_read_input", "plc_read_output", "plc_read_memory", "plc_list_connections",
        "net_ping",
    }
    has_real_evidence = any(
        (t[0] if isinstance(t, tuple) else str(t)) in _evidence_tools
        for t in called_tools
    )
    prior_evidence = ctx.state.get("pending_context", {}).get("evidence", [])
    has_evidence = has_real_evidence or bool(prior_evidence)

    output = WorkerOutputBuilder.troubleshooting(
        content=clean_result,
        problem_identified=f"Equipment diagnosis ({severity})",
        severity=severity,
        summary=f"Diagnosis complete - {len(called_tools)} manual searches",
        confidence=_compute_diagnostic_confidence(
            called_tools=called_tools,
            has_evidence=has_evidence,
            is_lab=ctx.is_lab,
            has_clarification=bool(ctx.clarification_text),
        ),
        status="ok" if len(called_tools) >= 2 else "partial",
    )
    output.metadata.completed_at = datetime.utcnow().isoformat()
    output.metadata.processing_time_ms = processing_time
    output.metadata.model_used = ctx.state.get("llm_model") or os.getenv("DEFAULT_MODEL", "gemini-2.0-flash")

    if not suggestions:
        suggestions = [
            "Search for a specific error code in the manual",
            "Check physical connections and indicators",
            "Try a different diagnostic approach",
        ]

    return {
        "worker_outputs": [output.model_dump()],
        "troubleshooting_result": output.model_dump_json(),
        "pending_context": {},
        "clarification_questions": [],
        "needs_human_input": False,
        "events": ctx.events,
        "follow_up_suggestions": suggestions,
        "token_usage": tokens_used,
    }

def _run_simple_diagnosis(ctx: TroubleshooterContext) -> Dict[str, Any]:
    logger.info("troubleshooter_node", f"[HANDLER] simple_diagnosis: is_lab={ctx.is_lab} equipment={ctx.equipment}")
    evidence_text = get_evidence_from_context(ctx.state)
    clarification_section = ctx.clarification_text if ctx.clarification_text else "No hay información adicional."

    try:
        llm = get_llm(ctx.state, temperature=0.3)
    except Exception as e:
        return _return_error(ctx, "LLM_INIT_ERROR", str(e))

    knowledge_context = ""
    if LAB_KNOWLEDGE_AVAILABLE:
        try:
            knowledge_context = get_knowledge_context(ctx.user_message)
        except Exception as e:
            logger.warning("troubleshooter_node", f"Error obteniendo conocimiento: {e}")

    full_message = ctx.user_message
    if ctx.clarification_text:
        full_message = f"{ctx.user_message}\n\n**Información del usuario:**\n{ctx.clarification_text}"

    prompt = TROUBLESHOOTER_PROMPT_SIMPLE.format(
        knowledge_context=knowledge_context,
        clarification_section=clarification_section,
        evidence_section=evidence_text,
        format_rules=MARKDOWN_FORMAT_RULES,
        user_name=ctx.user_name,
    )

    mode_instr = get_mode_instructions(ctx.state)
    if mode_instr:
        prompt += mode_instr

    # Inject equipment spec + troubleshoot skills
    eq_context = build_equipment_context_block(ctx.state, categories=["troubleshoot"])
    if eq_context:
        prompt = eq_context + "\n\n" + prompt

    tokens_used = 0
    try:
        response, tokens_used = invoke_and_track(llm, [
            SystemMessage(content=prompt),
            HumanMessage(content=full_message),
        ], "troubleshooter")
        result_text = (response.content or "").strip()
        processing_time = (datetime.utcnow() - ctx.start_time).total_seconds() * 1000
    except Exception as e:
        return _return_error(ctx, "LLM_ERROR", str(e))

    return _build_diagnosis_response(ctx, result_text, tokens_used, processing_time,
                                      called_tools=set(), evidence_text=evidence_text)


def _build_diagnosis_response(ctx: TroubleshooterContext, result_text: str,
                               tokens_used: int, processing_time: float,
                               called_tools: set = None, evidence_text: str = "") -> Dict[str, Any]:
    """Build the final diagnosis response dict (shared by ReAct and simple paths)."""
    severity = extract_severity(ctx.user_message + " " + result_text)
    clean_result, suggestions = extract_suggestions_from_text(result_text)

    _tools = called_tools or set()
    _has_ev = bool(evidence_text and evidence_text != "No hay documentación de referencia.")
    _status = "ok" if (len(_tools) >= 2 or _has_ev) else "partial"

    output = WorkerOutputBuilder.troubleshooting(
        content=clean_result,
        problem_identified=f"Issue at {'Station ' + str(ctx.station_num) if ctx.station_num else 'equipment'} ({severity})",
        severity=severity,
        summary=f"Diagnosis complete - Severity: {severity}",
        confidence=_compute_diagnostic_confidence(
            called_tools=_tools,
            has_evidence=_has_ev,
            is_lab=ctx.is_lab,
            has_clarification=bool(ctx.clarification_text),
        ),
        status=_status,
    )
    output.metadata.completed_at = datetime.utcnow().isoformat()
    output.metadata.processing_time_ms = processing_time
    output.metadata.model_used = ctx.state.get("llm_model") or os.getenv("DEFAULT_MODEL", "gemini-2.0-flash")

    logger.node_end("troubleshooter_node", {"severity": severity, "is_lab": ctx.is_lab})
    ctx.events.append(event_report("troubleshooting", f"Diagnosis ready (Severity: {severity})"))

    clean_context = ctx.pending_context.copy()
    clean_context.pop("user_clarification", None)
    clean_context.pop("original_query", None)

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
        "events": ctx.events,
        "follow_up_suggestions": suggestions,
        "token_usage": tokens_used,
    }


def troubleshooter_node(state: AgentState) -> Dict[str, Any]:
    """Main troubleshooting worker dispatcher."""
    ctx = _build_context(state)
    if ctx is None:
        events = [event_execute("troubleshooting", "Analizando problema...")]
        error_output = create_error_output("troubleshooting", "NO_MESSAGE", "No hay mensaje")
        return {
            "worker_outputs": [error_output.model_dump()],
            "troubleshooting_result": error_output.model_dump_json(),
            "events": events,
        }

    hitl_type = _read_hitl_type(ctx.pending_context)
    if hitl_type in ("cobot_confirmation", "repair_confirmation", "health_query"):
        return _run_autonomous_diagnosis(ctx)

    if ctx.is_command and not ctx.action_request and not ctx.query_request:
        return _handle_unrecognized_command(ctx)

    if not ctx.already_clarified and not ctx.is_command:
        result = _try_request_clarification(ctx)
        if result:
            return result

    if ctx.interaction_mode == "troubleshoot":
        return _run_autonomous_diagnosis(ctx)
    return _run_simple_diagnosis(ctx)
