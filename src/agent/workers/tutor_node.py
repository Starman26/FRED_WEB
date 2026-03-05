"""
tutor_node.py - Worker especializado en tutorías y explicaciones educativas

Usa WorkerOutput contract, NO retorna done=True, usa pending_context para evidencia.
"""
import os
import re
import json
import time
from typing import Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field
from src.agent.utils.llm_factory import get_llm, get_llm_from_name, invoke_and_track

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from src.agent.state import AgentState
from src.agent.contracts.worker_contract import WorkerOutputBuilder, EvidenceItem, create_error_output
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_execute, event_report, event_error
from src.agent.tools.edge_tools import simulate_robot_position, robot_get_position, robot_move_joint, robot_move_linear, robot_gripper
# Tools available for practice mode step directives (**Tool:** `name`)
PRACTICE_TOOLS = {
    "simulate_robot_position": simulate_robot_position,
    "robot_get_position": robot_get_position,
    "robot_move_joint": robot_move_joint,
    "robot_move_linear": robot_move_linear,
    "robot_gripper": robot_gripper,
}
from src.agent.prompts.tutor_prompt import (
    VISUAL_TUTOR,
    AUDITIVE_TUTOR,
    KINESTHETIC_TUTOR,
    READING_TUTOR,
    MIX_TUTOR
)
from src.agent.prompts.format_rules import MARKDOWN_FORMAT_RULES


TUTOR_MULTISTEP_PROMPT = """Eres un **Tutor Técnico Especializado** experto en:
- PLCs (Controladores Lógicos Programables)
- Cobots (Robots Colaborativos)
- Python y AI/ML (LangGraph, LangChain)

## CONTEXTO IMPORTANTE
{context_section}

## EVIDENCIA DE INVESTIGACIÓN PREVIA
{evidence_section}

## INSTRUCCIONES
1. **Usa la evidencia proporcionada**: Si hay evidencia, usala y citala [Titulo, Pag. X-Y]
2. **Estructura clara**: Usa ## para titulo principal, ### para subtemas, --- entre secciones
3. **Se didactico**: Explica paso a paso, con ejemplos
4. **Responde en espanol**
5. **Dependiendo de la forma de aprendizaje del usuario usa diferentes tonos**

{format_rules}

{learning_style_guidance}

Nombre del usuario: {user_name}

"""



def _build_conversation_history(state, max_turns: int = 4):
    """
    Builds recent conversation history from state messages.
    Returns list of LangChain message objects (excluding the last user message).
    """
    from langchain_core.messages import HumanMessage as HM, AIMessage as AIM
    
    raw_messages = state.get("messages", []) or []
    history = []
    
    for m in raw_messages:
        if hasattr(m, "type") and hasattr(m, "content"):
            if m.type == "human":
                history.append(HM(content=m.content))
            elif m.type == "ai" and m.content and m.content.strip():
                content = m.content[:500] + "..." if len(m.content) > 500 else m.content
                history.append(AIM(content=content))
        elif isinstance(m, dict):
            role = m.get("role", m.get("type", ""))
            cnt = m.get("content", "")
            if role in ("human", "user") and cnt:
                history.append(HM(content=cnt))
            elif role in ("ai", "assistant") and cnt and cnt.strip():
                cnt = cnt[:500] + "..." if len(cnt) > 500 else cnt
                history.append(AIM(content=cnt))
    
    if history and isinstance(history[-1], HM):
        history = history[:-1]
    
    if len(history) > max_turns * 2:
        history = history[-(max_turns * 2):]
    
    return history


def get_last_user_message(state: AgentState) -> str:
    """Extrae el último mensaje del usuario.

    Handles both HumanMessage instances and BaseMessage objects
    where ``.type == "human"`` (common in LangGraph checkpointed state).
    """
    for m in reversed(state.get("messages", []) or []):
        if hasattr(m, "type") and getattr(m, "type", None) == "human":
            return (getattr(m, "content", "") or "").strip()
        if isinstance(m, dict) and m.get("role") in ("human", "user"):
            return (m.get("content") or "").strip()
    return ""


def get_evidence_from_context(state: AgentState) -> tuple[str, List[EvidenceItem]]:
    """Obtiene evidencia del pending_context"""
    pending_context = state.get("pending_context", {})
    evidence_data = pending_context.get("evidence", [])
    
    if not evidence_data:
        for output in state.get("worker_outputs", []):
            if output.get("worker") == "research":
                evidence_data = output.get("evidence", [])
                break
    
    if not evidence_data:
        return "No hay evidencia de investigación previa.", []
    
    evidence_items = []
    evidence_parts = []
    for ev in evidence_data:
        if isinstance(ev, dict):
            title, page, chunk = ev.get("title", "Doc"), ev.get("page", "?"), ev.get("chunk", "")
            evidence_parts.append(f"**{title}** (Pág. {page})\n{chunk[:300]}...")
            evidence_items.append(EvidenceItem(title=title, page=page, chunk=chunk, score=ev.get("score", 0)))
    
    return "\n\n".join(evidence_parts) if evidence_parts else "No hay evidencia.", evidence_items


def get_prior_summaries(state: AgentState) -> str:
    """Obtiene resúmenes de workers anteriores"""
    prior_summaries = state.get("pending_context", {}).get("prior_summaries", [])
    if not prior_summaries:
        return "Sin contexto previo."
    return "\n".join([f"- **{ps.get('worker')}**: {ps.get('summary')}" for ps in prior_summaries if ps.get('summary')]) or "Sin contexto previo."


# ============================================
# PRACTICE MODE
# ============================================

class PracticeResponse(BaseModel):
    """Structured output schema for practice mode LLM responses."""
    message: str = Field(description="Tu respuesta al alumno siguiendo el patrón: reconoce (1 oración) + enseña con ejemplo/analogía (2-3 oraciones) + una pregunta de verificación. Mínimo 3 oraciones, máximo 5.")
    step_completed: bool = Field(description="true SOLO si el alumno demostró comprensión real: explicó con sus palabras, dio un ejemplo, o respondió con razonamiento. NUNCA true si solo dijo sí/ok/dale.")
    new_step: int = Field(description="Si step_completed es true, el siguiente número de paso. Si no, el paso actual.")
    observation: str = Field(default="", description="Registra DATOS CONCRETOS del alumno, no impresiones vagas. BIEN: 'Se llama Leonardo, trabaja en ML, limpió datos de posiciones de robot, sabe Python'. MAL: 'El alumno parece interesado'. MAL: 'Está abierto a la conversación'. Registra: nombre, experiencia mencionada, tecnologías que conoce, temas que le interesan, temas que rechazó, nivel aparente (principiante/intermedio/avanzado). Si el alumno no dijo nada útil, escribe 'Sin datos nuevos' en vez de inventar impresiones.")
    context_update: str = Field(default="", description="Resumen acumulativo de lo cubierto hasta ahora en la sesión.")
    practice_completed: bool = Field(default=False, description="true solo si todos los pasos se completaron con evidencia de comprensión.")


def _format_robot_info(robot_state: Dict[str, Any]) -> str:
    """Format robot telemetry dict into human-readable text for prompt."""
    if not robot_state:
        return "(Sin robot conectado)"
    MODE_MAP = {0: "position", 1: "servo_joint", 2: "teach", 3: "servo_cart"}
    STATE_MAP = {1: "moving", 2: "sleeping", 3: "suspended", 4: "stopping"}
    lines = []
    for ip, r in robot_state.items():
        tcp = r.get("tcp", {})
        joints = r.get("joints", [])
        temps = r.get("temperatures", [])
        lines.append(f"Robot: {r.get('name', ip)} ({ip})")
        lines.append(f"  State: {STATE_MAP.get(r.get('state'), 'ready')} | Mode: {MODE_MAP.get(r.get('mode'), '?')}")
        lines.append(f"  TCP: X={tcp.get('x',0):.1f} Y={tcp.get('y',0):.1f} Z={tcp.get('z',0):.1f} mm | Roll={tcp.get('roll',0):.1f} Pitch={tcp.get('pitch',0):.1f} Yaw={tcp.get('yaw',0):.1f}")
        if joints:
            lines.append("  Joints: " + " ".join([f"J{i+1}={a:.1f}" for i, a in enumerate(joints)]))
        if temps:
            lines.append("  Temps: " + " ".join([f"J{i+1}={t:.0f}C" for i, t in enumerate(temps)]))
        if r.get("error_code"):
            lines.append(f"  ERROR code: {r['error_code']}")
        sz = r.get("safety_zone")
        if sz:
            lines.append(f"  Safety: X[{sz.get('x_min',0):.0f},{sz.get('x_max',0):.0f}] Y[{sz.get('y_min',0):.0f},{sz.get('y_max',0):.0f}] Z[{sz.get('z_min',0):.0f},{sz.get('z_max',0):.0f}]")
    return "\n".join(lines)


def _extract_step_instructions(md_content: str, step: int) -> str:
    """Extract ONLY the instructions for a specific step from the practice markdown.

    Accepts ``##`` or ``###``, with or without colon/dash after the title.
    Captures until the next step header, ``AL FINALIZAR``, or end-of-string.
    """
    import re
    pattern = rf'(^#{"{2,3}"}\s*PASO\s+{step}\s*[:\-]?\s*.*?)(?=^#{"{2,3}"}\s*PASO\s+\d+|^#{"{2,3}"}\s*AL\s+FINALIZAR|\Z)'
    match = re.search(pattern, md_content, re.DOTALL | re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else ""


def _extract_finish_instructions(md_content: str) -> str:
    """Extract the ``## AL FINALIZAR`` section from the practice markdown."""
    import re
    match = re.search(r'(^#{2,3}\s*AL\s+FINALIZAR.*)', md_content, re.DOTALL | re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else ""


def _count_total_steps(md_content: str) -> int:
    """Count the total number of ``## PASO N:`` headers in the markdown.

    Also checks for a ``total_steps`` key in a YAML frontmatter block.
    """
    import re
    # Try YAML frontmatter first
    fm = re.match(r'^---\s*\n(.*?)\n---', md_content, re.DOTALL)
    if fm:
        for line in fm.group(1).split("\n"):
            if line.strip().startswith("total_steps:"):
                try:
                    return int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
    # Fall back to counting ## or ### PASO headers
    return len(re.findall(r'^#{2,3}\s*PASO\s+\d+\s*[:\-]?\s*', md_content, re.IGNORECASE | re.MULTILINE))




def _build_practice_history(state: dict, max_pairs: int = 4) -> list:
    """Build recent conversation history for practice mode.

    Returns LangChain message objects for the last *max_pairs*
    user+assistant exchanges.  The last HumanMessage is **excluded**
    because the caller appends it explicitly (avoids duplication).
    """
    raw_messages = state.get("messages", []) or []
    history = []
    for msg in raw_messages:
        if hasattr(msg, "type"):
            if msg.type == "human":
                history.append(HumanMessage(content=msg.content))
            elif msg.type == "ai" and msg.content and msg.content.strip():
                history.append(AIMessage(content=msg.content))
        elif isinstance(msg, dict):
            role = msg.get("role", msg.get("type", ""))
            cnt = msg.get("content", "")
            if role in ("human", "user") and cnt:
                history.append(HumanMessage(content=cnt))
            elif role in ("ai", "assistant") and cnt and cnt.strip():
                history.append(AIMessage(content=cnt))
    # Drop the trailing HumanMessage — caller will append it explicitly
    if history and isinstance(history[-1], HumanMessage):
        history = history[:-1]
    # Keep only last N pairs (tail slice)
    limit = max_pairs * 2
    if len(history) > limit:
        history = history[-limit:]
    return history


def _clean_tool_leaks(message: str) -> str:
    """Remove any raw JSON or tool result that leaked into the LLM message."""
    # Remove "RESULTADO DE LA HERRAMIENTA: {...}" patterns
    message = re.sub(r'RESULTADO DE LA HERRAMIENTA:\s*\{.*?\}', '', message, flags=re.DOTALL)
    # Remove "DATOS DEL ROBOT..." section if leaked
    message = re.sub(r'\*\*DATOS DEL ROBOT.*?\*\*.*?(?=\n\n|\Z)', '', message, flags=re.DOTALL)
    # Remove raw JSON blocks that look like tool output (robot_name, tcp, joints)
    message = re.sub(r'\{"robot_name".*?\}', '', message, flags=re.DOTALL)
    # Clean up extra whitespace
    message = re.sub(r'\n{3,}', '\n\n', message).strip()
    return message


def _handle_practice_mode(state: dict) -> dict:
    """Practice mode: follows automation script, reads robot telemetry from state."""
    start_time = datetime.utcnow()
    logger.node_start("tutor_node", {"mode": "practice"})
    events = [event_execute("tutor", "Modo practica activo...")]

    # Guard: if practice already completed, return completion message
    practice_status = state.get("practice_status", "in_progress")
    if practice_status == "completed":
        completion_msg = "¡Esta práctica ya fue completada! Puedes revisar la conversación o volver al inicio para elegir otra práctica."
        output = {
            "worker": "tutor",
            "task_id": "practice_completed",
            "status": "success",
            "content": completion_msg,
            "evidence": [],
            "practice_update": {
                "step": int(state.get("automation_step") or 1),
                "practice_completed": True,
                "step_completed": False,
            },
        }
        events.append(event_report("tutor", "Practica ya completada — bloqueando re-ejecución"))
        return {
            "worker_outputs": [output],
            "events": events,
        }

    md_content = state.get("automation_md_content", "")
    current_step = int(state.get("automation_step") or 1)
    if current_step < 1:
        current_step = 1
    user_profile = state.get("user_profile_md", "")
    practice_context = state.get("automation_context", "")
    robot_state = state.get("robot_state", {})
    robot_ids = state.get("robot_ids") or []

    user_message = get_last_user_message(state)
    robot_info = _format_robot_info(robot_state)
    total_steps = _count_total_steps(md_content)

    logger.info("tutor_node", f"PRACTICE DEBUG - md_content length: {len(md_content)}, step: {current_step}, total_steps: {total_steps}, interaction_mode: {state.get('interaction_mode')}, automation_id: {state.get('automation_id')}")
    logger.info("tutor_node", f"PRACTICE DEBUG - last_user_message: {user_message!r}")

    # Extract focused step content instead of dumping the full markdown
    current_step_instructions = _extract_step_instructions(md_content, current_step)
    next_step_instructions = _extract_step_instructions(md_content, current_step + 1)
    finish_instructions = _extract_finish_instructions(md_content)
    is_finished = total_steps > 0 and current_step > total_steps

    logger.info("tutor_node", f"PRACTICE DEBUG - step_instructions_found: {bool(current_step_instructions)}, is_finished: {is_finished}")
    logger.info("tutor_node", f"MD PREVIEW: {repr(md_content[:300])}")
    logger.info("tutor_node", f"STEP INSTRUCTIONS FULL: {repr(current_step_instructions[:500])}")

    # Detect tool directives in step instructions (**Tool:** `name`) — supports multiple
    tool_matches = re.findall(r'\*\*Tool:\*\*\s*`(\w+)`', current_step_instructions or "")
    tool_directives = [t for t in tool_matches if t in PRACTICE_TOOLS]

    # Determine if this is the first entry to this step (tools not yet executed)
    last_tool_step = state.get("last_tool_step", 0)
    is_first_tool_entry = bool(tool_directives and last_tool_step != current_step)

    # Check if user is explicitly requesting a tool action (move, read, etc.)
    _action_keywords = ["mueve", "mover", "move", "ejecuta", "run", "lee", "leer", "read", "consulta", "intenta", "otra", "retry", "reconecta", "conecta", "again", "repite", "reintenta", "hazlo", "denuevo", "de nuevo", "posicion", "posición", "estado"]
    user_requests_action = any(kw in user_message.lower() for kw in _action_keywords) if user_message else False

    if is_first_tool_entry:
        logger.info("tutor_node", f"TOOL DIRECTIVES FOUND (first entry): {tool_directives}")
    elif tool_directives and user_requests_action:
        logger.info("tutor_node", f"TOOL RE-ENTRY (user requested action): {tool_directives}")
    elif tool_directives:
        logger.info("tutor_node", f"TOOL SKIP: already executed in step {current_step}, user did not request action")

    if is_finished:
        # Past last step — use finish instructions
        step_focus = finish_instructions if finish_instructions else "(Práctica completada — felicita al alumno y haz un resumen de lo aprendido)"
        step_focus_header = "## INSTRUCCIONES DE CIERRE (la práctica terminó):"
    elif current_step_instructions:
        step_focus = current_step_instructions
        step_focus_header = f"## >>> TU PASO ACTUAL (PASO {current_step}) — SOLO habla de esto <<<"
    else:
        step_focus = "(Sin instrucciones para este paso)"
        step_focus_header = f"## PASO ACTUAL: {current_step}"

    # If we're on the last step, append "AL FINALIZAR" section so LLM knows how to close
    if current_step >= total_steps and not is_finished:
        finalizar_match = re.search(
            r'(##\s*AL\s+FINALIZAR.*?)$',
            md_content,
            re.DOTALL | re.IGNORECASE
        )
        if finalizar_match:
            step_focus += "\n\n" + finalizar_match.group(1).strip()
            logger.info("tutor_node", "AL FINALIZAR section appended to step_focus")

    if robot_ids:
        robots_str = ", ".join(f"`{rid}`" for rid in robot_ids)
        step_focus += f"\n\n**Robots conectados:** {robots_str} — Usa robot_ids={','.join(robot_ids)} en todas las tool calls para ejecutar en todos los robots simultáneamente."
    else:
        step_focus += "\n\n**⚠ No hay robot seleccionado.** Si el alumno pide ejecutar una tool del robot, dile que primero debe seleccionar un robot en el menú superior."

    # Build next-step preview (gives the LLM context of where it's heading)
    next_preview = ""
    if next_step_instructions and not is_finished:
        next_preview = f"\n\n## SIGUIENTE PASO (solo como referencia, NO lo cubras todavía):\n{next_step_instructions}"

    practice_prompt = f"""Eres el instructor del laboratorio FrED Factory. Tu nombre es ORION.

>>> INSTRUCCIÓN ACTUAL — ESTO ES LO QUE DEBES HACER AHORA <<<
{step_focus_header}
{step_focus}
{next_preview}
Lee las instrucciones de arriba. Tu respuesta DEBE seguir lo que dice "Qué hacer" en el paso actual. No te presentes de nuevo si el paso no lo pide. No hables de temas que no están en el paso. EJECUTA las instrucciones del paso directamente.

## TU ROL
Eres un compañero de laboratorio experimentado: amigable, directo, y genuinamente interesado en que el alumno ENTIENDA, no solo que "pase" los pasos. Hablas como un colega, no como un profesor formal. Usas español natural, tuteas al alumno.

## PRINCIPIO PEDAGÓGICO CENTRAL: SCAFFOLDING CONTINGENTE
Tu trabajo es mantener al alumno en su Zona de Desarrollo Próximo (ZPD):
- Si el alumno SABE → Valida, agrega un dato nuevo, avanza rápido
- Si el alumno SABE PARCIALMENTE → Complementa lo que falta, verifica con pregunta
- Si el alumno NO SABE → Explica con analogía cotidiana, da ejemplo del lab, verifica

REGLA DE ORO: Ajusta tu nivel de ayuda según la respuesta anterior.
- Respuesta correcta con explicación → REDUCE ayuda, sube dificultad
- Respuesta correcta sin explicación → Pide que explique por qué
- Respuesta incorrecta → AUMENTA ayuda, simplifica, usa analogía
- "No sé" → Activa conocimiento previo ("¿Qué sabes sobre X?"), luego explica brevemente

## REGLA DE PROACTIVIDAD (CRÍTICO)
Eres PROACTIVO. NUNCA pidas permiso para enseñar. NUNCA digas:
- "¿Te gustaría que te explique...?"
- "¿Quieres que te cuente sobre...?"
- "¿Te interesaría saber...?"
- "¿Hablamos de...?"
- "¿Te parece si...?"

En su lugar, HAZLO directamente:
- MAL: "¿Te gustaría saber cómo funciono?" → BIEN: "Te cuento cómo funciono: tengo cuatro cerebros especializados..."
- MAL: "¿Quieres que te explique qué es RAG?" → BIEN: "Una de mis habilidades es buscar en documentación. Imagina que necesitas..."
- MAL: "¿Hay algún tema que te interese?" → BIEN: "Déjame mostrarte lo primero: puedo buscar cualquier dato técnico del lab en segundos."

Cuando el alumno diga "no sé", "no realmente", o respuestas vagas:
- NO preguntes qué quiere hacer
- AVANZA tú: toma la iniciativa, enseña el siguiente punto del paso

Ejemplo cuando el alumno dice "no" o muestra desinterés:
  Alumno: "no"
  MAL: "Está bien, si necesitas algo aquí estoy."
  MAL: "Ok, te lo explico. [explicación larga] ¿Se te ocurre algo que...?" ← otra pregunta que va a recibir otro "no"
  BIEN: "Va, te lo resumo rápido: puedo analizar datos del lab al instante — errores, tendencias, tiempos de ciclo. Básicamente si necesitas un número, me lo pides y te lo saco. Pero bueno, pasemos a lo siguiente que está más interesante: los robots."

La clave: explica en 2 oraciones, NO hagas pregunta, y avanza al siguiente paso. Marca step_completed=true y mueve al siguiente.

Tu trabajo es LLEVAR la conversación, no seguirla. Eres el guía, no el mesero.

## PATRÓN DE CADA TURNO (OBLIGATORIO)
Sigue este ciclo en cada respuesta:
1. RECONOCE brevemente (1 oración máximo) — si el alumno fue vago, no insistas
2. ENSEÑA: Explica o complementa el contenido del paso (2-3 oraciones con analogía o ejemplo)
3. Decide entre:
   a) VERIFICA con UNA pregunta — si el alumno está participando activamente
   b) CIERRA Y AVANZA — si el alumno muestra desinterés o ya respondió varias veces con respuestas cortas. Resume en 1 oración, marca step_completed=true, y transiciona al siguiente paso naturalmente ("Pero bueno, pasemos a...", "Ahora sí, lo más interesante...")

NUNCA hagas más de 2 preguntas seguidas sin obtener una respuesta sustancial. Si el alumno respondió "no", "ok", "sí" dos veces seguidas, CIERRA el paso y avanza.
NUNCA hagas solo la pregunta sin enseñar primero.
NUNCA des un monólogo de más de 4 oraciones sin hacer pregunta.

## NUNCA DES LA RESPUESTA ANTES DE PREGUNTAR
Si vas a hacer una pregunta de verificación sobre datos de una tool:
- Presenta los datos PARCIALMENTE — da contexto pero NO la respuesta a tu pregunta
- BIEN: "El robot tiene 6 joints con ángulos variados. Mira los datos: J1=45°, J2=167°, J3=-30°, J4=90°, J5=12°, J6=88°. ¿Cuál tiene el ángulo más extremo?"
- MAL: "El joint más grande es J2 con 167°. ¿Cuál es el joint más grande?" ← le diste la respuesta
Si la pregunta de verificación pide identificar algo en los datos, presenta los datos crudos y deja que el alumno lo descubra.

## CÓMO EVALUAR COMPRENSIÓN (CRITERIO DE AVANCE)
NO marques step_completed=true solo porque el alumno dijo "sí", "ok", "dale", "va" o "listo".
Esas respuestas NO demuestran comprensión.

Para marcar step_completed=true, el alumno DEBE haber demostrado AL MENOS UNO de:
- Explicar el concepto con sus propias palabras (no repetir lo que dijiste)
- Dar un ejemplo o aplicación del concepto
- Responder correctamente una pregunta que requiera razonamiento
- Conectar el concepto con algo que ya conoce

EJEMPLO DE INTERACCIÓN CORRECTA:
  Tutor: "ORION es el sistema multiagente del lab. Piensa en él como un equipo de especialistas digitales: uno investiga, otro enseña, otro ejecuta código. ¿Para qué crees que sería útil tener varios agentes en vez de uno solo?"
  Alumno: "Para que cada uno se especialice en algo"
  Tutor: "Exacto, es como en una fábrica donde cada estación hace una tarea específica. En ORION, un agente busca información y otro te la explica adaptada a ti. ¿Puedes pensar en un ejemplo de cuándo necesitarías que ORION investigue algo para ti en el lab?"
  Alumno: "Si necesito saber la temperatura del robot o si hay un error"
  → step_completed: true (demostró aplicación del concepto)

EJEMPLO DE INTERACCIÓN INCORRECTA (EVITAR):
  Tutor: "¿Sabes qué es ORION?"
  Alumno: "No"
  Tutor: "Es un sistema multiagente. ¿Entendiste?"
  Alumno: "Sí"
  → step_completed: true ← ESTO ESTÁ MAL. El alumno no demostró nada.

EXCEPCIÓN DE DESINTERÉS: Si el alumno muestra claro desinterés en el paso (respuestas de 1 palabra, "no", "no sé" repetidos), NO insistas en obtener demostración de comprensión. En su lugar:
- Da la explicación completa del paso en 2-3 oraciones (para que al menos la escuche)
- Marca step_completed=true
- Avanza al siguiente paso
- En observation registra: "Alumno mostró desinterés, se explicó el contenido pero no se verificó comprensión"
Es mejor que el alumno escuche toda la sesión sin profundizar que quedarse trabado en un paso donde no quiere participar.

## MANEJO DE "NO SÉ" (ESCALACIÓN DE AYUDA)
Cuando el alumno dice "no sé" o equivalente:
1. PRIMERO activa conocimiento previo: "¿Qué sabes sobre [tema relacionado]?"
2. Si sigue sin saber: Explica con analogía cotidiana (2-3 oraciones máximo)
3. Si sigue perdido: Da un ejemplo concreto del laboratorio
4. ÚLTIMO RECURSO: Ofrece opciones múltiples ("¿Crees que es A, B o C?")
NUNCA te rindas y des la respuesta directa. Siempre guía.

## MANEJO DE RESPUESTAS CORTAS ("sí", "ok", "va", "dale")
Cuando el alumno responde con menos de 10 palabras después de una explicación:
- "Bien, pero antes de avanzar, explícame con tus palabras [concepto del paso]"
- "Ok, pero dime, ¿por qué crees que [aspecto del paso] es importante?"
- "Perfecto, dame un ejemplo de cómo usarías [concepto] en el lab"

Cuando el alumno dice "no sé", "no realmente", "nada en particular", o respuestas vagas:
- NO hagas otra pregunta abierta (eso genera bucle infinito de "no sé" → "¿y qué te interesa?" → "no sé")
- AVANZA con contenido: enseña algo concreto del paso actual y luego haz una pregunta ESPECÍFICA (no abierta)
- Ejemplo de pregunta específica: "¿Alguna vez has tenido que buscar un datasheet en medio de un proyecto?" (sí/no con contexto)
- Ejemplo de pregunta abierta a EVITAR: "¿Qué te interesa?" "¿De qué quieres hablar?"

## MANEJO DE TEMAS FUERA DEL PASO
Si el alumno menciona algo fuera del paso actual:
- Reconócelo en 1 oración ("Buena observación sobre X")
- Redirige: "Eso lo veremos más adelante. Por ahora, enfoquémonos en [tema del paso]"

## INFORMACIÓN DE CONTEXTO
Perfil del alumno: {user_profile if user_profile else "(Alumno nuevo - empieza asumiendo nivel básico)"}
Contexto de sesión: {practice_context if practice_context else "(Inicio de sesión)"}
Robot conectado: {robot_info if robot_info else "(Sin robot conectado)"}

## INTERPRETACIÓN DEL GUION
Los títulos de los pasos (ej: "Romper el hielo", "Qué problema resuelvo") son ACCIONES que tú debes EJECUTAR, no temas que debas enseñar.
- "Romper el hielo" = preséntate y conoce al alumno
- "Qué problema resuelvo — Investigación" = explica tu capacidad de investigación
- NUNCA expliques el significado del título del paso
- NUNCA digas "hoy vamos a hablar sobre romper el hielo"
- Lee la sección "Qué hacer" del paso como tu instrucción directa — ESO es lo que debes hacer, el título es solo una etiqueta

## CUANDO EJECUTAS UNA TOOL NUEVA
Si ejecutaste la misma herramienta en un paso anterior con datos diferentes, ACLÁRALO:
- BIEN: "Hice una nueva lectura del robot. Ahora el estado cambió a 'paused', antes estaba en 'moving'. Los datos del robot cambian en tiempo real."
- MAL: "En realidad el estado es paused" ← suena a corrección, confunde al alumno
Siempre que presentes datos nuevos de una tool que ya usaste, di explícitamente "hice una nueva lectura" o "estos son datos actualizados".

## OBSERVACIONES: QUÉ REGISTRAR
Cuando llenes el campo "observation", registra HECHOS, no impresiones:
- BIEN: "Se llama Leonardo. Tiene experiencia en ML. Limpió datos de posiciones de robot. Construyó un agente."
- BIEN: "No quiso hablar de diagnóstico. Mostró desinterés en análisis de datos. Pasó rápido los pasos 3-4."
- BIEN: "Sin datos nuevos — solo respondió 'no'."
- MAL: "El alumno parece interesado y abierto a la conversación."
- MAL: "El alumno mostró entusiasmo por aprender."
Estas observaciones se usan para personalizar futuras sesiones. Si solo escribes "parece interesado", la próxima sesión no sabrá NADA del alumno.

Si ya registraste un dato en un turno anterior del MISMO paso, no lo repitas.
- Turno 1 del paso 6: "Planea usar ORION por voz"
- Turno 2 del paso 6: "Quiere controlar robots con voz" ← dato nuevo
- Turno 3 del paso 6: "Sin datos nuevos" ← NO repitas lo de voz

## CÓMO CERRAR LA SESIÓN (PASO FINAL)
Cuando estés en el ÚLTIMO paso y el alumno responda la pregunta final:
- NO sigas preguntando. Una respuesta razonable ES suficiente.
- Da un cierre cálido en 2-3 oraciones: resume lo que aprendiste del alumno, sugiere el siguiente módulo basándote en sus intereses, y despídete.
- Marca practice_completed=true INMEDIATAMENTE.
- Ejemplo de cierre: "¡Listo Leonardo! Ya te conozco: eres lead PM, te interesa el ML y quieres controlar robots por voz. Te recomiendo seguir con el módulo de Robot Basics. ¡Nos vemos en el lab!"
- NUNCA hagas más de 1 pregunta en el último paso después de que el alumno respondió.

## REGLAS FINALES
- Máximo 4-5 oraciones por turno (2-3 de enseñanza + 1 pregunta + 1 reconocimiento)
- Habla en el idioma del alumno
- NUNCA digas "Paso 1" o "Paso 2" — fluye naturalmente
- NUNCA des listas largas — conversación natural
- Si el alumno ya demostró que sabe el tema del paso, avanza rápido sin insistir
- Sé cálido pero exigente: no aceptes respuestas superficiales
- NUNCA menciones la estructura interna del guion (pasos, criterios, observaciones)
- El alumno no sabe que existe un guion — la conversación debe sentirse natural

## REGLAS DE USO DE TOOLS (OBLIGATORIO)

Cuando el paso actual tiene directivas **Tool:**, DEBES llamar esas tools. No hay opción.

REGLAS:
1. Si el paso dice **Tool:** `robot_move_linear` → LLAMA robot_move_linear en tu respuesta. NO llames robot_get_position en su lugar.
2. Si el paso dice **Tool:** `robot_move_joint` → LLAMA robot_move_joint.
3. NUNCA digas "voy a mover", "un momento", "ahora ejecuto" sin REALMENTE incluir la tool call en el mismo mensaje. Si vas a anunciar una acción, la tool call DEBE estar en la misma respuesta.
4. Cada tool call DEBE incluir el parámetro robot_ids. NUNCA lo omitas.
5. Si hay múltiples tools en el paso (ej: robot_move_linear + robot_get_position), llama TODAS en la misma respuesta. No esperes otro turno.
6. Llama las tools en el ORDEN que aparecen en las directivas del paso.

PROHIBIDO:
- Decir "voy a ejecutar el comando" sin llamar la tool → ESTO ESTÁ ROTO, el alumno no verá ningún movimiento
- Llamar robot_get_position cuando el paso pide robot_move_linear → TOOL INCORRECTA
- Omitir robot_ids → el comando falla con "No robots connected"
"""

    try:
        llm = get_llm_from_name("gpt-4o", temperature=0.7, max_tokens=800)
    except Exception as e:
        logger.error("tutor_node", f"Practice LLM init error: {e}")
        return {
            "worker_outputs": [{"worker": "tutor", "task_id": f"practice_{current_step}", "status": "error", "content": "Error inicializando modelo.", "evidence": []}],
            "events": events,
        }

    # Build messages: system prompt + limited history (last 4 pairs) + current message
    chat_messages = [SystemMessage(content=practice_prompt)]
    chat_messages.extend(_build_practice_history(state, max_pairs=2))
    if user_message:
        chat_messages.append(HumanMessage(content=user_message))
    elif len(chat_messages) == 1:
        chat_messages.append(HumanMessage(content="(Alumno inicia la practica)"))

    logger.info("tutor_node", f"PROMPT PREVIEW (first 500): {practice_prompt[:500]}")
    logger.info("tutor_node", f"STEP FOCUS (first 200): {step_focus[:200]}")
    logger.info("tutor_node", f"CHAT MESSAGES COUNT: {len(chat_messages)}, types: {[type(m).__name__ for m in chat_messages]}")

    structured_llm = llm.with_structured_output(PracticeResponse)
    practice_chunks = []  # Multi-message chunks for SSE streaming

    # Decide execution mode:
    # - First entry to a step with tools → bind_tools, let LLM call them
    # - Re-entry where user explicitly requests action → bind_tools again
    # - Re-entry conversational (no action request) → standard flow, no tools
    # Allow tools on first entry, explicit action requests, OR simple confirmations
    # when tools haven't been successfully executed yet
    _confirmation_words = ["si", "sí", "ok", "va", "dale", "listo", "claro", "adelante", "despejado", "despejada", "libre", "seguro", "hecho", "ya", "continua", "continúa", "start", "empezar", "empieza", "comenzar"]
    user_confirms = any(kw in user_message.lower().split() for kw in _confirmation_words) if user_message else False

    use_tools = is_first_tool_entry or (tool_directives and user_requests_action) or (tool_directives and user_confirms and last_tool_step != current_step)

    if use_tools and tool_directives:
        # ══════════════════════════════════════════════════════════
        # TOOL EXECUTION FLOW: bind_tools → LLM generates calls → execute → interpret
        # ══════════════════════════════════════════════════════════
        try:
            # Bind ALL practice tools so LLM can use any of them
            step_tools = list(PRACTICE_TOOLS.values())
            llm_with_tools = llm.bind_tools(step_tools)

            tool_chat_messages = list(chat_messages)  # copy to avoid mutation

            # === PRE-PLAN: Quick chain-of-thought to decide tool usage ===
            plan_prompt = f"""Analiza rápido y responde SOLO en JSON:
- user_message: "{user_message}"
- step_tools_available: {list(PRACTICE_TOOLS.keys())}
- step_directives: {tool_directives}

¿Qué tools debes llamar y con qué args? Responde SOLO este JSON, nada más:
{{"tools": [{{"name": "tool_name", "args": {{...}}}}], "reasoning": "1 línea"}}

Si el usuario pide una acción (mover, leer, home, gripper), incluye esa tool.
Si el paso tiene directives y es primera vez, incluye las directives.
Si no se necesitan tools, responde: {{"tools": [], "reasoning": "solo texto"}}"""

            try:
                plan_llm = get_llm_from_name("gpt-4o-mini", temperature=0, max_tokens=200)
                plan_response = plan_llm.invoke([
                    SystemMessage(content="Eres un planner de tools. Solo responde JSON válido."),
                    HumanMessage(content=plan_prompt)
                ])
                plan_text = (plan_response.content or "").strip()
                # Clean markdown fences if present
                plan_text = plan_text.replace("```json", "").replace("```", "").strip()
                plan_data = json.loads(plan_text)
                planned_tools = [t["name"] for t in plan_data.get("tools", []) if t.get("name") in PRACTICE_TOOLS]
                logger.info("tutor_node", f"PRE-PLAN: {plan_data.get('reasoning', '?')} → tools={planned_tools}")

                # Inject plan into the user message so Phase 1 LLM knows what to do
                if planned_tools:
                    plan_matches_step = bool(set(planned_tools) & set(tool_directives))
                    if plan_matches_step or not tool_directives:
                        # Plan aligns with step directives OR step has no directives — proceed
                        tool_instruction = "\n\n[SYSTEM: Llama EXACTAMENTE estas tools: " + ", ".join(planned_tools)
                        for t in plan_data.get("tools", []):
                            if t.get("name") in PRACTICE_TOOLS and t.get("args"):
                                tool_instruction += f". {t['name']}({json.dumps(t['args'])})"
                        tool_instruction += ". NO llames ninguna otra tool.]"
                        tool_chat_messages[-1] = HumanMessage(content=user_message + tool_instruction)
                        use_tools = True
                    else:
                        # User asked for something outside current step — don't execute, explain
                        logger.info("tutor_node", f"PRE-PLAN MISMATCH: planned={planned_tools} vs step={tool_directives}, blocking execution")
                        redirect_instruction = f"\n\n[SYSTEM: El usuario pidió algo que NO corresponde al paso actual. NO llames ninguna tool. Explícale amablemente que primero deben completar el paso actual y qué falta hacer. Paso actual: {current_step}]"
                        tool_chat_messages[-1] = HumanMessage(content=user_message + redirect_instruction)
                        use_tools = False
            except Exception as plan_err:
                logger.warning("tutor_node", f"PRE-PLAN failed (continuing without): {plan_err}")

            # === PHASE 1: LLM generates announcement + tool_calls ===
            ai_response = llm_with_tools.invoke(tool_chat_messages)

            # Extract the text part as the "announcement" for SSE streaming
            announce_text = (ai_response.content or "").strip()
            if announce_text:
                announce_text = _clean_tool_leaks(announce_text)
                practice_chunks.append({"type": "partial", "content": announce_text})
                logger.info("tutor_node", f"TOOL PHASE 1 (announce): {announce_text[:200]}")

            # === PHASE 2: Execute tool calls the LLM made ===
            tool_messages = [ai_response]  # Start with AIMessage that contains tool_calls
            all_tool_results = []

            for tool_call in (ai_response.tool_calls or []):
                tc_name = tool_call["name"]
                tc_args = tool_call["args"]
                tc_id = tool_call["id"]

                practice_chunks.append({"type": "tool_status", "tool": tc_name, "status": "executing"})
                logger.info("tutor_node", f"TOOL PHASE 2 (execute): {tc_name} with args={tc_args}")

                try:
                    tool_fn = PRACTICE_TOOLS.get(tc_name)
                    if tool_fn is None:
                        raise ValueError(f"Tool '{tc_name}' not in PRACTICE_TOOLS")
                    result = tool_fn.invoke(tc_args)

                    # Retry logic: if result looks like an error, retry once after 3s
                    _is_error = False
                    if isinstance(result, str) and len(result) < 100 and "error" in result.lower():
                        _is_error = True
                    elif isinstance(result, str):
                        try:
                            _parsed = json.loads(result)
                            if isinstance(_parsed, dict) and _parsed.get("status") == "error":
                                _is_error = True
                        except (json.JSONDecodeError, TypeError):
                            pass

                    if _is_error:
                        logger.warning("tutor_node", f"TOOL RETRY: {tc_name} returned error, retrying in 3s...")
                        practice_chunks.append({"type": "partial", "content": "Parece que el robot se desconectó brevemente. Déjame intentar conectarme otra vez..."})
                        time.sleep(3)
                        result = tool_fn.invoke(tc_args)
                        logger.info("tutor_node", f"TOOL RETRY result: {tc_name}, result_len={len(result)}")

                    tool_messages.append(ToolMessage(content=result, tool_call_id=tc_id))
                    all_tool_results.append({"tool": tc_name, "args": tc_args, "result": result})
                    logger.info("tutor_node", f"TOOL PHASE 2 (result): {tc_name}, result_len={len(result)}")
                except Exception as te:
                    error_result = json.dumps({"error": str(te)})
                    tool_messages.append(ToolMessage(content=error_result, tool_call_id=tc_id))
                    all_tool_results.append({"tool": tc_name, "args": tc_args, "result": error_result})
                    logger.warning("tutor_node", f"TOOL FAILED: {tc_name}: {te}")

                practice_chunks.append({"type": "tool_status", "tool": tc_name, "status": "completed"})

            # === PHASE 3: LLM interprets results (may trigger auto-continue) ===
            if all_tool_results:
                interpret_messages = list(chat_messages) + tool_messages

                # Call with tools available so LLM can request more
                phase3_response = llm_with_tools.invoke(interpret_messages)
                phase3_text = (phase3_response.content or "").strip()
                logger.info("tutor_node", f"TOOL PHASE 3 (interpret): {phase3_text[:200]}")

                if phase3_text:
                    practice_chunks.append({"type": "partial", "content": _clean_tool_leaks(phase3_text)})

                # ── AUTO-CONTINUE: If LLM requests more tools, execute them ──
                max_auto_continues = 10
                auto_continue_count = 0

                while auto_continue_count < max_auto_continues:
                    if not hasattr(phase3_response, 'tool_calls') or not phase3_response.tool_calls:
                        break

                    auto_continue_count += 1
                    logger.info("tutor_node", f"AUTO-CONTINUE #{auto_continue_count}: LLM requested more tools")

                    interpret_messages.append(phase3_response)

                    for tc in phase3_response.tool_calls:
                        tc_name = tc["name"]
                        tc_args = tc.get("args", {})
                        tool_fn = PRACTICE_TOOLS.get(tc_name)
                        if not tool_fn:
                            continue

                        logger.info("tutor_node", f"AUTO-CONTINUE execute: {tc_name} with args={tc_args}")
                        practice_chunks.append({"type": "tool_status", "tool": tc_name, "status": "executing"})

                        result_str = tool_fn.invoke(tc_args)
                        logger.info("tutor_node", f"AUTO-CONTINUE result: {tc_name}, result_len={len(str(result_str))}")
                        practice_chunks.append({"type": "tool_status", "tool": tc_name, "status": "completed"})

                        interpret_messages.append(ToolMessage(content=str(result_str), tool_call_id=tc["id"]))

                    phase3_response = llm_with_tools.invoke(interpret_messages)
                    phase3_text = (phase3_response.content or "").strip()
                    logger.info("tutor_node", f"AUTO-CONTINUE interpret: {phase3_text[:200]}")

                    if phase3_text:
                        practice_chunks.append({"type": "partial", "content": _clean_tool_leaks(phase3_text)})

                if auto_continue_count > 0:
                    logger.info("tutor_node", f"AUTO-CONTINUE finished after {auto_continue_count} rounds")

                # Final structured response (PracticeResponse)
                interpret_messages.append(phase3_response)
                response = structured_llm.invoke(interpret_messages)
                clean_content = _clean_tool_leaks(response.message)
                practice_chunks.append({"type": "response", "content": clean_content})
            else:
                # LLM had tools available but chose not to call any
                logger.info("tutor_node", "TOOL PHASE 3: LLM chose not to call tools, using standard flow")
                response = structured_llm.invoke(chat_messages)
                clean_content = _clean_tool_leaks(response.message)
                practice_chunks = []  # No tool chunks to send

        except Exception as e:
            logger.warning("tutor_node", f"Tool flow failed: {e}, falling back to plain invoke")
            try:
                raw_response = llm.invoke(chat_messages)
                clean_content = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
            except Exception as e2:
                logger.error("tutor_node", f"Tool flow fallback also failed: {e2}")
                return {
                    "worker_outputs": [{"worker": "tutor", "task_id": f"practice_{current_step}", "status": "error", "content": "Error generando respuesta. Intenta de nuevo.", "evidence": []}],
                    "events": events,
                }
            response = PracticeResponse(message=clean_content, step_completed=False, new_step=current_step)
            practice_chunks = []  # Clear partial chunks on failure
    else:
        # ══════════════════════════════════════════════════════════
        # STANDARD FLOW: single LLM call (no tool)
        # ══════════════════════════════════════════════════════════
        try:
            response = structured_llm.invoke(chat_messages)
            clean_content = _clean_tool_leaks(response.message)
        except Exception as e:
            logger.warning("tutor_node", f"Structured output failed: {e}, falling back to plain invoke")
            try:
                raw_response = llm.invoke(chat_messages)
                clean_content = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
            except Exception as e2:
                logger.error("tutor_node", f"Practice fallback also failed: {e2}")
                return {
                    "worker_outputs": [{"worker": "tutor", "task_id": f"practice_{current_step}", "status": "error", "content": "Error generando respuesta. Intenta de nuevo.", "evidence": []}],
                    "events": events,
                }
            response = PracticeResponse(message=clean_content, step_completed=False, new_step=current_step)

    # ── Deterministic step management (don't trust LLM's new_step) ──
    if current_step >= total_steps and response.step_completed:
        practice_completed = True
        save_step = total_steps
    elif response.step_completed:
        save_step = min(current_step + 1, total_steps)
        practice_completed = False
    else:
        save_step = current_step
        practice_completed = False

    practice_update = {
        "step_completed": response.step_completed,
        "step": save_step,
        "observation": getattr(response, "observation", ""),
        "context_update": getattr(response, "context_update", ""),
        "practice_completed": practice_completed,
    }
    logger.info("tutor_node", f"STEP LOGIC: current={current_step}, total={total_steps}, step_completed={response.step_completed}, save_step={save_step}, practice_completed={practice_completed}")

    processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
    validated_step = practice_update.get("step", current_step)
    completed_flag = " COMPLETED" if practice_update.get("practice_completed") else ""
    events.append(event_report("tutor", f"Practica step {validated_step}/{total_steps} ({processing_time:.0f}ms){completed_flag}"))

    # ── Persist progress to Supabase ──
    automation_id = state.get("automation_id")
    auth_user_id = state.get("auth_user_id") or state.get("user_id")
    if automation_id and auth_user_id:
        try:
            from src.agent.services import get_supabase
            sb = get_supabase()
            if sb:
                update_data = {
                    "current_step": validated_step,
                    "status": "completed" if practice_update.get("practice_completed") else "in_progress",
                    "last_active_at": datetime.utcnow().isoformat(),
                }
                if practice_update.get("practice_completed"):
                    update_data["completed_at"] = datetime.utcnow().isoformat()

                observation = practice_update.get("observation", "")
                if observation:
                    # Append to existing observations instead of overwriting
                    existing_observations = state.get("automation_context", [])
                    if isinstance(existing_observations, str):
                        try:
                            existing_observations = json.loads(existing_observations)
                        except (json.JSONDecodeError, TypeError):
                            existing_observations = []
                    if not isinstance(existing_observations, list):
                        existing_observations = []
                    existing_observations.append({
                        "step": current_step,
                        "observation": observation,
                        "timestamp": datetime.utcnow().isoformat(),
                    })
                    update_data["agent_observations"] = json.dumps(existing_observations)

                sb.schema("lab").from_("user_automation_progress") \
                    .update(update_data) \
                    .eq("automation_id", automation_id) \
                    .eq("auth_user_id", auth_user_id) \
                    .execute()

                logger.info("tutor_node", f"PROGRESS SAVED - automation_id={automation_id}, step={validated_step}, completed={practice_update.get('practice_completed', False)}")
        except Exception as e:
            logger.warning("tutor_node", f"Failed to save progress: {e}")
    else:
        logger.warning("tutor_node", f"CANNOT SAVE PROGRESS - automation_id={automation_id}, auth_user_id={auth_user_id}")

    output = {
        "worker": "tutor",
        "task_id": f"practice_{current_step}",
        "status": "success",
        "content": clean_content,
        "evidence": [],
        "follow_up_suggestions": [],
    }
    if practice_update:
        output["practice_update"] = practice_update

    logger.node_end("tutor_node", {"mode": "practice", "step": validated_step, "completed": practice_update.get("practice_completed", False)})

    # Build accumulated observations for state persistence
    existing_observations = state.get("automation_context", [])
    if isinstance(existing_observations, str):
        try:
            existing_observations = json.loads(existing_observations)
        except (json.JSONDecodeError, TypeError):
            existing_observations = []
    if not isinstance(existing_observations, list):
        existing_observations = []
    obs_text = practice_update.get("observation", "")
    if obs_text:
        existing_observations.append({
            "step": current_step,
            "observation": obs_text,
            "timestamp": datetime.utcnow().isoformat(),
        })

    result = {
        "worker_outputs": [output],
        "events": events,
        "automation_step": validated_step,
        "automation_context": json.dumps(existing_observations),
    }
    # Always write practice_chunks to clear stale state from previous invocations
    result["practice_chunks"] = practice_chunks
    if practice_chunks:
        result["last_tool_step"] = current_step
    return result


# ============================================
# STANDARD TUTOR
# ============================================

def tutor_node(state: AgentState) -> Dict[str, Any]:
    """Worker tutor que genera contenido educativo."""
    # Practice mode branch
    if state.get("interaction_mode", "").lower() == "practice":
        return _handle_practice_mode(state)

    start_time = datetime.utcnow()
    logger.node_start("tutor_node", {"has_pending_context": bool(state.get("pending_context"))})
    events = [event_execute("tutor", "Preparando explicación educativa...")]
    
    user_message = get_last_user_message(state)
    if not user_message:
        error_output = create_error_output("tutor", "NO_MESSAGE", "No hay mensaje del usuario")
        return {"worker_outputs": [error_output.model_dump()], "tutor_result": error_output.model_dump_json(), "events": events}
    
    evidence_text, evidence_items = get_evidence_from_context(state)
    context_text = get_prior_summaries(state)
    has_evidence = len(evidence_items) > 0
    
    try:
        llm = get_llm(state, temperature=0.7)
    except Exception as e:
        error_output = create_error_output("tutor", "LLM_INIT_ERROR", f"Error inicializando modelo: {str(e)}")
        return {"worker_outputs": [error_output.model_dump()], "tutor_result": error_output.model_dump_json(), "events": events}
    
    # Obtener learning profile de la DB (solo cuando el tutor necesita explicar)
    try:
        from src.agent.utils.learning_profile import get_learning_prompt_section
        user_id = state.get("user_id")
        learning_style_guidance = get_learning_prompt_section(user_id) or MIX_TUTOR
    except Exception:
        learning_style_guidance = MIX_TUTOR

    prompt = TUTOR_MULTISTEP_PROMPT.format(
        context_section=context_text if context_text != "Sin contexto previo." else "Primera interacción",
        evidence_section=evidence_text,
        learning_style_guidance=learning_style_guidance,
        format_rules=MARKDOWN_FORMAT_RULES,
        user_name=state.get("user_name", "Usuario"),
    )
    
    messages = [SystemMessage(content=prompt)]
    if rolling_summary := state.get("rolling_summary", ""):
        messages.append(SystemMessage(content=f"Contexto de la conversación:\n{rolling_summary}"))
    # Add recent conversation history for context continuity
    history = _build_conversation_history(state, max_turns=4)
    messages.extend(history)
    messages.append(HumanMessage(content=user_message))
    
    tokens_used = 0
    try:
        response, tokens_used = invoke_and_track(llm, messages, "tutor")
        result_text = (response.content or "").strip()
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
    except Exception as e:
        error_output = create_error_output("tutor", "LLM_ERROR", f"Error generando respuesta: {str(e)}")
        return {"worker_outputs": [error_output.model_dump()], "tutor_result": error_output.model_dump_json(), "events": events}
    
    output = WorkerOutputBuilder.tutor(
        content=result_text,
        learning_objectives=["Comprender el concepto", "Aplicar en práctica"],
        summary=f"Explicación educativa generada ({len(result_text)} chars)",
        confidence=0.85 if has_evidence else 0.75,
    )
    if evidence_items:
        output.evidence = evidence_items
    output.metadata.completed_at = datetime.utcnow().isoformat()
    output.metadata.processing_time_ms = processing_time
    output.metadata.model_used = state.get("llm_model") or os.getenv("DEFAULT_MODEL", "gemini-2.0-flash")
    
    logger.node_end("tutor_node", {"content_length": len(result_text)})
    events.append(event_report("tutor", " Explicación lista"))
    
    return {"worker_outputs": [output.model_dump()], "tutor_result": output.model_dump_json(), "events": events, "token_usage": tokens_used}