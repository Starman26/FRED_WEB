"""
orchestrator.py - Supervisor Orchestrator Multi-Step

Este módulo implementa la lógica de orquestación que permite:
1. Planificar una secuencia de workers basada en la solicitud del usuario
2. Ejecutar workers en orden, pasando contexto entre ellos
3. Detectar cuando se necesita human-in-the-loop
4. Sintetizar los resultados de múltiples workers

FLUJO:
Usuario → plan_node → [worker1 → route → worker2 → route → ...] → synthesize → Usuario
"""
import os
import json
from typing import Dict, Any, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from src.agent.state import AgentState
from src.agent.contracts.worker_contract import WorkerOutput, EvidenceItem
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_plan, event_route, event_report, event_error


# ============================================
# PROMPTS PARA ORCHESTRATION
# ============================================

ORCHESTRATOR_PLAN_PROMPT = """Eres el **Orchestrator** de un sistema multi-agente especializado en:
- Documentos técnicos y papers (PLCs, Cobots, AI/ML)
- Troubleshooting de sistemas industriales
- Tutorías y explicaciones educativas

Tu trabajo es ANALIZAR la solicitud del usuario y crear un PLAN de ejecución.

WORKERS DISPONIBLES:
1. **chat**: Conversación general, saludos, presentaciones, preguntas simples.
   - Usar cuando: saludos ("hola", "me llamo X"), preguntas casuales, agradecimientos, despedidas
   - NO usar para: preguntas técnicas, búsqueda de información

2. **research**: Busca en documentos/papers internos usando RAG. Retorna evidencia con citas y páginas.
   - Usar cuando: mencionen "paper", "documento", "buscar", "referencia", "según el paper"

3. **troubleshooting**: Diagnostica problemas, obtiene datos de máquinas/logs/alertas.
   - Usar cuando: mencionen "error", "falla", "problema", "log", "máquina", "diagnóstico"

4. **tutor**: Explica conceptos, sintetiza información de forma educativa.
   - Usar cuando: pidan "explicar", "cómo funciona", "enséñame", o necesiten sintetizar info de research

5. **summarizer**: Comprime memoria cuando hay muchos mensajes (automático, no incluir en plan normal)

REGLAS DE PLANIFICACIÓN:

1. **Saludo/presentación/conversación casual**:
   → Plan: ["chat"]
   Ejemplo: "Hola", "Me llamo Juan", "Gracias" → ["chat"]

2. **Solicitud simple** (una sola tarea clara):
   → Plan de 1 worker
   Ejemplo: "Explícame async/await" → ["tutor"]

3. **Buscar información en papers**:
   → Plan: ["research"]
   Si piden SOLO buscar, no añadir tutor

4. **Buscar Y explicar un paper**:
   → Plan: ["research", "tutor"]
   Primero buscar evidencia, luego explicar/sintetizar

5. **Diagnóstico técnico**:
   → Plan: ["troubleshooting"]
   Si mencionan documentación también: ["research", "troubleshooting"]

FORMATO DE RESPUESTA (JSON estricto):
{{
  "plan": ["worker1", "worker2"],
  "reasoning": "Breve explicación del por qué",
  "initial_context": {{"key": "value si hay contexto específico"}}
}}

IMPORTANTE:
- Máximo 3 workers en el plan
- Para saludos y conversación casual, SIEMPRE usar ["chat"]
- Si el usuario solo quiere buscar info, usar solo ["research"]
- Solo añadir tutor si el usuario pide explicación
"""

ORCHESTRATOR_SYNTHESIZE_PROMPT = """Eres un **Sintetizador** que combina resultados de múltiples workers.

Tu tarea:
1. Combinar la información de forma coherente
2. Mantener las citas y referencias originales
3. Estructurar la respuesta de forma clara
4. NO inventar información que no esté en los outputs

OUTPUTS DE WORKERS:
{worker_outputs}

CONSULTA ORIGINAL DEL USUARIO:
{user_query}

Genera una respuesta final que:
- Integre toda la información relevante
- Mantenga las citas en formato [Título, Pág. X-Y]
- Sea clara y bien estructurada
- Incluya una sección de "Fuentes" al final si hay evidencia citada
"""


# ============================================
# NODOS DE ORCHESTRATION
# ============================================

def orchestrator_plan_node(state: AgentState) -> Dict[str, Any]:
    """
    Nodo inicial: Analiza la solicitud y crea el plan de ejecución.
    
    NUEVO: Usa intent_analysis del state para mejor decisión.
    
    Este nodo:
    1. Revisa el intent_analysis (ya calculado por intent_analyzer_node)
    2. Usa el suggested_worker como base
    3. Puede usar LLM para casos complejos
    4. Retorna el primer worker a ejecutar
    """
    logger.node_start("orchestrator_plan", {
        "messages_count": len(state.get("messages", []))
    })
    
    events = [event_plan("orchestrator", "Analizando solicitud y creando plan...")]
    
    # Obtener intent_analysis (calculado por intent_analyzer_node)
    intent_analysis = state.get("intent_analysis", {})
    
    # Obtener último mensaje del usuario
    messages = state.get("messages", [])
    user_message = ""
    
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_message = m.content
            break
        if isinstance(m, dict):
            role = m.get("role") or m.get("type")
            if role in ("user", "human"):
                user_message = m.get("content", "")
                break
    
    if not user_message:
        logger.warning("orchestrator_plan", "No se encontró mensaje del usuario")
        return {
            "orchestration_plan": ["chat"],
            "current_step": 0,
            "worker_outputs": [],
            "clarification_questions": [],
            "needs_human_input": False,
            "next": "chat",
            "events": events + [event_error("orchestrator", "Sin mensaje del usuario")],
        }
    
    # ==========================================
    # USAR INTENT ANALYSIS SI ESTÁ DISPONIBLE
    # ==========================================
    if intent_analysis:
        intent = intent_analysis.get("intent", "chat")
        suggested_worker = intent_analysis.get("suggested_worker", "chat")
        action = intent_analysis.get("action")
        entities = intent_analysis.get("entities", {})
        needs_clarification = intent_analysis.get("needs_clarification", False)
        
        # Construir contexto inicial desde el análisis
        initial_context = {
            "intent": intent,
            "action": action,
            "entities": entities,
            "urgency": intent_analysis.get("urgency", "low"),
            "sentiment": intent_analysis.get("sentiment", "neutral"),
        }
        
        # Mapear intent a plan
        if intent == "command":
            plan = ["troubleshooting"]
            reasoning = f"Comando detectado: {action}"
        elif intent == "query":
            plan = ["troubleshooting"]
            reasoning = f"Consulta de estado: {action}"
        elif intent == "troubleshoot":
            plan = ["troubleshooting"]
            reasoning = "Reporte de problema técnico"
        elif intent == "learn":
            # Si es sobre el lab, podría necesitar research + tutor
            if any(kw in user_message.lower() for kw in ["paper", "documento", "buscar"]):
                plan = ["research", "tutor"]
                reasoning = "Aprendizaje con búsqueda de documentos"
            else:
                plan = ["tutor"]
                reasoning = "Explicación educativa"
        else:
            plan = ["chat"]
            reasoning = "Conversación general"
        
        logger.info("orchestrator_plan", f"Plan desde intent_analysis: {plan} - {reasoning}")
        
        # Crear anuncio
        announcement = _create_plan_announcement(plan)
        
        events.append(event_plan("orchestrator", f"Plan: {' → '.join(plan)}"))
        
        return {
            "orchestration_plan": plan,
            "current_step": 0,
            "pending_context": initial_context,
            "worker_outputs": [],
            "clarification_questions": [],
            "needs_human_input": False,
            "next": plan[0],
            "task_type": plan[0],
            "messages": [AIMessage(content=announcement)] if announcement else [],
            "events": events,
        }
    
    # ==========================================
    # FALLBACK: Usar LLM si no hay intent_analysis
    # ==========================================
    model_name = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-20250514")
    
    try:
        # Intentar con Anthropic primero
        if "claude" in model_name.lower():
            llm = ChatAnthropic(model=model_name, temperature=0.3)
        else:
            llm = ChatOpenAI(model=model_name, temperature=0.3)
        
        response = llm.invoke([
            SystemMessage(content=ORCHESTRATOR_PLAN_PROMPT),
            HumanMessage(content=f"SOLICITUD DEL USUARIO:\n{user_message}")
        ])
        
        response_text = response.content.strip()
        
        # Parsear JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        plan_data = json.loads(response_text)
        plan = plan_data.get("plan", ["tutor"])
        reasoning = plan_data.get("reasoning", "")
        initial_context = plan_data.get("initial_context", {})
        
        # Validar que los workers existan
        valid_workers = {"chat", "research", "tutor", "troubleshooting", "summarizer"}
        plan = [w for w in plan if w in valid_workers]
        
        if not plan:
            plan = ["chat"]
        
        logger.info("orchestrator_plan", f"Plan creado: {plan} - Razón: {reasoning}")
        
        # Crear anuncio para el usuario
        announcement = _create_plan_announcement(plan)
        
        events.append(event_plan("orchestrator", f"Plan: {' → '.join(plan)}"))
        
        return {
            "orchestration_plan": plan,
            "current_step": 0,
            "pending_context": initial_context,
            "worker_outputs": [],  # RESETEAR para esta ejecución
            "clarification_questions": [],  # LIMPIAR
            "needs_human_input": False,
            "next": plan[0],
            "task_type": plan[0],
            "messages": [AIMessage(content=announcement)] if announcement else [],
            "events": events,
        }
        
    except json.JSONDecodeError as e:
        logger.error("orchestrator_plan", f"Error parseando plan JSON: {e}")
        # Fallback: plan simple basado en keywords
        plan = _fallback_plan_from_keywords(user_message)
        
        return {
            "orchestration_plan": plan,
            "current_step": 0,
            "worker_outputs": [],
            "clarification_questions": [],  # LIMPIAR
            "needs_human_input": False,
            "next": plan[0] if plan else "chat",
            "events": events + [event_error("orchestrator", "Usando plan fallback")],
        }
        
    except Exception as e:
        logger.error("orchestrator_plan", f"Error creando plan: {e}")
        return {
            "orchestration_plan": ["chat"],
            "current_step": 0,
            "worker_outputs": [],
            "clarification_questions": [],  # LIMPIAR
            "needs_human_input": False,
            "next": "chat",
            "events": events + [event_error("orchestrator", str(e))],
        }


def orchestrator_route_node(state: AgentState) -> Dict[str, Any]:
    """
    Nodo de decisión: Después de cada worker, decide el siguiente paso.
    """
    logger.node_start("orchestrator_route", {
        "plan": state.get("orchestration_plan"),
        "current_step": state.get("current_step"),
    })
    
    plan = state.get("orchestration_plan", [])
    current_step = state.get("current_step", 0)
    worker_outputs = state.get("worker_outputs", [])
    pending_context = state.get("pending_context", {}) or {}
    
    # IMPORTANTE: Buscar preguntas en el STATE también
    state_questions = state.get("clarification_questions", [])
    
    events = [event_route("orchestrator", "Evaluando siguiente paso...", route="pending")]
    
    # Obtener el último output del worker
    last_output = worker_outputs[-1] if worker_outputs else None
    
    # Verificar si ya tenemos clarificación del usuario
    has_user_clarification = bool(pending_context.get("user_clarification"))
    wizard_completed = pending_context.get("wizard_completed", False)
    
    logger.info("orchestrator_route", f"has_clarification={has_user_clarification}, wizard_completed={wizard_completed}, last_output_status={last_output.get('status') if last_output else 'None'}")
    
    # ==========================================
    # CASO 1: Usuario ya respondió - re-ejecutar worker
    # ==========================================
    if has_user_clarification or wizard_completed:
        current_worker = plan[current_step] if current_step < len(plan) else None
        if current_worker:
            logger.info("orchestrator_route", f"✅ Re-ejecutando {current_worker} con clarificación del usuario")
            
            # Limpiar outputs con needs_context para evitar loop
            updated_outputs = [o for o in worker_outputs if o.get("status") != "needs_context"]
            
            return {
                "worker_outputs": updated_outputs,
                "clarification_questions": [],  # Limpiar
                "needs_human_input": False,
                "next": current_worker,
                "task_type": current_worker,
                "events": events + [event_route("orchestrator", f"Continuando {current_worker} con info del usuario", route=current_worker)],
            }
    
    # ==========================================
    # CASO 2: Worker necesita input - ir a human_input
    # ==========================================
    if last_output and last_output.get("status") == "needs_context":
        # Buscar preguntas: primero en worker_output, luego en state
        questions = last_output.get("clarification_questions", [])
        if not questions:
            questions = state_questions
        
        logger.info("orchestrator_route", f"Worker necesita contexto, {len(questions)} preguntas")
        
        return {
            "needs_human_input": True,
            "clarification_questions": questions,
            "next": "human_input",
            "events": events + [event_route("orchestrator", "Requiere input humano", route="human_input")],
        }
    
    # ==========================================
    # CASO 3: Avanzar al siguiente step
    # ==========================================
    next_step = current_step + 1
    
    if next_step >= len(plan):
        logger.info("orchestrator_route", "Plan completado, sintetizando...")
        return {
            "orchestration_plan": plan,
            "current_step": next_step,
            "done": True,
            "next": "synthesize",
            "events": events + [event_route("orchestrator", "Plan completado", route="synthesize")],
        }
    
    # Preparar contexto para el siguiente worker
    for output in worker_outputs:
        evidence = output.get("evidence", [])
        if evidence:
            pending_context["evidence"] = pending_context.get("evidence", []) + evidence
    
    next_worker = plan[next_step]
    logger.info("orchestrator_route", f"Siguiente worker: {next_worker}")
    
    return {
        "orchestration_plan": plan,
        "current_step": next_step,
        "pending_context": pending_context,
        "next": next_worker,
        "task_type": next_worker,
        "events": events + [event_route("orchestrator", f"Continuando con {next_worker}", route=next_worker)],
    }

def synthesize_node(state: AgentState) -> Dict[str, Any]:
    """
    Nodo final: Combina todos los outputs de workers en una respuesta coherente.
    
    Este nodo:
    1. Recopila todos los worker_outputs
    2. Extrae contenido y evidencia
    3. Genera una respuesta sintetizada
    4. Incluye fuentes al final
    """
    logger.node_start("synthesize", {
        "outputs_count": len(state.get("worker_outputs", []))
    })
    
    events = [event_report("synthesize", "Combinando resultados...")]
    
    worker_outputs = state.get("worker_outputs", [])
    
    if not worker_outputs:
        logger.warning("synthesize", "No hay outputs de workers")
        return {
            "messages": [AIMessage(content="No se generaron resultados para tu solicitud.")],
            "done": False,
            "next": "END",
            "events": events,
        }
    
    # ==========================================
    # Recopilar contenido y fuentes
    # ==========================================
    content_parts = []
    all_sources = set()
    
    for output in worker_outputs:
        content = output.get("content", "")
        worker_name = output.get("worker", "unknown")
        
        if content:
            # Si hay múltiples workers, añadir separador
            if len(worker_outputs) > 1:
                content_parts.append(f"**{_worker_display_name(worker_name)}:**\n{content}")
            else:
                content_parts.append(content)
        
        # Recopilar fuentes de la evidencia
        for ev in output.get("evidence", []):
            title = ev.get("title", "")
            page = ev.get("page", "?")
            if title:
                all_sources.add(f"{title} (Pág. {page})")
    
    # ==========================================
    # Construir respuesta final
    # ==========================================
    if len(content_parts) == 1:
        combined_content = content_parts[0]
    else:
        combined_content = "\n\n---\n\n".join(content_parts)
    
    # Añadir fuentes si las hay
    if all_sources:
        combined_content += "\n\n---\n**Fuentes consultadas:**\n"
        for source in sorted(all_sources):
            combined_content += f"- {source}\n"
    
    logger.node_end("synthesize", {"content_length": len(combined_content)})
    
    return {
        "messages": [AIMessage(content=combined_content)],
        "worker_outputs": [],  # LIMPIAR para evitar acumulación
        "pending_context": {},  # LIMPIAR contexto
        "clarification_questions": [],  # LIMPIAR
        "orchestration_plan": [],  # LIMPIAR plan
        "current_step": 0,
        "done": False,
        "next": "END",
        "events": events + [event_report("synthesize", "✅ Respuesta sintetizada")],
    }


# ============================================
# HELPERS
# ============================================

def _create_plan_announcement(plan: List[str]) -> str:
    """Crea un mini anuncio del plan para el usuario"""
    if len(plan) == 1:
        worker = plan[0]
        announcements = {
            "chat": "",  # Sin anuncio para chat (respuesta directa)
            "research": "Voy a buscar en los documentos...",
            "tutor": "Voy a prepararte una explicación.",
            "troubleshooting": "",  # Sin anuncio - las acciones hablan por sí mismas
            "summarizer": "Comprimiendo memoria...",
        }
        return announcements.get(worker, "")
    else:
        steps = [_worker_display_name(w) for w in plan if w != "chat"]
        if steps:
            return f"Voy a {' → '.join(steps).lower()}."
        return ""


def _worker_display_name(worker: str) -> str:
    """Nombre legible del worker"""
    names = {
        "chat": "Responder",
        "research": "Investigar",
        "tutor": "Explicar",
        "troubleshooting": "Diagnosticar",
        "summarizer": "Resumir",
    }
    return names.get(worker, worker.capitalize())


def _fallback_plan_from_keywords(text: str) -> List[str]:
    """
    Plan fallback basado en keywords cuando el LLM falla.
    """
    text_lower = text.lower()
    plan = []
    
    # Detectar chat/saludo
    chat_keywords = ["hola", "hey", "buenas", "gracias", "me llamo", "mi nombre es", "adios", "bye"]
    if any(kw in text_lower for kw in chat_keywords):
        return ["chat"]
    
    # Detectar research
    research_keywords = ["paper", "documento", "buscar", "referencia", "según", "cita", "fuente"]
    if any(kw in text_lower for kw in research_keywords):
        plan.append("research")
    
    # Detectar troubleshooting
    trouble_keywords = ["error", "falla", "problema", "log", "máquina", "diagnóstico", "no funciona"]
    if any(kw in text_lower for kw in trouble_keywords):
        plan.append("troubleshooting")
    
    # Detectar tutor
    tutor_keywords = ["explicar", "cómo", "enséñame", "qué es", "tutorial", "aprender"]
    if any(kw in text_lower for kw in tutor_keywords):
        plan.append("tutor")
    
    # Fallback default
    if not plan:
        plan = ["chat"]
    
    return plan


def get_orchestration_status(state: AgentState) -> Dict[str, Any]:
    """
    Helper para debugging: obtiene el estado actual de la orquestación.
    """
    plan = state.get("orchestration_plan", [])
    step = state.get("current_step", 0)
    
    return {
        "plan": plan,
        "current_step": step,
        "current_worker": plan[step] if step < len(plan) else "DONE",
        "remaining_workers": plan[step + 1:] if step < len(plan) else [],
        "outputs_count": len(state.get("worker_outputs", [])),
        "has_pending_context": bool(state.get("pending_context")),
        "needs_human_input": state.get("needs_human_input", False),
    }
