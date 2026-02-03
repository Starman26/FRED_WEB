"""
intent_analyzer.py - Nodo de análisis de intención del usuario

Este nodo procesa el mensaje del usuario y extrae:
- Intención principal (query, command, troubleshoot, chat, learn)
- Acción específica si es un comando
- Entidades mencionadas (estación, equipo, rutina, etc.)
- Urgencia y sentimiento
- Si necesita clarificación

El análisis se comparte con orchestrator y workers para mejor comprensión.
"""
import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from src.agent.state import AgentState
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_execute, event_report, event_error


# ============================================
# SCHEMA DE ANÁLISIS DE INTENCIÓN
# ============================================

INTENT_ANALYSIS_PROMPT = '''Eres un analizador de intenciones para un sistema de control de laboratorio industrial.

Tu tarea es analizar el mensaje del usuario y extraer información estructurada.

**Mensaje del usuario:**
"{user_message}"

**Historial reciente (si hay):**
{conversation_history}

**Contexto del sistema:**
- Este es un laboratorio con 6 estaciones de manufactura
- Cada estación tiene: PLC (controlador), Cobot (robot colaborativo), sensores de puerta
- El usuario puede: consultar estado, ejecutar comandos, reportar problemas, hacer preguntas

**Analiza y responde SOLO con este JSON:**
{{
  "intent": "<tipo de intención>",
  "action": "<acción específica o null>",
  "entities": {{
    "station": <número 1-6 o null>,
    "equipment": "<plc|cobot|door|sensor o null>",
    "routine": <número de rutina o null>,
    "error_type": "<tipo de error mencionado o null>"
  }},
  "context_clues": ["<pistas contextuales importantes>"],
  "urgency": "<low|medium|high|critical>",
  "sentiment": "<neutral|confused|frustrated|urgent|casual>",
  "suggested_worker": "<chat|troubleshooting|research|tutor>",
  "needs_clarification": <true|false>,
  "clarification_reason": "<razón si necesita clarificación o null>",
  "summary": "<resumen breve de lo que el usuario quiere>"
}}

**Tipos de intención (intent):**
- "command": El usuario quiere ejecutar una acción (iniciar, parar, cerrar, reset)
- "query": El usuario pregunta por estado/información (¿hay errores?, estado del lab)
- "troubleshoot": El usuario reporta un problema o pide diagnóstico
- "learn": El usuario quiere aprender/entender algo (¿cómo funciona?, explícame)
- "chat": Conversación general, saludos, agradecimientos

**Acciones reconocidas (action):**
- "start_cobot": Iniciar cobot/rutina
- "stop_cobot": Parar cobot/rutina  
- "check_errors": Consultar errores activos
- "check_status": Ver estado general o de estación
- "close_doors": Cerrar puertas
- "reconnect_plc": Reconectar PLC
- "reset_lab": Reset completo del laboratorio
- "resolve_errors": Limpiar/resolver errores
- "auto_fix": Reparación automática
- null: Si no es un comando específico

**Ejemplos:**

Mensaje: "inicia la rutina 2 en estación 3"
→ intent: "command", action: "start_cobot", entities: {{station: 3, routine: 2}}

Mensaje: "hay algun error en el lab?"  
→ intent: "query", action: "check_errors", needs_clarification: false

Mensaje: "la plc no conecta"
→ intent: "troubleshoot", action: null, entities: {{equipment: "plc"}}, needs_clarification: true (¿qué estación?)

Mensaje: "gracias!"
→ intent: "chat", action: null, sentiment: "casual"

Mensaje: "como funciona el cobot?"
→ intent: "learn", action: null, suggested_worker: "tutor"

Mensaje: "quiero iniciar la práctica de ABB" o "iniciar práctica" o "prácticas disponibles"
→ intent: "learn", action: "start_practice", suggested_worker: "tutor"
(IMPORTANTE: "práctica" siempre es intent=learn, NO es un comando de cobot)

Mensaje: "siguiente paso" o "continuar práctica"
→ intent: "learn", action: "continue_practice", suggested_worker: "tutor"

Ahora analiza el mensaje del usuario:'''


def get_conversation_history(state: AgentState, max_messages: int = 4) -> str:
    """Extrae historial reciente de la conversación"""
    messages = state.get("messages", [])
    if not messages:
        return "No hay historial previo."
    
    history_lines = []
    recent = messages[-max_messages:] if len(messages) > max_messages else messages
    
    for msg in recent:
        if isinstance(msg, HumanMessage):
            history_lines.append(f"Usuario: {msg.content[:100]}...")
        elif isinstance(msg, AIMessage):
            history_lines.append(f"Asistente: {msg.content[:100]}...")
        elif isinstance(msg, dict):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:100]
            history_lines.append(f"{role.capitalize()}: {content}...")
    
    return "\n".join(history_lines) if history_lines else "No hay historial previo."


def intent_analyzer_node(state: AgentState) -> Dict[str, Any]:
    """
    Nodo que analiza la intención del usuario usando LLM.
    
    Extrae:
    - intent: Tipo de intención (command, query, troubleshoot, learn, chat)
    - action: Acción específica si es comando
    - entities: Entidades mencionadas (estación, equipo, etc.)
    - urgency/sentiment: Contexto emocional
    - suggested_worker: Worker recomendado
    - needs_clarification: Si necesita más info
    """
    start_time = datetime.utcnow()
    logger.node_start("intent_analyzer", {})
    
    events = [event_execute("intent_analyzer", "Analizando intención del usuario...")]
    
    # Obtener último mensaje del usuario
    user_message = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
        elif isinstance(msg, dict) and msg.get("role") in ("human", "user"):
            user_message = msg.get("content", "")
            break
    
    if not user_message:
        logger.warning("intent_analyzer", "No se encontró mensaje del usuario")
        return {
            "intent_analysis": {
                "intent": "chat",
                "action": None,
                "entities": {},
                "urgency": "low",
                "sentiment": "neutral",
                "suggested_worker": "chat",
                "needs_clarification": False,
                "summary": "Mensaje vacío"
            },
            "events": events,
        }
    
    # Obtener historial
    conversation_history = get_conversation_history(state)
    
    # Crear LLM
    model_name = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-20250514")
    try:
        if "claude" in model_name.lower():
            llm = ChatAnthropic(model=model_name, temperature=0.1)
        else:
            llm = ChatOpenAI(model=model_name, temperature=0.1)
    except Exception as e:
        logger.error("intent_analyzer", f"Error creando LLM: {e}")
        # Fallback básico
        return {
            "intent_analysis": _fallback_analysis(user_message),
            "events": events + [event_error("intent_analyzer", f"LLM error: {e}")],
        }
    
    # Ejecutar análisis
    try:
        prompt = INTENT_ANALYSIS_PROMPT.format(
            user_message=user_message,
            conversation_history=conversation_history
        )
        
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        
        # Extraer JSON
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            logger.warning("intent_analyzer", f"No JSON en respuesta: {content[:200]}")
            return {
                "intent_analysis": _fallback_analysis(user_message),
                "events": events,
            }
        
        json_str = content[start_idx:end_idx]
        analysis = json.loads(json_str)
        
        # Validar campos requeridos
        analysis.setdefault("intent", "chat")
        analysis.setdefault("action", None)
        analysis.setdefault("entities", {})
        analysis.setdefault("urgency", "low")
        analysis.setdefault("sentiment", "neutral")
        analysis.setdefault("suggested_worker", "chat")
        analysis.setdefault("needs_clarification", False)
        analysis.setdefault("summary", user_message[:50])
        
        # =================================================================
        # OVERRIDE: Detectar intents de PRÁCTICA (siempre va a tutor)
        # =================================================================
        analysis = _override_practice_intent(user_message, analysis)
        
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        logger.info("intent_analyzer", f"Análisis completado en {elapsed:.2f}s: intent={analysis['intent']}, action={analysis['action']}")
        
        events.append(event_report("intent_analyzer", f"✅ Intent: {analysis['intent']} | Action: {analysis['action']}"))
        
        return {
            "intent_analysis": analysis,
            "events": events,
        }
        
    except json.JSONDecodeError as e:
        logger.error("intent_analyzer", f"Error parseando JSON: {e}")
        return {
            "intent_analysis": _fallback_analysis(user_message),
            "events": events + [event_error("intent_analyzer", "JSON parse error")],
        }
    except Exception as e:
        logger.error("intent_analyzer", f"Error en análisis: {e}")
        return {
            "intent_analysis": _fallback_analysis(user_message),
            "events": events + [event_error("intent_analyzer", str(e))],
        }


def _override_practice_intent(message: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detecta si el mensaje es sobre prácticas de laboratorio y ajusta el intent.
    Las prácticas SIEMPRE van al tutor, nunca a troubleshooting.
    """
    msg = message.lower()
    
    # Palabras clave de práctica
    practice_keywords = [
        "práctica", "practica", "practice",
        "prácticas disponibles", "lista de prácticas",
        "iniciar práctica", "comenzar práctica", "empezar práctica",
        "siguiente paso", "continuar práctica", "siguiente",
        "pista", "hint", "ayuda con la práctica",
        "terminar práctica", "finalizar práctica",
    ]
    
    # Verificar si es una solicitud de práctica
    is_practice_request = any(kw in msg for kw in practice_keywords)
    
    if is_practice_request:
        # Determinar acción específica
        if any(kw in msg for kw in ["lista", "disponibles", "qué prácticas", "cuáles prácticas"]):
            action = "list_practices"
        elif any(kw in msg for kw in ["iniciar", "comenzar", "empezar", "start"]):
            action = "start_practice"
        elif any(kw in msg for kw in ["siguiente", "continuar", "next", "listo", "ok"]):
            action = "continue_practice"
        elif any(kw in msg for kw in ["pista", "hint", "ayuda"]):
            action = "practice_hint"
        elif any(kw in msg for kw in ["terminar", "finalizar", "complete", "finish"]):
            action = "complete_practice"
        else:
            action = "start_practice"
        
        # Override el análisis
        analysis["intent"] = "learn"
        analysis["action"] = action
        analysis["suggested_worker"] = "tutor"
        analysis["needs_clarification"] = False
        analysis["context_clues"].append("practice_mode")
        analysis["summary"] = f"Solicitud de práctica: {action}"
        
        logger.info("intent_analyzer", f"[OVERRIDE] Práctica detectada: {action}")
    
    return analysis


def _fallback_analysis(message: str) -> Dict[str, Any]:
    """
    Análisis básico sin LLM como fallback.
    Usa reglas simples para detectar intent.
    """
    msg = message.lower()
    
    analysis = {
        "intent": "chat",
        "action": None,
        "entities": {
            "station": None,
            "equipment": None,
            "routine": None,
        },
        "context_clues": [],
        "urgency": "low",
        "sentiment": "neutral",
        "suggested_worker": "chat",
        "needs_clarification": False,
        "clarification_reason": None,
        "summary": message[:50]
    }
    
    # Detectar estación
    for i in range(1, 7):
        if f"estacion {i}" in msg or f"estación {i}" in msg or f"est {i}" in msg or f"est. {i}" in msg:
            analysis["entities"]["station"] = i
            break
    
    # Detectar equipo
    if "plc" in msg:
        analysis["entities"]["equipment"] = "plc"
    elif "cobot" in msg or "robot" in msg:
        analysis["entities"]["equipment"] = "cobot"
    elif "puerta" in msg or "door" in msg:
        analysis["entities"]["equipment"] = "door"
    
    # Detectar rutina
    for i in range(1, 5):
        if f"rutina {i}" in msg or f"routine {i}" in msg or f"modo {i}" in msg:
            analysis["entities"]["routine"] = i
            break
    
    # =================================================================
    # PRIMERO: Detectar prácticas (tienen prioridad sobre comandos)
    # =================================================================
    practice_keywords = ["práctica", "practica", "practice", "prácticas"]
    if any(kw in msg for kw in practice_keywords):
        analysis["intent"] = "learn"
        analysis["suggested_worker"] = "tutor"
        if any(kw in msg for kw in ["lista", "disponibles", "cuáles"]):
            analysis["action"] = "list_practices"
        elif any(kw in msg for kw in ["iniciar", "comenzar", "empezar", "start"]):
            analysis["action"] = "start_practice"
        elif any(kw in msg for kw in ["siguiente", "continuar", "next"]):
            analysis["action"] = "continue_practice"
        else:
            analysis["action"] = "start_practice"
        analysis["context_clues"].append("practice_mode")
        return analysis  # Retornar inmediatamente, no procesar como comando
    
    # Detectar intención y acción
    command_keywords = {
        "start_cobot": ["iniciar", "inicia", "arranca", "ejecuta", "start", "enciende", "activa"],
        "stop_cobot": ["parar", "para", "detener", "deten", "stop", "apaga", "apagar"],
        "close_doors": ["cierra", "cerrar", "close"],
        "reconnect_plc": ["reconecta", "reconectar", "reconnect"],
        "reset_lab": ["reset", "reinicia", "reiniciar", "restaura"],
        "resolve_errors": ["resolver", "resuelve", "limpia", "clear"],
        "auto_fix": ["arregla", "arréglalo", "fix", "repara"],
    }
    
    for action, keywords in command_keywords.items():
        if any(kw in msg for kw in keywords):
            analysis["intent"] = "command"
            analysis["action"] = action
            analysis["suggested_worker"] = "troubleshooting"
            break
    
    # Detectar query
    query_keywords = {
        "check_errors": ["errores", "error", "falla", "problemas"],
        "check_status": ["estado", "status", "como está", "como esta"],
    }
    
    if analysis["intent"] == "chat":
        for action, keywords in query_keywords.items():
            if any(kw in msg for kw in keywords):
                analysis["intent"] = "query"
                analysis["action"] = action
                analysis["suggested_worker"] = "troubleshooting"
                break
    
    # Detectar troubleshoot
    problem_keywords = ["no conecta", "no funciona", "no responde", "falló", "fallo", "problema", "ayuda"]
    if any(kw in msg for kw in problem_keywords):
        analysis["intent"] = "troubleshoot"
        analysis["suggested_worker"] = "troubleshooting"
        if not analysis["entities"]["station"]:
            analysis["needs_clarification"] = True
            analysis["clarification_reason"] = "No se especificó la estación"
    
    # Detectar learn
    learn_keywords = ["como funciona", "cómo funciona", "explica", "qué es", "que es", "enseña", "aprend"]
    if any(kw in msg for kw in learn_keywords):
        analysis["intent"] = "learn"
        analysis["suggested_worker"] = "tutor"
    
    # Detectar urgencia
    if any(kw in msg for kw in ["urgente", "crítico", "critico", "emergencia", "grave"]):
        analysis["urgency"] = "high"
    elif any(kw in msg for kw in ["error", "falla", "no funciona"]):
        analysis["urgency"] = "medium"
    
    # Detectar sentimiento
    if any(kw in msg for kw in ["gracias", "genial", "perfecto", "excelente"]):
        analysis["sentiment"] = "casual"
    elif any(kw in msg for kw in ["no entiendo", "confundido", "cual", "cuál"]):
        analysis["sentiment"] = "confused"
    elif any(kw in msg for kw in ["ya intenté", "otra vez", "sigue sin", "todavía"]):
        analysis["sentiment"] = "frustrated"
    
    return analysis


# ============================================
# HELPERS PARA USAR EL ANÁLISIS
# ============================================

def get_intent_analysis(state: AgentState) -> Dict[str, Any]:
    """Helper para obtener el análisis de intención del state"""
    return state.get("intent_analysis", {})


def is_command_intent(state: AgentState) -> bool:
    """Verifica si la intención es un comando"""
    analysis = get_intent_analysis(state)
    return analysis.get("intent") == "command"


def get_detected_action(state: AgentState) -> Optional[str]:
    """Obtiene la acción detectada"""
    analysis = get_intent_analysis(state)
    return analysis.get("action")


def get_detected_entities(state: AgentState) -> Dict[str, Any]:
    """Obtiene las entidades detectadas"""
    analysis = get_intent_analysis(state)
    return analysis.get("entities", {})


def needs_clarification(state: AgentState) -> bool:
    """Verifica si se necesita clarificación"""
    analysis = get_intent_analysis(state)
    return analysis.get("needs_clarification", False)
