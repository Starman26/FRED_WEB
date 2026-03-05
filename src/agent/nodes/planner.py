"""
planner.py - ReAct Planner: análisis de intención + planificación en un solo nodo

Reemplaza intent_analyzer_node + orchestrator_plan_node con un diseño más inteligente:

  Fast-path (~70%): Regex → plan directo en <1ms, 0 LLM calls
  Smart-path (~30%): 1 LLM call con chain-of-thought reasoning

Flujo en el grafo:
  bootstrap → planner → [worker₁ → route → worker₂ → ...] → synthesize → END
"""
import os
import re
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage

from src.agent.state import AgentState
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_plan, event_report, event_error


# ============================================
# CONSTANTS
# ============================================

FAST_CONFIDENCE_THRESHOLD = 0.85

VALID_WORKERS = {
    "chat", "research", "tutor", "troubleshooting",
    "summarizer", "robot_operator", "analysis",
}


# ============================================
# REACT PLANNER PROMPT (smart-path)
# ============================================

REACT_PLANNER_PROMPT = """You are the Planner for ORION, a multi-agent system for a manufacturing laboratory.

AVAILABLE WORKERS:
1. **chat**: General conversation, coding help, advice, greetings, anything NOT about the lab/robot/papers
2. **research**: Search internal documents/papers using RAG (PLCs, Cobots, AI/ML, industrial automation)
3. **troubleshooting**: Diagnose lab problems, check equipment status, execute lab commands (6 stations with PLC + Cobot + sensors)
4. **tutor**: Explain technical concepts pedagogically, synthesize information for learning
5. **robot_operator**: Control xArm robot (move, home, gripper, emergency stop)
6. **summarizer**: Compress conversation memory (automatic only, NEVER include in a plan)

THINK step by step:
1. INTENT: What is the user actually asking? Classify as: command | query | troubleshoot | learn | chat
2. REASONING: What information or actions are needed to fully answer this? Consider what each worker can provide.
3. PLAN: Which workers should execute, in what order? (max 3). Later workers receive output from earlier ones.
4. CONTEXT: What specific focus should each worker have?

RULES:
- Greetings, code help, general questions, emotions, opinions → ["chat"]
- Search documents/papers → ["research"]
- Search AND explain a concept from documents → ["research", "tutor"]
- Search documents AND apply/relate to a specific station/equipment → ["research", "troubleshooting"] (troubleshooting fetches real station data, synthesis merges both)
- Search documents AND apply/relate AND explain pedagogically → ["research", "troubleshooting", "tutor"]
- Lab diagnosis, equipment status, execute lab actions → ["troubleshooting"]
- Robot xArm control (move, home, gripper) → ["robot_operator"]
- Explain a general concept (no lab-specific documents needed) → ["tutor"]
- NEVER include "summarizer" in a plan
- Maximum 3 workers

USER MESSAGE: "{user_message}"
CONVERSATION CONTEXT: {conversation_context}
INTERACTION MODE: {interaction_mode}

Respond ONLY with valid JSON:
{{
  "intent": "<command|query|troubleshoot|learn|chat>",
  "reasoning": "<2-3 sentences explaining your chain of thought>",
  "plan": ["worker1", "worker2"],
  "worker_context": {{
    "worker1": "<what this worker should focus on>",
    "worker2": "<what this worker should focus on>"
  }},
  "action": "<specific action name or null>",
  "entities": {{
    "station": null,
    "equipment": null,
    "routine": null,
    "error_type": null
  }},
  "urgency": "<low|medium|high|critical>",
  "confidence": 0.90
}}"""


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  FAST INTENT: COMPILED REGEX & LOOKUP SETS (module-level, once)    ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── Sets for O(1) exact-match lookups ──

_GREETINGS: set = {
    "hola", "hey", "buenas", "buen día", "buen dia", "buenos días",
    "buenos dias", "qué tal", "que tal", "saludos", "buenas tardes",
    "buenas noches", "qué onda", "que onda", "quiubo", "qué hay",
    "que hay", "ey", "epa",
    "hi", "hello", "sup", "yo",
}
_GREETING_PREFIXES: tuple = tuple(_GREETINGS)

_THANKS: set = {
    "gracias", "muchas gracias", "mil gracias", "perfecto", "genial",
    "excelente", "ok gracias", "vale gracias", "ok perfecto",
    "listo gracias", "listo", "entendido", "ok", "okay", "okey",
    "dale", "de acuerdo", "enterado", "10-4", "thanks", "thank you",
}

_FAREWELLS: set = {
    "adiós", "adios", "bye", "hasta luego", "nos vemos", "chao",
    "chau", "hasta mañana", "hasta manana", "bye bye", "nos vemos luego",
    "hasta pronto", "me voy", "ya me voy",
}

# ── Compiled regex patterns (compiled ONCE at import time) ──

_RE_NAME = re.compile(r"(?:me llamo|mi nombre es|soy|me dicen|dime)\s+(\w+)")

# Entities
_RE_STATION = re.compile(r"estaci[oó]n\s*([1-6])|est\.?\s*([1-6])")
_RE_EQUIPMENT_COBOT = re.compile(r"cobot|robot(?!\s*xarm)|brazo\s*robot")
_RE_EQUIPMENT_DOOR = re.compile(r"puerta|door")
_RE_ROUTINE = re.compile(r"rutina\s*(\d+)|modo\s*(\d+)")

# Robot xARM
_ROBOT_PATTERNS: list = [
    (re.compile(r"(?:mueve?|mover)\s+(?:el\s+)?(?:robot|xarm|brazo).*?([xyz]).*?([-+]?\d+)"), "robot_move"),
    (re.compile(r"(?:mueve?|mover)\s+([xyz])\s*([-+]?\d+)"), "robot_move"),
    (re.compile(r"(?:robot|xarm|brazo).*?(?:a\s+)?(?:home|posici[oó]n\s+inicial|inicio)"), "robot_home"),
    (re.compile(r"(?:home|posici[oó]n\s+inicial).*?(?:robot|xarm|brazo)"), "robot_home"),
    (re.compile(r"(?:conecta|conectar)\s*(?:el\s+)?(?:robot|xarm|brazo)"), "robot_connect"),
    (re.compile(r"(?:posici[oó]n|donde\s+est[aá]|coordenadas?).*?(?:robot|xarm|brazo)"), "robot_get_position"),
    (re.compile(r"(?:robot|xarm|brazo).*?(?:posici[oó]n|donde|coordenadas?)"), "robot_get_position"),
    (re.compile(r"(?:abre?|abrir)\s*(?:el\s+)?(?:gripper|pinza|garra)"), "robot_gripper_open"),
    (re.compile(r"(?:cierra|cerrar)\s*(?:el\s+)?(?:gripper|pinza|garra)"), "robot_gripper_close"),
    (re.compile(r"(?:estado|status)\s*(?:del?\s+)?(?:robot|xarm|brazo)"), "robot_status"),
    (re.compile(r"(?:velocidad|speed)\s*(?:del?\s+)?(?:robot|xarm|brazo)"), "robot_get_speed"),
    (re.compile(r"(?:cambia|cambiar|set|pon|poner)\s*(?:la\s+)?(?:velocidad|speed).*?(\d+)"), "robot_set_speed"),
]

_RE_EMERGENCY = re.compile(
    r"(?:paro|emergencia|emergency|e[\s\-]?stop|"
    r"stop\s*(?:robot|xarm|todo)|"
    r"det[eé]n(?:er|te|gan|lo)?\s*(?:el\s+)?(?:robot|xarm|brazo|todo)|"
    r"para\s+todo|frena)"
)

# PLC / Health
_RE_PLC_PING = re.compile(
    r"ping\s*pong|ping\s+(?:la\s+|el\s+)?plc|"
    r"health\s*check|chequeo\s+(?:de\s+)?salud|"
    r"salud\s+(?:del?\s+)?(?:plc|estaci[oó]n|lab)|"
    r"hacer\s+(?:un\s+)?ping|hazle\s+(?:un\s+)?ping"
)

# Lab commands
_LAB_COMMANDS: list = [
    (re.compile(r"(?:inici|arranca|ejecuta|enciend|activa|corre|lanza)\w*\s+(?:el\s+|la\s+)?(?:cobot|robot|brazo|rutina|estaci[oó]n)"), "start_cobot"),
    (re.compile(r"(?:cobot|robot|brazo|rutina|estaci[oó]n)\s+(?:inici|arranca|ejecuta|enciend|activa)"), "start_cobot"),
    (re.compile(r"(?:pon\s+a\s+(?:correr|funcionar)|echa\s+a\s+andar)\s+(?:el\s+|la\s+)?(?:cobot|robot|brazo|estaci[oó]n)"), "start_cobot"),
    (re.compile(r"(?:par[ae]|deten\w*|apag\w*)\s+(?:el\s+|la\s+)?(?:cobot|robot|brazo|rutina|estaci[oó]n)"), "stop_cobot"),
    (re.compile(r"(?:cobot|robot|brazo|rutina|estaci[oó]n)\s+(?:par[ae]|deten|stop|apag)"), "stop_cobot"),
    (re.compile(r"(?:^|\s)stop\s+(?:cobot|robot|brazo|estaci[oó]n)"), "stop_cobot"),
    (re.compile(r"(?:cierra|cerrar)\s*(?:las?\s+)?puertas?"), "close_doors"),
    (re.compile(r"(?:abre?|abrir)\s*(?:las?\s+)?puertas?"), "open_doors"),
    (re.compile(r"reconect\w*\s*(?:el\s+)?plc"), "reconnect_plc"),
    (re.compile(r"(?:reset|reinici\w*|restaur\w*)\s*(?:el\s+)?lab"), "reset_lab"),
    (re.compile(r"(?:resolver|resuelve|limpi\w*|clear|borra)\s*(?:los?\s+)?error"), "resolve_errors"),
    (re.compile(r"(?:arregl\w*|fix|repar\w*|compone?r?)\s+(?:el\s+)?(?:cobot|robot|plc|estaci[oó]n|lab)"), "auto_fix"),
]

# Status queries
_RE_CHECK_ERRORS = re.compile(r"(?:hay|tiene|existen?)\s*(?:alg[uú]n|alg[uú]nos?)?\s*error")
_RE_DOOR_STATUS = re.compile(
    r"(?:hay|tiene|est[aá]n?|cu[aá]l|checa|checar|revisa|revisar|verifica|verificar)\s*(?:las?\s+)?puertas?|"
    r"puertas?\w*\s+(?:abiert|cerrad|estado|status)|"
    r"las?\s+puertas?\s+(?:est[aá]n?|abiert|cerrad)"
)
_RE_PLC_STATUS = re.compile(
    r"(?:estado|status|conectad)\w*\s+(?:del?\s+)?plc|"
    r"plc\s+(?:conectad|estado|status|activ)"
)
_RE_COBOT_STATUS = re.compile(
    r"(?:hay|tiene|est[aá]n?|cu[aá]l|qu[eé])\s*(?:los?\s+)?(?:cobot|rutina)|"
    r"(?:cobot|rutina)\w*\s+(?:corr|ejecut|activ|estado)"
)
_RE_GENERAL_STATUS = re.compile(
    r"estado\s+(?:del?|actual|general)|status\s+(?:del?|of)|"
    r"c[oó]mo\s+est[aá]|cu[aá]l\s+es\s+el\s+estado|"
    r"dame\s+(?:el\s+)?(?:estado|status|resumen)"
)

# Troubleshoot
_RE_PROBLEMS = re.compile(
    r"no\s+(?:conecta|funciona|responde|enciende|arranca|prende|carga|sirve)|"
    r"fall[oó]|problema|se\s+(?:trab[oó]|cay[oó]|congel[oó]|qued[oó])|"
    r"est[aá]\s+(?:fallando|muerto|ca[ií]do|trabado)"
)

# Learn / Tutor
_RE_LEARN = re.compile(
    r"c[oó]mo\s+funciona|explic(?:a|ar|ame)|qu[eé]\s+es|"
    r"ense[ñn](?:a|ar|ame)|aprender|tutorial|"
    r"para\s+qu[eé]\s+sirve|qu[eé]\s+hace|cu[aá]l\s+es\s+la\s+diferencia|"
    r"c[oó]mo\s+se\s+(?:usa|utiliza|hace)"
)
_RE_NEEDS_RESEARCH = re.compile(r"paper|documento|buscar|seg[uú]n|art[ií]culo|estudio")

# Research
_RE_RESEARCH = re.compile(
    r"paper|documento|busca(?:r)?|referencia|seg[uú]n|cita|fuente|"
    r"art[ií]culo|investigaci[oó]n|estudio|bibliograf"
)

# Urgency / Sentiment
_RE_URGENCY_HIGH = re.compile(r"urgente|cr[ií]tico|emergencia|grave|ya\s*!|ayuda\s*!")
_RE_URGENCY_MEDIUM = re.compile(r"error|falla|no funciona")
_RE_SENTIMENT_CONFUSED = re.compile(r"no entiendo|confundid[oa]|cu[aá]l|no\s+s[eé]|me\s+perd[ií]")
_RE_SENTIMENT_FRUSTRATED = re.compile(
    r"ya\s+intent[eé]|otra\s+vez|sigue\s+sin|todav[ií]a|"
    r"de\s+nuevo|siempre\s+(?:pasa|falla)|no\s+sirve\s+nada"
)


# ============================================
# FAST INTENT: HELPERS
# ============================================

def _make_base(message: str) -> Dict[str, Any]:
    """Crea el dict base con valores por defecto."""
    return {
        "intent": "chat",
        "action": None,
        "entities": {"station": None, "equipment": None, "routine": None, "error_type": None},
        "context_clues": [],
        "urgency": "low",
        "sentiment": "neutral",
        "suggested_worker": "chat",
        "needs_clarification": False,
        "clarification_reason": None,
        "summary": message[:60],
        "confidence": 0.3,
    }


def _extract_entities(msg: str) -> Dict[str, Any]:
    """Extrae entidades comunes del mensaje (estación, equipo, rutina)."""
    entities: Dict[str, Any] = {
        "station": None, "equipment": None, "routine": None, "error_type": None,
    }
    station_match = _RE_STATION.search(msg)
    if station_match:
        entities["station"] = int(station_match.group(1) or station_match.group(2))
    if "plc" in msg:
        entities["equipment"] = "plc"
    elif _RE_EQUIPMENT_COBOT.search(msg):
        entities["equipment"] = "cobot"
    elif _RE_EQUIPMENT_DOOR.search(msg):
        entities["equipment"] = "door"
    routine_match = _RE_ROUTINE.search(msg)
    if routine_match:
        entities["routine"] = int(routine_match.group(1) or routine_match.group(2))
    return entities


def _flag_clarification(base: Dict[str, Any], reason: str) -> None:
    """Marca que se necesita clarificación si no hay estación."""
    if not base["entities"].get("station"):
        base["needs_clarification"] = True
        base["clarification_reason"] = reason


def _enrich_urgency_sentiment(base: Dict[str, Any], msg: str) -> None:
    """Enriquece urgencia y sentimiento (mutación in-place)."""
    if _RE_URGENCY_HIGH.search(msg):
        base["urgency"] = "high"
    elif _RE_URGENCY_MEDIUM.search(msg):
        base["urgency"] = "medium"
    if _RE_SENTIMENT_CONFUSED.search(msg):
        base["sentiment"] = "confused"
    elif _RE_SENTIMENT_FRUSTRATED.search(msg):
        base["sentiment"] = "frustrated"


# ============================================
# FAST INTENT ANALYSIS (compiled regex version)
# ============================================

def _fast_intent_analysis(message: str) -> Dict[str, Any]:
    """
    Análisis basado en patrones — cubre ~70-80% de los casos sin LLM.
    Retorna dict con 'confidence' que indica si se puede usar sin LLM.

    Usa regex precompiladas y sets O(1) definidos a nivel de módulo.
    """
    msg = message.lower().strip()
    base = _make_base(message)

    # ── 1. SALUDOS Y CHAT RÁPIDO (O(1) lookup) ──
    if msg in _GREETINGS:
        base.update(intent="chat", suggested_worker="chat", confidence=0.95, summary="Saludo")
        return base
    if msg.startswith(_GREETING_PREFIXES):
        base.update(intent="chat", suggested_worker="chat", confidence=0.95, summary="Saludo")
        return base
    if msg in _THANKS:
        base.update(intent="chat", suggested_worker="chat", confidence=0.95,
                    sentiment="casual", summary="Agradecimiento")
        return base
    if any(msg.startswith(t) for t in _THANKS if len(t) > 2):
        base.update(intent="chat", suggested_worker="chat", confidence=0.90,
                    sentiment="casual", summary="Agradecimiento")
        return base
    if msg in _FAREWELLS:
        base.update(intent="chat", suggested_worker="chat", confidence=0.95, summary="Despedida")
        return base
    if any(msg.startswith(f) for f in _FAREWELLS):
        base.update(intent="chat", suggested_worker="chat", confidence=0.95, summary="Despedida")
        return base

    # Presentación
    name_match = _RE_NAME.match(msg)
    if name_match:
        base.update(intent="chat", suggested_worker="chat", confidence=0.95,
                    summary=f"Presentación: {name_match.group(1)}")
        return base

    # ── 2. EXTRAER ENTIDADES (una sola vez) ──
    entities = _extract_entities(msg)
    base["entities"] = entities

    # ── 3. EMERGENCY STOP (máxima prioridad) ──
    if _RE_EMERGENCY.search(msg):
        base.update(
            intent="command", action="robot_emergency_stop",
            suggested_worker="robot_operator", urgency="critical",
            confidence=0.95, summary="Emergency stop",
        )
        return base

    # ── 4. ROBOT xARM ──
    for pattern, action in _ROBOT_PATTERNS:
        match = pattern.search(msg)
        if match:
            if action == "robot_move" and match.lastindex and match.lastindex >= 2:
                entities["axis"] = match.group(1)
                entities["distance"] = int(match.group(2))
            elif action == "robot_set_speed" and match.lastindex:
                entities["speed"] = int(match.group(1))
            base.update(
                intent="command", action=action,
                suggested_worker="robot_operator",
                confidence=0.90,
                summary=f"Robot command: {action}",
            )
            return base

    # ── 5. PLC HEALTH CHECK ──
    if _RE_PLC_PING.search(msg):
        base.update(intent="query", action="ping_plc",
                    suggested_worker="troubleshooting", confidence=0.95,
                    summary="PLC health check")
        _flag_clarification(base, "No se especificó la estación")
        return base

    # ── 6. LAB COMMANDS ──
    for pattern, action in _LAB_COMMANDS:
        if pattern.search(msg):
            base.update(
                intent="command", action=action,
                suggested_worker="troubleshooting", confidence=0.90,
                summary=f"Lab command: {action}",
            )
            return base

    # ── 7. STATUS QUERIES ──
    if _RE_CHECK_ERRORS.search(msg):
        base.update(intent="query", action="check_errors",
                    suggested_worker="troubleshooting", confidence=0.88)
        return base
    if _RE_DOOR_STATUS.search(msg):
        base.update(intent="query", action="check_door_status",
                    suggested_worker="troubleshooting", confidence=0.88)
        return base
    if _RE_PLC_STATUS.search(msg):
        base.update(intent="query", action="check_plc_status",
                    suggested_worker="troubleshooting", confidence=0.88)
        return base
    if _RE_COBOT_STATUS.search(msg):
        base.update(intent="query", action="check_cobot_status",
                    suggested_worker="troubleshooting", confidence=0.88)
        return base
    if _RE_GENERAL_STATUS.search(msg[:200]):
        base.update(intent="query", action="check_status",
                    suggested_worker="troubleshooting", confidence=0.85)
        return base

    # ── 8. TROUBLESHOOT ──
    if _RE_PROBLEMS.search(msg):
        base.update(
            intent="troubleshoot", suggested_worker="troubleshooting",
            urgency="medium", confidence=0.80,
        )
        _flag_clarification(base, "No se especificó la estación")
        return base

    # ── 9. LEARN / TUTOR ──
    if _RE_LEARN.search(msg):
        base.update(intent="learn", suggested_worker="tutor", confidence=0.82)
        if _RE_NEEDS_RESEARCH.search(msg):
            base["context_clues"].append("needs_research")
            base["confidence"] = 0.85
        return base

    # ── 10. RESEARCH ──
    if _RE_RESEARCH.search(msg):
        base.update(intent="query", action="search_docs",
                    suggested_worker="research", confidence=0.85)
        return base

    # ── 11. FALLBACK: enriquecer urgencia/sentimiento ──
    _enrich_urgency_sentiment(base, msg)
    return base


# ============================================
# PLAN MAPPING TABLE (replaces if/elif chain)
# ============================================

_LAB_KEYWORDS = [
    "laboratorio", "estacion", "estación", "plc", "cobot", "puerta",
    "door", "sensor", "equipo", "alarma", "alarm", "falla",
    "station", "conveyor", "banda",
]

_LAB_ACTIONS = [
    "check_status", "check_cobot", "check_door", "check_plc",
    "get_status", "lab_overview", "check_errors",
    "ping_plc", "plc_health", "health_check", "station_health",
]

_RESEARCH_NEEDED_KEYWORDS = [
    "paper", "documento", "buscar", "laboratorio", "lab ",
    "estacion", "estación", "plc", "cobot", "robot",
    "equipo", "máquina", "maquina", "station",
    "equipment", "factory", "fred",
]

_ROBOT_ACTION_KEYWORDS = {
    "robot", "move", "mover", "mueve", "gripper", "pinza",
    "home", "xarm", "emergency", "emergencia", "paro",
    "position", "posicion", "posición", "step", "paso",
}


def _is_robot_action(fast_result: Dict[str, Any], message: str) -> bool:
    """Determina si la acción es de robot basándose en el análisis rápido."""
    suggested = fast_result.get("suggested_worker", "")
    action = (fast_result.get("action") or "").lower()
    intent = fast_result.get("intent", "")

    if suggested == "robot_operator":
        return True
    if action and action.startswith("robot_"):
        return True
    if any(kw in action for kw in _ROBOT_ACTION_KEYWORDS):
        return True
    if intent == "command":
        msg_lower = message.lower()
        has_robot_mention = "robot" in msg_lower or "xarm" in msg_lower or "brazo" in msg_lower
        has_robot_verb = any(kw in msg_lower for kw in ["mueve", "mover", "move", "home", "gripper", "paro", "posicion", "conecta"])
        if has_robot_mention and has_robot_verb:
            return True
    return False


def _has_application_intent(msg_lower: str) -> bool:
    """Detecta si el usuario pide aplicar/conectar conocimiento con algo práctico."""
    return bool(re.search(
        r"(?:aplic|implement|usar|uso|utiliz|integr|conect|relacion|combin|merg|mezcl|junt)\w*"
        r"|(?:de\s+qu[eé]\s+manera|c[oó]mo\s+(?:lo|se)\s+(?:aplic|us|implement))",
        msg_lower,
    ))


def _mentions_lab_entity(msg_lower: str) -> bool:
    """Detecta si el mensaje menciona entidades del lab (estaciones, equipos)."""
    return bool(re.search(
        r"estaci[oó]n\s*\d|est\.\s*\d|laboratorio|lab\b|plc|cobot|puerta|sensor|conveyor|banda",
        msg_lower,
    ))


def _map_intent_to_plan(fast_result: Dict[str, Any], message: str) -> Tuple[List[str], str]:
    """
    Tabla declarativa: mapea intent_analysis → plan de ejecución.
    Retorna (plan, reasoning).
    """
    intent = fast_result.get("intent", "chat")
    action = fast_result.get("action")
    msg_lower = message[:300].lower()

    if _is_robot_action(fast_result, message):
        return ["robot_operator"], f"Robot command detected: {action or 'robot action'}"

    if intent == "command":
        return ["troubleshooting"], f"Lab command: {action or 'lab action'}"

    if intent == "query":
        if action == "search_docs":
            if _mentions_lab_entity(msg_lower):
                if _has_application_intent(msg_lower):
                    return ["research", "troubleshooting"], "Document search + lab application (needs station data to merge)"
                return ["research", "troubleshooting"], "Document search mentioning lab entity (needs real data)"
            return ["research"], "Document search query"
        is_lab_query = (
            any(kw in msg_lower for kw in _LAB_KEYWORDS)
            or (action and action in _LAB_ACTIONS)
        )
        if is_lab_query:
            return ["troubleshooting"], f"Lab status query: {action or 'check'}"
        return ["chat"], f"General query: {action or 'question'}"

    if intent == "troubleshoot":
        return ["troubleshooting"], "Technical problem reported"

    if intent == "learn":
        needs_research = (
            any(kw in message.lower() for kw in _RESEARCH_NEEDED_KEYWORDS)
            or (action and any(x in (action or "") for x in ["lab", "equipment", "station", "describe", "overview"]))
        )
        if needs_research or "needs_research" in fast_result.get("context_clues", []):
            if _mentions_lab_entity(msg_lower):
                return ["research", "troubleshooting", "tutor"], "Learning about docs + applying to specific lab equipment"
            return ["research", "tutor"], "Learning request requiring document context"
        if _mentions_lab_entity(msg_lower):
            return ["troubleshooting", "tutor"], "Learning about specific lab equipment"
        return ["tutor"], "Educational explanation request"

    return ["chat"], "General conversation"


# ============================================
# LLM PLAN (smart-path, 1 call)
# ============================================

def _get_conversation_context(state: AgentState, max_messages: int = 4) -> str:
    """Construye contexto de conversación reciente para el prompt del planner."""
    messages = state.get("messages", [])
    if not messages:
        return "No prior conversation."

    lines = []
    recent = messages[-max_messages:] if len(messages) > max_messages else messages
    for msg in recent:
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content[:100]}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {msg.content[:100]}")
        elif isinstance(msg, dict):
            role = msg.get("role", "unknown")
            lines.append(f"{role.capitalize()}: {msg.get('content', '')[:100]}")
    return "\n".join(lines) if lines else "No prior conversation."


def _llm_plan(user_message: str, state: AgentState) -> Dict[str, Any]:
    """
    Smart-path: 1 LLM call con chain-of-thought reasoning.
    Produce intent analysis + plan + reasoning en una sola invocación.
    """
    from src.agent.utils.llm_factory import get_llm

    llm = get_llm(state, temperature=0.3, max_tokens=500)

    context = _get_conversation_context(state)
    interaction_mode = state.get("interaction_mode", "chat")

    prompt = REACT_PLANNER_PROMPT.format(
        user_message=user_message,
        conversation_context=context,
        interaction_mode=interaction_mode,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()

        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        if start_idx == -1 or end_idx == 0:
            logger.error("planner", "LLM returned no JSON, falling back to keywords")
            return _build_fallback_plan(user_message)

        data = json.loads(content[start_idx:end_idx])

        raw_plan = data.get("plan", ["chat"])
        plan = [w for w in raw_plan if w in VALID_WORKERS and w != "summarizer"] or ["chat"]

        reasoning = data.get("reasoning", "LLM-planned execution")
        intent = data.get("intent", "chat")
        action = data.get("action")
        entities = data.get("entities", {})
        urgency = data.get("urgency", "low")
        confidence = min(data.get("confidence", 0.90), 1.0)

        worker_context = data.get("worker_context", {})

        intent_analysis = {
            "intent": intent,
            "action": action,
            "entities": entities,
            "urgency": urgency,
            "sentiment": data.get("sentiment", "neutral"),
            "suggested_worker": plan[0] if plan else "chat",
            "needs_clarification": False,
            "summary": reasoning[:60],
            "confidence": confidence,
        }

        return {
            "plan": plan,
            "reasoning": reasoning,
            "intent_analysis": intent_analysis,
            "pending_context": {
                "intent": intent,
                "action": action,
                "entities": entities,
                "urgency": urgency,
                "worker_context": worker_context,
            },
        }

    except (json.JSONDecodeError, Exception) as e:
        logger.error("planner", f"LLM plan error: {e}")
        return _build_fallback_plan(user_message)


def _build_fallback_plan(text: str) -> Dict[str, Any]:
    """Safety-net: keyword-based plan when all else fails."""
    text_lower = text.lower()

    chat_kw = ["hola", "hey", "buenas", "gracias", "me llamo", "adios", "bye"]
    if any(kw in text_lower for kw in chat_kw):
        plan = ["chat"]
        reasoning = "Fallback: greeting detected"
    elif any(kw in text_lower for kw in ["xarm", "robot xarm", "mover robot", "gripper", "home robot"]):
        plan = ["robot_operator"]
        reasoning = "Fallback: robot keyword detected"
    else:
        plan = []
        if any(kw in text_lower for kw in ["paper", "documento", "buscar", "referencia"]):
            plan.append("research")
        if any(kw in text_lower for kw in ["error", "falla", "problema", "log", "máquina"]):
            plan.append("troubleshooting")
        if any(kw in text_lower for kw in ["explicar", "cómo", "enséñame", "qué es"]):
            plan.append("tutor")
        plan = plan or ["chat"]
        reasoning = f"Fallback: keyword matching → {plan}"

    fast = _fast_intent_analysis(text)
    return {
        "plan": plan,
        "reasoning": reasoning,
        "intent_analysis": fast,
        "pending_context": {
            "intent": fast.get("intent", "chat"),
            "action": fast.get("action"),
            "entities": fast.get("entities", {}),
            "urgency": fast.get("urgency", "low"),
        },
    }


# ============================================
# DISPLAY HELPERS
# ============================================

_WORKER_DISPLAY = {
    "chat": "responder",
    "research": "buscar en los documentos",
    "tutor": "preparar una explicación",
    "troubleshooting": "consultar el laboratorio",
    "robot_operator": "operar el robot",
    "summarizer": "comprimir memoria",
    "analysis": "analizar los datos",
}


def _extract_topic(user_message: str) -> str:
    """Extract the main topic/subject from the user message for display."""
    msg = user_message.strip()
    starters = [
        r"^(?:qu[eé]\s+dice\s+(?:el\s+)?(?:paper|documento)\s+(?:de|sobre)\s+)",
        r"^(?:busca(?:r)?\s+(?:informaci[oó]n\s+)?(?:sobre|de)\s+)",
        r"^(?:qu[eé]\s+(?:es|son|significa)\s+)",
        r"^(?:c[oó]mo\s+(?:funciona|se\s+usa|aplico)\s+)",
        r"^(?:expl[ií]ca(?:me)?\s+(?:qu[eé]\s+es\s+|c[oó]mo\s+(?:funciona\s+|se\s+)?)?)",
        r"^(?:investiga(?:r)?\s+(?:sobre\s+)?)",
        r"^(?:dime\s+(?:sobre|qu[eé])\s+)",
        r"^(?:qu[eé]\s+es\s+)",
        r"^(?:cu[eé]nta(?:me)?\s+(?:sobre|de)\s+)",
    ]
    cleaned = msg
    for pat in starters:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)

    if len(cleaned) > 60:
        for sep in [",", " y ", " para ", " de que ", " de qué "]:
            idx = cleaned.find(sep)
            if 10 < idx < 60:
                cleaned = cleaned[:idx]
                break
        else:
            cleaned = cleaned[:60].rsplit(" ", 1)[0]

    return cleaned.strip(" .,?¿!¡")


def _entity_label(entities: Dict[str, Any]) -> str:
    """Build a human-readable label for detected entities."""
    parts = []
    station = entities.get("station")
    equipment = entities.get("equipment")
    if station:
        parts.append(f"estación {station}")
    if equipment:
        equip_names = {"plc": "el PLC", "cobot": "el cobot", "door": "las puertas"}
        parts.append(equip_names.get(equipment, equipment))
    return " y ".join(parts) if parts else ""


def _create_rich_announcement(
    plan: List[str],
    fast_result: Dict[str, Any],
    reasoning: str,
    user_message: str,
) -> str:
    """
    Genera anuncio con razonamiento visible (0 LLM calls).
    """
    if plan == ["chat"] or plan == ["robot_operator"]:
        return ""
    if plan == ["summarizer"]:
        return "Comprimiendo memoria..."

    topic = _extract_topic(user_message)
    entities = fast_result.get("entities", {})
    entity = _entity_label(entities)
    plan_key = tuple(plan)

    # Single-worker plans
    if plan_key == ("research",):
        return f"Voy a buscar en los documentos sobre {topic}..." if topic else "Voy a buscar en los documentos..."

    if plan_key == ("troubleshooting",):
        action = fast_result.get("action", "")
        if action and (action.startswith("check_") or action in ("ping_plc", "plc_health", "health_check", "station_health")):
            check_label = {
                "check_door_status": "el estado de las puertas",
                "check_plc_status": "el estado de los PLCs",
                "check_cobot_status": "el estado de los cobots",
                "check_errors": "los errores activos",
                "check_status": f"el estado {'de la ' + entity if entity else 'del laboratorio'}",
                "ping_plc": "la salud del PLC" + (f" de {entity}" if entity else ""),
                "plc_health": "la salud del PLC" + (f" de {entity}" if entity else ""),
                "health_check": "la salud completa" + (f" de {entity}" if entity else " de la estación"),
                "station_health": "la salud completa" + (f" de {entity}" if entity else " de la estación"),
            }.get(action, f"el estado de {entity}" if entity else "el laboratorio")
            return f"Voy a consultar {check_label}..."
        if entity:
            return f"Voy a consultar {entity} en el laboratorio..."
        return ""

    if plan_key == ("tutor",):
        return f"Voy a prepararte una explicación sobre {topic}..." if topic else "Voy a prepararte una explicación."

    # Multi-worker plans
    if plan_key == ("research", "troubleshooting"):
        if entity and topic:
            return (
                f"Detecto que preguntas sobre {topic} y cómo se relaciona con {entity}. "
                f"Voy a buscar en los documentos y luego consultar los datos de {entity} para darte una respuesta integrada."
            )
        if entity:
            return f"Voy a buscar en los documentos y consultar los datos de {entity} para integrar la información."
        return f"Voy a buscar en los documentos sobre {topic} y consultar el laboratorio."

    if plan_key == ("research", "tutor"):
        return f"Voy a investigar {topic} en los documentos y luego prepararte una explicación adaptada."

    if plan_key == ("research", "troubleshooting", "tutor"):
        if entity and topic:
            return (
                f"Detecto que quieres aprender sobre {topic} aplicado a {entity}. "
                f"Voy a buscar en los documentos, consultar datos reales de {entity}, "
                f"y prepararte una explicación integrada."
            )
        return f"Voy a investigar, consultar el laboratorio, y prepararte una explicación completa."

    if plan_key == ("troubleshooting", "tutor"):
        if entity:
            return f"Voy a obtener datos de {entity} del laboratorio y prepararte una explicación."
        return "Voy a consultar el laboratorio y prepararte una explicación."

    # Fallback: generic multi-step
    steps = [_WORKER_DISPLAY.get(w, w) for w in plan if w != "chat"]
    if steps:
        return f"Voy a {', luego '.join(steps)}."
    return ""


# ============================================
# MAIN NODE
# ============================================

def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    ReAct Planner: nodo que fusiona intent_analyzer + orchestrator_plan.

    Fast-path (~70%): regex → plan directo, 0 LLM calls, <1ms
    Smart-path (~30%): 1 LLM call con chain-of-thought reasoning
    """
    start_time = datetime.utcnow()
    logger.node_start("planner", {"action": "planning"})
    events = [event_plan("planner", "Planning execution...")]

    # ── Get last user message ──
    user_message = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
        elif isinstance(msg, dict) and msg.get("role") in ("human", "user"):
            user_message = msg.get("content", "")
            break

    if not user_message:
        return {
            "intent_analysis": {
                "intent": "chat", "action": None, "entities": {},
                "urgency": "low", "sentiment": "neutral",
                "suggested_worker": "chat", "needs_clarification": False,
                "summary": "Empty message", "confidence": 1.0,
            },
            "orchestration_plan": ["chat"],
            "current_step": 0,
            "worker_outputs": [],
            "pending_context": {},
            "clarification_questions": [],
            "needs_human_input": False,
            "next": "chat",
            "task_type": "chat",
            "plan_reasoning": "No user message found",
            "planner_method": "fast",
            "events": events,
        }

    # ══════════════════════════════════════════
    # MODE DETECTION
    # ══════════════════════════════════════════
    interaction_mode = state.get("interaction_mode", "chat").lower()
    _is_analysis_mode = interaction_mode == "analysis"

    # ══════════════════════════════════════════
    # PRACTICE MODE: bypass everything, direct to tutor
    # ══════════════════════════════════════════
    if interaction_mode == "practice":
        return {
            "orchestration_plan": ["tutor"],
            "current_step": 0,
            "intent_analysis": {"primary_intent": "practice", "confidence": 1.0, "requires_workers": ["tutor"]},
            "plan_reasoning": "Practice mode -- direct to tutor",
            "planner_method": "fast_path_practice",
            "pending_context": {},
            "worker_outputs": [],
            "clarification_questions": [],
            "needs_human_input": False,
            "next": "tutor",
            "task_type": "tutor",
            "messages": [],
            "events": events,
        }

    # ══════════════════════════════════════════
    # FAST PATH: regex → plan directo
    # ══════════════════════════════════════════
    fast_result = _fast_intent_analysis(user_message)

    # Mode override: code/voice con intent chat → mantener en chat con confianza alta
    if interaction_mode in ("code", "voice") and fast_result["intent"] == "chat":
        fast_result["confidence"] = max(fast_result["confidence"], 0.90)

    if fast_result["confidence"] >= FAST_CONFIDENCE_THRESHOLD:
        plan, reasoning = _map_intent_to_plan(fast_result, user_message)

        # ANALYSIS MODE: append analysis as final enrichment/formatting step
        if _is_analysis_mode and "analysis" not in plan:
            plan.append("analysis")
            reasoning += " + SQL analysis enrichment"
            logger.info("planner", f"ANALYSIS MODE: appended analysis → {plan}")

        elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        entities = fast_result.get("entities", {})
        entity_parts = [f"{k}={v}" for k, v in entities.items() if v]
        entities_str = ", ".join(entity_parts) if entity_parts else "none"
        events.append(event_plan("planner",
                                 f"Detected: {fast_result['intent']} | {fast_result.get('action') or 'general'} | entities: {entities_str}"))
        events.append(event_plan("planner",
                                 f"Plan: {' → '.join(plan)} ({elapsed_ms:.0f}ms) | {reasoning}"))

        logger.info("planner",
                     f"FAST ({elapsed_ms:.0f}ms): {fast_result['intent']}|{fast_result['action']} → {plan}")

        announcement = _create_rich_announcement(plan, fast_result, reasoning, user_message)

        return {
            "intent_analysis": fast_result,
            "orchestration_plan": plan,
            "current_step": 0,
            "pending_context": {
                "intent": fast_result["intent"],
                "action": fast_result.get("action"),
                "entities": fast_result.get("entities", {}),
                "urgency": fast_result.get("urgency", "low"),
                "sentiment": fast_result.get("sentiment", "neutral"),
            },
            "worker_outputs": [],
            "clarification_questions": [],
            "needs_human_input": False,
            "next": plan[0],
            "task_type": plan[0],
            "plan_reasoning": reasoning,
            "planner_method": "fast",
            "messages": [AIMessage(content=announcement)] if announcement else [],
            "events": events,
        }

    # ══════════════════════════════════════════
    # SMART PATH: 1 LLM call con chain-of-thought
    # ══════════════════════════════════════════
    logger.info("planner",
                f"Confidence {fast_result['confidence']:.2f} < {FAST_CONFIDENCE_THRESHOLD}, using LLM planning...")
    events.append(event_plan("planner", "Complex query — reasoning with LLM..."))

    llm_result = _llm_plan(user_message, state)
    elapsed_s = (datetime.utcnow() - start_time).total_seconds()

    plan = llm_result["plan"]
    reasoning = llm_result["reasoning"]

    # ANALYSIS MODE: append analysis as final enrichment/formatting step
    if _is_analysis_mode and "analysis" not in plan:
        plan.append("analysis")
        reasoning += " + SQL analysis enrichment"
        logger.info("planner", f"ANALYSIS MODE: appended analysis → {plan}")

    logger.info("planner", f"LLM ({elapsed_s:.2f}s): {reasoning[:80]} → {plan}")
    events.append(event_plan("planner",
                             f"Plan: {' → '.join(plan)} ({elapsed_s:.2f}s) | {reasoning}"))

    intent_analysis = llm_result["intent_analysis"]
    announcement = _create_rich_announcement(plan, intent_analysis, reasoning, user_message)
    if reasoning and len(reasoning) > len(announcement):
        display_names = [_WORKER_DISPLAY.get(w, w) for w in plan if w != "chat"]
        announcement = f"{reasoning}. Voy a {', luego '.join(display_names)}." if display_names else reasoning

    return {
        "intent_analysis": intent_analysis,
        "orchestration_plan": plan,
        "current_step": 0,
        "pending_context": llm_result["pending_context"],
        "worker_outputs": [],
        "clarification_questions": [],
        "needs_human_input": False,
        "next": plan[0],
        "task_type": plan[0],
        "plan_reasoning": reasoning,
        "planner_method": "llm",
        "messages": [AIMessage(content=announcement)] if announcement else [],
        "events": events,
    }


# ============================================
# BACKWARD-COMPAT HELPERS
# ============================================

def get_intent_analysis(state: AgentState) -> Dict[str, Any]:
    return state.get("intent_analysis", {})

def is_command_intent(state: AgentState) -> bool:
    return get_intent_analysis(state).get("intent") == "command"

def get_detected_action(state: AgentState) -> Optional[str]:
    return get_intent_analysis(state).get("action")

def get_detected_entities(state: AgentState) -> Dict[str, Any]:
    return get_intent_analysis(state).get("entities", {})

def needs_clarification(state: AgentState) -> bool:
    return get_intent_analysis(state).get("needs_clarification", False)