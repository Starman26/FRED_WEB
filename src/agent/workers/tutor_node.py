"""
tutor_node.py - Worker especializado en tutorías y explicaciones educativas

Usa WorkerOutput contract, NO retorna done=True, usa pending_context para evidencia.
Incluye soporte para imágenes educativas del banco de imágenes.
Integrado con sistema de prácticas guiadas.
"""
import os
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from src.agent.state import AgentState
from src.agent.contracts.worker_contract import WorkerOutputBuilder, EvidenceItem, create_error_output
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_execute, event_report, event_error

# Practice system integration
try:
    from src.agent.practices import (
        get_practice_manager,
        PracticeSession,
        StepType,
    )
    PRACTICES_AVAILABLE = True
except ImportError:
    PRACTICES_AVAILABLE = False

# Import format_helpers with fallback
try:
    from src.agent.utils.format_helpers import format_learning_style
except ImportError:
    def format_learning_style(style: Dict) -> str:
        if not style or not isinstance(style, dict):
            return "mixed"
        return f"Type: {style.get('type', 'mixed')}, Pace: {style.get('pace', 'medium')}, Depth: {style.get('depth', 'intermediate')}"

from src.agent.prompts.tutor_prompt import (
    VISUAL_TUTOR,
    AUDITIVE_TUTOR,
    KINESTHETIC_TUTOR,
    READING_TUTOR,
    MIX_TUTOR
)

# Image bank integration
try:
    from src.agent.media.images import get_image_sourcer, ImageCategory
    from src.agent.media.images.metadata import ImageRequest
    IMAGES_AVAILABLE = True
except ImportError:
    IMAGES_AVAILABLE = False


# Keywords for image category detection
IMAGE_CATEGORY_KEYWORDS = {
    ImageCategory.PLC_SIEMENS if IMAGES_AVAILABLE else "plc-siemens": [
        "siemens", "s7-1200", "s7-1500", "s7-300", "s7-400", "tia portal", "step 7"
    ],
    ImageCategory.PLC_ALLEN_BRADLEY if IMAGES_AVAILABLE else "plc-allen-bradley": [
        "allen bradley", "rockwell", "controllogix", "compactlogix", "micrologix", "rslogix"
    ],
    ImageCategory.COBOT_UR if IMAGES_AVAILABLE else "cobot-ur": [
        "universal robots", "ur5", "ur10", "ur3", "ur5e", "ur10e", "ur3e", "polyscope"
    ],
    ImageCategory.COBOT_FANUC if IMAGES_AVAILABLE else "cobot-fanuc": [
        "fanuc", "crx", "fanuc robot"
    ],
    ImageCategory.COBOT_GENERAL if IMAGES_AVAILABLE else "cobot-general": [
        "cobot", "robot colaborativo", "collaborative robot", "brazo robotico"
    ],
    ImageCategory.LADDER_LOGIC if IMAGES_AVAILABLE else "ladder-logic": [
        "ladder", "escalera", "ladder logic", "diagrama ladder", "contactos", "bobinas"
    ],
    ImageCategory.HMI_SCREEN if IMAGES_AVAILABLE else "hmi-screen": [
        "hmi", "pantalla", "scada", "interfaz", "panel operador"
    ],
    ImageCategory.PROFINET if IMAGES_AVAILABLE else "profinet": [
        "profinet", "profibus", "ethernet industrial", "red industrial"
    ],
    ImageCategory.SAFETY_EQUIPMENT if IMAGES_AVAILABLE else "safety-equipment": [
        "seguridad", "safety", "e-stop", "paro emergencia", "cortina luz", "light curtain"
    ],
}


TUTOR_MULTISTEP_PROMPT = """You are SENTINEL's Technical Education Module specializing in:
- PLCs (Programmable Logic Controllers)
- Cobots (Collaborative Robots)  
- Python and AI/ML (LangGraph, LangChain)

LANGUAGE: ALWAYS respond in the same language the user writes in.

## IMPORTANT CONTEXT
{context_section}

## PRIOR RESEARCH EVIDENCE
{evidence_section}

## INSTRUCTIONS
1. Use provided evidence: If evidence exists, use and cite it [Title, Page X-Y]
2. Clear structure: Use headers and lists when helpful
3. Be didactic: Explain step by step with examples
4. Adapt to the user's learning style as specified below
5. Maintain professional tone - no emojis

{learning_style_guidance}

User name: {user_name}
Learning profile: {learning_style}

Always end your response with exactly 3 follow-up suggestions:
---SUGGESTIONS---
1. [First related topic or deeper exploration]
2. [Second suggestion for practice or application]
3. [Third suggestion for expanding knowledge]
---END_SUGGESTIONS---
"""


def extract_suggestions_from_text(text: str) -> tuple[str, list[str]]:
    """
    Extracts follow-up suggestions from the LLM response text.
    
    Args:
        text: Raw LLM response text
        
    Returns:
        Tuple of (content_without_suggestions, list_of_suggestions)
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


def detect_image_category(text: str) -> Optional[Any]:
    """
    Detects the most relevant image category based on text.

    Args:
        text: User text or context

    Returns:
        Most relevant ImageCategory or None
    """
    if not IMAGES_AVAILABLE:
        return None

    text_lower = text.lower()

    for category, keywords in IMAGE_CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                return category

    return None


def extract_image_search_terms(user_message: str, evidence_text: str = "") -> str:
    """
    Extracts relevant search terms for images.

    Args:
        user_message: User message
        evidence_text: Prior evidence text

    Returns:
        Search query for images
    """
    technical_keywords = [
        "plc", "siemens", "allen bradley", "rockwell", "cobot", "robot",
        "ladder", "hmi", "scada", "profinet", "profibus", "sensor",
        "actuador", "variador", "motor", "encoder", "io", "modbus",
        "ethernet", "tia portal", "step 7", "ur5", "ur10", "fanuc",
        "seguridad", "safety", "diagrama", "conexion", "cableado"
    ]

    combined_text = f"{user_message} {evidence_text}".lower()
    found_keywords = []

    for keyword in technical_keywords:
        if keyword in combined_text:
            found_keywords.append(keyword)

    if found_keywords:
        return " ".join(found_keywords[:4])

    words = re.findall(r'\b\w{4,}\b', user_message.lower())
    return " ".join(words[:3]) if words else "industrial automation"


def search_relevant_images(
    user_message: str,
    evidence_text: str = "",
    learning_style: str = "",
    max_images: int = 3
) -> List[Dict[str, Any]]:
    """
    Searches for relevant images for the educational explanation.

    Args:
        user_message: User message
        evidence_text: Evidence text
        learning_style: User's learning style
        max_images: Maximum number of images to return

    Returns:
        List of dictionaries with image metadata
    """
    if not IMAGES_AVAILABLE:
        return []

    try:
        sourcer = get_image_sourcer()

        category = detect_image_category(user_message)
        search_query = extract_image_search_terms(user_message, evidence_text)

        request = ImageRequest(
            query=search_query,
            category=category,
            max_results=max_images,
            require_commercial_license=False
        )

        result = sourcer.search(request, include_online=True)

        images = []
        for img in result.images:
            images.append({
                "id": img.id,
                "title": img.title,
                "source": img.source.value if hasattr(img.source, 'value') else str(img.source),
                "source_url": img.source_url,
                "source_page": img.source_page,
                "local_path": img.local_path,
                "author": img.author,
                "license": {
                    "name": img.license.name,
                    "requires_attribution": img.license.requires_attribution,
                    "allows_commercial": img.license.allows_commercial,
                },
                "category": img.category.value if hasattr(img.category, 'value') else str(img.category) if img.category else None,
                "alt_text": img.alt_text or img.title,
                "width": img.width,
                "height": img.height,
            })

        logger.info("tutor", f"Found {len(images)} relevant images for query: {search_query}")
        return images

    except Exception as e:
        logger.warning("tutor", f"Error searching images: {e}")
        return []


def get_learning_style_prompt(learning_style_dict: Dict[str, Any]) -> str:
    """
    Selects the appropriate learning style prompt based on type.

    Args:
        learning_style_dict: Dictionary with user's learning_style

    Returns:
        Specific prompt for that learning style
    """
    if not learning_style_dict or not isinstance(learning_style_dict, dict):
        return MIX_TUTOR

    learning_type = learning_style_dict.get("type", "").lower()

    style_map = {
        "visual": VISUAL_TUTOR,
        "auditory": AUDITIVE_TUTOR,
        "auditivo": AUDITIVE_TUTOR,
        "kinesthetic": KINESTHETIC_TUTOR,
        "kinestésico": KINESTHETIC_TUTOR,
        "kinesthesic": KINESTHETIC_TUTOR,
        "reading": READING_TUTOR,
        "lectura": READING_TUTOR,
        "mixed": MIX_TUTOR,
        "mixto": MIX_TUTOR,
        "both": MIX_TUTOR,
        "ambos": MIX_TUTOR,
    }

    return style_map.get(learning_type, MIX_TUTOR)


def get_last_user_message(state: AgentState) -> str:
    """Extracts the last user message"""
    for m in reversed(state.get("messages", []) or []):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
        if isinstance(m, dict) and m.get("role") in ("human", "user"):
            return (m.get("content") or "").strip()
    return ""


def get_evidence_from_context(state: AgentState) -> tuple[str, List[EvidenceItem]]:
    """Gets evidence from pending_context"""
    pending_context = state.get("pending_context", {})
    evidence_data = pending_context.get("evidence", [])
    
    if not evidence_data:
        for output in state.get("worker_outputs", []):
            if output.get("worker") == "research":
                evidence_data = output.get("evidence", [])
                break
    
    if not evidence_data:
        return "No prior research evidence available.", []
    
    evidence_items = []
    evidence_parts = []
    for ev in evidence_data:
        if isinstance(ev, dict):
            title, page, chunk = ev.get("title", "Doc"), ev.get("page", "?"), ev.get("chunk", "")
            evidence_parts.append(f"**{title}** (Page {page})\n{chunk[:300]}...")
            evidence_items.append(EvidenceItem(title=title, page=page, chunk=chunk, score=ev.get("score", 0)))
    
    return "\n\n".join(evidence_parts) if evidence_parts else "No evidence available.", evidence_items


def get_prior_summaries(state: AgentState) -> str:
    """Gets summaries from previous workers"""
    prior_summaries = state.get("pending_context", {}).get("prior_summaries", [])
    if not prior_summaries:
        return "No prior context."
    return "\n".join([f"- **{ps.get('worker')}**: {ps.get('summary')}" for ps in prior_summaries if ps.get('summary')]) or "No prior context."


# =============================================================================
# PRACTICE SYSTEM INTEGRATION
# =============================================================================

def detect_practice_intent(user_message: str) -> Optional[Tuple[str, str]]:
    """
    Detecta si el usuario quiere iniciar, continuar o completar una práctica.
    
    Returns:
        Tuple (action, practice_id) o None
        action puede ser: 'start', 'continue', 'next', 'hint', 'complete', 'list'
    """
    msg_lower = user_message.lower()
    
    # Listar prácticas disponibles
    if any(kw in msg_lower for kw in ["qué prácticas", "lista de prácticas", "prácticas disponibles", "available practices"]):
        return ("list", None)
    
    # Iniciar práctica específica
    start_patterns = [
        r"(?:iniciar|comenzar|empezar|start)\s+(?:la\s+)?práctica\s+(?:de\s+)?(.+)",
        r"(?:quiero|vamos a)\s+(?:hacer|realizar)\s+(?:la\s+)?práctica\s+(?:de\s+)?(.+)",
        r"practice[:\s]+(.+)",
    ]
    for pattern in start_patterns:
        match = re.search(pattern, msg_lower)
        if match:
            practice_hint = match.group(1).strip()
            # Mapear a practice_id conocido
            if any(kw in practice_hint for kw in ["abb", "pick", "place", "irb"]):
                return ("start", "abb_irb1100_pick_and_place")
            return ("start", practice_hint)
    
    # Continuar/siguiente paso
    if any(kw in msg_lower for kw in ["siguiente", "next", "continuar", "continue", "listo", "done", "siguiente paso"]):
        return ("next", None)
    
    # Pedir pista
    if any(kw in msg_lower for kw in ["pista", "hint", "ayuda", "help", "no entiendo", "explica más"]):
        return ("hint", None)
    
    # Completar práctica
    if any(kw in msg_lower for kw in ["terminar práctica", "finalizar práctica", "complete practice", "finish practice"]):
        return ("complete", None)
    
    return None


def get_active_practice_session(state: AgentState) -> Optional[PracticeSession]:
    """Obtiene la sesión de práctica activa del estado"""
    if not PRACTICES_AVAILABLE:
        return None
    
    pending_context = state.get("pending_context", {})
    practice_session_data = pending_context.get("active_practice_session")
    
    if practice_session_data:
        return PracticeSession.from_dict(practice_session_data)
    
    return None


def handle_practice_request(
    state: AgentState,
    user_message: str,
    intent: Tuple[str, str]
) -> Optional[Dict[str, Any]]:
    """
    Maneja solicitudes relacionadas con prácticas.
    
    Returns:
        Dict con respuesta si se manejó, None si debe continuar normal
    """
    if not PRACTICES_AVAILABLE:
        return None
    
    action, practice_id = intent
    manager = get_practice_manager()
    user_id = state.get("thread_id", "default_user")
    
    events = [event_execute("tutor", f"Practice action: {action}")]
    
    # LISTAR PRÁCTICAS
    if action == "list":
        practices = manager.get_available_practices()
        if not practices:
            content = "No hay prácticas disponibles en este momento."
        else:
            content = "## Prácticas Disponibles\n\n"
            for p in practices:
                content += f"### {p['title']}\n"
                content += f"- **Dificultad:** {p['difficulty']}\n"
                content += f"- **Duración estimada:** {p['estimated_time']} minutos\n"
                content += f"- **Descripción:** {p['description'][:200]}...\n\n"
            content += "\nPara iniciar una práctica, escribe: **'Iniciar práctica de [nombre]'**"
        
        output = WorkerOutputBuilder.tutor(
            content=content,
            learning_objectives=["Conocer prácticas disponibles"],
            summary="Lista de prácticas disponibles",
            confidence=1.0,
        )
        
        return {
            "worker_outputs": [output.model_dump()],
            "tutor_result": output.model_dump_json(),
            "events": events,
            "follow_up_suggestions": [
                "Start the ABB Pick and Place practice",
                "What will I learn in each practice?",
                "What prerequisites do I need?"
            ],
        }
    
    # INICIAR PRÁCTICA
    if action == "start" and practice_id:
        practice = manager.get_practice(practice_id)
        if not practice:
            # Buscar práctica similar
            available = manager.get_available_practices()
            suggestions = [p['id'] for p in available]
            content = f"No encontré la práctica '{practice_id}'. Prácticas disponibles: {', '.join(suggestions)}"
            
            output = WorkerOutputBuilder.tutor(
                content=content,
                learning_objectives=[],
                summary="Práctica no encontrada",
                confidence=0.5,
            )
            return {
                "worker_outputs": [output.model_dump()],
                "tutor_result": output.model_dump_json(),
                "events": events,
                "follow_up_suggestions": [f"Iniciar práctica {s}" for s in suggestions[:3]],
            }
        
        # Crear sesión
        session = manager.get_or_create_session(user_id, practice_id)
        
        # Obtener primer paso
        step_context = manager.get_step_context_for_tutor(session)
        step_display = manager.format_step_for_display(session)
        
        content = f"""## ¡Iniciando Práctica!

**{practice.title}**

{practice.description}

**Duración estimada:** {practice.estimated_total_time_minutes} minutos
**Dificultad:** {practice.difficulty_level}

---

{step_context}

---

Cuando estés listo para continuar, escribe **"siguiente"** o **"listo"**.
Si necesitas ayuda, escribe **"pista"** o **"ayuda"**.
"""
        
        output = WorkerOutputBuilder.tutor(
            content=content,
            learning_objectives=practice.sections[0].learning_objectives if practice.sections else [],
            summary=f"Práctica iniciada: {practice.title}",
            confidence=1.0,
        )
        
        # Guardar sesión en pending_context
        updated_context = state.get("pending_context", {}).copy()
        updated_context["active_practice_session"] = session.to_dict()
        updated_context["practice_mode"] = True
        
        return {
            "worker_outputs": [output.model_dump()],
            "tutor_result": output.model_dump_json(),
            "events": events,
            "pending_context": updated_context,
            "follow_up_suggestions": [
                "I'm ready, next step",
                "I need a hint",
                "Explain this in more detail"
            ],
        }
    
    # SIGUIENTE PASO
    if action == "next":
        session = get_active_practice_session(state)
        if not session:
            return None  # No hay práctica activa, continuar normal
        
        practice = manager.get_practice(session.practice_id)
        if not practice:
            return None
        
        current_step_id = session.current_step_id
        
        # Marcar paso actual como completado
        next_step = manager.complete_step(session, current_step_id, response="completed", score=1.0)
        
        events.append(event_report("tutor", f"Step completed: {current_step_id}"))
        
        if next_step:
            # Hay más pasos
            step_context = manager.get_step_context_for_tutor(session)
            
            content = f"""## Paso Completado ✓

¡Muy bien! Avanzando al siguiente paso...

---

{step_context}

---

Escribe **"siguiente"** cuando estés listo, o **"pista"** si necesitas ayuda.
"""
            
            output = WorkerOutputBuilder.tutor(
                content=content,
                learning_objectives=[],
                summary=f"Avanzando en práctica ({len(session.completed_step_ids)}/{practice.get_total_steps()})",
                confidence=1.0,
            )
            
            # Actualizar sesión en context
            updated_context = state.get("pending_context", {}).copy()
            updated_context["active_practice_session"] = session.to_dict()
            
            return {
                "worker_outputs": [output.model_dump()],
                "tutor_result": output.model_dump_json(),
                "events": events,
                "pending_context": updated_context,
                "follow_up_suggestions": [
                    "Next step",
                    "I need more explanation",
                    "Show me an example"
                ],
            }
        else:
            # Práctica completada
            summary = manager.complete_practice(session)
            
            content = f"""## 🎉 ¡Práctica Completada!

**{practice.title}**

### Resumen de tu desempeño:

- **Puntuación:** {summary['average_score']:.0f}%
- **Pasos completados:** {summary['completion_rate']:.0f}%
- **Pistas solicitadas:** {summary['hints_requested']}
- **Estado:** {'✅ APROBADO' if summary['passed'] else '⚠️ Necesita más práctica'}

### Fortalezas:
{chr(10).join(['- ' + s for s in summary['strengths']])}

### Áreas de mejora:
{chr(10).join(['- ' + s for s in summary['areas_to_improve']])}

### Recomendaciones:
{chr(10).join(['- ' + r for r in summary['recommendations']])}

---

¡Felicidades por completar la práctica! Tu progreso ha sido guardado.
"""
            
            output = WorkerOutputBuilder.tutor(
                content=content,
                learning_objectives=["Completar práctica exitosamente"],
                summary=f"Práctica completada: {summary['average_score']:.0f}%",
                confidence=1.0,
            )
            
            # Limpiar práctica del context
            updated_context = state.get("pending_context", {}).copy()
            updated_context.pop("active_practice_session", None)
            updated_context["practice_mode"] = False
            
            return {
                "worker_outputs": [output.model_dump()],
                "tutor_result": output.model_dump_json(),
                "events": events,
                "pending_context": updated_context,
                "follow_up_suggestions": [
                    "Start another practice",
                    "Review what I learned",
                    "What should I practice next?"
                ],
            }
    
    # PISTA
    if action == "hint":
        session = get_active_practice_session(state)
        if not session:
            return None
        
        manager.record_hint_request(session)
        
        # Obtener contexto con hints
        step_context = manager.get_step_context_for_tutor(session, include_hints=True)
        
        content = f"""## 💡 Aquí tienes una pista

{step_context}

---

Recuerda: pedir ayuda es parte del aprendizaje. Tómate tu tiempo.
"""
        
        output = WorkerOutputBuilder.tutor(
            content=content,
            learning_objectives=[],
            summary="Pista proporcionada",
            confidence=1.0,
        )
        
        # Actualizar sesión
        updated_context = state.get("pending_context", {}).copy()
        updated_context["active_practice_session"] = session.to_dict()
        
        return {
            "worker_outputs": [output.model_dump()],
            "tutor_result": output.model_dump_json(),
            "events": events,
            "pending_context": updated_context,
            "follow_up_suggestions": [
                "I understand now, next step",
                "I still need more help",
                "Can you show me an example?"
            ],
        }
    
    return None


def tutor_node(state: AgentState) -> Dict[str, Any]:
    """Worker tutor that generates educational content with relevant images."""
    start_time = datetime.utcnow()
    logger.node_start("tutor_node", {"has_pending_context": bool(state.get("pending_context"))})
    events = [event_execute("tutor", "Preparing educational content...")]

    user_message = get_last_user_message(state)
    if not user_message:
        error_output = create_error_output("tutor", "NO_MESSAGE", "No user message found")
        return {
            "worker_outputs": [error_output.model_dump()],
            "tutor_result": error_output.model_dump_json(),
            "events": events,
            "follow_up_suggestions": ["Ask a technical question", "Request an explanation", "Inquire about lab equipment"],
        }

    # ==========================================================================
    # PRACTICE MODE: Detectar y manejar solicitudes de práctica
    # ==========================================================================
    if PRACTICES_AVAILABLE:
        # Verificar si hay práctica activa o si el usuario quiere iniciar una
        practice_intent = detect_practice_intent(user_message)
        active_session = get_active_practice_session(state)
        
        # Si hay práctica activa y el mensaje es simple, tratarlo como "siguiente"
        if active_session and not practice_intent:
            # Mensajes simples en práctica activa = avanzar
            simple_confirmations = ["ok", "sí", "si", "yes", "listo", "entendido", "vale", "de acuerdo"]
            if user_message.lower().strip() in simple_confirmations:
                practice_intent = ("next", None)
        
        if practice_intent or active_session:
            # Si hay intent de práctica, procesarlo
            if practice_intent:
                practice_response = handle_practice_request(state, user_message, practice_intent)
                if practice_response:
                    return practice_response
            
            # Si hay práctica activa pero el mensaje es una pregunta sobre el contenido,
            # continuar con el flujo normal del tutor pero agregar contexto de práctica
            if active_session:
                events.append(event_report("tutor", f"Practice mode active: {active_session.practice_id}"))

    # ==========================================================================
    # FLUJO NORMAL DEL TUTOR
    # ==========================================================================
    evidence_text, evidence_items = get_evidence_from_context(state)
    context_text = get_prior_summaries(state)
    has_evidence = len(evidence_items) > 0

    # Format user's learning_style
    learning_style_raw = state.get("learning_style", {})
    learning_style_text = format_learning_style(learning_style_raw)

    # Search for relevant images (only 1 best match)
    relevant_images = search_relevant_images(
        user_message=user_message,
        evidence_text=evidence_text,
        learning_style=learning_style_text,
        max_images=1
    )
    if relevant_images:
        events.append(event_report("tutor", f"Found {len(relevant_images)} relevant images"))

    model_name = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-20250514")
    try:
        llm = ChatAnthropic(model=model_name, temperature=0.7) if "claude" in model_name.lower() else ChatOpenAI(model=model_name, temperature=0.7)
    except Exception as e:
        error_output = create_error_output("tutor", "LLM_INIT_ERROR", f"Error initializing model: {str(e)}")
        return {
            "worker_outputs": [error_output.model_dump()],
            "tutor_result": error_output.model_dump_json(),
            "events": events,
            "follow_up_suggestions": ["Try again", "Check system status", "Report the issue"],
        }

    # Select appropriate learning style prompt
    learning_style_guidance = get_learning_style_prompt(learning_style_raw)

    prompt = TUTOR_MULTISTEP_PROMPT.format(
        context_section=context_text if context_text != "No prior context." else "First interaction",
        evidence_section=evidence_text,
        learning_style_guidance=learning_style_guidance,
        user_name=state.get("user_name", "User"),
        learning_style=learning_style_text
    )

    messages = [SystemMessage(content=prompt)]
    if rolling_summary := state.get("rolling_summary", ""):
        messages.append(SystemMessage(content=f"Conversation context:\n{rolling_summary}"))
    messages.append(HumanMessage(content=user_message))

    suggestions = []
    try:
        response = llm.invoke(messages)
        raw_result = (response.content or "").strip()
        # Extract suggestions from result
        result_text, suggestions = extract_suggestions_from_text(raw_result)
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
    except Exception as e:
        error_output = create_error_output("tutor", "LLM_ERROR", f"Error generating response: {str(e)}")
        return {
            "worker_outputs": [error_output.model_dump()],
            "tutor_result": error_output.model_dump_json(),
            "events": events,
            "follow_up_suggestions": ["Try rephrasing your question", "Check connection", "Report issue"],
        }

    # Build output with images in extra
    output = WorkerOutputBuilder.tutor(
        content=result_text,
        learning_objectives=["Understand the concept", "Apply in practice"],
        summary=f"Educational explanation generated ({len(result_text)} chars)",
        confidence=0.85 if has_evidence else 0.75,
    )

    # Add images to extra
    if relevant_images:
        output.extra["images"] = relevant_images
        output.extra["images_count"] = len(relevant_images)

    if evidence_items:
        output.evidence = evidence_items
    output.metadata.completed_at = datetime.utcnow().isoformat()
    output.metadata.processing_time_ms = processing_time
    output.metadata.model_used = model_name

    logger.node_end("tutor_node", {"content_length": len(result_text), "images_found": len(relevant_images)})
    events.append(event_report("tutor", "Explanation ready"))

    # Default suggestions if none extracted
    if not suggestions:
        suggestions = [
            "Explore related concepts in depth",
            "Request practical examples",
            "Ask about advanced applications"
        ]

    return {
        "worker_outputs": [output.model_dump()],
        "tutor_result": output.model_dump_json(),
        "events": events,
        "follow_up_suggestions": suggestions,
    }
