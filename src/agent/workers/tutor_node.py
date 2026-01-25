"""
tutor_node.py - Worker especializado en tutorías y explicaciones educativas

Usa WorkerOutput contract, NO retorna done=True, usa pending_context para evidencia.
Incluye soporte para imágenes educativas del banco de imágenes.
"""
import os
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from src.agent.state import AgentState
from src.agent.contracts.worker_contract import WorkerOutputBuilder, EvidenceItem, create_error_output
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_execute, event_report, event_error
from src.agent.utils.format_helpers import format_learning_style
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


TUTOR_MULTISTEP_PROMPT = """Eres un **Tutor Técnico Especializado** experto en:
- PLCs (Controladores Lógicos Programables)
- Cobots (Robots Colaborativos)
- Python y AI/ML (LangGraph, LangChain)

## CONTEXTO IMPORTANTE
{context_section}

## EVIDENCIA DE INVESTIGACIÓN PREVIA
{evidence_section}

## INSTRUCCIONES
1. **Usa la evidencia proporcionada**: Si hay evidencia, úsala y cítala [Título, Pág. X-Y]
2. **Estructura clara**: Usa encabezados y listas cuando ayuden
3. **Sé didáctico**: Explica paso a paso, con ejemplos
4. **Responde en español**
5. **Dependiendo de la forma de aprendizaje del usuario usa diferentes tonos**

{learning_style_guidance}

Nombre del usuario: {user_name}
Perfil de aprendizaje: {learning_style}

"""


def detect_image_category(text: str) -> Optional[Any]:
    """
    Detecta la categoría de imagen más relevante basada en el texto.

    Args:
        text: Texto del usuario o contexto

    Returns:
        ImageCategory más relevante o None
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
    Extrae términos de búsqueda relevantes para imágenes.

    Args:
        user_message: Mensaje del usuario
        evidence_text: Texto de evidencia previa

    Returns:
        Query de búsqueda para imágenes
    """
    # Palabras clave técnicas relevantes
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

    # Usar los primeros 3-4 keywords más relevantes
    if found_keywords:
        return " ".join(found_keywords[:4])

    # Fallback: usar las primeras palabras significativas del mensaje
    words = re.findall(r'\b\w{4,}\b', user_message.lower())
    return " ".join(words[:3]) if words else "industrial automation"


def search_relevant_images(
    user_message: str,
    evidence_text: str = "",
    learning_style: str = "",
    max_images: int = 3
) -> List[Dict[str, Any]]:
    """
    Busca imágenes relevantes para la explicación educativa.

    Args:
        user_message: Mensaje del usuario
        evidence_text: Texto de evidencia
        learning_style: Estilo de aprendizaje del usuario
        max_images: Número máximo de imágenes a retornar

    Returns:
        Lista de diccionarios con metadatos de imagen
    """
    if not IMAGES_AVAILABLE:
        return []

    # Los estudiantes visuales obtienen más imágenes
    if "visual" in learning_style.lower():
        max_images = min(max_images + 2, 5)

    try:
        sourcer = get_image_sourcer()

        # Detectar categoría
        category = detect_image_category(user_message)

        # Extraer términos de búsqueda
        search_query = extract_image_search_terms(user_message, evidence_text)

        # Crear solicitud de imagen
        request = ImageRequest(
            query=search_query,
            category=category,
            max_results=max_images,
            require_commercial_license=False  # Para uso educativo
        )

        # Buscar (primero local, luego online)
        result = sourcer.search(request, include_online=True)

        # Convertir a diccionarios serializables
        images = []
        for img in result.images:
            images.append({
                "id": img.id,
                "title": img.title,
                "source": img.source.value if hasattr(img.source, 'value') else str(img.source),
                "source_url": img.source_url,
                "source_page": img.source_page,
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
    Selecciona el prompt de estilo de aprendizaje apropiado basado en el tipo.

    Args:
        learning_style_dict: Diccionario con el learning_style del usuario

    Returns:
        El prompt específico para ese estilo de aprendizaje
    """
    if not learning_style_dict or not isinstance(learning_style_dict, dict):
        return MIX_TUTOR  # Default: mixto

    learning_type = learning_style_dict.get("type", "").lower()

    # Mapeo de tipos a prompts
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
    """Extrae el último mensaje del usuario"""
    for m in reversed(state.get("messages", []) or []):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
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


def tutor_node(state: AgentState) -> Dict[str, Any]:
    """Worker tutor que genera contenido educativo con imágenes relevantes."""
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

    # Formatear el learning_style del usuario
    learning_style_raw = state.get("learning_style", {})
    learning_style_text = format_learning_style(learning_style_raw)

    # Buscar imágenes relevantes para la explicación
    relevant_images = search_relevant_images(
        user_message=user_message,
        evidence_text=evidence_text,
        learning_style=learning_style_text,
        max_images=3
    )
    if relevant_images:
        events.append(event_report("tutor", f"Encontradas {len(relevant_images)} imágenes relevantes"))

    model_name = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-20250514")
    try:
        llm = ChatAnthropic(model=model_name, temperature=0.7) if "claude" in model_name.lower() else ChatOpenAI(model=model_name, temperature=0.7)
    except Exception as e:
        error_output = create_error_output("tutor", "LLM_INIT_ERROR", f"Error inicializando modelo: {str(e)}")
        return {"worker_outputs": [error_output.model_dump()], "tutor_result": error_output.model_dump_json(), "events": events}

    # Seleccionar el prompt de estilo de aprendizaje apropiado
    learning_style_guidance = get_learning_style_prompt(learning_style_raw)

    prompt = TUTOR_MULTISTEP_PROMPT.format(
        context_section=context_text if context_text != "Sin contexto previo." else "Primera interacción",
        evidence_section=evidence_text,
        learning_style_guidance=learning_style_guidance,
        user_name=state.get("user_name", "Usuario"),
        learning_style=learning_style_text
    )

    messages = [SystemMessage(content=prompt)]
    if rolling_summary := state.get("rolling_summary", ""):
        messages.append(SystemMessage(content=f"Contexto de la conversación:\n{rolling_summary}"))
    messages.append(HumanMessage(content=user_message))

    try:
        response = llm.invoke(messages)
        result_text = (response.content or "").strip()
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
    except Exception as e:
        error_output = create_error_output("tutor", "LLM_ERROR", f"Error generando respuesta: {str(e)}")
        return {"worker_outputs": [error_output.model_dump()], "tutor_result": error_output.model_dump_json(), "events": events}

    # Construir output con imágenes en extra
    output = WorkerOutputBuilder.tutor(
        content=result_text,
        learning_objectives=["Comprender el concepto", "Aplicar en práctica"],
        summary=f"Explicación educativa generada ({len(result_text)} chars)",
        confidence=0.85 if has_evidence else 0.75,
    )

    # Agregar imágenes al extra
    if relevant_images:
        output.extra["images"] = relevant_images
        output.extra["images_count"] = len(relevant_images)

    if evidence_items:
        output.evidence = evidence_items
    output.metadata.completed_at = datetime.utcnow().isoformat()
    output.metadata.processing_time_ms = processing_time
    output.metadata.model_used = model_name

    logger.node_end("tutor_node", {"content_length": len(result_text), "images_found": len(relevant_images)})
    events.append(event_report("tutor", "Explicación lista"))

    return {"worker_outputs": [output.model_dump()], "tutor_result": output.model_dump_json(), "events": events}
