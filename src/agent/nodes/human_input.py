"""
human_input.py - Nodo de Human-in-the-Loop con soporte de Wizard Interactivo

Este nodo se activa cuando un worker necesita más información del usuario.
Usa interrupt() de LangGraph para pausar el grafo.

Características:
- Wizard mode: Preguntas una por una con navegación (atrás, saltar, cancelar)
- Formulario rápido: Todas las preguntas a la vez
- Validación de respuestas en tiempo real
- Progreso visual del wizard
- Soporte para preguntas condicionales

Para reanudar:
    from langgraph.types import Command
    result = graph.invoke(Command(resume="respuesta"), config=config)
"""
from typing import Dict, Any, List, Optional, Union
import json

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from src.agent.state import AgentState
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_read, event_report
from src.agent.contracts.question_schema import (
    QuestionSet,
    WizardState,
    WizardAction,
    QuestionSetResponse,
    ClarificationQuestion,
    QuestionType,
)


def human_input_node(state: AgentState) -> Dict[str, Any]:
    """
    Nodo Human-in-the-Loop que interrumpe el workflow para pedir input al usuario.

    Soporta dos modos:
    1. Wizard mode: Preguntas una por una con navegación
    2. Simple mode: Todas las preguntas juntas (legacy)

    Usa interrupt() de LangGraph. El grafo se pausa hasta que se invoque
    con Command(resume="respuesta del usuario").
    """
    logger.node_start("human_input", {
        "questions_count": len(state.get("clarification_questions", []))
    })

    events = [event_read("human_input", "Solicitando información al usuario...")]
    pending_context = state.get("pending_context", {})

    # Verificar si hay un wizard en progreso
    wizard_state = _get_or_create_wizard_state(state, pending_context)

    if wizard_state:
        return _handle_wizard_flow(state, wizard_state, pending_context, events)
    else:
        return _handle_simple_flow(state, pending_context, events)


def _get_or_create_wizard_state(
    state: AgentState,
    pending_context: Dict[str, Any]
) -> Optional[WizardState]:
    """Obtiene o crea el estado del wizard si corresponde."""
    # Si ya hay un wizard en progreso, restaurarlo
    if "wizard_state" in pending_context:
        try:
            wizard_data = pending_context["wizard_state"]
            if isinstance(wizard_data, str):
                wizard_data = json.loads(wizard_data)
            return WizardState.model_validate(wizard_data)
        except Exception as e:
            logger.warning("human_input", f"Error restaurando wizard: {e}")

    # Si hay un question_set estructurado, crear wizard
    if "question_set" in pending_context:
        try:
            qs_data = pending_context["question_set"]
            if isinstance(qs_data, str):
                qs_data = json.loads(qs_data)
            question_set = QuestionSet.model_validate(qs_data)
            return question_set.to_wizard_state()
        except Exception as e:
            logger.warning("human_input", f"Error creando wizard: {e}")

    # Verificar si las preguntas son objetos estructurados
    questions = state.get("clarification_questions", [])
    if questions and isinstance(questions[0], dict) and "type" in questions[0]:
        try:
            parsed_questions = [
                ClarificationQuestion.model_validate(q) if isinstance(q, dict) else q
                for q in questions
            ]
            question_set = QuestionSet(
                questions=parsed_questions,
                worker=pending_context.get("current_worker", "unknown"),
                wizard_mode=len(parsed_questions) > 1,
                max_questions=5
            )
            return question_set.to_wizard_state()
        except Exception as e:
            logger.warning("human_input", f"Error parseando preguntas: {e}")

    return None


def _handle_wizard_flow(
    state: AgentState,
    wizard_state: WizardState,
    pending_context: Dict[str, Any],
    events: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Maneja el flujo de wizard interactivo."""
    messages = []

    # Loop del wizard - continúa hasta completar o cancelar
    while not wizard_state.is_complete and not wizard_state.is_cancelled:
        # Generar prompt para la pregunta actual
        prompt = wizard_state.get_display_prompt()
        logger.info("human_input", f"Wizard paso {wizard_state.current_step + 1}: {prompt[:100]}...")

        # Guardar estado del wizard antes de interrumpir
        pending_context["wizard_state"] = wizard_state.model_dump_json()

        # ==========================================
        # INTERRUPT: Pausa el grafo y espera input
        # ==========================================
        user_response = interrupt(prompt)

        # ==========================================
        # REANUDACIÓN: Procesar respuesta del usuario
        # ==========================================
        logger.info("human_input", f"Usuario respondió: {str(user_response)[:100]}...")
        events.append(event_report("human_input", f"Respuesta recibida: {str(user_response)[:50]}..."))

        # Agregar mensajes al historial
        messages.append(AIMessage(content=prompt))
        messages.append(HumanMessage(content=str(user_response)))

        # Procesar la entrada del usuario
        action = wizard_state.process_input(str(user_response))

        if action.action_type == "back":
            if wizard_state.go_back():
                events.append(event_report("human_input", "⬅️ Volviendo a pregunta anterior"))
            else:
                events.append(event_report("human_input", "⚠️ No puedes volver más atrás"))

        elif action.action_type == "skip":
            if wizard_state.skip_current():
                events.append(event_report("human_input", "⏭️ Pregunta omitida"))
            else:
                current_q = wizard_state.get_current_question()
                if current_q and current_q.required:
                    events.append(event_report("human_input", "⚠️ Esta pregunta es obligatoria"))
                else:
                    events.append(event_report("human_input", "⚠️ No se puede omitir esta pregunta"))

        elif action.action_type == "cancel":
            wizard_state.cancel()
            events.append(event_report("human_input", "❌ Wizard cancelado"))

        elif action.action_type == "restart":
            wizard_state.restart()
            events.append(event_report("human_input", "🔄 Wizard reiniciado"))

        elif action.action_type == "answer":
            success, error = wizard_state.submit_answer(action.value)
            if success:
                events.append(event_report("human_input", "✅ Respuesta guardada"))
            else:
                # Mostrar error y repetir la pregunta
                events.append(event_report("human_input", f"⚠️ {error}"))
                # El loop continuará y mostrará la misma pregunta

    # Wizard completado o cancelado
    if wizard_state.is_complete:
        events.append(event_report("human_input", "✅ Wizard completado"))
        completion_msg = wizard_state.question_set.completion_message or "✅ ¡Gracias por la información!"
        messages.append(AIMessage(content=completion_msg))
    elif wizard_state.is_cancelled:
        events.append(event_report("human_input", "❌ Proceso cancelado por el usuario"))

    # Convertir respuestas a formato para el worker
    response = wizard_state.to_response()
    clarification_context = response.to_context_string()

    # Limpiar wizard del pending_context
    pending_context.pop("wizard_state", None)
    pending_context.pop("question_set", None)

    # Guardar respuestas estructuradas y string
    pending_context["user_clarification"] = clarification_context
    pending_context["wizard_responses"] = response.model_dump()
    pending_context["wizard_completed"] = response.completed
    pending_context["wizard_cancelled"] = response.cancelled

    return {
        "messages": messages,
        "needs_human_input": False,
        "clarification_questions": [],
        "pending_context": pending_context,
        "next": "route",
        "events": events,
    }


def _handle_simple_flow(
    state: AgentState,
    pending_context: Dict[str, Any],
    events: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Maneja el flujo simple (legacy) de preguntas."""
    questions = state.get("clarification_questions", [])

    # Construir mensaje de clarificación
    if questions:
        questions_to_ask = questions[:5]
        question_text = "\n".join([f"• {q}" for q in questions_to_ask])
        prompt = f"""📋 Necesito más información para ayudarte mejor:

{question_text}

💡 Por favor, proporciona los detalles que puedas.
_(Puedes responder todas las preguntas en un solo mensaje)_"""
    else:
        prompt = "❓ ¿Podrías darme más contexto sobre tu solicitud?"

    logger.info("human_input", f"Preguntando (modo simple): {prompt[:100]}...")

    # ==========================================
    # INTERRUPT: Pausa el grafo y espera input
    # ==========================================
    user_response = interrupt(prompt)

    # ==========================================
    # REANUDACIÓN: Se ejecuta cuando el usuario responde
    # ==========================================
    logger.info("human_input", f"Usuario respondió: {str(user_response)[:100]}...")
    events.append(event_report("human_input", "✅ Respuesta del usuario recibida"))

    # Guardar respuesta en pending_context para que el worker la use
    pending_context["user_clarification"] = str(user_response)

    return {
        "messages": [
            AIMessage(content=prompt),
            HumanMessage(content=str(user_response))
        ],
        "needs_human_input": False,
        "clarification_questions": [],
        "pending_context": pending_context,
        "next": "route",
        "events": events,
    }


def create_clarification_request(
    worker: str,
    questions: Union[List[str], List[ClarificationQuestion], QuestionSet],
    partial_content: str = "",
    wizard_mode: bool = True
) -> Dict[str, Any]:
    """
    Helper para que workers creen una solicitud de clarificación.

    Args:
        worker: Nombre del worker que solicita
        questions: Lista de strings, ClarificationQuestions o un QuestionSet completo
        partial_content: Contenido parcial para mostrar mientras se espera
        wizard_mode: Si True, usa el wizard interactivo (default)

    Returns:
        Dict con los campos necesarios para activar human-in-the-loop
    """
    from src.agent.contracts.worker_contract import create_needs_context_output

    # Convertir a lista de strings para el output legacy
    if isinstance(questions, QuestionSet):
        question_set = questions
        question_strings = [q.question for q in questions.questions]
    elif questions and isinstance(questions[0], ClarificationQuestion):
        question_set = QuestionSet(
            questions=questions,
            worker=worker,
            wizard_mode=wizard_mode,
            max_questions=5
        )
        question_strings = [q.question for q in questions]
    else:
        question_set = None
        question_strings = questions

    output = create_needs_context_output(
        worker=worker,
        questions=question_strings,
        partial_content=partial_content
    )

    result = {
        "worker_outputs": [output.model_dump()],
        f"{worker}_result": output.model_dump_json(),
        "needs_human_input": True,
        "clarification_questions": question_strings,
    }

    # Si hay question_set estructurado, agregarlo al pending_context
    if question_set:
        result["pending_context"] = {
            "question_set": question_set.model_dump_json(),
            "current_worker": worker,
        }

    return result


def create_wizard_request(
    worker: str,
    question_set: QuestionSet,
    partial_content: str = ""
) -> Dict[str, Any]:
    """
    Helper para crear una solicitud de wizard interactivo.

    Ejemplo:
        from src.agent.contracts.question_schema import get_troubleshooting_questions

        wizard = get_troubleshooting_questions(["modelo", "error"], wizard_mode=True)
        return create_wizard_request("troubleshooting", wizard)
    """
    return create_clarification_request(
        worker=worker,
        questions=question_set,
        partial_content=partial_content,
        wizard_mode=True
    )


def create_confirmation_request(
    worker: str,
    action_description: str,
    require_backup_check: bool = False
) -> Dict[str, Any]:
    """
    Helper para crear una solicitud de confirmación de acción.

    Ejemplo:
        return create_confirmation_request(
            worker="troubleshooting",
            action_description="Reiniciar el PLC S7-1500",
            require_backup_check=True
        )
    """
    from src.agent.contracts.question_schema import get_repair_confirmation_wizard

    wizard = get_repair_confirmation_wizard(action_description)
    return create_wizard_request(worker, wizard)


def build_context_from_response(
    original_query: str,
    clarification_response: str
) -> str:
    """
    Helper para combinar la query original con la respuesta de clarificación.
    """
    return f"""Consulta original: {original_query}

Información adicional proporcionada: {clarification_response}"""


def build_context_from_wizard_response(
    original_query: str,
    wizard_responses: Dict[str, Any]
) -> str:
    """
    Helper para construir contexto desde respuestas del wizard.

    Args:
        original_query: Consulta original del usuario
        wizard_responses: Dict con las respuestas del wizard (de pending_context["wizard_responses"])

    Returns:
        String formateado con toda la información
    """
    lines = [f"Consulta original: {original_query}", "", "Información proporcionada:"]

    if isinstance(wizard_responses, dict) and "responses" in wizard_responses:
        for resp in wizard_responses["responses"]:
            q_id = resp.get("question_id", "?")
            answer = resp.get("answer_label") or resp.get("answer", "(omitido)")
            if resp.get("is_skipped"):
                lines.append(f"- {q_id}: (omitido)")
            else:
                lines.append(f"- {q_id}: {answer}")
    else:
        lines.append(str(wizard_responses))

    return "\n".join(lines)


def parse_quick_reply(
    user_input: str,
    question: ClarificationQuestion
) -> Optional[str]:
    """
    Verifica si el input del usuario coincide con una quick reply.

    Returns:
        El valor de la quick reply si coincide, None si no
    """
    if not question.quick_replies:
        return None

    input_lower = user_input.strip().lower()

    for qr in question.quick_replies:
        if input_lower == qr.label.lower() or input_lower == qr.id.lower():
            return qr.value

    return None


def validate_and_sanitize_input(
    user_input: str,
    question: ClarificationQuestion
) -> tuple[bool, Any, Optional[str]]:
    """
    Valida y sanitiza el input del usuario para una pregunta.

    Returns:
        (is_valid, sanitized_value, error_message)
    """
    # Primero verificar quick replies
    quick_value = parse_quick_reply(user_input, question)
    if quick_value:
        return True, quick_value, None

    # Validar con el esquema de la pregunta
    result = question.validate_answer(user_input)

    if result.is_valid:
        return True, result.sanitized_value, None
    else:
        return False, None, result.error_message
