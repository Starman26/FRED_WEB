"""
human_input.py - Nodo de Human-in-the-Loop

Este nodo se activa cuando un worker necesita más información del usuario.
Usa interrupt() de LangGraph para pausar el grafo.

Para reanudar:
    from langgraph.types import Command
    result = graph.invoke(Command(resume="respuesta"), config=config)
"""
from typing import Dict, Any, List

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from src.agent.state import AgentState
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_read, event_report


def format_questions_for_display(questions: List[Any]) -> str:
    """
    Formatea las preguntas para mostrar al usuario.
    Soporta tanto strings como dicts estructurados.
    """
    lines = []
    for i, q in enumerate(questions[:5], 1):  # Máximo 5 preguntas
        if isinstance(q, dict):
            # Pregunta estructurada
            question_text = q.get("question", str(q))
            q_type = q.get("type", "text")
            options = q.get("options", [])
            
            lines.append(f"**Pregunta {i}:** {question_text}")
            
            if options and q_type == "choice":
                for opt in options:
                    if isinstance(opt, dict):
                        opt_label = opt.get("label", "")
                        opt_desc = opt.get("description", "")
                        if opt_desc:
                            lines.append(f"  - {opt_label}: {opt_desc}")
                        else:
                            lines.append(f"  - {opt_label}")
            lines.append("")
        else:
            # String simple
            lines.append(f"**Pregunta {i}:** {q}")
            lines.append("")
    
    return "\n".join(lines)


def human_input_node(state: AgentState) -> Dict[str, Any]:
    """
    Nodo Human-in-the-Loop que interrumpe el workflow para pedir input al usuario.
    
    Usa interrupt() de LangGraph. El grafo se pausa hasta que se invoque
    con Command(resume="respuesta del usuario").
    """
    questions = state.get("clarification_questions", [])
    pending_context = state.get("pending_context", {}) or {}
    
    logger.node_start("human_input", {
        "questions_count": len(questions)
    })
    
    events = [event_read("human_input", "Solicitando información al usuario...")]
    
    # Construir prompt para el interrupt
    if questions:
        # Obtener contexto del wizard si existe
        wizard_context = ""
        if pending_context.get("question_set"):
            try:
                import json
                qs_data = json.loads(pending_context.get("question_set", "{}"))
                wizard_context = qs_data.get("context", "")
                wizard_title = qs_data.get("title", "Información necesaria")
            except:
                wizard_title = "Información necesaria"
        else:
            wizard_title = "Información necesaria"
        
        # Formatear preguntas
        formatted_questions = format_questions_for_display(questions)
        
        prompt = f"""## {wizard_title}

{wizard_context}

{formatted_questions}

Por favor, proporciona la información solicitada."""
    else:
        prompt = "¿Podrías darme más contexto sobre tu solicitud?"
    
    logger.info("human_input", f"Wizard paso 1: {prompt[:100]}...")
    
    # ==========================================
    # INTERRUPT: Pausa el grafo y espera input
    # ==========================================
    user_response = interrupt(prompt)
    
    # ==========================================
    # REANUDACIÓN: Se ejecuta cuando el usuario responde
    # ==========================================
    logger.info("human_input", f"Usuario respondió: {str(user_response)[:100]}...")
    events.append(event_report("human_input", "✅ Respuesta del usuario recibida"))
    
    # Preservar contexto existente y agregar respuesta
    updated_context = pending_context.copy()
    updated_context["user_clarification"] = str(user_response)
    
    # Marcar que el wizard se completó
    updated_context["wizard_completed"] = True
    
    # Determinar a dónde ir después
    # Si hay un worker específico que solicitó la info, volver a él
    current_worker = updated_context.get("current_worker", "")
    orchestration_plan = state.get("orchestration_plan", [])
    
    # El route decidirá, pero indicamos que debe continuar el plan
    next_destination = "route"
    
    logger.info("human_input", f"Contexto actualizado, continuando a: {next_destination}")
    
    return {
        "messages": [
            AIMessage(content=prompt),
            HumanMessage(content=str(user_response))
        ],
        "needs_human_input": False,
        "clarification_questions": [],  # Limpiar preguntas
        "pending_context": updated_context,
        "next": next_destination,
        "events": events,
    }


def create_clarification_request(
    worker: str,
    questions: List[str],
    partial_content: str = ""
) -> Dict[str, Any]:
    """
    Helper para que workers creen una solicitud de clarificación.
    """
    from src.agent.contracts.worker_contract import create_needs_context_output
    
    output = create_needs_context_output(
        worker=worker,
        questions=questions,
        partial_content=partial_content
    )
    
    return {
        "worker_outputs": [output.model_dump()],
        f"{worker}_result": output.model_dump_json(),
        "needs_human_input": True,
        "clarification_questions": questions,
    }


def build_context_from_response(
    original_query: str,
    clarification_response: str
) -> str:
    """
    Helper para combinar la query original con la respuesta de clarificación.
    """
    return f"""Consulta original: {original_query}

Información adicional proporcionada: {clarification_response}"""
