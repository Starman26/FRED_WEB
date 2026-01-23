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


def human_input_node(state: AgentState) -> Dict[str, Any]:
    """
    Nodo Human-in-the-Loop que interrumpe el workflow para pedir input al usuario.
    
    Usa interrupt() de LangGraph. El grafo se pausa hasta que se invoque
    con Command(resume="respuesta del usuario").
    """
    logger.node_start("human_input", {
        "questions_count": len(state.get("clarification_questions", []))
    })
    
    events = [event_read("human_input", "Solicitando información al usuario...")]
    
    # Obtener preguntas de clarificación
    questions = state.get("clarification_questions", [])
    
    # Construir mensaje de clarificación
    if questions:
        questions_to_ask = questions[:3]
        question_text = "\n".join([f"• {q}" for q in questions_to_ask])
        prompt = f"""Necesito más información para ayudarte mejor:

{question_text}

Por favor, proporciona los detalles que puedas."""
    else:
        prompt = "¿Podrías darme más contexto sobre tu solicitud?"
    
    logger.info("human_input", f"Preguntando: {prompt[:100]}...")
    
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
    pending_context = state.get("pending_context", {})
    pending_context["user_clarification"] = str(user_response)
    
    return {
        "messages": [
            AIMessage(content=prompt),
            HumanMessage(content=str(user_response))
        ],
        "needs_human_input": False,
        "clarification_questions": [],
        "pending_context": pending_context,
        "next": "route",  # Volver al route para continuar el plan
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
