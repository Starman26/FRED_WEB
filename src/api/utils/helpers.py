"""
Funciones auxiliares para procesamiento de eventos y mensajes
"""
from typing import Optional, List, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage

from src.api.models import Question, QuestionOption, QuestionType


def extract_message_content(msg: Any) -> Optional[str]:
    """Extrae contenido de un mensaje LangChain o dict."""
    if isinstance(msg, (AIMessage, HumanMessage)):
        return msg.content if hasattr(msg, "content") else None
    elif isinstance(msg, dict):
        return msg.get("content")
    elif isinstance(msg, str):
        return msg
    elif hasattr(msg, "content"):
        return msg.content
    return None


def extract_questions_from_event(event: Dict) -> List[Dict]:
    """Extrae preguntas estructuradas de un evento del grafo."""
    for node_name in ["troubleshooting", "research", "tutor", "chat"]:
        if node_name in event and isinstance(event[node_name], dict):
            questions = event[node_name].get("clarification_questions", [])
            if questions:
                return questions
    return []


def extract_worker_content(events: List[Dict]) -> str:
    """Extrae el contenido del worker antes de las preguntas."""
    for event in events:
        for node_name in ["troubleshooting", "research", "tutor", "chat"]:
            if node_name in event and isinstance(event[node_name], dict):
                worker_outputs = event[node_name].get("worker_outputs", [])
                for wo in worker_outputs:
                    if isinstance(wo, dict) and wo.get("content"):
                        return wo.get("content", "")
    return ""


def format_event_display(event: Dict) -> Dict[str, str]:
    """Formatea un evento para mostrar en el cliente."""
    
    event_type = event.get("type", "").lower()
    node = event.get("source", event.get("node", "?")).upper()
    message = event.get("content", event.get("message", ""))
    
    return {
        "node": node,
        "message": message,
        "type": event_type
    }


def parse_questions(raw_questions: List[Dict]) -> List[Question]:
    """Convierte preguntas raw del agente a modelos Pydantic."""
    questions = []
    
    for q in raw_questions:
        # Parsear opciones si existen
        options = None
        if q.get("options"):
            options = [
                QuestionOption(
                    id=opt.get("id", ""),
                    label=opt.get("label", ""),
                    description=opt.get("description")
                )
                for opt in q.get("options", [])
            ]
        
        # Determinar tipo
        q_type = QuestionType.choice
        if q.get("type") == "boolean":
            q_type = QuestionType.boolean
        elif q.get("type") == "text":
            q_type = QuestionType.text
        
        questions.append(Question(
            id=q.get("id", f"q{len(questions)}"),
            question=q.get("question", ""),
            type=q_type,
            options=options,
            placeholder=q.get("placeholder")
        ))
    
    return questions


def format_answers_for_agent(questions: List[Dict], answers: Dict[str, str]) -> str:
    """Formatea las respuestas del usuario para enviar al agente."""
    lines = []
    for q in questions:
        q_id = q.get("id", "")
        q_text = q.get("question", "")
        answer = answers.get(q_id, "No respondida")
        lines.append(f"- {q_text}: {answer}")
    return "\n".join(lines)
