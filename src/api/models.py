"""
Pydantic models para la API
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class EventType(str, Enum):
    message = "message"           
    log = "log"                  
    questions = "questions"     
    error = "error"           
    done = "done"               
    ping = "ping"          


class QuestionType(str, Enum):
    choice = "choice"
    boolean = "boolean"
    text = "text"


# REQUEST MODELS

class ChatRequest(BaseModel):
    """Request para iniciar/continuar una conversación"""
    query: str = Field(..., min_length=1, max_length=10000, description="Mensaje del usuario")
    thread_id: Optional[str] = Field(None, description="ID del thread (se genera si no se proporciona)")
    user_name: Optional[str] = Field("Usuario", description="Nombre del usuario")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "¿Cómo configuro TIA Portal?",
                    "thread_id": "my-session-123",
                    "user_name": "Daniela"
                }
            ]
        }
    }


class ResumeRequest(BaseModel):
    """Request para responder a preguntas del agente (interrupt)"""
    thread_id: str = Field(..., description="ID del thread con preguntas pendientes")
    answers: Dict[str, str] = Field(..., description="Respuestas: {question_id: respuesta}")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "thread_id": "my-session-123",
                    "answers": {
                        "q1": "TIA Portal V17",
                        "q2": "Sí",
                        "q3": "Error de conexión"
                    }
                }
            ]
        }
    }

# RESPONSE MODELS

class QuestionOption(BaseModel):
    """Opción para preguntas de selección"""
    id: str
    label: str
    description: Optional[str] = None


class Question(BaseModel):
    """Pregunta estructurada del agente"""
    id: str
    question: str
    type: QuestionType = QuestionType.choice
    options: Optional[List[QuestionOption]] = None
    placeholder: Optional[str] = None


class StreamEvent(BaseModel):
    """Evento enviado via SSE"""
    type: EventType
    data: Any
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    def to_sse(self) -> str:
        """Formatea como Server-Sent Event"""
        import json
        payload = {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    

class ThreadStatus(BaseModel):
    """Estado de un thread"""
    thread_id: str
    has_pending_questions: bool
    questions: Optional[List[Question]] = None
    message_count: int = 0


class HealthResponse(BaseModel):
    """Respuesta de health check"""
    status: str = "ok"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: Dict[str, bool] = {}


class ErrorResponse(BaseModel):
    """Respuesta de error"""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
