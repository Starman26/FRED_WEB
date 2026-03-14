"""
worker_contract.py

Universal contract for worker-supervisor communication.
All workers must return WorkerOutput for the supervisor to accumulate
evidence, detect HITL needs, and chain multi-step orchestration.
"""
import uuid
import json
from typing import List, Optional, Literal, Any, Dict
from pydantic import BaseModel, Field
from datetime import datetime


class EvidenceItem(BaseModel):
    """A retrieved evidence fragment (primarily from RAG)."""
    source_id: Optional[str] = Field(default=None, description="ID del chunk/documento")
    title: str = Field(default="", description="Título del documento fuente")
    chunk: str = Field(default="", description="Contenido del fragmento")
    page: Optional[str] = Field(default=None, description="Página(s) del documento")
    score: float = Field(default=0.0, description="Score de relevancia (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata adicional")


class ActionItem(BaseModel):
    """Suggested action for the orchestrator."""
    type: Literal["ask_user", "call_worker", "call_tool", "end"]
    target: Optional[str] = Field(default=None, description="Nombre del worker/tool objetivo")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Datos para la acción")
    reason: str = Field(default="", description="Razón de la acción sugerida")
    priority: int = Field(default=0, description="Prioridad (mayor = más urgente)")


class ErrorItem(BaseModel):
    """Structured error for debugging."""
    code: str = Field(description="Código del error (ej: 'RAG_NO_RESULTS')")
    message: str = Field(description="Mensaje legible del error")
    severity: Literal["warning", "error", "critical"] = Field(default="error")
    debug: Dict[str, Any] = Field(default_factory=dict, description="Info de debug")
    recoverable: bool = Field(default=True, description="Si el error permite continuar")


class WorkerMetadata(BaseModel):
    """Execution metadata common to all workers."""
    started_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    tokens_used: int = Field(default=0, description="Tokens consumidos")
    model_used: str = Field(default="", description="Modelo LLM usado")
    processing_time_ms: float = Field(default=0.0, description="Tiempo de procesamiento")
    retries: int = Field(default=0, description="Número de reintentos")


class WorkerOutput(BaseModel):
    """Universal output contract for all workers."""

    worker: Literal["chat", "research", "tutor", "troubleshooting", "summarizer", "robot_operator", "analysis", "practice"]
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    status: Literal["ok", "needs_context", "partial", "error"] = Field(
        default="ok",
        description="ok=completo, needs_context=falta info, partial=resultado incompleto, error=falló"
    )
    
    summary: str = Field(
        default="",
        description="Resumen CORTO (1-2 oraciones) para el orchestrator"
    )
    content: str = Field(
        default="",
        description="Contenido COMPLETO para mostrar al usuario"
    )
    
    evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Fragmentos de evidencia recuperados"
    )
    
    next_actions: List[ActionItem] = Field(
        default_factory=list,
        description="Acciones que el orchestrator debería considerar"
    )
    
    clarification_questions: List[Any] = Field(
        default_factory=list,
        description="Preguntas para el usuario si status=needs_context. Puede ser strings o ClarificationQuestion dicts"
    )

    wizard_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuración del wizard interactivo (wizard_mode, max_questions, etc.)"
    )
    
    errors: List[ErrorItem] = Field(
        default_factory=list,
        description="Errores ocurridos durante la ejecución"
    )
    
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confianza en el resultado (0-1)"
    )
    metadata: WorkerMetadata = Field(default_factory=WorkerMetadata)
    
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Datos adicionales específicos del worker"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "worker": "research",
                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "ok",
                "summary": "Encontré 3 documentos relevantes sobre métricas RAGAS",
                "content": "Según el paper FrEDie...",
                "evidence": [
                    {
                        "source_id": "chunk_123",
                        "title": "FrEDie Paper",
                        "chunk": "Las métricas RAGAS incluyen...",
                        "page": "5-6",
                        "score": 0.92
                    }
                ],
                "confidence": 0.85,
                "next_actions": [
                    {
                        "type": "call_worker",
                        "target": "tutor",
                        "reason": "Sintetizar la evidencia encontrada"
                    }
                ]
            }
        }


def serialize_worker_output(output: WorkerOutput) -> str:
    return output.model_dump_json(indent=2)


def parse_worker_output(json_str: str) -> Optional[WorkerOutput]:
    """Parse JSON string to WorkerOutput, returns None on failure."""
    try:
        data = json.loads(json_str)
        return WorkerOutput(**data)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[WorkerContract] Error parseando output: {e}")
        return None


def create_error_output(
    worker: str,
    error_code: str,
    error_message: str,
    debug_info: Optional[Dict] = None
) -> WorkerOutput:
    return WorkerOutput(
        worker=worker,
        status="error",
        summary=f"Error: {error_message[:50]}",
        content=f"Ocurrió un error durante el procesamiento: {error_message}",
        errors=[
            ErrorItem(
                code=error_code,
                message=error_message,
                debug=debug_info or {},
                recoverable=True
            )
        ],
        confidence=0.0
    )


def create_needs_context_output(
    worker: str,
    questions: List[Any],
    partial_content: str = "",
    wizard_mode: bool = False,
    max_questions: int = 5
) -> WorkerOutput:
    """Build a needs_context output with clarification questions."""
    serialized_questions = []
    for q in questions:
        if hasattr(q, "model_dump"):
            serialized_questions.append(q.model_dump())
        elif isinstance(q, dict):
            serialized_questions.append(q)
        else:
            serialized_questions.append(str(q))

    wizard_config = None
    if wizard_mode or (serialized_questions and isinstance(serialized_questions[0], dict)):
        wizard_config = {
            "wizard_mode": wizard_mode,
            "max_questions": max_questions,
            "allow_back": True,
            "allow_skip": True,
            "show_progress": True
        }

    return WorkerOutput(
        worker=worker,
        status="needs_context",
        summary="Se necesita más información del usuario",
        content=partial_content,
        clarification_questions=serialized_questions,
        wizard_config=wizard_config,
        confidence=0.3
    )


def create_wizard_context_output(
    worker: str,
    question_set_data: Dict[str, Any],
    partial_content: str = ""
) -> WorkerOutput:
    """Build a needs_context output configured for wizard interaction."""
    questions = question_set_data.get("questions", [])
    wizard_config = {
        "wizard_mode": question_set_data.get("wizard_mode", True),
        "max_questions": question_set_data.get("max_questions", 5),
        "allow_back": question_set_data.get("allow_back", True),
        "allow_skip": question_set_data.get("allow_skip", True),
        "show_progress": question_set_data.get("show_progress", True),
        "title": question_set_data.get("title"),
        "completion_message": question_set_data.get("completion_message"),
    }

    return WorkerOutput(
        worker=worker,
        status="needs_context",
        summary="Se necesita más información del usuario (wizard)",
        content=partial_content,
        clarification_questions=questions,
        wizard_config=wizard_config,
        confidence=0.3,
        extra={"question_set": question_set_data}
    )


class WorkerOutputBuilder:
    """Factory helpers for building WorkerOutput per worker type."""

    @staticmethod
    def chat(
        content: str,
        summary: str = "",
        confidence: float = 0.8,
        status: str = "ok",
        **kwargs
    ) -> WorkerOutput:
        return WorkerOutput(
            worker="chat",
            status=status,
            summary=summary or "Conversación general",
            content=content,
            confidence=confidence,
            **kwargs
        )

    @staticmethod
    def research(
        content: str,
        evidence: List[Dict] = None,
        summary: str = "",
        confidence: float = 0.8,
        status: str = "ok",
        next_actions: List[Dict] = None,
        **kwargs
    ) -> WorkerOutput:
        evidence_items = []
        if evidence:
            for ev in evidence:
                evidence_items.append(EvidenceItem(**ev))
        
        action_items = []
        if next_actions:
            for action in next_actions:
                action_items.append(ActionItem(**action))
        
        return WorkerOutput(
            worker="research",
            status=status,
            summary=summary or f"Investigación completada con {len(evidence_items)} fuentes",
            content=content,
            evidence=evidence_items,
            next_actions=action_items,
            confidence=confidence,
            **kwargs
        )
    
    @staticmethod
    def tutor(
        content: str,
        learning_objectives: List[str] = None,
        next_steps: List[str] = None,
        resources: List[str] = None,
        summary: str = "",
        confidence: float = 0.85,
        **kwargs
    ) -> WorkerOutput:
        extra = {
            "learning_objectives": learning_objectives or [],
            "next_steps": next_steps or [],
            "resources": resources or []
        }
        
        return WorkerOutput(
            worker="tutor",
            status="ok",
            summary=summary or "Explicación educativa generada",
            content=content,
            confidence=confidence,
            extra=extra,
            **kwargs
        )
    
    @staticmethod
    def troubleshooting(
        content: str,
        problem_identified: str = "",
        root_cause: str = "",
        solution_steps: List[str] = None,
        severity: str = "medium",
        summary: str = "",
        confidence: float = 0.8,
        status: str = "ok",
        **kwargs
    ) -> WorkerOutput:
        extra = {
            "problem_identified": problem_identified,
            "root_cause": root_cause,
            "solution_steps": solution_steps or [],
            "severity": severity
        }
        
        return WorkerOutput(
            worker="troubleshooting",
            status=status,
            summary=summary or f"Diagnóstico completado - Severidad: {severity}",
            content=content,
            confidence=confidence,
            extra=extra,
            **kwargs
        )
    
    @staticmethod
    def summarizer(
        content: str,
        key_points: List[str] = None,
        messages_compressed: int = 0,
        compression_ratio: float = 0.0,
        summary: str = "",
        **kwargs
    ) -> WorkerOutput:
        extra = {
            "key_points": key_points or [],
            "messages_compressed": messages_compressed,
            "compression_ratio": compression_ratio
        }
        
        return WorkerOutput(
            worker="summarizer",
            status="ok",
            summary=summary or f"Comprimidos {messages_compressed} mensajes",
            content=content,
            confidence=1.0,
            extra=extra,
            **kwargs
        )
