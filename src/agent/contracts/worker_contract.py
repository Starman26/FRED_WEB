"""
worker_contract.py - Contrato UNIVERSAL para comunicación entre workers y supervisor

Este módulo define el schema JSON que TODOS los workers deben cumplir.
El supervisor SIEMPRE espera este formato para poder:
1. Acumular evidencia entre workers
2. Detectar cuando se necesita human-in-the-loop
3. Encadenar workers en multi-step orchestration
"""
import uuid
import json
from typing import List, Optional, Literal, Any, Dict
from pydantic import BaseModel, Field
from datetime import datetime


class EvidenceItem(BaseModel):
    """Un fragmento de evidencia recuperado (principalmente de RAG)"""
    source_id: Optional[str] = Field(default=None, description="ID del chunk/documento")
    title: str = Field(default="", description="Título del documento fuente")
    chunk: str = Field(default="", description="Contenido del fragmento")
    page: Optional[str] = Field(default=None, description="Página(s) del documento")
    score: float = Field(default=0.0, description="Score de relevancia (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata adicional")


class ActionItem(BaseModel):
    """Acción que el orchestrator debe considerar"""
    type: Literal["ask_user", "call_worker", "call_tool", "end"]
    target: Optional[str] = Field(default=None, description="Nombre del worker/tool objetivo")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Datos para la acción")
    reason: str = Field(default="", description="Razón de la acción sugerida")
    priority: int = Field(default=0, description="Prioridad (mayor = más urgente)")


class ErrorItem(BaseModel):
    """Error estructurado para debugging"""
    code: str = Field(description="Código del error (ej: 'RAG_NO_RESULTS')")
    message: str = Field(description="Mensaje legible del error")
    severity: Literal["warning", "error", "critical"] = Field(default="error")
    debug: Dict[str, Any] = Field(default_factory=dict, description="Info de debug")
    recoverable: bool = Field(default=True, description="Si el error permite continuar")


class WorkerMetadata(BaseModel):
    """Metadata común de ejecución del worker"""
    started_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    tokens_used: int = Field(default=0, description="Tokens consumidos")
    model_used: str = Field(default="", description="Modelo LLM usado")
    processing_time_ms: float = Field(default=0.0, description="Tiempo de procesamiento")
    retries: int = Field(default=0, description="Número de reintentos")


class WorkerOutput(BaseModel):
    """
    Contrato UNIVERSAL para todos los workers.
    
    Este es el formato que TODOS los workers deben retornar.
    El supervisor/orchestrator usa este contrato para:
    - Decidir el siguiente paso
    - Acumular evidencia
    - Detectar necesidad de human-in-the-loop
    - Renderizar respuesta final
    
    Campos obligatorios:
    - worker: Nombre del worker que generó el output
    - status: Estado del resultado
    - content: Contenido para el usuario
    
    Campos opcionales pero importantes:
    - summary: Resumen corto para el orchestrator
    - evidence: Evidencia recuperada (para RAG)
    - next_actions: Acciones sugeridas
    - clarification_questions: Preguntas para human-in-the-loop
    """
    
    # Identificación
    worker: Literal["chat", "research", "tutor", "troubleshooting", "summarizer"]
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Estado del resultado
    status: Literal["ok", "needs_context", "partial", "error"] = Field(
        default="ok",
        description="ok=completo, needs_context=falta info, partial=resultado incompleto, error=falló"
    )
    
    # Contenido principal
    summary: str = Field(
        default="",
        description="Resumen CORTO (1-2 oraciones) para el orchestrator"
    )
    content: str = Field(
        default="",
        description="Contenido COMPLETO para mostrar al usuario"
    )
    
    # Evidencia (principalmente para research)
    evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Fragmentos de evidencia recuperados"
    )
    
    # Acciones sugeridas (para multi-step)
    next_actions: List[ActionItem] = Field(
        default_factory=list,
        description="Acciones que el orchestrator debería considerar"
    )
    
    # Preguntas para human-in-the-loop
    clarification_questions: List[str] = Field(
        default_factory=list,
        description="Preguntas para el usuario si status=needs_context"
    )
    
    # Errores
    errors: List[ErrorItem] = Field(
        default_factory=list,
        description="Errores ocurridos durante la ejecución"
    )
    
    # Confianza y metadata
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confianza en el resultado (0-1)"
    )
    metadata: WorkerMetadata = Field(default_factory=WorkerMetadata)
    
    # Datos extra para casos específicos
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
    """Serializa un WorkerOutput a JSON string"""
    return output.model_dump_json(indent=2)


def parse_worker_output(json_str: str) -> Optional[WorkerOutput]:
    """
    Parsea un JSON string a WorkerOutput.
    Retorna None si el parsing falla.
    """
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
    """Helper para crear un output de error rápidamente"""
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
    questions: List[str],
    partial_content: str = ""
) -> WorkerOutput:
    """Helper para crear un output que necesita más contexto del usuario"""
    return WorkerOutput(
        worker=worker,
        status="needs_context",
        summary="Se necesita más información del usuario",
        content=partial_content,
        clarification_questions=questions,
        confidence=0.3
    )


# ============================================
# Builders específicos por worker (compatibilidad)
# ============================================

class WorkerOutputBuilder:
    """Factory para construir WorkerOutput de forma más ergonómica"""
    
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
        """Construye output para research worker"""
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
        """Construye output para tutor worker"""
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
        """Construye output para troubleshooting worker"""
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
        """Construye output para summarizer worker"""
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
            confidence=1.0,  # Summarizer siempre tiene alta confianza
            extra=extra,
            **kwargs
        )
