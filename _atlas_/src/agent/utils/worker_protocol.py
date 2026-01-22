"""Protocolo de comunicación JSON entre workers y supervisor"""
import json
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


# ============================================================================
# SCHEMAS PYDANTIC (validación automática)
# ============================================================================

class WorkerMetadata(BaseModel):
    """Metadata común a todos los workers"""
    tokens_used: int = Field(default=0, description="Tokens consumidos en LLM call")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confianza en la respuesta (0-1)")
    processing_time_ms: float = Field(default=0.0, description="Tiempo de procesamiento en ms")
    model_used: str = Field(default="", description="Modelo LLM usado")
    error: Optional[str] = Field(default=None, description="Error si ocurrió")


class TutorResponse(BaseModel):
    """Respuesta del Tutor Worker"""
    content: str = Field(description="Explicación/tutorial para el usuario")
    learning_objectives: List[str] = Field(default=[], description="Objetivos de aprendizaje cubiertos")
    next_steps: List[str] = Field(default=[], description="Pasos sugeridos para profundizar")
    resources: List[str] = Field(default=[], description="Recursos recomendados")
    metadata: WorkerMetadata = Field(default_factory=WorkerMetadata)
    reasoning: str = Field(default="", description="Por qué se eligió este enfoque")


class TroubleshooterResponse(BaseModel):
    """Respuesta del Troubleshooter Worker"""
    content: str = Field(description="Diagnóstico y solución del problema")
    problem_identified: str = Field(default="", description="Problema identificado")
    root_cause: str = Field(default="", description="Causa raíz")
    solution_steps: List[str] = Field(default=[], description="Pasos para resolver")
    severity: str = Field(default="medium", description="Severidad: low, medium, high, critical")
    metadata: WorkerMetadata = Field(default_factory=WorkerMetadata)
    reasoning: str = Field(default="", description="Razonamiento del diagnóstico")


class SummarizerResponse(BaseModel):
    """Respuesta del Summarizer Worker"""
    content: str = Field(description="Resumen comprimido de la conversación")
    key_points: List[str] = Field(default=[], description="Puntos clave del resumen")
    messages_compressed: int = Field(default=0, description="Cantidad de mensajes comprimidos")
    compression_ratio: float = Field(default=0.0, description="Ratio de compresión")
    metadata: WorkerMetadata = Field(default_factory=WorkerMetadata)
    reasoning: str = Field(default="", description="Estrategia de compresión")


# ============================================================================
# BUILDERS (construir respuestas JSON)
# ============================================================================

class WorkerResponseBuilder:
    """Factory para construir respuestas JSON de workers"""
    
    @staticmethod
    def tutor(
        content: str,
        learning_objectives: List[str] = None,
        next_steps: List[str] = None,
        resources: List[str] = None,
        tokens_used: int = 0,
        confidence: float = 0.8,
        model_used: str = "",
        reasoning: str = ""
    ) -> dict:
        """Construye respuesta del Tutor"""
        response = TutorResponse(
            content=content,
            learning_objectives=learning_objectives or [],
            next_steps=next_steps or [],
            resources=resources or [],
            metadata=WorkerMetadata(
                tokens_used=tokens_used,
                confidence=confidence,
                model_used=model_used
            ),
            reasoning=reasoning
        )
        return json.loads(response.model_dump_json())
    
    @staticmethod
    def troubleshooter(
        content: str,
        problem_identified: str = "",
        root_cause: str = "",
        solution_steps: List[str] = None,
        severity: str = "medium",
        tokens_used: int = 0,
        confidence: float = 0.8,
        model_used: str = "",
        reasoning: str = ""
    ) -> dict:
        """Construye respuesta del Troubleshooter"""
        response = TroubleshooterResponse(
            content=content,
            problem_identified=problem_identified,
            root_cause=root_cause,
            solution_steps=solution_steps or [],
            severity=severity,
            metadata=WorkerMetadata(
                tokens_used=tokens_used,
                confidence=confidence,
                model_used=model_used
            ),
            reasoning=reasoning
        )
        return json.loads(response.model_dump_json())
    
    @staticmethod
    def summarizer(
        content: str,
        key_points: List[str] = None,
        messages_compressed: int = 0,
        compression_ratio: float = 0.0,
        tokens_used: int = 0,
        model_used: str = "",
        reasoning: str = ""
    ) -> dict:
        """Construye respuesta del Summarizer"""
        response = SummarizerResponse(
            content=content,
            key_points=key_points or [],
            messages_compressed=messages_compressed,
            compression_ratio=compression_ratio,
            metadata=WorkerMetadata(
                tokens_used=tokens_used,
                model_used=model_used
            ),
            reasoning=reasoning
        )
        return json.loads(response.model_dump_json())


# ============================================================================
# PARSERS (parsear respuestas JSON)
# ============================================================================

class WorkerResponseParser:
    """Parser para extraer y validar respuestas JSON de workers"""
    
    @staticmethod
    def safe_parse(json_str: str, worker_type: str) -> Optional[dict]:
        """
        Parsea JSON de worker de forma segura.
        
        Args:
            json_str: string JSON del worker
            worker_type: "tutor", "troubleshooter", "summarizer"
        
        Returns:
            dict parseado o None si falla
        """
        try:
            data = json.loads(json_str)
            
            # Validar según tipo
            if worker_type == "tutor":
                response = TutorResponse(**data)
                return json.loads(response.model_dump_json())
            
            elif worker_type == "troubleshooter":
                response = TroubleshooterResponse(**data)
                return json.loads(response.model_dump_json())
            
            elif worker_type == "summarizer":
                response = SummarizerResponse(**data)
                return json.loads(response.model_dump_json())
            
            else:
                # Fallback: devolver como está
                return data
        
        except Exception as e:
            print(f"Error parseando JSON del worker: {e}")
            return None
    
    @staticmethod
    def extract_content(parsed: dict) -> str:
        """Extrae el campo 'content' de respuesta parseada"""
        return parsed.get("content", "")
    
    @staticmethod
    def extract_metadata(parsed: dict) -> dict:
        """Extrae el campo 'metadata' de respuesta parseada"""
        return parsed.get("metadata", {})
