"""
question_schema.py - Schema para preguntas estructuradas de Human-in-the-Loop

Las preguntas pueden ser:
1. Opción múltiple (choice) - El usuario selecciona una opción
2. Texto libre (text) - El usuario escribe la respuesta
3. Sí/No (boolean) - Pregunta binaria

Máximo 3 preguntas por interacción.
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class QuestionOption(BaseModel):
    """Una opción de respuesta para preguntas de opción múltiple"""
    id: str  # "1", "2", "3", "other"
    label: str  # "TIA Portal V17"
    description: Optional[str] = None  # Descripción adicional opcional


class ClarificationQuestion(BaseModel):
    """
    Una pregunta de clarificación estructurada.
    
    Ejemplos:
    - choice: "¿Qué versión de TIA Portal usas?" con opciones V15, V16, V17, Otra
    - text: "¿Cuál es el mensaje de error exacto?"
    - boolean: "¿El PLC funcionaba correctamente antes?"
    """
    id: str = Field(..., description="ID único de la pregunta, ej: 'q1', 'tia_version'")
    question: str = Field(..., description="Texto de la pregunta")
    type: Literal["choice", "text", "boolean"] = Field(
        default="choice",
        description="Tipo de pregunta"
    )
    options: List[QuestionOption] = Field(
        default_factory=list,
        description="Opciones para preguntas tipo 'choice'"
    )
    required: bool = Field(default=True, description="Si la pregunta es obligatoria")
    placeholder: Optional[str] = Field(
        default=None,
        description="Placeholder para preguntas tipo 'text'"
    )
    
    def to_display_text(self) -> str:
        """Convierte la pregunta a texto legible para mostrar"""
        lines = [f"**{self.question}**"]
        
        if self.type == "choice" and self.options:
            for opt in self.options:
                desc = f" - {opt.description}" if opt.description else ""
                lines.append(f"  {opt.id}) {opt.label}{desc}")
        elif self.type == "boolean":
            lines.append("  1) Sí")
            lines.append("  2) No")
        elif self.type == "text":
            lines.append(f"  _(Escribe tu respuesta)_")
        
        return "\n".join(lines)


class QuestionSet(BaseModel):
    """
    Conjunto de preguntas para una interacción de clarificación.
    Máximo 3 preguntas por set.
    """
    questions: List[ClarificationQuestion] = Field(
        ...,
        max_length=3,
        description="Lista de preguntas (máximo 3)"
    )
    context: Optional[str] = Field(
        default=None,
        description="Contexto adicional para el usuario"
    )
    worker: str = Field(..., description="Worker que solicita la clarificación")
    
    def to_display_text(self) -> str:
        """Convierte el set de preguntas a texto legible"""
        lines = []
        
        if self.context:
            lines.append(self.context)
            lines.append("")
        
        for i, q in enumerate(self.questions, 1):
            lines.append(f"**Pregunta {i}:**")
            lines.append(q.to_display_text())
            lines.append("")
        
        return "\n".join(lines)
    
    def to_dict_list(self) -> List[dict]:
        """Convierte a lista de dicts para guardar en state"""
        return [q.model_dump() for q in self.questions]


class QuestionResponse(BaseModel):
    """Respuesta a una pregunta"""
    question_id: str
    answer: str  # ID de opción seleccionada o texto libre
    answer_label: Optional[str] = None  # Label de la opción si aplica


class QuestionSetResponse(BaseModel):
    """Respuestas a un set de preguntas"""
    responses: List[QuestionResponse]
    
    def to_context_string(self) -> str:
        """Convierte las respuestas a un string para el contexto del worker"""
        lines = []
        for r in self.responses:
            if r.answer_label:
                lines.append(f"- {r.question_id}: {r.answer_label}")
            else:
                lines.append(f"- {r.question_id}: {r.answer}")
        return "\n".join(lines)


# ============================================
# HELPERS PARA CREAR PREGUNTAS COMUNES
# ============================================

def create_choice_question(
    id: str,
    question: str,
    options: List[tuple],  # [(id, label), ...] o [(id, label, description), ...]
    include_other: bool = True
) -> ClarificationQuestion:
    """
    Helper para crear preguntas de opción múltiple.
    
    Ejemplo:
        create_choice_question(
            "plc_model",
            "¿Cuál es el modelo del PLC?",
            [("1", "S7-1200"), ("2", "S7-1500"), ("3", "S7-300")]
        )
    """
    opts = []
    for opt in options:
        if len(opt) == 2:
            opts.append(QuestionOption(id=opt[0], label=opt[1]))
        else:
            opts.append(QuestionOption(id=opt[0], label=opt[1], description=opt[2]))
    
    if include_other:
        opts.append(QuestionOption(id="other", label="Otro (especificar)"))
    
    return ClarificationQuestion(
        id=id,
        question=question,
        type="choice",
        options=opts
    )


def create_text_question(
    id: str,
    question: str,
    placeholder: str = "Escribe tu respuesta..."
) -> ClarificationQuestion:
    """Helper para crear preguntas de texto libre"""
    return ClarificationQuestion(
        id=id,
        question=question,
        type="text",
        placeholder=placeholder
    )


def create_boolean_question(
    id: str,
    question: str
) -> ClarificationQuestion:
    """Helper para crear preguntas Sí/No"""
    return ClarificationQuestion(
        id=id,
        question=question,
        type="boolean",
        options=[
            QuestionOption(id="yes", label="Sí"),
            QuestionOption(id="no", label="No"),
        ]
    )


# ============================================
# PREGUNTAS PREDEFINIDAS PARA TROUBLESHOOTING
# ============================================

TROUBLESHOOTING_QUESTIONS = {
    "plc_model": create_choice_question(
        "plc_model",
        "¿Cuál es el modelo exacto del PLC?",
        [
            ("1", "S7-1200", "Serie compacta"),
            ("2", "S7-1500", "Serie avanzada"),
            ("3", "S7-300", "Serie clásica"),
            ("4", "S7-400", "Serie de alto rendimiento"),
        ]
    ),
    "tia_version": create_choice_question(
        "tia_version",
        "¿Qué versión de TIA Portal usas?",
        [
            ("1", "V15"),
            ("2", "V16"),
            ("3", "V17"),
            ("4", "V18"),
            ("5", "V19"),
        ]
    ),
    "connection_type": create_choice_question(
        "connection_type",
        "¿Cómo está conectado el PLC?",
        [
            ("1", "Cable directo PC-PLC", "Conexión punto a punto"),
            ("2", "A través de switch/router", "Red local"),
            ("3", "Red corporativa/VPN", "Conexión remota"),
        ]
    ),
    "worked_before": create_boolean_question(
        "worked_before",
        "¿El PLC funcionaba correctamente antes de este problema?"
    ),
    "error_message": create_text_question(
        "error_message",
        "¿Cuál es el mensaje de error exacto?",
        "Copia y pega el mensaje de error o descríbelo..."
    ),
    "recent_changes": create_text_question(
        "recent_changes",
        "¿Qué cambios recientes se hicieron?",
        "Actualizaciones, cambios de red, nuevo programa..."
    ),
    "led_status": create_choice_question(
        "led_status",
        "¿Cuál es el estado de los LEDs del PLC?",
        [
            ("1", "RUN verde fijo", "Operación normal"),
            ("2", "STOP amarillo/rojo", "CPU detenida"),
            ("3", "ERROR parpadeando", "Hay un fallo"),
            ("4", "Todos apagados", "Sin alimentación"),
        ]
    ),
}


def get_troubleshooting_questions(missing_info: List[str]) -> QuestionSet:
    """
    Genera un QuestionSet basado en la información faltante detectada.
    
    Args:
        missing_info: Lista de tipos de info faltante ["plc_model", "error_message", ...]
    
    Returns:
        QuestionSet con máximo 3 preguntas
    """
    questions = []
    
    # Mapeo de info faltante a preguntas predefinidas
    info_to_question = {
        "modelo": "plc_model",
        "plc": "plc_model",
        "version": "tia_version",
        "tia": "tia_version",
        "error": "error_message",
        "mensaje": "error_message",
        "conexion": "connection_type",
        "conexión": "connection_type",
        "led": "led_status",
        "cambios": "recent_changes",
    }
    
    added = set()
    for info in missing_info:
        info_lower = info.lower()
        for keyword, question_key in info_to_question.items():
            if keyword in info_lower and question_key not in added:
                if question_key in TROUBLESHOOTING_QUESTIONS:
                    questions.append(TROUBLESHOOTING_QUESTIONS[question_key])
                    added.add(question_key)
                    break
        
        if len(questions) >= 3:
            break
    
    # Si no encontró preguntas predefinidas, usar las básicas
    if not questions:
        questions = [
            TROUBLESHOOTING_QUESTIONS["plc_model"],
            TROUBLESHOOTING_QUESTIONS["error_message"],
        ]
    
    return QuestionSet(
        questions=questions[:3],
        context="Para diagnosticar tu problema necesito algunos detalles:",
        worker="troubleshooting"
    )
