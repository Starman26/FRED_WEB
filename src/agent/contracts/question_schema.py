"""
question_schema.py - Schema para preguntas estructuradas de Human-in-the-Loop

Las preguntas pueden ser:
1. Opción múltiple (choice) - El usuario selecciona una opción
2. Multi-selección (multi_choice) - El usuario selecciona múltiples opciones
3. Texto libre (text) - El usuario escribe la respuesta
4. Sí/No (boolean) - Pregunta binaria
5. Numérico (number) - El usuario ingresa un número
6. Confirmación (confirm) - Confirmar una acción

Características:
- Validación de respuestas (regex, min/max, rangos numéricos)
- Preguntas condicionales (mostrar según respuestas anteriores)
- Workflow tipo wizard con navegación
- Quick replies para respuestas rápidas
- Máximo configurable de preguntas por interacción
"""
from typing import List, Optional, Literal, Dict, Any, Callable, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import re


# ============================================
# ENUMS Y CONSTANTES
# ============================================

class QuestionType(str, Enum):
    """Tipos de preguntas soportados"""
    CHOICE = "choice"
    MULTI_CHOICE = "multi_choice"
    TEXT = "text"
    BOOLEAN = "boolean"
    NUMBER = "number"
    CONFIRM = "confirm"


class ValidationErrorType(str, Enum):
    """Tipos de errores de validación"""
    REQUIRED = "required"
    PATTERN = "pattern"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    MIN_VALUE = "min_value"
    MAX_VALUE = "max_value"
    INVALID_OPTION = "invalid_option"
    MIN_SELECTIONS = "min_selections"
    MAX_SELECTIONS = "max_selections"


# ============================================
# CLASES DE VALIDACIÓN
# ============================================

class ValidationRule(BaseModel):
    """Reglas de validación para respuestas"""
    pattern: Optional[str] = Field(
        default=None,
        description="Regex pattern para validar texto"
    )
    min_length: Optional[int] = Field(
        default=None,
        description="Longitud mínima para texto"
    )
    max_length: Optional[int] = Field(
        default=None,
        description="Longitud máxima para texto"
    )
    min_value: Optional[float] = Field(
        default=None,
        description="Valor mínimo para números"
    )
    max_value: Optional[float] = Field(
        default=None,
        description="Valor máximo para números"
    )
    min_selections: Optional[int] = Field(
        default=None,
        description="Mínimo de selecciones para multi_choice"
    )
    max_selections: Optional[int] = Field(
        default=None,
        description="Máximo de selecciones para multi_choice"
    )
    custom_error: Optional[str] = Field(
        default=None,
        description="Mensaje de error personalizado"
    )

    def validate_text(self, value: str) -> tuple[bool, Optional[str]]:
        """Valida una respuesta de texto"""
        if self.min_length and len(value) < self.min_length:
            return False, self.custom_error or f"Mínimo {self.min_length} caracteres"
        if self.max_length and len(value) > self.max_length:
            return False, self.custom_error or f"Máximo {self.max_length} caracteres"
        if self.pattern:
            if not re.match(self.pattern, value):
                return False, self.custom_error or "Formato inválido"
        return True, None

    def validate_number(self, value: float) -> tuple[bool, Optional[str]]:
        """Valida una respuesta numérica"""
        if self.min_value is not None and value < self.min_value:
            return False, self.custom_error or f"Valor mínimo: {self.min_value}"
        if self.max_value is not None and value > self.max_value:
            return False, self.custom_error or f"Valor máximo: {self.max_value}"
        return True, None

    def validate_selections(self, count: int) -> tuple[bool, Optional[str]]:
        """Valida número de selecciones"""
        if self.min_selections and count < self.min_selections:
            return False, self.custom_error or f"Selecciona al menos {self.min_selections}"
        if self.max_selections and count > self.max_selections:
            return False, self.custom_error or f"Máximo {self.max_selections} selecciones"
        return True, None


class ValidationResult(BaseModel):
    """Resultado de validación"""
    is_valid: bool
    error_type: Optional[ValidationErrorType] = None
    error_message: Optional[str] = None
    sanitized_value: Optional[Any] = None


# ============================================
# CONDICIONES PARA PREGUNTAS
# ============================================

class QuestionCondition(BaseModel):
    """
    Condición para mostrar una pregunta basada en respuestas anteriores.

    Ejemplos:
    - Mostrar "¿Cuál es el error?" solo si "¿Hay algún error?" = Sí
    - Mostrar "Versión TIA" solo si eligió "TIA Portal" como herramienta
    """
    depends_on: str = Field(..., description="ID de la pregunta de la que depende")
    operator: Literal["equals", "not_equals", "contains", "in", "not_in"] = Field(
        default="equals",
        description="Operador de comparación"
    )
    value: Union[str, List[str]] = Field(..., description="Valor(es) para comparar")

    def evaluate(self, answer: Any) -> bool:
        """Evalúa si la condición se cumple"""
        if answer is None:
            return False

        answer_str = str(answer).lower() if not isinstance(answer, list) else answer
        compare_value = self.value.lower() if isinstance(self.value, str) else self.value

        if self.operator == "equals":
            return answer_str == compare_value
        elif self.operator == "not_equals":
            return answer_str != compare_value
        elif self.operator == "contains":
            return compare_value in answer_str
        elif self.operator == "in":
            return answer_str in [v.lower() for v in compare_value] if isinstance(compare_value, list) else False
        elif self.operator == "not_in":
            return answer_str not in [v.lower() for v in compare_value] if isinstance(compare_value, list) else True
        return False


# ============================================
# QUICK REPLIES
# ============================================

class QuickReply(BaseModel):
    """Respuesta rápida predefinida (botón)"""
    id: str
    label: str
    value: str  # Valor que se envía como respuesta
    icon: Optional[str] = None  # Emoji o icono
    style: Literal["default", "primary", "success", "warning", "danger"] = "default"


# ============================================
# OPCIONES Y PREGUNTAS
# ============================================

class QuestionOption(BaseModel):
    """Una opción de respuesta para preguntas de opción múltiple"""
    id: str  # "1", "2", "3", "other"
    label: str  # "TIA Portal V17"
    description: Optional[str] = None  # Descripción adicional opcional
    icon: Optional[str] = None  # Emoji o icono
    disabled: bool = False  # Si la opción está deshabilitada


class ClarificationQuestion(BaseModel):
    """
    Una pregunta de clarificación estructurada con soporte completo de validación.

    Ejemplos:
    - choice: "¿Qué versión de TIA Portal usas?" con opciones V15, V16, V17, Otra
    - multi_choice: "¿Qué módulos tiene el PLC?" (selección múltiple)
    - text: "¿Cuál es el mensaje de error exacto?"
    - boolean: "¿El PLC funcionaba correctamente antes?"
    - number: "¿Cuántos módulos tiene?"
    - confirm: "¿Confirmas que quieres ejecutar esta reparación?"
    """
    id: str = Field(..., description="ID único de la pregunta, ej: 'q1', 'tia_version'")
    question: str = Field(..., description="Texto de la pregunta")
    type: QuestionType = Field(
        default=QuestionType.CHOICE,
        description="Tipo de pregunta"
    )
    options: List[QuestionOption] = Field(
        default_factory=list,
        description="Opciones para preguntas tipo 'choice' o 'multi_choice'"
    )
    required: bool = Field(default=True, description="Si la pregunta es obligatoria")
    placeholder: Optional[str] = Field(
        default=None,
        description="Placeholder para preguntas tipo 'text' o 'number'"
    )

    # Nuevos campos para validación y condiciones
    validation: Optional[ValidationRule] = Field(
        default=None,
        description="Reglas de validación"
    )
    condition: Optional[QuestionCondition] = Field(
        default=None,
        description="Condición para mostrar la pregunta"
    )
    quick_replies: List[QuickReply] = Field(
        default_factory=list,
        description="Respuestas rápidas predefinidas"
    )
    help_text: Optional[str] = Field(
        default=None,
        description="Texto de ayuda adicional"
    )
    default_value: Optional[str] = Field(
        default=None,
        description="Valor por defecto"
    )
    skip_label: Optional[str] = Field(
        default=None,
        description="Etiqueta del botón para saltar (si no es requerida)"
    )

    # Metadatos para el wizard
    group: Optional[str] = Field(
        default=None,
        description="Grupo/sección de la pregunta"
    )
    order: int = Field(
        default=0,
        description="Orden de la pregunta en el wizard"
    )

    def should_show(self, previous_answers: Dict[str, Any]) -> bool:
        """Determina si la pregunta debe mostrarse según las respuestas anteriores"""
        if not self.condition:
            return True
        return self.condition.evaluate(previous_answers.get(self.condition.depends_on))

    def validate_answer(self, answer: Any) -> ValidationResult:
        """Valida la respuesta dada"""
        # Verificar requerido
        if self.required and (answer is None or answer == ""):
            return ValidationResult(
                is_valid=False,
                error_type=ValidationErrorType.REQUIRED,
                error_message="Esta pregunta es obligatoria"
            )

        # Si no hay respuesta y no es requerida, es válida
        if answer is None or answer == "":
            return ValidationResult(is_valid=True, sanitized_value=None)

        # Validar según tipo
        if self.type == QuestionType.CHOICE:
            valid_ids = [opt.id for opt in self.options]
            if str(answer) not in valid_ids:
                return ValidationResult(
                    is_valid=False,
                    error_type=ValidationErrorType.INVALID_OPTION,
                    error_message=f"Opción inválida. Opciones válidas: {', '.join(valid_ids)}"
                )
            return ValidationResult(is_valid=True, sanitized_value=str(answer))

        elif self.type == QuestionType.MULTI_CHOICE:
            selections = answer if isinstance(answer, list) else [answer]
            valid_ids = [opt.id for opt in self.options]
            for sel in selections:
                if str(sel) not in valid_ids:
                    return ValidationResult(
                        is_valid=False,
                        error_type=ValidationErrorType.INVALID_OPTION,
                        error_message=f"Opción inválida: {sel}"
                    )
            if self.validation:
                is_valid, error = self.validation.validate_selections(len(selections))
                if not is_valid:
                    return ValidationResult(
                        is_valid=False,
                        error_type=ValidationErrorType.MIN_SELECTIONS,
                        error_message=error
                    )
            return ValidationResult(is_valid=True, sanitized_value=selections)

        elif self.type == QuestionType.TEXT:
            if self.validation:
                is_valid, error = self.validation.validate_text(str(answer))
                if not is_valid:
                    return ValidationResult(
                        is_valid=False,
                        error_type=ValidationErrorType.PATTERN,
                        error_message=error
                    )
            return ValidationResult(is_valid=True, sanitized_value=str(answer))

        elif self.type == QuestionType.NUMBER:
            try:
                num_value = float(answer)
                if self.validation:
                    is_valid, error = self.validation.validate_number(num_value)
                    if not is_valid:
                        return ValidationResult(
                            is_valid=False,
                            error_type=ValidationErrorType.MIN_VALUE,
                            error_message=error
                        )
                return ValidationResult(is_valid=True, sanitized_value=num_value)
            except ValueError:
                return ValidationResult(
                    is_valid=False,
                    error_type=ValidationErrorType.PATTERN,
                    error_message="Ingresa un número válido"
                )

        elif self.type == QuestionType.BOOLEAN:
            bool_value = str(answer).lower() in ["yes", "sí", "si", "true", "1", "y", "s"]
            return ValidationResult(is_valid=True, sanitized_value=bool_value)

        elif self.type == QuestionType.CONFIRM:
            confirmed = str(answer).lower() in ["yes", "sí", "si", "confirm", "confirmar", "1", "y", "s"]
            return ValidationResult(is_valid=True, sanitized_value=confirmed)

        return ValidationResult(is_valid=True, sanitized_value=answer)

    def to_display_text(self, step: Optional[int] = None, total: Optional[int] = None) -> str:
        """Convierte la pregunta a texto legible para mostrar"""
        lines = []

        # Indicador de progreso
        if step is not None and total is not None:
            lines.append(f"📍 Paso {step} de {total}")
            lines.append("")

        # Indicador de obligatoria
        required_mark = " *" if self.required else ""
        lines.append(f"**{self.question}**{required_mark}")

        # Texto de ayuda
        if self.help_text:
            lines.append(f"  _💡 {self.help_text}_")

        # Opciones según tipo
        if self.type in [QuestionType.CHOICE, QuestionType.MULTI_CHOICE] and self.options:
            if self.type == QuestionType.MULTI_CHOICE:
                lines.append("  _(Puedes seleccionar múltiples opciones)_")
            for opt in self.options:
                if not opt.disabled:
                    icon = f"{opt.icon} " if opt.icon else ""
                    desc = f" - {opt.description}" if opt.description else ""
                    lines.append(f"  {opt.id}) {icon}{opt.label}{desc}")
        elif self.type == QuestionType.BOOLEAN:
            lines.append("  1) ✅ Sí")
            lines.append("  2) ❌ No")
        elif self.type == QuestionType.TEXT:
            placeholder = self.placeholder or "Escribe tu respuesta..."
            lines.append(f"  📝 _{placeholder}_")
        elif self.type == QuestionType.NUMBER:
            placeholder = self.placeholder or "Ingresa un número"
            lines.append(f"  🔢 _{placeholder}_")
            if self.validation:
                if self.validation.min_value is not None and self.validation.max_value is not None:
                    lines.append(f"  _(Rango: {self.validation.min_value} - {self.validation.max_value})_")
        elif self.type == QuestionType.CONFIRM:
            lines.append("  ✅ Escribe 'sí' o 'confirmar' para continuar")
            lines.append("  ❌ Escribe 'no' o 'cancelar' para rechazar")

        # Quick replies
        if self.quick_replies:
            lines.append("")
            lines.append("  Respuestas rápidas:")
            for qr in self.quick_replies:
                icon = f"{qr.icon} " if qr.icon else ""
                lines.append(f"  [{icon}{qr.label}]")

        # Opción de saltar
        if not self.required and self.skip_label:
            lines.append("")
            lines.append(f"  ⏭️ {self.skip_label}")

        return "\n".join(lines)


class QuestionSet(BaseModel):
    """
    Conjunto de preguntas para una interacción de clarificación.
    Soporta modo wizard para preguntas secuenciales.
    """
    questions: List[ClarificationQuestion] = Field(
        ...,
        description="Lista de preguntas"
    )
    context: Optional[str] = Field(
        default=None,
        description="Contexto adicional para el usuario"
    )
    worker: str = Field(..., description="Worker que solicita la clarificación")

    # Configuración del wizard
    max_questions: int = Field(
        default=5,
        description="Máximo de preguntas a mostrar (configurable)"
    )
    wizard_mode: bool = Field(
        default=False,
        description="Si True, muestra preguntas una por una"
    )
    allow_back: bool = Field(
        default=True,
        description="Permite volver a preguntas anteriores"
    )
    allow_skip: bool = Field(
        default=False,
        description="Permite saltar preguntas opcionales"
    )
    show_progress: bool = Field(
        default=True,
        description="Muestra indicador de progreso"
    )
    title: Optional[str] = Field(
        default=None,
        description="Título del wizard"
    )
    completion_message: Optional[str] = Field(
        default=None,
        description="Mensaje al completar todas las preguntas"
    )

    def get_visible_questions(self, answers: Dict[str, Any] = None) -> List[ClarificationQuestion]:
        """Obtiene las preguntas visibles basado en condiciones y respuestas anteriores"""
        answers = answers or {}
        visible = []
        for q in self.questions:
            if q.should_show(answers):
                visible.append(q)
            if len(visible) >= self.max_questions:
                break
        return visible

    def to_display_text(self, current_step: int = 0, answers: Dict[str, Any] = None) -> str:
        """Convierte el set de preguntas a texto legible"""
        lines = []
        visible_questions = self.get_visible_questions(answers)
        total = len(visible_questions)

        # Título del wizard
        if self.title:
            lines.append(f"## {self.title}")
            lines.append("")

        if self.context:
            lines.append(self.context)
            lines.append("")

        # Barra de progreso
        if self.show_progress and self.wizard_mode and total > 1:
            progress_bar = self._generate_progress_bar(current_step + 1, total)
            lines.append(progress_bar)
            lines.append("")

        if self.wizard_mode:
            # Mostrar solo la pregunta actual
            if current_step < len(visible_questions):
                q = visible_questions[current_step]
                lines.append(q.to_display_text(current_step + 1, total))
                lines.append("")

                # Navegación
                nav_options = []
                if self.allow_back and current_step > 0:
                    nav_options.append("⬅️ 'atrás' para volver")
                if self.allow_skip and not q.required:
                    nav_options.append("⏭️ 'saltar' para omitir")

                if nav_options:
                    lines.append(f"_({' | '.join(nav_options)})_")
        else:
            # Mostrar todas las preguntas
            for i, q in enumerate(visible_questions, 1):
                lines.append(f"**Pregunta {i}:**")
                lines.append(q.to_display_text())
                lines.append("")

        return "\n".join(lines)

    def _generate_progress_bar(self, current: int, total: int) -> str:
        """Genera una barra de progreso visual"""
        filled = "●" * current
        empty = "○" * (total - current)
        percentage = int((current / total) * 100)
        return f"[{filled}{empty}] {current}/{total} ({percentage}%)"

    def to_dict_list(self) -> List[dict]:
        """Convierte a lista de dicts para guardar en state"""
        return [q.model_dump() for q in self.questions]

    def to_wizard_state(self) -> "WizardState":
        """Convierte a estado de wizard para el human_input node"""
        return WizardState(
            question_set=self,
            current_step=0,
            answers={},
            validation_errors={},
            is_complete=False,
            navigation_history=[]
        )


class QuestionResponse(BaseModel):
    """Respuesta a una pregunta"""
    question_id: str
    answer: Any  # ID de opción seleccionada, texto, número, lista, etc.
    answer_label: Optional[str] = None  # Label de la opción si aplica
    is_skipped: bool = False  # Si la pregunta fue saltada
    validation_result: Optional[ValidationResult] = None


class QuestionSetResponse(BaseModel):
    """Respuestas a un set de preguntas"""
    responses: List[QuestionResponse]
    completed: bool = True
    cancelled: bool = False

    def to_context_string(self) -> str:
        """Convierte las respuestas a un string para el contexto del worker"""
        lines = []
        for r in self.responses:
            if r.is_skipped:
                lines.append(f"- {r.question_id}: (omitido)")
            elif r.answer_label:
                lines.append(f"- {r.question_id}: {r.answer_label}")
            else:
                lines.append(f"- {r.question_id}: {r.answer}")
        return "\n".join(lines)

    def get_answer(self, question_id: str) -> Optional[Any]:
        """Obtiene la respuesta a una pregunta específica"""
        for r in self.responses:
            if r.question_id == question_id:
                return r.answer
        return None


# ============================================
# WIZARD STATE - ESTADO DEL FLUJO INTERACTIVO
# ============================================

class WizardState(BaseModel):
    """
    Estado del wizard para gestionar el flujo interactivo de preguntas.
    Se almacena en pending_context para persistir entre interrupciones.
    """
    question_set: QuestionSet
    current_step: int = 0
    answers: Dict[str, Any] = Field(default_factory=dict)
    validation_errors: Dict[str, str] = Field(default_factory=dict)
    is_complete: bool = False
    is_cancelled: bool = False
    navigation_history: List[int] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def get_current_question(self) -> Optional[ClarificationQuestion]:
        """Obtiene la pregunta actual"""
        visible = self.question_set.get_visible_questions(self.answers)
        if self.current_step < len(visible):
            return visible[self.current_step]
        return None

    def get_total_questions(self) -> int:
        """Obtiene el total de preguntas visibles"""
        return len(self.question_set.get_visible_questions(self.answers))

    def process_input(self, user_input: str) -> "WizardAction":
        """
        Procesa la entrada del usuario y retorna la acción a tomar.
        """
        input_lower = user_input.strip().lower()

        # Comandos de navegación
        if input_lower in ["atrás", "atras", "back", "anterior", "<"]:
            return WizardAction(action_type="back")
        if input_lower in ["saltar", "skip", "omitir", "siguiente", ">"]:
            return WizardAction(action_type="skip")
        if input_lower in ["cancelar", "cancel", "exit", "salir"]:
            return WizardAction(action_type="cancel")
        if input_lower in ["reiniciar", "restart", "reset"]:
            return WizardAction(action_type="restart")

        # Es una respuesta normal
        return WizardAction(action_type="answer", value=user_input)

    def submit_answer(self, answer: Any) -> tuple[bool, Optional[str]]:
        """
        Envía una respuesta para la pregunta actual.
        Returns: (success, error_message)
        """
        question = self.get_current_question()
        if not question:
            return False, "No hay pregunta actual"

        # Validar la respuesta
        result = question.validate_answer(answer)
        if not result.is_valid:
            self.validation_errors[question.id] = result.error_message
            return False, result.error_message

        # Guardar respuesta
        self.answers[question.id] = result.sanitized_value
        self.validation_errors.pop(question.id, None)
        self.navigation_history.append(self.current_step)

        # Avanzar al siguiente paso
        self.current_step += 1

        # Verificar si completamos
        if self.current_step >= self.get_total_questions():
            self.is_complete = True

        return True, None

    def go_back(self) -> bool:
        """Retrocede a la pregunta anterior. Returns: success"""
        if not self.question_set.allow_back or self.current_step <= 0:
            return False

        self.current_step -= 1
        if self.navigation_history:
            self.navigation_history.pop()
        return True

    def skip_current(self) -> bool:
        """Salta la pregunta actual si es opcional. Returns: success"""
        question = self.get_current_question()
        if not question or question.required:
            return False

        if not self.question_set.allow_skip:
            return False

        self.answers[question.id] = None
        self.navigation_history.append(self.current_step)
        self.current_step += 1

        if self.current_step >= self.get_total_questions():
            self.is_complete = True

        return True

    def cancel(self):
        """Cancela el wizard"""
        self.is_cancelled = True

    def restart(self):
        """Reinicia el wizard"""
        self.current_step = 0
        self.answers = {}
        self.validation_errors = {}
        self.navigation_history = []
        self.is_complete = False
        self.is_cancelled = False

    def to_response(self) -> QuestionSetResponse:
        """Convierte el estado final a QuestionSetResponse"""
        responses = []
        for q in self.question_set.questions:
            if q.id in self.answers:
                answer = self.answers[q.id]
                label = None
                is_skipped = answer is None

                # Obtener label si es choice
                if q.type in [QuestionType.CHOICE, QuestionType.MULTI_CHOICE] and answer:
                    for opt in q.options:
                        if opt.id == str(answer):
                            label = opt.label
                            break

                responses.append(QuestionResponse(
                    question_id=q.id,
                    answer=answer,
                    answer_label=label,
                    is_skipped=is_skipped
                ))

        return QuestionSetResponse(
            responses=responses,
            completed=self.is_complete,
            cancelled=self.is_cancelled
        )

    def get_display_prompt(self) -> str:
        """Genera el prompt para mostrar al usuario"""
        if self.is_complete:
            msg = self.question_set.completion_message or "✅ ¡Gracias! He recibido toda la información."
            return msg

        if self.is_cancelled:
            return "❌ Proceso cancelado."

        return self.question_set.to_display_text(self.current_step, self.answers)


class WizardAction(BaseModel):
    """Acción resultante de procesar input del usuario en el wizard"""
    action_type: Literal["answer", "back", "skip", "cancel", "restart"]
    value: Optional[str] = None


# ============================================
# HELPERS PARA CREAR PREGUNTAS COMUNES
# ============================================

def create_choice_question(
    id: str,
    question: str,
    options: List[tuple],  # [(id, label), ...] o [(id, label, description), ...]
    include_other: bool = True,
    required: bool = True,
    help_text: Optional[str] = None,
    condition: Optional[QuestionCondition] = None,
    quick_replies: Optional[List[QuickReply]] = None
) -> ClarificationQuestion:
    """
    Helper para crear preguntas de opción múltiple.

    Ejemplo:
        create_choice_question(
            "plc_model",
            "¿Cuál es el modelo del PLC?",
            [("1", "S7-1200"), ("2", "S7-1500"), ("3", "S7-300")],
            help_text="Puedes encontrarlo en la etiqueta del equipo"
        )
    """
    opts = []
    for opt in options:
        if len(opt) == 2:
            opts.append(QuestionOption(id=opt[0], label=opt[1]))
        elif len(opt) == 3:
            opts.append(QuestionOption(id=opt[0], label=opt[1], description=opt[2]))
        elif len(opt) == 4:
            opts.append(QuestionOption(id=opt[0], label=opt[1], description=opt[2], icon=opt[3]))

    if include_other:
        opts.append(QuestionOption(id="other", label="Otro (especificar)", icon="✏️"))

    return ClarificationQuestion(
        id=id,
        question=question,
        type=QuestionType.CHOICE,
        options=opts,
        required=required,
        help_text=help_text,
        condition=condition,
        quick_replies=quick_replies or []
    )


def create_multi_choice_question(
    id: str,
    question: str,
    options: List[tuple],
    min_selections: int = 1,
    max_selections: Optional[int] = None,
    required: bool = True,
    help_text: Optional[str] = None,
    condition: Optional[QuestionCondition] = None
) -> ClarificationQuestion:
    """
    Helper para crear preguntas de selección múltiple.

    Ejemplo:
        create_multi_choice_question(
            "modules",
            "¿Qué módulos tiene el PLC?",
            [("1", "DI"), ("2", "DO"), ("3", "AI"), ("4", "AO")],
            min_selections=1,
            max_selections=4
        )
    """
    opts = []
    for opt in options:
        if len(opt) == 2:
            opts.append(QuestionOption(id=opt[0], label=opt[1]))
        elif len(opt) == 3:
            opts.append(QuestionOption(id=opt[0], label=opt[1], description=opt[2]))

    validation = ValidationRule(
        min_selections=min_selections,
        max_selections=max_selections
    )

    return ClarificationQuestion(
        id=id,
        question=question,
        type=QuestionType.MULTI_CHOICE,
        options=opts,
        required=required,
        validation=validation,
        help_text=help_text,
        condition=condition
    )


def create_text_question(
    id: str,
    question: str,
    placeholder: str = "Escribe tu respuesta...",
    required: bool = True,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    pattern: Optional[str] = None,
    pattern_error: Optional[str] = None,
    help_text: Optional[str] = None,
    condition: Optional[QuestionCondition] = None,
    quick_replies: Optional[List[QuickReply]] = None
) -> ClarificationQuestion:
    """
    Helper para crear preguntas de texto libre con validación.

    Ejemplo:
        create_text_question(
            "error_message",
            "¿Cuál es el mensaje de error exacto?",
            min_length=5,
            max_length=500,
            help_text="Copia el mensaje completo si es posible"
        )
    """
    validation = None
    if min_length or max_length or pattern:
        validation = ValidationRule(
            min_length=min_length,
            max_length=max_length,
            pattern=pattern,
            custom_error=pattern_error
        )

    return ClarificationQuestion(
        id=id,
        question=question,
        type=QuestionType.TEXT,
        placeholder=placeholder,
        required=required,
        validation=validation,
        help_text=help_text,
        condition=condition,
        quick_replies=quick_replies or []
    )


def create_number_question(
    id: str,
    question: str,
    placeholder: str = "Ingresa un número",
    required: bool = True,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    help_text: Optional[str] = None,
    condition: Optional[QuestionCondition] = None
) -> ClarificationQuestion:
    """
    Helper para crear preguntas numéricas con validación de rango.

    Ejemplo:
        create_number_question(
            "module_count",
            "¿Cuántos módulos de E/S tiene el rack?",
            min_value=1,
            max_value=32,
            help_text="Cuenta todos los módulos incluyendo el CPU"
        )
    """
    validation = None
    if min_value is not None or max_value is not None:
        validation = ValidationRule(
            min_value=min_value,
            max_value=max_value
        )

    return ClarificationQuestion(
        id=id,
        question=question,
        type=QuestionType.NUMBER,
        placeholder=placeholder,
        required=required,
        validation=validation,
        help_text=help_text,
        condition=condition
    )


def create_boolean_question(
    id: str,
    question: str,
    required: bool = True,
    help_text: Optional[str] = None,
    condition: Optional[QuestionCondition] = None
) -> ClarificationQuestion:
    """Helper para crear preguntas Sí/No"""
    return ClarificationQuestion(
        id=id,
        question=question,
        type=QuestionType.BOOLEAN,
        options=[
            QuestionOption(id="yes", label="Sí", icon="✅"),
            QuestionOption(id="no", label="No", icon="❌"),
        ],
        required=required,
        help_text=help_text,
        condition=condition
    )


def create_confirm_question(
    id: str,
    question: str,
    help_text: Optional[str] = None,
    condition: Optional[QuestionCondition] = None
) -> ClarificationQuestion:
    """
    Helper para crear preguntas de confirmación (para acciones importantes).

    Ejemplo:
        create_confirm_question(
            "confirm_repair",
            "¿Confirmas que quieres ejecutar esta reparación en el PLC?",
            help_text="Esta acción modificará el estado del equipo"
        )
    """
    return ClarificationQuestion(
        id=id,
        question=question,
        type=QuestionType.CONFIRM,
        required=True,
        help_text=help_text,
        condition=condition
    )


def create_conditional_question(
    id: str,
    question: str,
    question_type: QuestionType,
    depends_on: str,
    show_when_equals: Union[str, List[str]],
    options: Optional[List[tuple]] = None,
    **kwargs
) -> ClarificationQuestion:
    """
    Helper para crear preguntas condicionales.

    Ejemplo:
        create_conditional_question(
            "tia_version",
            "¿Qué versión de TIA Portal usas?",
            QuestionType.CHOICE,
            depends_on="uses_tia",
            show_when_equals="yes",
            options=[("1", "V17"), ("2", "V18"), ("3", "V19")]
        )
    """
    condition = QuestionCondition(
        depends_on=depends_on,
        operator="in" if isinstance(show_when_equals, list) else "equals",
        value=show_when_equals
    )

    opts = []
    if options:
        for opt in options:
            if len(opt) == 2:
                opts.append(QuestionOption(id=opt[0], label=opt[1]))
            else:
                opts.append(QuestionOption(id=opt[0], label=opt[1], description=opt[2]))

    return ClarificationQuestion(
        id=id,
        question=question,
        type=question_type,
        options=opts,
        condition=condition,
        **kwargs
    )


# ============================================
# HELPERS PARA CREAR WIZARDS
# ============================================

def create_wizard(
    worker: str,
    questions: List[ClarificationQuestion],
    title: Optional[str] = None,
    context: Optional[str] = None,
    max_questions: int = 10,
    allow_back: bool = True,
    allow_skip: bool = True,
    completion_message: Optional[str] = None
) -> QuestionSet:
    """
    Crea un QuestionSet configurado como wizard interactivo.

    Ejemplo:
        wizard = create_wizard(
            worker="troubleshooting",
            title="🔧 Diagnóstico de PLC",
            questions=[
                create_choice_question("plc_model", "¿Cuál es el modelo?", ...),
                create_boolean_question("has_error", "¿Hay algún error?"),
                create_conditional_question("error_msg", "¿Cuál es el error?", ...),
            ],
            context="Vamos a diagnosticar el problema paso a paso",
            completion_message="✅ ¡Listo! Ahora puedo analizar tu problema."
        )
    """
    # Asignar orden a las preguntas
    for i, q in enumerate(questions):
        q.order = i

    return QuestionSet(
        questions=questions,
        worker=worker,
        title=title,
        context=context,
        max_questions=max_questions,
        wizard_mode=True,
        allow_back=allow_back,
        allow_skip=allow_skip,
        show_progress=True,
        completion_message=completion_message
    )


def create_quick_form(
    worker: str,
    questions: List[ClarificationQuestion],
    context: Optional[str] = None,
    max_questions: int = 3
) -> QuestionSet:
    """
    Crea un QuestionSet para mostrar todas las preguntas a la vez (formulario rápido).

    Ejemplo:
        form = create_quick_form(
            worker="troubleshooting",
            questions=[
                create_text_question("error", "¿Cuál es el error?"),
                create_choice_question("urgency", "¿Qué tan urgente es?", ...),
            ],
            context="Por favor responde estas preguntas:"
        )
    """
    return QuestionSet(
        questions=questions[:max_questions],
        worker=worker,
        context=context,
        max_questions=max_questions,
        wizard_mode=False,
        show_progress=False
    )


# ============================================
# PREGUNTAS PREDEFINIDAS PARA TROUBLESHOOTING
# ============================================

TROUBLESHOOTING_QUESTIONS = {
    "plc_model": create_choice_question(
        "plc_model",
        "¿Cuál es el modelo exacto del PLC?",
        [
            ("1", "S7-1200", "Serie compacta", "🔹"),
            ("2", "S7-1500", "Serie avanzada", "🔷"),
            ("3", "S7-300", "Serie clásica", "⬜"),
            ("4", "S7-400", "Serie de alto rendimiento", "🔶"),
        ],
        help_text="Lo puedes encontrar en la etiqueta frontal del equipo"
    ),
    "tia_version": create_choice_question(
        "tia_version",
        "¿Qué versión de TIA Portal usas?",
        [
            ("1", "V15", None, "📦"),
            ("2", "V16", None, "📦"),
            ("3", "V17", None, "📦"),
            ("4", "V18", None, "📦"),
            ("5", "V19", "Última versión", "✨"),
        ],
        help_text="Menú Ayuda → Acerca de TIA Portal"
    ),
    "connection_type": create_choice_question(
        "connection_type",
        "¿Cómo está conectado el PLC?",
        [
            ("1", "Cable directo PC-PLC", "Conexión punto a punto", "🔌"),
            ("2", "A través de switch/router", "Red local", "🌐"),
            ("3", "Red corporativa/VPN", "Conexión remota", "🏢"),
        ],
        help_text="Esto ayuda a identificar problemas de comunicación"
    ),
    "worked_before": create_boolean_question(
        "worked_before",
        "¿El PLC funcionaba correctamente antes de este problema?",
        help_text="Nos ayuda a saber si es un problema nuevo o recurrente"
    ),
    "error_message": create_text_question(
        "error_message",
        "¿Cuál es el mensaje de error exacto?",
        placeholder="Copia y pega el mensaje de error o descríbelo...",
        min_length=3,
        max_length=1000,
        help_text="Entre más detallado, mejor podemos ayudarte",
        quick_replies=[
            QuickReply(id="no_error", label="No hay mensaje", value="No hay mensaje de error visible", icon="🔇"),
            QuickReply(id="cant_see", label="No lo puedo ver", value="No tengo acceso al mensaje ahora", icon="👁️"),
        ]
    ),
    "recent_changes": create_text_question(
        "recent_changes",
        "¿Qué cambios recientes se hicieron?",
        placeholder="Actualizaciones, cambios de red, nuevo programa...",
        required=False,
        help_text="Cualquier cambio en las últimas horas/días puede ser relevante",
        quick_replies=[
            QuickReply(id="no_changes", label="Ningún cambio", value="No se han realizado cambios recientes", icon="✖️"),
            QuickReply(id="not_sure", label="No estoy seguro", value="No estoy seguro de qué cambios se hicieron", icon="❓"),
        ]
    ),
    "led_status": create_choice_question(
        "led_status",
        "¿Cuál es el estado de los LEDs del PLC?",
        [
            ("1", "RUN verde fijo", "Operación normal", "🟢"),
            ("2", "STOP amarillo/rojo", "CPU detenida", "🟡"),
            ("3", "ERROR parpadeando", "Hay un fallo", "🔴"),
            ("4", "Todos apagados", "Sin alimentación", "⚫"),
        ],
        help_text="Los LEDs del frente indican el estado del equipo"
    ),
    "urgency": create_choice_question(
        "urgency",
        "¿Qué tan urgente es resolver este problema?",
        [
            ("1", "Crítico - Producción parada", "Necesito solución inmediata", "🚨"),
            ("2", "Alto - Afecta operación", "Impacta significativamente", "⚠️"),
            ("3", "Medio - Funciona parcialmente", "Puedo continuar con limitaciones", "📊"),
            ("4", "Bajo - Preventivo/mejora", "No hay impacto inmediato", "📝"),
        ],
        required=False,
        help_text="Nos ayuda a priorizar la atención"
    ),
}


def get_troubleshooting_questions(
    missing_info: List[str],
    wizard_mode: bool = True,
    max_questions: int = 5
) -> QuestionSet:
    """
    Genera un QuestionSet basado en la información faltante detectada.

    Args:
        missing_info: Lista de tipos de info faltante ["plc_model", "error_message", ...]
        wizard_mode: Si True, muestra preguntas una por una
        max_questions: Máximo de preguntas a incluir

    Returns:
        QuestionSet configurado como wizard o formulario
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
        "urgencia": "urgency",
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

        if len(questions) >= max_questions:
            break

    # Si no encontró preguntas predefinidas, usar las básicas
    if not questions:
        questions = [
            TROUBLESHOOTING_QUESTIONS["plc_model"],
            TROUBLESHOOTING_QUESTIONS["error_message"],
        ]

    if wizard_mode:
        return create_wizard(
            worker="troubleshooting",
            title="🔧 Diagnóstico de problema",
            questions=questions[:max_questions],
            context="Vamos a identificar el problema paso a paso. Puedes escribir 'atrás' para volver a la pregunta anterior.",
            max_questions=max_questions,
            allow_back=True,
            allow_skip=True,
            completion_message="✅ ¡Gracias! Ya tengo la información necesaria para analizar tu problema."
        )
    else:
        return create_quick_form(
            worker="troubleshooting",
            questions=questions[:max_questions],
            context="Para diagnosticar tu problema necesito algunos detalles:",
            max_questions=max_questions
        )


# ============================================
# WIZARDS PREDEFINIDOS
# ============================================

def get_verification_wizard() -> QuestionSet:
    """Wizard para verificación de identidad del usuario"""
    return create_wizard(
        worker="verify_info",
        title="🔐 Verificación de identidad",
        questions=[
            create_choice_question(
                "id_type",
                "¿Cómo prefieres identificarte?",
                [
                    ("1", "Correo electrónico", None, "📧"),
                    ("2", "Número de teléfono", None, "📱"),
                    ("3", "ID de cliente", None, "🆔"),
                ],
                include_other=False
            ),
            create_conditional_question(
                "email",
                "¿Cuál es tu correo electrónico?",
                QuestionType.TEXT,
                depends_on="id_type",
                show_when_equals="1",
                placeholder="ejemplo@empresa.com"
            ),
            create_conditional_question(
                "phone",
                "¿Cuál es tu número de teléfono?",
                QuestionType.TEXT,
                depends_on="id_type",
                show_when_equals="2",
                placeholder="+52 555 123 4567"
            ),
            create_conditional_question(
                "customer_id",
                "¿Cuál es tu ID de cliente?",
                QuestionType.TEXT,
                depends_on="id_type",
                show_when_equals="3",
                placeholder="CLI-XXXXX"
            ),
        ],
        context="Para acceder a información de tu cuenta, necesito verificar tu identidad.",
        completion_message="✅ Verificación completada. Buscando tu información..."
    )


def get_repair_confirmation_wizard(repair_description: str) -> QuestionSet:
    """Wizard para confirmar una acción de reparación"""
    return create_wizard(
        worker="troubleshooting",
        title="⚠️ Confirmación de reparación",
        questions=[
            create_confirm_question(
                "confirm_action",
                f"¿Confirmas que quieres ejecutar esta acción?\n\n📋 {repair_description}",
                help_text="Esta acción modificará el estado del equipo"
            ),
            create_conditional_question(
                "backup_confirmed",
                "¿Has realizado un respaldo del programa actual?",
                QuestionType.BOOLEAN,
                depends_on="confirm_action",
                show_when_equals=["yes", "sí", "si", "true"],
                help_text="Es recomendable tener un respaldo antes de cualquier cambio"
            ),
        ],
        context="Antes de ejecutar la reparación, necesito tu confirmación.",
        allow_back=True,
        allow_skip=False,
        completion_message="✅ Confirmación recibida. Ejecutando reparación..."
    )


def get_learning_style_wizard() -> QuestionSet:
    """Wizard para determinar el estilo de aprendizaje del usuario"""
    return create_wizard(
        worker="tutor",
        title="📚 Tu estilo de aprendizaje",
        questions=[
            create_choice_question(
                "experience_level",
                "¿Cuál es tu nivel de experiencia con PLCs?",
                [
                    ("1", "Principiante", "Estoy empezando a aprender", "🌱"),
                    ("2", "Intermedio", "Tengo experiencia básica", "📈"),
                    ("3", "Avanzado", "Trabajo regularmente con PLCs", "⭐"),
                    ("4", "Experto", "Diseño y desarrollo sistemas", "🏆"),
                ],
                include_other=False
            ),
            create_choice_question(
                "preferred_format",
                "¿Cómo prefieres que te explique los conceptos?",
                [
                    ("1", "Paso a paso detallado", "Explicaciones completas", "📝"),
                    ("2", "Ejemplos prácticos", "Aprendo haciendo", "🔧"),
                    ("3", "Conceptos teóricos primero", "Base sólida", "📖"),
                    ("4", "Directo al punto", "Solo lo esencial", "🎯"),
                ],
                include_other=False
            ),
            create_multi_choice_question(
                "topics_interest",
                "¿Qué temas te interesan más? (puedes elegir varios)",
                [
                    ("1", "Programación LAD/FBD"),
                    ("2", "Programación SCL/ST"),
                    ("3", "Comunicaciones industriales"),
                    ("4", "Diagnóstico y troubleshooting"),
                    ("5", "HMI y visualización"),
                    ("6", "Seguridad funcional"),
                ],
                min_selections=1,
                max_selections=3,
                required=False
            ),
        ],
        context="Conocer tu estilo de aprendizaje me ayudará a explicarte mejor los conceptos.",
        completion_message="✅ ¡Perfecto! Ahora puedo adaptar mis explicaciones a tu estilo."
    )
