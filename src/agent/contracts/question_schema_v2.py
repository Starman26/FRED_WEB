"""
question_schema_v2.py

Fluent HITL question API for FastAPI + SSE. Pure dataclasses, no Pydantic.
QuestionBuilder builds question sets; AnswerSet parses resume data.
"""
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json


class QuestionType(str, Enum):
    CHOICE = "choice"
    MULTI_CHOICE = "multi_choice"
    TEXT = "text"
    BOOLEAN = "boolean"
    NUMBER = "number"
    CONFIRM = "confirm"


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Condition:
    """Conditional display: show question only if a previous answer matches."""
    depends_on: str
    value: Union[str, List[str]]
    operator: str = "equals"  # equals, not_equals, in, not_in, contains

    def evaluate(self, answer: Any) -> bool:
        if answer is None:
            return False
        answer_str = str(answer).lower() if not isinstance(answer, list) else answer
        compare = self.value.lower() if isinstance(self.value, str) else self.value

        if self.operator == "equals":
            return answer_str == compare
        elif self.operator == "not_equals":
            return answer_str != compare
        elif self.operator == "in":
            return answer_str in [v.lower() for v in compare] if isinstance(compare, list) else False
        elif self.operator == "not_in":
            return answer_str not in [v.lower() for v in compare] if isinstance(compare, list) else True
        elif self.operator == "contains":
            return compare in answer_str
        return False


@dataclass
class Option:
    id: str
    label: str
    description: str = ""
    icon: str = ""

    def to_dict(self) -> dict:
        d = {"id": self.id, "label": self.label}
        if self.description:
            d["description"] = self.description
        if self.icon:
            d["icon"] = self.icon
        return d


@dataclass
class Question:
    id: str
    question: str
    type: QuestionType = QuestionType.TEXT
    options: List[Option] = field(default_factory=list)
    required: bool = True
    placeholder: str = ""
    help_text: str = ""
    default_value: str = ""
    condition: Optional[Condition] = None

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "question": self.question,
            "type": self.type.value,
            "required": self.required,
        }
        if self.options:
            d["options"] = [o.to_dict() for o in self.options]
        if self.placeholder:
            d["placeholder"] = self.placeholder
        if self.help_text:
            d["help_text"] = self.help_text
        if self.default_value:
            d["default_value"] = self.default_value
        if self.condition:
            d["condition"] = {
                "depends_on": self.condition.depends_on,
                "value": self.condition.value,
                "operator": self.condition.operator,
            }
        return d


@dataclass
class QuestionSet:
    """A set of questions ready to send to the frontend."""
    worker: str
    questions: List[Question]
    title: str = ""
    context: str = ""
    wizard_mode: bool = False
    allow_skip: bool = True
    on_complete: str = ""
    urgency: Urgency = Urgency.MEDIUM

    def to_interrupt_payload(self) -> dict:
        """Dict for interrupt() and SSE. Frontend receives as-is."""
        return {
            "type": "clarification",
            "worker": self.worker,
            "title": self.title,
            "context": self.context,
            "questions": [q.to_dict() for q in self.questions],
            "wizard_mode": self.wizard_mode,
            "allow_skip": self.allow_skip,
            "on_complete": self.on_complete,
            "urgency": self.urgency.value,
        }

    def to_dict_list(self) -> list:
        """Backward compat: list of question dicts."""
        return [q.to_dict() for q in self.questions]

    def to_display_text(self, current_step: int = 0) -> str:
        """Text fallback for worker_output.content."""
        lines = []
        if self.title:
            lines.append(f"## {self.title}\n")
        if self.context:
            lines.append(f"{self.context}\n")
        for i, q in enumerate(self.questions, 1):
            lines.append(f"**Pregunta {i}:** {q.question}")
            if q.options:
                for opt in q.options:
                    desc = f" - {opt.description}" if opt.description else ""
                    lines.append(f"  {opt.id}) {opt.label}{desc}")
            lines.append("")
        return "\n".join(lines)

    def model_dump_json(self) -> str:
        """Backward compat: JSON string for pending_context."""
        return json.dumps(self.to_interrupt_payload(), ensure_ascii=False)


@dataclass
class AnswerSet:
    """Parsed answers from the frontend resume payload."""
    answers: Dict[str, Any] = field(default_factory=dict)
    completed: bool = True
    cancelled: bool = False

    @classmethod
    def _list_to_dict(cls, items: list) -> dict:
        """Convert a list of answer dicts/strings to a flat dict."""
        answers: Dict[str, Any] = {}
        for item in items:
            if isinstance(item, dict):
                q = item.get("question", item.get("id", f"q{len(answers)}"))
                a = item.get("answer", item.get("value", ""))
                answers[q] = a
            elif isinstance(item, str):
                answers[f"q{len(answers)}"] = item
        return answers

    @classmethod
    def from_resume(cls, data: Any) -> "AnswerSet":
        """Parse from Command(resume=...) data.
        Supports:
        - v2 dict: {"answers": {"plc_model": "s71200", ...}, "completed": true}
        - v2 dict with list answers: {"answers": [{"question": "...", "answer": "..."}], ...}
        - Legacy list: [{"question": "...", "answer": "..."}]
        - Plain string: "user typed this"
        """
        print(f"[HITL DEBUG] from_resume input type={type(data).__name__}, value={str(data)[:200]}", flush=True)
        if isinstance(data, dict):
            raw_answers = data.get("answers", data)
            # answers can arrive as dict or list — normalize to dict
            if isinstance(raw_answers, list):
                answers = cls._list_to_dict(raw_answers)
            elif isinstance(raw_answers, dict):
                answers = raw_answers
            else:
                answers = {"user_response": str(raw_answers)}
            return cls(
                answers=answers,
                completed=data.get("completed", True),
                cancelled=data.get("cancelled", False),
            )
        if isinstance(data, list):
            return cls(answers=cls._list_to_dict(data))
        if isinstance(data, str):
            return cls(answers={"user_response": data})
        return cls()

    def to_context_string(self) -> str:
        """Convert to a human-readable string for LLM context."""
        return "\n".join(f"- {k}: {v}" for k, v in self.answers.items())

    def to_user_clarification(self) -> str:
        """Single answer returns just the value; multiple returns key: value lines."""
        if not self.answers:
            return ""
        if len(self.answers) == 1:
            return str(list(self.answers.values())[0])
        return "\n".join(f"{k}: {v}" for k, v in self.answers.items())


class QuestionBuilder:
    """Fluent API to build QuestionSets."""

    def __init__(self, worker: str):
        self._worker = worker
        self._title = ""
        self._context = ""
        self._wizard_mode = False
        self._allow_skip = True
        self._on_complete = ""
        self._urgency = Urgency.MEDIUM
        self._questions: List[Question] = []

    def title(self, t: str) -> "QuestionBuilder":
        self._title = t
        return self

    def context(self, c: str) -> "QuestionBuilder":
        self._context = c
        return self

    def wizard(self, allow_skip: bool = True) -> "QuestionBuilder":
        self._wizard_mode = True
        self._allow_skip = allow_skip
        return self

    def on_complete(self, message: str) -> "QuestionBuilder":
        self._on_complete = message
        return self

    def set_urgency(self, u: Urgency) -> "QuestionBuilder":
        self._urgency = u
        return self

    def choice(
        self, id: str, question: str, options: List[tuple],
        include_other: bool = True, required: bool = True,
        help_text: str = "", condition: Optional[Condition] = None,
    ) -> "QuestionBuilder":
        opts = []
        for opt in options:
            if len(opt) == 2:
                opts.append(Option(id=str(opt[0]), label=str(opt[1])))
            elif len(opt) >= 3:
                opts.append(Option(id=str(opt[0]), label=str(opt[1]), description=str(opt[2])))
        if include_other:
            opts.append(Option(id="other", label="Otro (especificar)"))
        self._questions.append(Question(
            id=id, question=question, type=QuestionType.CHOICE,
            options=opts, required=required, help_text=help_text, condition=condition,
        ))
        return self

    def text(
        self, id: str, question: str, placeholder: str = "Escribe tu respuesta...",
        required: bool = True, help_text: str = "",
        condition: Optional[Condition] = None,
    ) -> "QuestionBuilder":
        self._questions.append(Question(
            id=id, question=question, type=QuestionType.TEXT,
            placeholder=placeholder, required=required, help_text=help_text, condition=condition,
        ))
        return self

    def boolean(
        self, id: str, question: str, required: bool = True,
        help_text: str = "", condition: Optional[Condition] = None,
    ) -> "QuestionBuilder":
        self._questions.append(Question(
            id=id, question=question, type=QuestionType.BOOLEAN,
            options=[Option(id="yes", label="Sí"), Option(id="no", label="No")],
            required=required, help_text=help_text, condition=condition,
        ))
        return self

    def confirm(
        self, id: str, question: str, help_text: str = "",
        condition: Optional[Condition] = None,
    ) -> "QuestionBuilder":
        self._questions.append(Question(
            id=id, question=question, type=QuestionType.CONFIRM,
            required=True, help_text=help_text, condition=condition,
        ))
        return self

    def number(
        self, id: str, question: str, placeholder: str = "Ingresa un número",
        required: bool = True, help_text: str = "",
        condition: Optional[Condition] = None,
    ) -> "QuestionBuilder":
        self._questions.append(Question(
            id=id, question=question, type=QuestionType.NUMBER,
            placeholder=placeholder, required=required, help_text=help_text, condition=condition,
        ))
        return self

    def add_question(self, question: Question) -> "QuestionBuilder":
        """Add a pre-built Question object (useful with registry)."""
        self._questions.append(question)
        return self

    def build(self) -> QuestionSet:
        return QuestionSet(
            worker=self._worker,
            questions=self._questions,
            title=self._title,
            context=self._context,
            wizard_mode=self._wizard_mode,
            allow_skip=self._allow_skip,
            on_complete=self._on_complete,
            urgency=self._urgency,
        )


class QuestionRegistry:
    """Registry of predefined question templates, keyed by string."""

    def __init__(self):
        self._templates: Dict[str, Question] = {}

    def register(self, key: str, question: Question) -> None:
        self._templates[key] = question

    def get(self, key: str) -> Optional[Question]:
        return self._templates.get(key)

    def get_many(self, keys: List[str]) -> List[Question]:
        return [self._templates[k] for k in keys if k in self._templates]

    def keys(self) -> List[str]:
        return list(self._templates.keys())


def _build_troubleshooting_registry() -> QuestionRegistry:
    r = QuestionRegistry()

    r.register("plc_model", Question(
        id="plc_model",
        question="¿Cuál es el modelo exacto del PLC?",
        type=QuestionType.CHOICE,
        options=[
            Option(id="1", label="S7-1200", description="Serie compacta"),
            Option(id="2", label="S7-1500", description="Serie avanzada"),
            Option(id="3", label="S7-300", description="Serie clásica"),
            Option(id="4", label="S7-400", description="Serie de alto rendimiento"),
            Option(id="other", label="Otro (especificar)"),
        ],
        help_text="Lo puedes encontrar en la etiqueta frontal del equipo",
    ))

    r.register("tia_version", Question(
        id="tia_version",
        question="¿Qué versión de TIA Portal usas?",
        type=QuestionType.CHOICE,
        options=[
            Option(id="1", label="V15"),
            Option(id="2", label="V16"),
            Option(id="3", label="V17"),
            Option(id="4", label="V18"),
            Option(id="5", label="V19", description="Última versión"),
            Option(id="other", label="Otro (especificar)"),
        ],
        help_text="Menú Ayuda → Acerca de TIA Portal",
    ))

    r.register("connection_type", Question(
        id="connection_type",
        question="¿Cómo está conectado el PLC?",
        type=QuestionType.CHOICE,
        options=[
            Option(id="1", label="Cable directo PC-PLC", description="Conexión punto a punto"),
            Option(id="2", label="A través de switch/router", description="Red local"),
            Option(id="3", label="Red corporativa/VPN", description="Conexión remota"),
            Option(id="other", label="Otro (especificar)"),
        ],
        help_text="Esto ayuda a identificar problemas de comunicación",
    ))

    r.register("worked_before", Question(
        id="worked_before",
        question="¿El PLC funcionaba correctamente antes de este problema?",
        type=QuestionType.BOOLEAN,
        options=[Option(id="yes", label="Sí"), Option(id="no", label="No")],
        help_text="Nos ayuda a saber si es un problema nuevo o recurrente",
    ))

    r.register("error_message", Question(
        id="error_message",
        question="¿Cuál es el mensaje de error exacto?",
        type=QuestionType.TEXT,
        placeholder="Copia y pega el mensaje de error o descríbelo...",
        help_text="Entre más detallado, mejor podemos ayudarte",
    ))

    r.register("recent_changes", Question(
        id="recent_changes",
        question="¿Qué cambios recientes se hicieron?",
        type=QuestionType.TEXT,
        placeholder="Actualizaciones, cambios de red, nuevo programa...",
        required=False,
        help_text="Cualquier cambio en las últimas horas/días puede ser relevante",
    ))

    r.register("led_status", Question(
        id="led_status",
        question="¿Cuál es el estado de los LEDs del PLC?",
        type=QuestionType.CHOICE,
        options=[
            Option(id="1", label="RUN verde fijo", description="Operación normal"),
            Option(id="2", label="STOP amarillo/rojo", description="CPU detenida"),
            Option(id="3", label="ERROR parpadeando", description="Hay un fallo"),
            Option(id="4", label="Todos apagados", description="Sin alimentación"),
            Option(id="other", label="Otro (especificar)"),
        ],
        help_text="Los LEDs del frente indican el estado del equipo",
    ))

    r.register("urgency", Question(
        id="urgency",
        question="¿Qué tan urgente es resolver este problema?",
        type=QuestionType.CHOICE,
        options=[
            Option(id="1", label="Crítico - Producción parada", description="Necesito solución inmediata"),
            Option(id="2", label="Alto - Afecta operación", description="Impacta significativamente"),
            Option(id="3", label="Medio - Funciona parcialmente", description="Puedo continuar con limitaciones"),
            Option(id="4", label="Bajo - Preventivo/mejora", description="No hay impacto inmediato"),
        ],
        required=False,
        help_text="Nos ayuda a priorizar la atención",
    ))

    return r


troubleshooting_registry = _build_troubleshooting_registry()


def quick_questions(
    worker: str,
    missing_info: List[str],
    wizard_mode: bool = True,
    max_questions: int = 5,
) -> QuestionSet:
    """Generate a QuestionSet from predefined templates based on missing info keys."""
    info_to_key = {
        "modelo": "plc_model", "plc": "plc_model",
        "version": "tia_version", "tia": "tia_version",
        "error": "error_message", "mensaje": "error_message",
        "conexion": "connection_type", "conexión": "connection_type",
        "led": "led_status",
        "cambios": "recent_changes",
        "urgencia": "urgency",
    }

    questions: List[Question] = []
    added: set = set()

    for info in missing_info:
        info_lower = info.lower()
        for keyword, question_key in info_to_key.items():
            if keyword in info_lower and question_key not in added:
                q = troubleshooting_registry.get(question_key)
                if q:
                    questions.append(q)
                    added.add(question_key)
                    break
        if len(questions) >= max_questions:
            break

    if not questions:
        for key in ["plc_model", "error_message"]:
            q = troubleshooting_registry.get(key)
            if q:
                questions.append(q)

    builder = QuestionBuilder(worker)
    builder.title("Diagnóstico de problema")
    builder.context("Vamos a identificar el problema paso a paso.")
    if wizard_mode:
        builder.wizard(allow_skip=True)
    builder.on_complete("Gracias. Ya tengo la información necesaria para analizar tu problema.")

    for q in questions[:max_questions]:
        builder.add_question(q)

    return builder.build()
