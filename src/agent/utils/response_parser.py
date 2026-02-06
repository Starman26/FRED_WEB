"""
response_parser.py - Utilidades para parsear respuestas de usuario en el wizard

Este módulo proporciona funciones para:
1. Parsear respuestas de texto libre a opciones estructuradas
2. Extraer respuestas de mensajes conversacionales
3. Normalizar respuestas para diferentes tipos de preguntas
4. Manejar respuestas múltiples en un solo mensaje
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import re
from src.agent.contracts.question_schema import (
    ClarificationQuestion,
    QuestionType,
    QuestionOption,
    QuestionResponse,
    QuestionSetResponse,
    ValidationResult,
)


class ResponseParser:
    """Parser para respuestas de usuario en el wizard interactivo."""

    # Patrones para detectar respuestas booleanas
    YES_PATTERNS = [
        r"^s[íi]$", r"^yes$", r"^y$", r"^1$", r"^true$",
        r"^afirmativo$", r"^correcto$", r"^claro$", r"^ok$",
        r"^así es$", r"^efectivamente$", r"^por supuesto$"
    ]
    NO_PATTERNS = [
        r"^no$", r"^n$", r"^0$", r"^false$",
        r"^negativo$", r"^incorrecto$", r"^para nada$"
    ]

    # Patrones para detectar navegación
    BACK_PATTERNS = [r"^atr[aá]s$", r"^back$", r"^anterior$", r"^<$", r"^volver$"]
    SKIP_PATTERNS = [r"^saltar$", r"^skip$", r"^omitir$", r"^siguiente$", r"^>$", r"^pasar$"]
    CANCEL_PATTERNS = [r"^cancelar$", r"^cancel$", r"^exit$", r"^salir$", r"^abort$"]
    RESTART_PATTERNS = [r"^reiniciar$", r"^restart$", r"^reset$", r"^empezar$"]

    @classmethod
    def parse_boolean(cls, text: str) -> Optional[bool]:
        """
        Parsea texto a booleano.

        Returns:
            True si es afirmativo, False si es negativo, None si no es reconocible
        """
        text_lower = text.strip().lower()

        for pattern in cls.YES_PATTERNS:
            if re.match(pattern, text_lower):
                return True

        for pattern in cls.NO_PATTERNS:
            if re.match(pattern, text_lower):
                return False

        return None

    @classmethod
    def parse_choice(
        cls,
        text: str,
        options: List[QuestionOption]
    ) -> Optional[str]:
        """
        Parsea texto a una opción de choice.

        Busca coincidencias por:
        1. ID exacto (ej: "1", "2", "other")
        2. Label parcial (ej: "S7-1200" match "1200")
        3. Descripción parcial

        Returns:
            ID de la opción seleccionada o None
        """
        text_lower = text.strip().lower()

        # Buscar por ID exacto
        for opt in options:
            if text_lower == opt.id.lower():
                return opt.id

        # Buscar por label exacto
        for opt in options:
            if text_lower == opt.label.lower():
                return opt.id

        # Buscar por label parcial (contenido)
        for opt in options:
            if text_lower in opt.label.lower() or opt.label.lower() in text_lower:
                return opt.id

        # Buscar en descripción
        for opt in options:
            if opt.description:
                if text_lower in opt.description.lower():
                    return opt.id

        return None

    @classmethod
    def parse_multi_choice(
        cls,
        text: str,
        options: List[QuestionOption],
        separator: str = ","
    ) -> List[str]:
        """
        Parsea texto a múltiples opciones.

        Soporta formatos:
        - "1, 2, 3"
        - "1 2 3"
        - "opción1, opción2"
        - "1 y 2 y 3"

        Returns:
            Lista de IDs de opciones seleccionadas
        """
        # Normalizar separadores
        text_normalized = text.replace(" y ", ",").replace(" and ", ",")
        text_normalized = re.sub(r"\s+", ",", text_normalized)

        parts = [p.strip() for p in text_normalized.split(separator) if p.strip()]
        selected_ids = []

        for part in parts:
            option_id = cls.parse_choice(part, options)
            if option_id and option_id not in selected_ids:
                selected_ids.append(option_id)

        return selected_ids

    @classmethod
    def parse_number(cls, text: str) -> Optional[float]:
        """
        Parsea texto a número.

        Soporta:
        - Enteros: "42"
        - Decimales: "3.14" o "3,14"
        - Negativos: "-5"

        Returns:
            Número parseado o None
        """
        text_clean = text.strip().replace(",", ".")
        # Remover espacios y caracteres de moneda
        text_clean = re.sub(r"[$€£¥]", "", text_clean)
        text_clean = re.sub(r"\s", "", text_clean)

        try:
            return float(text_clean)
        except ValueError:
            return None

    @classmethod
    def detect_navigation_command(cls, text: str) -> Optional[str]:
        """
        Detecta si el texto es un comando de navegación.

        Returns:
            "back", "skip", "cancel", "restart", o None
        """
        text_lower = text.strip().lower()

        for pattern in cls.BACK_PATTERNS:
            if re.match(pattern, text_lower):
                return "back"

        for pattern in cls.SKIP_PATTERNS:
            if re.match(pattern, text_lower):
                return "skip"

        for pattern in cls.CANCEL_PATTERNS:
            if re.match(pattern, text_lower):
                return "cancel"

        for pattern in cls.RESTART_PATTERNS:
            if re.match(pattern, text_lower):
                return "restart"

        return None

    @classmethod
    def parse_for_question(
        cls,
        text: str,
        question: ClarificationQuestion
    ) -> Tuple[bool, Any, Optional[str]]:
        """
        Parsea texto según el tipo de pregunta.

        Returns:
            (success, parsed_value, error_message)
        """
        text = text.strip()

        # Verificar quick replies primero
        for qr in question.quick_replies:
            if text.lower() == qr.label.lower() or text.lower() == qr.id.lower():
                return True, qr.value, None

        if question.type == QuestionType.BOOLEAN:
            result = cls.parse_boolean(text)
            if result is not None:
                return True, result, None
            return False, None, "Por favor responde 'sí' o 'no'"

        elif question.type == QuestionType.CHOICE:
            result = cls.parse_choice(text, question.options)
            if result:
                return True, result, None
            option_list = ", ".join([f"{o.id}) {o.label}" for o in question.options])
            return False, None, f"Opción no reconocida. Opciones válidas: {option_list}"

        elif question.type == QuestionType.MULTI_CHOICE:
            results = cls.parse_multi_choice(text, question.options)
            if results:
                return True, results, None
            return False, None, "No se reconocieron opciones válidas"

        elif question.type == QuestionType.NUMBER:
            result = cls.parse_number(text)
            if result is not None:
                # Validar con las reglas de la pregunta
                validation = question.validate_answer(result)
                if validation.is_valid:
                    return True, result, None
                return False, None, validation.error_message
            return False, None, "Por favor ingresa un número válido"

        elif question.type == QuestionType.CONFIRM:
            result = cls.parse_boolean(text)
            if result is not None:
                return True, result, None
            # También aceptar "confirmar"
            if text.lower() in ["confirmar", "confirm", "confirmo"]:
                return True, True, None
            return False, None, "Por favor confirma con 'sí' o 'confirmar'"

        else:  # TEXT
            # Validar si hay reglas
            validation = question.validate_answer(text)
            if validation.is_valid:
                return True, text, None
            return False, None, validation.error_message


class MultiAnswerParser:
    """Parser para respuestas múltiples en un solo mensaje."""

    @classmethod
    def parse_numbered_responses(cls, text: str) -> Dict[int, str]:
        """
        Extrae respuestas numeradas de un texto.

        Ejemplo:
            "1. S7-1200\n2. V17\n3. Cable directo"
            -> {1: "S7-1200", 2: "V17", 3: "Cable directo"}
        """
        responses = {}
        # Patrón para "1. respuesta" o "1) respuesta" o "1: respuesta"
        pattern = r"(\d+)[.\):\-]\s*(.+?)(?=\n\d+[.\):\-]|\Z)"
        matches = re.findall(pattern, text, re.DOTALL)

        for num_str, answer in matches:
            try:
                num = int(num_str)
                responses[num] = answer.strip()
            except ValueError:
                continue

        return responses

    @classmethod
    def parse_labeled_responses(cls, text: str) -> Dict[str, str]:
        """
        Extrae respuestas etiquetadas de un texto.

        Ejemplo:
            "Modelo: S7-1200\nVersión: V17\nConexión: Cable directo"
            -> {"modelo": "S7-1200", "versión": "V17", "conexión": "Cable directo"}
        """
        responses = {}
        # Patrón para "etiqueta: respuesta"
        pattern = r"([a-záéíóúñ\s]+)[:=]\s*(.+?)(?=\n[a-záéíóúñ\s]+[:=]|\Z)"
        matches = re.findall(pattern, text.lower(), re.DOTALL | re.IGNORECASE)

        for label, answer in matches:
            responses[label.strip().lower()] = answer.strip()

        return responses

    @classmethod
    def parse_all_at_once(
        cls,
        text: str,
        questions: List[ClarificationQuestion]
    ) -> QuestionSetResponse:
        """
        Intenta parsear respuestas para múltiples preguntas de un solo mensaje.

        Estrategia:
        1. Primero intenta respuestas numeradas
        2. Luego respuestas etiquetadas
        3. Finalmente divide por líneas

        Returns:
            QuestionSetResponse con las respuestas parseadas
        """
        responses = []

        # Intentar respuestas numeradas
        numbered = cls.parse_numbered_responses(text)
        if numbered:
            for i, question in enumerate(questions, 1):
                if i in numbered:
                    success, value, _ = ResponseParser.parse_for_question(
                        numbered[i], question
                    )
                    if success:
                        label = None
                        if question.type in [QuestionType.CHOICE, QuestionType.MULTI_CHOICE]:
                            for opt in question.options:
                                if opt.id == str(value):
                                    label = opt.label
                                    break
                        responses.append(QuestionResponse(
                            question_id=question.id,
                            answer=value,
                            answer_label=label
                        ))
            if responses:
                return QuestionSetResponse(responses=responses)

        # Intentar respuestas etiquetadas
        labeled = cls.parse_labeled_responses(text)
        if labeled:
            for question in questions:
                # Buscar por ID o pregunta parcial
                for label, answer in labeled.items():
                    if (question.id.lower() in label or
                        label in question.question.lower() or
                        any(word in label for word in question.id.lower().split("_"))):
                        success, value, _ = ResponseParser.parse_for_question(
                            answer, question
                        )
                        if success:
                            responses.append(QuestionResponse(
                                question_id=question.id,
                                answer=value
                            ))
                            break
            if responses:
                return QuestionSetResponse(responses=responses)

        # Dividir por líneas y asignar en orden
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for i, line in enumerate(lines):
            if i < len(questions):
                question = questions[i]
                success, value, _ = ResponseParser.parse_for_question(line, question)
                if success:
                    responses.append(QuestionResponse(
                        question_id=question.id,
                        answer=value
                    ))

        return QuestionSetResponse(responses=responses, completed=len(responses) == len(questions))


def extract_structured_info(
    text: str,
    expected_fields: List[str]
) -> Dict[str, Optional[str]]:
    """
    Extrae información estructurada de texto libre.

    Útil para extraer campos como modelo, versión, error, etc.
    de un mensaje conversacional.

    Args:
        text: Texto del usuario
        expected_fields: Campos a buscar ["plc_model", "tia_version", "error_message"]

    Returns:
        Dict con campos encontrados y sus valores
    """
    result = {field: None for field in expected_fields}

    # Patrones comunes para cada tipo de campo
    patterns = {
        "plc_model": [
            r"s7[- ]?(1200|1500|300|400)",
            r"(1200|1500|s7\d+)",
            r"plc\s+(\S+)",
        ],
        "tia_version": [
            r"v(15|16|17|18|19)",
            r"tia\s*portal?\s*v?(15|16|17|18|19)",
            r"versión?\s*(\d+)",
        ],
        "error_message": [
            r"error[:\s]+[\"']?(.+?)[\"']?(?:\n|$)",
            r"mensaje[:\s]+[\"']?(.+?)[\"']?(?:\n|$)",
        ],
        "connection_type": [
            r"(cable directo|ethernet|profinet|profibus)",
            r"conexión?\s*(\w+)",
        ],
    }

    text_lower = text.lower()

    for field in expected_fields:
        if field in patterns:
            for pattern in patterns[field]:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    result[field] = match.group(1) if match.groups() else match.group(0)
                    break

    return result
