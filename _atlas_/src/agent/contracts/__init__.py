"""Contratos de comunicación entre workers y supervisor"""
from .worker_contract import (
    WorkerOutput,
    EvidenceItem,
    ActionItem,
    ErrorItem,
    serialize_worker_output,
    parse_worker_output,
)

from .question_schema import (
    QuestionOption,
    ClarificationQuestion,
    QuestionSet,
    QuestionResponse,
    QuestionSetResponse,
    TROUBLESHOOTING_QUESTIONS,
    create_choice_question,
    create_text_question,
    create_boolean_question,
)

__all__ = [
    # Worker contract
    "WorkerOutput",
    "EvidenceItem",
    "ActionItem",
    "ErrorItem",
    "serialize_worker_output",
    "parse_worker_output",
    # Question schema
    "QuestionOption",
    "ClarificationQuestion",
    "QuestionSet",
    "QuestionResponse",
    "QuestionSetResponse",
    "TROUBLESHOOTING_QUESTIONS",
    "create_choice_question",
    "create_text_question",
    "create_boolean_question",
]
