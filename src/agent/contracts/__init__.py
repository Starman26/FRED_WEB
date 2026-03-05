"""Contratos de comunicación entre workers y supervisor"""
from .worker_contract import (
    WorkerOutput,
    EvidenceItem,
    ActionItem,
    ErrorItem,
    serialize_worker_output,
    parse_worker_output,
)

from .question_schema_v2 import (
    QuestionType,
    Question,
    Option,
    QuestionSet,
    AnswerSet,
    QuestionBuilder,
    QuestionRegistry,
    Condition,
    Urgency,
    quick_questions,
    troubleshooting_registry,
)

__all__ = [
    # Worker contract
    "WorkerOutput",
    "EvidenceItem",
    "ActionItem",
    "ErrorItem",
    "serialize_worker_output",
    "parse_worker_output",
    # Question schema v2
    "QuestionType",
    "Question",
    "Option",
    "QuestionSet",
    "AnswerSet",
    "QuestionBuilder",
    "QuestionRegistry",
    "Condition",
    "Urgency",
    "quick_questions",
    "troubleshooting_registry",
]
