"""
practice_schema.py - Schema for structured lab practices

A practice is a guided learning experience with:
- Steps that guide the user through a procedure
- Checkpoints to verify understanding
- Comprehension questions
- Final evaluation criteria

Practices are stored as JSON/YAML files and loaded by the tutor.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json
import yaml
import os
from datetime import datetime


class StepType(Enum):
    """Types of practice steps"""
    THEORY = "theory"           # Conceptual explanation
    SAFETY = "safety"           # Safety warning/instruction
    PROCEDURE = "procedure"     # Hands-on step
    CHECKPOINT = "checkpoint"   # Comprehension check
    OBSERVATION = "observation" # User observes/reports something
    REFLECTION = "reflection"   # User reflects on what they learned
    PRACTICAL = "practical"     # Hands-on practice exercise


class QuestionType(Enum):
    """Types of comprehension questions"""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    PRACTICAL = "practical"     # "Do X and tell me what happens"


@dataclass
class ComprehensionQuestion:
    """A question to check understanding"""
    id: str
    question: str
    question_type: QuestionType
    options: List[str] = field(default_factory=list)  # For multiple choice
    correct_answer: Optional[str] = None  # For auto-grading
    hints: List[str] = field(default_factory=list)
    points: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "question_type": self.question_type.value,
            "options": self.options,
            "hints": self.hints,
            "points": self.points,
        }


@dataclass
class PracticeStep:
    """A single step in a practice"""
    id: str
    title: str
    step_type: StepType
    content: str                    # Main instruction/explanation
    
    # Optional enrichment
    details: Optional[str] = None   # Expanded explanation if user needs more
    image_ref: Optional[str] = None # Reference to figure (e.g., "fig_7")
    video_ref: Optional[str] = None # Reference to video if available
    
    # For checkpoint steps
    questions: List[ComprehensionQuestion] = field(default_factory=list)
    
    # Safety emphasis
    is_critical: bool = False       # If True, MUST confirm before proceeding
    safety_warning: Optional[str] = None
    
    # Adaptive hints
    common_mistakes: List[str] = field(default_factory=list)
    tips: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "step_type": self.step_type.value,
            "content": self.content,
            "details": self.details,
            "image_ref": self.image_ref,
            "is_critical": self.is_critical,
            "safety_warning": self.safety_warning,
            "questions": [q.to_dict() for q in self.questions],
            "common_mistakes": self.common_mistakes,
            "tips": self.tips,
        }


@dataclass
class PracticeSection:
    """A logical section grouping related steps"""
    id: str
    title: str
    description: str
    learning_objectives: List[str]
    steps: List[PracticeStep]
    estimated_time_minutes: int = 15
    
    # Competencies this section develops
    competencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "learning_objectives": self.learning_objectives,
            "steps": [s.to_dict() for s in self.steps],
            "estimated_time_minutes": self.estimated_time_minutes,
            "competencies": self.competencies,
        }


@dataclass
class Practice:
    """A complete lab practice"""
    id: str
    title: str
    description: str
    version: str
    author: str
    
    # Prerequisites
    prerequisites: List[str] = field(default_factory=list)  # Competency IDs
    required_equipment: List[str] = field(default_factory=list)
    
    # Content
    sections: List[PracticeSection] = field(default_factory=list)
    
    # Metadata
    difficulty_level: str = "beginner"  # beginner, intermediate, advanced
    estimated_total_time_minutes: int = 60
    competencies_developed: List[str] = field(default_factory=list)
    
    # Evaluation
    passing_score: float = 0.7  # 70% to pass
    
    # Timestamps
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "prerequisites": self.prerequisites,
            "required_equipment": self.required_equipment,
            "sections": [s.to_dict() for s in self.sections],
            "difficulty_level": self.difficulty_level,
            "estimated_total_time_minutes": self.estimated_total_time_minutes,
            "competencies_developed": self.competencies_developed,
            "passing_score": self.passing_score,
        }
    
    def get_total_steps(self) -> int:
        return sum(len(section.steps) for section in self.sections)
    
    def get_step_by_id(self, step_id: str) -> Optional[PracticeStep]:
        for section in self.sections:
            for step in section.steps:
                if step.id == step_id:
                    return step
        return None
    
    def get_section_for_step(self, step_id: str) -> Optional[PracticeSection]:
        for section in self.sections:
            for step in section.steps:
                if step.id == step_id:
                    return section
        return None
    
    def get_next_step(self, current_step_id: str) -> Optional[PracticeStep]:
        """Get the next step after the given step"""
        found_current = False
        for section in self.sections:
            for step in section.steps:
                if found_current:
                    return step
                if step.id == current_step_id:
                    found_current = True
        return None
    
    def get_progress_percentage(self, completed_step_ids: List[str]) -> float:
        total = self.get_total_steps()
        if total == 0:
            return 0.0
        return len(completed_step_ids) / total * 100


@dataclass
class PracticeSession:
    """Tracks a user's progress through a practice"""
    session_id: str
    user_id: str
    practice_id: str
    
    # Progress
    current_section_id: Optional[str] = None
    current_step_id: Optional[str] = None
    completed_step_ids: List[str] = field(default_factory=list)
    
    # Responses and scores
    responses: Dict[str, Any] = field(default_factory=dict)  # step_id -> response
    scores: Dict[str, float] = field(default_factory=dict)   # step_id -> score
    
    # Timing
    started_at: Optional[str] = None
    last_activity_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # Adaptive tracking
    hints_requested: int = 0
    mistakes_made: int = 0
    time_per_step: Dict[str, float] = field(default_factory=dict)  # step_id -> seconds
    
    # Final summary (generated at completion)
    final_summary: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "practice_id": self.practice_id,
            "current_section_id": self.current_section_id,
            "current_step_id": self.current_step_id,
            "completed_step_ids": self.completed_step_ids,
            "responses": self.responses,
            "scores": self.scores,
            "started_at": self.started_at,
            "last_activity_at": self.last_activity_at,
            "completed_at": self.completed_at,
            "hints_requested": self.hints_requested,
            "mistakes_made": self.mistakes_made,
            "time_per_step": self.time_per_step,
            "final_summary": self.final_summary,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PracticeSession":
        return cls(
            session_id=data.get("session_id", ""),
            user_id=data.get("user_id", ""),
            practice_id=data.get("practice_id", ""),
            current_section_id=data.get("current_section_id"),
            current_step_id=data.get("current_step_id"),
            completed_step_ids=data.get("completed_step_ids", []),
            responses=data.get("responses", {}),
            scores=data.get("scores", {}),
            started_at=data.get("started_at"),
            last_activity_at=data.get("last_activity_at"),
            completed_at=data.get("completed_at"),
            hints_requested=data.get("hints_requested", 0),
            mistakes_made=data.get("mistakes_made", 0),
            time_per_step=data.get("time_per_step", {}),
            final_summary=data.get("final_summary"),
        )
    
    def mark_step_complete(self, step_id: str, response: Any = None, score: float = 1.0):
        """Mark a step as completed"""
        if step_id not in self.completed_step_ids:
            self.completed_step_ids.append(step_id)
        if response is not None:
            self.responses[step_id] = response
        self.scores[step_id] = score
        self.last_activity_at = datetime.utcnow().isoformat()
    
    def get_overall_score(self) -> float:
        """Calculate overall score from all graded steps"""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)
    
    def is_completed(self, practice: Practice) -> bool:
        """Check if all steps are completed"""
        total_steps = practice.get_total_steps()
        return len(self.completed_step_ids) >= total_steps


def load_practice_from_file(filepath: str) -> Practice:
    """Load a practice from JSON or YAML file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    
    return _parse_practice_data(data)


def _parse_practice_data(data: Dict[str, Any]) -> Practice:
    """Parse practice data from dict"""
    sections = []
    for section_data in data.get("sections", []):
        steps = []
        for step_data in section_data.get("steps", []):
            questions = []
            for q_data in step_data.get("questions", []):
                questions.append(ComprehensionQuestion(
                    id=q_data.get("id", ""),
                    question=q_data.get("question", ""),
                    question_type=QuestionType(q_data.get("question_type", "short_answer")),
                    options=q_data.get("options", []),
                    correct_answer=q_data.get("correct_answer"),
                    hints=q_data.get("hints", []),
                    points=q_data.get("points", 1),
                ))
            
            steps.append(PracticeStep(
                id=step_data.get("id", ""),
                title=step_data.get("title", ""),
                step_type=StepType(step_data.get("step_type", "procedure")),
                content=step_data.get("content", ""),
                details=step_data.get("details"),
                image_ref=step_data.get("image_ref"),
                video_ref=step_data.get("video_ref"),
                questions=questions,
                is_critical=step_data.get("is_critical", False),
                safety_warning=step_data.get("safety_warning"),
                common_mistakes=step_data.get("common_mistakes", []),
                tips=step_data.get("tips", []),
            ))
        
        sections.append(PracticeSection(
            id=section_data.get("id", ""),
            title=section_data.get("title", ""),
            description=section_data.get("description", ""),
            learning_objectives=section_data.get("learning_objectives", []),
            steps=steps,
            estimated_time_minutes=section_data.get("estimated_time_minutes", 15),
            competencies=section_data.get("competencies", []),
        ))
    
    return Practice(
        id=data.get("id", ""),
        title=data.get("title", ""),
        description=data.get("description", ""),
        version=data.get("version", "1.0"),
        author=data.get("author", ""),
        prerequisites=data.get("prerequisites", []),
        required_equipment=data.get("required_equipment", []),
        sections=sections,
        difficulty_level=data.get("difficulty_level", "beginner"),
        estimated_total_time_minutes=data.get("estimated_total_time_minutes", 60),
        competencies_developed=data.get("competencies_developed", []),
        passing_score=data.get("passing_score", 0.7),
        created_at=data.get("created_at"),
        updated_at=data.get("updated_at"),
    )
