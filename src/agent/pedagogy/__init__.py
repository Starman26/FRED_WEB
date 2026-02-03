"""
pedagogy - Pedagogical Agent Components

Implements adaptive learning features:
- Scaffolding: Dynamic support based on user competency
- Exercise Generation: Progressive exercises targeting knowledge gaps
- Learning Path: Recommended sequence of learning activities

Components:
- scaffolding: Adaptive hints and guided problem-solving
- exercise_generator: Creates exercises based on user level and errors
"""

from .scaffolding import (
    AdaptiveScaffolder,
    get_scaffolder,
    SupportLevel,
    HintType,
    Hint,
    ScaffoldingContext,
)

from .exercise_generator import (
    ExerciseGenerator,
    get_exercise_generator,
    Exercise,
    ExerciseOption,
    ExerciseType,
    DifficultyLevel,
)

__all__ = [
    # Scaffolding
    "AdaptiveScaffolder",
    "get_scaffolder",
    "SupportLevel",
    "HintType",
    "Hint",
    "ScaffoldingContext",
    # Exercise Generator
    "ExerciseGenerator",
    "get_exercise_generator",
    "Exercise",
    "ExerciseOption",
    "ExerciseType",
    "DifficultyLevel",
]
