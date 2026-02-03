"""
exercise_generator.py - Progressive Exercise Generator

Generates adaptive exercises based on:
- User's current competency level
- Detected knowledge gaps
- Error patterns from troubleshooting history
- Learning objectives
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import random

from src.agent.memory.user_competency import (
    UserCompetencyTracker,
    MasteryLevel,
    get_user_tracker,
)
from src.agent.memory.knowledge_graph import (
    get_knowledge_graph,
    EntityType,
)


class ExerciseType(Enum):
    """Types of exercises"""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    FILL_BLANK = "fill_blank"
    SEQUENCE = "sequence"           # Put steps in order
    MATCHING = "matching"           # Match concepts to definitions
    SIMULATION = "simulation"       # Interactive simulation
    TROUBLESHOOT = "troubleshoot"   # Diagnose a problem
    PRACTICAL = "practical"         # Hands-on task


class DifficultyLevel(Enum):
    """Exercise difficulty levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class ExerciseOption:
    """An option in a multiple choice or matching exercise"""
    id: str
    text: str
    is_correct: bool = False
    feedback: str = ""  # Feedback if this option is selected


@dataclass
class Exercise:
    """An exercise/assessment item"""
    id: str
    type: ExerciseType
    competency_id: str
    difficulty: DifficultyLevel
    
    # Content
    title: str
    instructions: str
    question: str
    
    # Options/answers
    options: List[ExerciseOption] = field(default_factory=list)
    correct_answer: Any = None
    explanation: str = ""
    
    # Metadata
    time_limit_seconds: Optional[int] = None
    points: int = 10
    tags: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    
    # Hints (progressive)
    hints: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "competency_id": self.competency_id,
            "difficulty": self.difficulty.value,
            "title": self.title,
            "instructions": self.instructions,
            "question": self.question,
            "options": [{"id": o.id, "text": o.text} for o in self.options],
            "time_limit_seconds": self.time_limit_seconds,
            "points": self.points,
            "tags": self.tags,
            "hints_available": len(self.hints),
        }
    
    def check_answer(self, answer: Any) -> Tuple[bool, str, int]:
        """
        Check if an answer is correct.
        Returns: (is_correct, feedback, points_earned)
        """
        if self.type == ExerciseType.MULTIPLE_CHOICE:
            selected_option = next((o for o in self.options if o.id == answer), None)
            if selected_option:
                if selected_option.is_correct:
                    return True, self.explanation or "Correct!", self.points
                else:
                    return False, selected_option.feedback or "Incorrect.", 0
            return False, "Invalid selection.", 0
        
        elif self.type == ExerciseType.TRUE_FALSE:
            is_correct = str(answer).lower() == str(self.correct_answer).lower()
            if is_correct:
                return True, self.explanation or "Correct!", self.points
            return False, f"Incorrect. The answer is {self.correct_answer}.", 0
        
        elif self.type == ExerciseType.SEQUENCE:
            if answer == self.correct_answer:
                return True, self.explanation or "Correct sequence!", self.points
            # Partial credit for partially correct sequence
            correct_positions = sum(1 for i, a in enumerate(answer) 
                                   if i < len(self.correct_answer) and a == self.correct_answer[i])
            partial_points = int(self.points * correct_positions / len(self.correct_answer))
            return False, f"Not quite right. {correct_positions}/{len(self.correct_answer)} in correct position.", partial_points
        
        elif self.type == ExerciseType.FILL_BLANK:
            # Case-insensitive comparison
            if str(answer).lower().strip() == str(self.correct_answer).lower().strip():
                return True, self.explanation or "Correct!", self.points
            return False, f"Incorrect. The answer is: {self.correct_answer}", 0
        
        else:
            # For simulation/practical exercises, return neutral
            return True, "Exercise completed.", self.points


class ExerciseGenerator:
    """
    Generates exercises adapted to user competency.
    
    Features:
    - Progressive difficulty based on mastery
    - Error-targeted exercises
    - Variety of exercise types
    - Contextual exercises using lab scenarios
    """
    
    def __init__(self):
        self._exercise_bank: Dict[str, List[Exercise]] = {}
        self._initialize_exercise_bank()
    
    def _initialize_exercise_bank(self):
        """Initialize exercise bank with base exercises"""
        
        # Safety exercises
        self._exercise_bank["comp_safety_basics"] = [
            Exercise(
                id="safety_mc_1",
                type=ExerciseType.MULTIPLE_CHOICE,
                competency_id="comp_safety_basics",
                difficulty=DifficultyLevel.EASY,
                title="Safety Door Function",
                instructions="Select the correct answer",
                question="What happens when a safety door is opened while a cobot is running?",
                options=[
                    ExerciseOption("a", "The cobot speeds up", feedback="Incorrect. Opening a safety door triggers a protective response."),
                    ExerciseOption("b", "The cobot stops immediately", is_correct=True, feedback="Correct! Safety doors trigger an immediate stop."),
                    ExerciseOption("c", "Nothing happens", feedback="Incorrect. Safety systems always respond to door state."),
                    ExerciseOption("d", "An alarm sounds but motion continues", feedback="Incorrect. Motion must stop for safety."),
                ],
                explanation="Safety doors are connected to the safety circuit. Opening them triggers an immediate protective stop to prevent injury.",
                hints=["Think about what the primary purpose of a safety door is.", "What would be the safest response?"],
            ),
            Exercise(
                id="safety_seq_1",
                type=ExerciseType.SEQUENCE,
                competency_id="comp_safety_basics",
                difficulty=DifficultyLevel.MEDIUM,
                title="Safety Check Sequence",
                instructions="Put these safety check steps in the correct order",
                question="Order the steps for a pre-operation safety check:",
                options=[
                    ExerciseOption("a", "Verify E-stop is not engaged"),
                    ExerciseOption("b", "Check all safety doors are closed"),
                    ExerciseOption("c", "Clear the work area of personnel"),
                    ExerciseOption("d", "Verify safety light curtains are active"),
                ],
                correct_answer=["c", "b", "d", "a"],
                explanation="First clear people from danger, then verify all barriers, then check final safety controls.",
                hints=["What should you check first before anything else?", "Physical barriers come before electronic checks."],
            ),
        ]
        
        # Troubleshooting exercises
        self._exercise_bank["comp_troubleshooting"] = [
            Exercise(
                id="ts_mc_1",
                type=ExerciseType.MULTIPLE_CHOICE,
                competency_id="comp_troubleshooting",
                difficulty=DifficultyLevel.EASY,
                title="Communication Error Diagnosis",
                instructions="Select the most likely cause",
                question="A PLC shows 'Communication Timeout'. What should you check first?",
                options=[
                    ExerciseOption("a", "PLC program logic", feedback="Program logic doesn't cause timeouts if the PLC was working before."),
                    ExerciseOption("b", "Ethernet cable connection", is_correct=True, feedback="Correct! Physical connections are the most common cause of timeouts."),
                    ExerciseOption("c", "Room temperature", feedback="Temperature rarely causes immediate communication issues."),
                    ExerciseOption("d", "Time of day settings", feedback="Time settings don't affect basic communication."),
                ],
                explanation="Always start with the simplest, most common causes. Physical connections fail more often than software.",
                hints=["Start with the most common cause.", "What's the most basic requirement for communication?"],
            ),
            Exercise(
                id="ts_troubleshoot_1",
                type=ExerciseType.TROUBLESHOOT,
                competency_id="comp_troubleshooting",
                difficulty=DifficultyLevel.MEDIUM,
                title="Cobot Won't Start",
                instructions="Diagnose the problem based on the symptoms",
                question="""
Station 3 cobot won't start a program. Symptoms:
- Power LED is green
- Safety status shows 'Normal'
- Polyscope shows 'Remote Mode'
- Program is loaded

What is the most likely issue?""",
                options=[
                    ExerciseOption("a", "Power supply failure", feedback="Power LED is green, so power is OK."),
                    ExerciseOption("b", "Safety interlock active", feedback="Safety status shows Normal."),
                    ExerciseOption("c", "External control not enabled", is_correct=True, feedback="Correct! In Remote Mode, external start signal is required."),
                    ExerciseOption("d", "Program needs to be reloaded", feedback="Program is already loaded."),
                ],
                explanation="In Remote Mode, the cobot waits for an external start signal from the PLC or control system. Check the PLC program for the start condition.",
                hints=["What does 'Remote Mode' mean for control?", "Where does the start command come from in Remote Mode?"],
            ),
            Exercise(
                id="ts_seq_1",
                type=ExerciseType.SEQUENCE,
                competency_id="comp_troubleshooting",
                difficulty=DifficultyLevel.HARD,
                title="Systematic Troubleshooting",
                instructions="Order these troubleshooting steps correctly",
                question="You receive a report of 'intermittent sensor readings' at Station 2. Order your diagnostic steps:",
                options=[
                    ExerciseOption("a", "Check sensor wiring connections"),
                    ExerciseOption("b", "Review recent changes to the system"),
                    ExerciseOption("c", "Monitor live sensor values"),
                    ExerciseOption("d", "Check for electrical interference sources"),
                    ExerciseOption("e", "Replace the sensor"),
                ],
                correct_answer=["b", "c", "a", "d", "e"],
                explanation="Start with information gathering (recent changes, monitoring), then physical checks, environmental factors, and replacement only as a last resort.",
            ),
        ]
        
        # PLC exercises
        self._exercise_bank["comp_plc_basics"] = [
            Exercise(
                id="plc_tf_1",
                type=ExerciseType.TRUE_FALSE,
                competency_id="comp_plc_basics",
                difficulty=DifficultyLevel.EASY,
                title="PLC Scan Cycle",
                instructions="Answer True or False",
                question="A PLC executes its program continuously in a scan cycle, reading inputs at the start and writing outputs at the end of each cycle.",
                correct_answer="true",
                explanation="This is the fundamental operation of a PLC: Input scan → Program execution → Output update → Repeat.",
            ),
            Exercise(
                id="plc_mc_1",
                type=ExerciseType.MULTIPLE_CHOICE,
                competency_id="comp_plc_basics",
                difficulty=DifficultyLevel.MEDIUM,
                title="PLC Status LEDs",
                instructions="Select the correct interpretation",
                question="A Siemens S7-1200 shows: RUN LED blinking orange, ERROR LED off. What does this indicate?",
                options=[
                    ExerciseOption("a", "PLC is in STOP mode", feedback="STOP mode shows solid LED states."),
                    ExerciseOption("b", "PLC is starting up", is_correct=True, feedback="Correct! Blinking orange during RUN indicates startup/initialization."),
                    ExerciseOption("c", "PLC has a hardware fault", feedback="Hardware faults light the ERROR LED."),
                    ExerciseOption("d", "PLC firmware needs update", feedback="Firmware status isn't shown this way."),
                ],
                explanation="During startup, the RUN LED blinks orange while the PLC initializes. Once running normally, it turns solid green.",
            ),
        ]
        
        # PID tuning exercises
        self._exercise_bank["comp_pid_tuning"] = [
            Exercise(
                id="pid_mc_1",
                type=ExerciseType.MULTIPLE_CHOICE,
                competency_id="comp_pid_tuning",
                difficulty=DifficultyLevel.MEDIUM,
                title="PID Oscillation",
                instructions="Select the best action",
                question="Your temperature controller is oscillating around the setpoint with consistent amplitude. Which PID parameter should you adjust?",
                options=[
                    ExerciseOption("a", "Increase Ki (Integral)", feedback="Increasing Ki would make oscillations worse."),
                    ExerciseOption("b", "Increase Kp (Proportional)", feedback="Higher Kp typically increases oscillation."),
                    ExerciseOption("c", "Increase Kd (Derivative)", is_correct=True, feedback="Correct! Derivative action dampens oscillations."),
                    ExerciseOption("d", "Set all parameters to zero", feedback="This would disable control entirely."),
                ],
                explanation="The derivative term (Kd) acts as a damper, reducing the rate of change and suppressing oscillations. Alternatively, you could reduce Kp.",
                hints=["Which term reacts to the rate of change?", "What would 'dampen' the oscillation?"],
            ),
            Exercise(
                id="pid_fill_1",
                type=ExerciseType.FILL_BLANK,
                competency_id="comp_pid_tuning",
                difficulty=DifficultyLevel.EASY,
                title="PID Components",
                instructions="Fill in the blank",
                question="In a PID controller, the _____ term eliminates steady-state error by accumulating error over time.",
                correct_answer="integral",
                explanation="The Integral (I) term sums up error over time, which eliminates any persistent offset between setpoint and actual value.",
            ),
        ]
        
        # Cobot operation exercises
        self._exercise_bank["comp_cobot_operation"] = [
            Exercise(
                id="cobot_mc_1",
                type=ExerciseType.MULTIPLE_CHOICE,
                competency_id="comp_cobot_operation",
                difficulty=DifficultyLevel.EASY,
                title="Cobot vs Industrial Robot",
                instructions="Select the key difference",
                question="What is the main feature that distinguishes a collaborative robot (cobot) from a traditional industrial robot?",
                options=[
                    ExerciseOption("a", "Cobots are always smaller", feedback="Size isn't the defining characteristic."),
                    ExerciseOption("b", "Cobots have built-in safety features for human proximity", is_correct=True, feedback="Correct! Cobots are designed to work safely alongside humans."),
                    ExerciseOption("c", "Cobots are always faster", feedback="Speed isn't the distinguishing feature."),
                    ExerciseOption("d", "Cobots use different programming languages", feedback="Many use similar languages."),
                ],
                explanation="Collaborative robots have force-torque sensing, rounded edges, and safety-rated monitoring that allows them to work near humans without full safety enclosures.",
            ),
            Exercise(
                id="cobot_match_1",
                type=ExerciseType.MATCHING,
                competency_id="comp_cobot_operation",
                difficulty=DifficultyLevel.MEDIUM,
                title="UR Cobot Modes",
                instructions="Match each mode to its description",
                question="Match the Universal Robots operating modes:",
                options=[
                    ExerciseOption("local", "Program can only be started from teach pendant"),
                    ExerciseOption("remote", "Program controlled by external PLC/system"),
                    ExerciseOption("freedrive", "Robot can be moved by hand for teaching"),
                    ExerciseOption("running", "Actively executing a program"),
                ],
                correct_answer={"local": "teach pendant", "remote": "external PLC", "freedrive": "moved by hand", "running": "executing program"},
            ),
        ]
    
    # ==========================================
    # Exercise Generation
    # ==========================================
    
    def generate_exercise(
        self,
        tracker: UserCompetencyTracker,
        competency_id: str,
        preferred_type: Optional[ExerciseType] = None
    ) -> Optional[Exercise]:
        """
        Generate an exercise appropriate for the user's level.
        """
        # Get appropriate difficulty
        difficulty = self._get_appropriate_difficulty(tracker, competency_id)
        
        # Get exercises for this competency
        exercises = self._exercise_bank.get(competency_id, [])
        if not exercises:
            return None
        
        # Filter by difficulty (allow one level up or down)
        difficulty_order = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, 
                          DifficultyLevel.HARD, DifficultyLevel.EXPERT]
        diff_idx = difficulty_order.index(difficulty)
        allowed_difficulties = set()
        for i in range(max(0, diff_idx - 1), min(len(difficulty_order), diff_idx + 2)):
            allowed_difficulties.add(difficulty_order[i])
        
        filtered = [e for e in exercises if e.difficulty in allowed_difficulties]
        
        # Filter by type if specified
        if preferred_type:
            type_filtered = [e for e in filtered if e.type == preferred_type]
            if type_filtered:
                filtered = type_filtered
        
        if not filtered:
            filtered = exercises  # Fall back to all exercises
        
        # Select randomly from filtered exercises
        return random.choice(filtered)
    
    def generate_error_targeted_exercise(
        self,
        tracker: UserCompetencyTracker,
        error_type: str
    ) -> Optional[Exercise]:
        """
        Generate an exercise targeting a specific error type.
        """
        # Map error types to relevant competencies
        error_to_competency = {
            "err_door_open": "comp_safety_basics",
            "err_estop": "comp_safety_basics",
            "err_plc_disconnect": "comp_plc_basics",
            "err_cobot_fault": "comp_cobot_operation",
            "err_sensor_timeout": "comp_troubleshooting",
            "err_temp_out_of_range": "comp_pid_tuning",
        }
        
        competency_id = error_to_competency.get(error_type, "comp_troubleshooting")
        return self.generate_exercise(tracker, competency_id)
    
    def generate_exercise_set(
        self,
        tracker: UserCompetencyTracker,
        competency_id: str,
        count: int = 5
    ) -> List[Exercise]:
        """
        Generate a set of exercises with progressive difficulty.
        """
        exercises = []
        used_ids = set()
        
        for i in range(count):
            exercise = self.generate_exercise(tracker, competency_id)
            if exercise and exercise.id not in used_ids:
                exercises.append(exercise)
                used_ids.add(exercise.id)
        
        # Sort by difficulty
        difficulty_order = {
            DifficultyLevel.EASY: 0,
            DifficultyLevel.MEDIUM: 1,
            DifficultyLevel.HARD: 2,
            DifficultyLevel.EXPERT: 3,
        }
        exercises.sort(key=lambda e: difficulty_order[e.difficulty])
        
        return exercises
    
    def _get_appropriate_difficulty(
        self,
        tracker: UserCompetencyTracker,
        competency_id: str
    ) -> DifficultyLevel:
        """Determine appropriate difficulty based on user mastery"""
        comp = tracker.get_competency(competency_id)
        
        if not comp or comp.total_attempts == 0:
            return DifficultyLevel.EASY
        
        # Map mastery to difficulty
        mastery_to_difficulty = {
            MasteryLevel.NOVICE: DifficultyLevel.EASY,
            MasteryLevel.BEGINNER: DifficultyLevel.EASY,
            MasteryLevel.INTERMEDIATE: DifficultyLevel.MEDIUM,
            MasteryLevel.ADVANCED: DifficultyLevel.HARD,
            MasteryLevel.EXPERT: DifficultyLevel.EXPERT,
        }
        
        base_difficulty = mastery_to_difficulty.get(comp.mastery_level, DifficultyLevel.MEDIUM)
        
        # Adjust based on recent performance
        if comp.improvement_rate > 0.1:
            # User is improving, challenge them more
            if base_difficulty == DifficultyLevel.EASY:
                return DifficultyLevel.MEDIUM
            elif base_difficulty == DifficultyLevel.MEDIUM:
                return DifficultyLevel.HARD
        elif comp.improvement_rate < -0.1:
            # User is struggling, make it easier
            if base_difficulty == DifficultyLevel.EXPERT:
                return DifficultyLevel.HARD
            elif base_difficulty == DifficultyLevel.HARD:
                return DifficultyLevel.MEDIUM
        
        return base_difficulty
    
    # ==========================================
    # Exercise Bank Management
    # ==========================================
    
    def add_exercise(self, exercise: Exercise):
        """Add an exercise to the bank"""
        comp_id = exercise.competency_id
        if comp_id not in self._exercise_bank:
            self._exercise_bank[comp_id] = []
        self._exercise_bank[comp_id].append(exercise)
    
    def get_exercise_count(self, competency_id: Optional[str] = None) -> int:
        """Get count of exercises, optionally filtered by competency"""
        if competency_id:
            return len(self._exercise_bank.get(competency_id, []))
        return sum(len(exercises) for exercises in self._exercise_bank.values())
    
    def get_available_competencies(self) -> List[str]:
        """Get list of competencies with available exercises"""
        return list(self._exercise_bank.keys())


# Singleton instance
_generator: Optional[ExerciseGenerator] = None


def get_exercise_generator() -> ExerciseGenerator:
    """Get the singleton exercise generator"""
    global _generator
    if _generator is None:
        _generator = ExerciseGenerator()
    return _generator
