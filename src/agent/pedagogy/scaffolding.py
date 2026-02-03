"""
scaffolding.py - Adaptive Scaffolding System

Implements pedagogical scaffolding that adapts to user competency:
- Dynamic hint generation
- Progressive difficulty adjustment
- Contextual explanations
- Guided problem-solving
- Fading support as mastery increases
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.agent.memory.user_competency import (
    UserCompetencyTracker,
    MasteryLevel,
    get_user_tracker,
)


class SupportLevel(Enum):
    """Level of scaffolding support to provide"""
    FULL = "full"           # Maximum support - detailed guidance
    HIGH = "high"           # High support - step-by-step hints
    MEDIUM = "medium"       # Medium support - general hints
    LOW = "low"             # Low support - minimal hints
    NONE = "none"           # No support - independent work


class HintType(Enum):
    """Types of hints that can be provided"""
    CONCEPTUAL = "conceptual"       # Explain underlying concept
    PROCEDURAL = "procedural"       # Show steps to follow
    STRATEGIC = "strategic"         # Suggest approach/strategy
    METACOGNITIVE = "metacognitive" # Help reflect on thinking
    CORRECTIVE = "corrective"       # Point out specific error
    EXAMPLE = "example"             # Provide worked example


@dataclass
class Hint:
    """A pedagogical hint"""
    id: str
    type: HintType
    content: str
    level: int  # 1 = most general, 5 = most specific
    related_competency: str
    prerequisites: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "level": self.level,
            "related_competency": self.related_competency,
            "prerequisites": self.prerequisites,
        }


@dataclass
class ScaffoldingContext:
    """Context for scaffolding decisions"""
    user_id: str
    task_type: str
    competency_id: str
    current_step: int = 0
    total_steps: int = 1
    errors_in_task: List[str] = field(default_factory=list)
    hints_given: List[str] = field(default_factory=list)
    time_on_task_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "task_type": self.task_type,
            "competency_id": self.competency_id,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "errors_in_task": self.errors_in_task,
            "hints_given": self.hints_given,
            "time_on_task_seconds": self.time_on_task_seconds,
        }


class AdaptiveScaffolder:
    """
    Provides adaptive scaffolding based on user competency and task context.
    
    Features:
    - Determines appropriate support level
    - Generates progressive hints
    - Provides contextual explanations
    - Implements fading (reducing support as mastery increases)
    """
    
    def __init__(self):
        # Hint banks for different competencies and situations
        self._hint_banks: Dict[str, List[Hint]] = {}
        self._initialize_hint_banks()
    
    def _initialize_hint_banks(self):
        """Initialize hint banks for common scenarios"""
        
        # Safety hints
        self._hint_banks["comp_safety_basics"] = [
            Hint("safety_1", HintType.CONCEPTUAL, 
                 "Safety systems protect both humans and equipment. Always verify safety conditions before any operation.",
                 1, "comp_safety_basics"),
            Hint("safety_2", HintType.PROCEDURAL,
                 "Step 1: Check all safety doors are closed. Step 2: Verify E-stop is not engaged. Step 3: Check for any active alarms.",
                 2, "comp_safety_basics"),
            Hint("safety_3", HintType.STRATEGIC,
                 "When facing a safety interlock, work backwards: what triggered it? Check the HMI for specific error codes.",
                 3, "comp_safety_basics"),
        ]
        
        # Troubleshooting hints
        self._hint_banks["comp_troubleshooting"] = [
            Hint("ts_1", HintType.CONCEPTUAL,
                 "Troubleshooting follows a systematic approach: Identify symptoms → Isolate cause → Apply solution → Verify fix.",
                 1, "comp_troubleshooting"),
            Hint("ts_2", HintType.STRATEGIC,
                 "Start with the most common causes. Check: Power, Connections, Configuration, Software state.",
                 2, "comp_troubleshooting"),
            Hint("ts_3", HintType.PROCEDURAL,
                 "For communication errors: 1) Check physical connections 2) Verify IP/Network settings 3) Check firewall/ports 4) Restart services",
                 3, "comp_troubleshooting"),
            Hint("ts_4", HintType.METACOGNITIVE,
                 "What changed recently? New updates, configuration changes, or physical modifications often cause issues.",
                 2, "comp_troubleshooting"),
            Hint("ts_5", HintType.EXAMPLE,
                 "Example: PLC disconnect → Check Ethernet cable → Ping test → Verify TIA Portal connection → Check PLC status LEDs",
                 4, "comp_troubleshooting"),
        ]
        
        # Cobot operation hints
        self._hint_banks["comp_cobot_operation"] = [
            Hint("cobot_1", HintType.CONCEPTUAL,
                 "Cobots are designed for human collaboration. They have built-in safety features that stop motion on contact.",
                 1, "comp_cobot_operation"),
            Hint("cobot_2", HintType.PROCEDURAL,
                 "To start a routine: 1) Put cobot in Remote mode 2) Load the program 3) Verify start conditions 4) Press Play",
                 2, "comp_cobot_operation"),
            Hint("cobot_3", HintType.CORRECTIVE,
                 "Protective stop usually means unexpected contact or force. Check for obstacles in the work envelope.",
                 3, "comp_cobot_operation"),
            Hint("cobot_4", HintType.STRATEGIC,
                 "When a cobot won't start, check: Mode (Local/Remote), Program loaded, Safety status, Previous errors cleared",
                 3, "comp_cobot_operation"),
        ]
        
        # PLC hints
        self._hint_banks["comp_plc_basics"] = [
            Hint("plc_1", HintType.CONCEPTUAL,
                 "PLCs execute programs in a scan cycle: Read inputs → Execute logic → Update outputs → Repeat",
                 1, "comp_plc_basics"),
            Hint("plc_2", HintType.PROCEDURAL,
                 "To connect to a Siemens PLC: 1) Configure TIA Portal 2) Set PC IP in same subnet 3) Go Online 4) Download/Upload",
                 2, "comp_plc_basics"),
            Hint("plc_3", HintType.STRATEGIC,
                 "When debugging PLC logic, use Watch Tables to monitor variable values in real-time",
                 3, "comp_plc_basics"),
        ]
        
        # PID tuning hints
        self._hint_banks["comp_pid_tuning"] = [
            Hint("pid_1", HintType.CONCEPTUAL,
                 "PID control: P reacts to current error, I eliminates steady-state error, D dampens oscillations",
                 1, "comp_pid_tuning"),
            Hint("pid_2", HintType.PROCEDURAL,
                 "Ziegler-Nichols tuning: 1) Set Ki=Kd=0 2) Increase Kp until oscillation 3) Note Ku and Tu 4) Calculate parameters",
                 2, "comp_pid_tuning"),
            Hint("pid_3", HintType.STRATEGIC,
                 "If system oscillates: reduce Kp or increase Kd. If slow response: increase Kp. If steady-state error: increase Ki",
                 3, "comp_pid_tuning"),
            Hint("pid_4", HintType.EXAMPLE,
                 "Al_FrED_0 temperature control: Kp=35, Ki=0.8, Kd=12 achieves <1°C error with 30s settling time",
                 4, "comp_pid_tuning"),
        ]
    
    # ==========================================
    # Support Level Determination
    # ==========================================
    
    def determine_support_level(
        self,
        tracker: UserCompetencyTracker,
        competency_id: str,
        context: Optional[ScaffoldingContext] = None
    ) -> SupportLevel:
        """
        Determine appropriate support level based on user mastery and context.
        
        Implements fading: support decreases as mastery increases.
        """
        comp = tracker.get_competency(competency_id)
        
        if comp is None:
            return SupportLevel.FULL
        
        # Base support level on mastery
        mastery_to_support = {
            MasteryLevel.NOVICE: SupportLevel.FULL,
            MasteryLevel.BEGINNER: SupportLevel.HIGH,
            MasteryLevel.INTERMEDIATE: SupportLevel.MEDIUM,
            MasteryLevel.ADVANCED: SupportLevel.LOW,
            MasteryLevel.EXPERT: SupportLevel.NONE,
        }
        base_support = mastery_to_support.get(comp.mastery_level, SupportLevel.MEDIUM)
        
        # Adjust based on context
        if context:
            # Increase support if many errors
            if len(context.errors_in_task) >= 3:
                base_support = self._increase_support(base_support)
            
            # Increase support if stuck (long time on task)
            expected_time = 120  # 2 minutes expected
            if context.time_on_task_seconds > expected_time * 3:
                base_support = self._increase_support(base_support)
            
            # Decrease support if progressing well
            if context.current_step > 0 and len(context.errors_in_task) == 0:
                base_support = self._decrease_support(base_support)
        
        # Adjust based on improvement rate
        if comp.improvement_rate > 0.1:
            base_support = self._decrease_support(base_support)
        elif comp.improvement_rate < -0.1:
            base_support = self._increase_support(base_support)
        
        return base_support
    
    def _increase_support(self, level: SupportLevel) -> SupportLevel:
        """Increase support level by one step"""
        order = [SupportLevel.NONE, SupportLevel.LOW, SupportLevel.MEDIUM, 
                 SupportLevel.HIGH, SupportLevel.FULL]
        idx = order.index(level)
        return order[min(idx + 1, len(order) - 1)]
    
    def _decrease_support(self, level: SupportLevel) -> SupportLevel:
        """Decrease support level by one step"""
        order = [SupportLevel.NONE, SupportLevel.LOW, SupportLevel.MEDIUM,
                 SupportLevel.HIGH, SupportLevel.FULL]
        idx = order.index(level)
        return order[max(idx - 1, 0)]
    
    # ==========================================
    # Hint Generation
    # ==========================================
    
    def get_next_hint(
        self,
        competency_id: str,
        context: ScaffoldingContext,
        error_type: Optional[str] = None
    ) -> Optional[Hint]:
        """
        Get the next appropriate hint based on context.
        
        Implements progressive disclosure: hints become more specific
        as user requests more help.
        """
        hints = self._hint_banks.get(competency_id, [])
        if not hints:
            return None
        
        # Filter out already given hints
        available_hints = [h for h in hints if h.id not in context.hints_given]
        if not available_hints:
            return None
        
        # Determine hint level based on hints already given
        hint_level = len(context.hints_given) + 1
        
        # Prefer corrective hints if there's an error
        if error_type and len(context.errors_in_task) > 0:
            corrective = [h for h in available_hints if h.type == HintType.CORRECTIVE]
            if corrective:
                return corrective[0]
        
        # Get hint at appropriate level
        level_hints = [h for h in available_hints if h.level <= hint_level]
        if level_hints:
            # Sort by level and return lowest level hint not yet given
            level_hints.sort(key=lambda h: h.level)
            return level_hints[0]
        
        return available_hints[0] if available_hints else None
    
    def get_all_hints_for_competency(self, competency_id: str) -> List[Hint]:
        """Get all available hints for a competency"""
        return self._hint_banks.get(competency_id, [])
    
    # ==========================================
    # Contextual Explanations
    # ==========================================
    
    def generate_explanation(
        self,
        action: str,
        context: Dict[str, Any],
        tracker: UserCompetencyTracker,
        competency_id: str
    ) -> Dict[str, Any]:
        """
        Generate a contextual explanation for an action/decision.
        
        Adapts detail level based on user mastery.
        """
        support_level = self.determine_support_level(tracker, competency_id)
        
        explanation = {
            "action": action,
            "context": context,
            "detail_level": support_level.value,
        }
        
        # Generate explanation based on support level
        if support_level == SupportLevel.FULL:
            explanation["sections"] = [
                {"type": "what", "content": f"What is happening: {action}"},
                {"type": "why", "content": "Why this is important: [Generated based on action]"},
                {"type": "how", "content": "How it works: [Step by step explanation]"},
                {"type": "example", "content": "Example: [Concrete example from the lab]"},
            ]
        elif support_level == SupportLevel.HIGH:
            explanation["sections"] = [
                {"type": "what", "content": f"Action: {action}"},
                {"type": "why", "content": "Reason: [Brief explanation]"},
                {"type": "how", "content": "Steps: [Key steps only]"},
            ]
        elif support_level == SupportLevel.MEDIUM:
            explanation["sections"] = [
                {"type": "summary", "content": f"{action} - [Brief rationale]"},
                {"type": "note", "content": "[One important consideration]"},
            ]
        elif support_level == SupportLevel.LOW:
            explanation["sections"] = [
                {"type": "summary", "content": f"{action}"},
            ]
        else:  # NONE
            explanation["sections"] = []
        
        return explanation
    
    # ==========================================
    # Guided Problem-Solving
    # ==========================================
    
    def get_guided_steps(
        self,
        problem_type: str,
        tracker: UserCompetencyTracker,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get guided problem-solving steps adapted to user level.
        
        Returns steps with appropriate detail level.
        """
        # Define step templates for common problems
        problem_steps = {
            "plc_disconnect": [
                {"step": 1, "action": "Check physical connection", 
                 "detail": "Verify Ethernet cable is properly connected at both ends",
                 "verification": "Cable LED indicators should be lit"},
                {"step": 2, "action": "Verify network configuration",
                 "detail": "Ensure PC and PLC are on same subnet",
                 "verification": "Ping PLC IP address successfully"},
                {"step": 3, "action": "Check PLC status",
                 "detail": "Look at PLC front panel LEDs for error indicators",
                 "verification": "RUN LED should be green"},
                {"step": 4, "action": "Restart communication",
                 "detail": "In TIA Portal, go offline then online again",
                 "verification": "Connection status shows 'Online'"},
            ],
            "cobot_fault": [
                {"step": 1, "action": "Identify fault type",
                 "detail": "Check Polyscope for specific error message",
                 "verification": "Error code and description visible"},
                {"step": 2, "action": "Clear workspace",
                 "detail": "Ensure no obstacles in cobot's work envelope",
                 "verification": "360° visual inspection complete"},
                {"step": 3, "action": "Reset fault",
                 "detail": "Press the fault reset button or use teach pendant",
                 "verification": "Fault indicator clears"},
                {"step": 4, "action": "Test motion",
                 "detail": "Manually jog cobot at low speed to verify freedom of movement",
                 "verification": "Cobot moves smoothly in all directions"},
            ],
            "temperature_issue": [
                {"step": 1, "action": "Check current readings",
                 "detail": "Read temperature from sensor and compare to setpoint",
                 "verification": "Note the error magnitude"},
                {"step": 2, "action": "Verify heater operation",
                 "detail": "Check if heater cartridge is receiving power",
                 "verification": "Measure voltage at heater terminals"},
                {"step": 3, "action": "Check thermistor",
                 "detail": "Measure thermistor resistance and compare to expected value",
                 "verification": "Resistance matches temperature curve"},
                {"step": 4, "action": "Adjust PID if needed",
                 "detail": "Fine-tune Kp, Ki, Kd parameters based on response",
                 "verification": "Temperature stabilizes within ±1°C"},
            ],
        }
        
        steps = problem_steps.get(problem_type, [])
        if not steps:
            return []
        
        # Adapt detail level based on user mastery
        comp = tracker.get_competency("comp_troubleshooting")
        mastery = comp.mastery_level if comp else MasteryLevel.NOVICE
        
        adapted_steps = []
        for step in steps:
            adapted = {"step": step["step"], "action": step["action"]}
            
            # Include more detail for lower mastery levels
            if mastery in [MasteryLevel.NOVICE, MasteryLevel.BEGINNER]:
                adapted["detail"] = step["detail"]
                adapted["verification"] = step["verification"]
            elif mastery == MasteryLevel.INTERMEDIATE:
                adapted["detail"] = step["detail"]
            # Advanced and Expert get minimal guidance
            
            adapted_steps.append(adapted)
        
        return adapted_steps
    
    # ==========================================
    # Add Custom Hints
    # ==========================================
    
    def add_hint(self, competency_id: str, hint: Hint):
        """Add a custom hint to a competency's hint bank"""
        if competency_id not in self._hint_banks:
            self._hint_banks[competency_id] = []
        self._hint_banks[competency_id].append(hint)


# Singleton instance
_scaffolder: Optional[AdaptiveScaffolder] = None


def get_scaffolder() -> AdaptiveScaffolder:
    """Get the singleton scaffolder instance"""
    global _scaffolder
    if _scaffolder is None:
        _scaffolder = AdaptiveScaffolder()
    return _scaffolder
