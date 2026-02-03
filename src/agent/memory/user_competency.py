"""
user_competency.py - User Competency Tracking System

Monitorea el nivel de competencia del usuario:
- Tracks skill levels across different domains
- Records performance on exercises/tasks
- Calculates mastery levels
- Identifies knowledge gaps
- Suggests learning paths
"""
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import math


class MasteryLevel(Enum):
    """Levels of mastery for a competency"""
    NOVICE = "novice"           # 0-20%: Just starting
    BEGINNER = "beginner"       # 20-40%: Basic understanding
    INTERMEDIATE = "intermediate"  # 40-60%: Can apply with guidance
    ADVANCED = "advanced"       # 60-80%: Can apply independently
    EXPERT = "expert"           # 80-100%: Can teach others
    
    @classmethod
    def from_score(cls, score: float) -> "MasteryLevel":
        """Convert a 0-1 score to a mastery level"""
        if score < 0.2:
            return cls.NOVICE
        elif score < 0.4:
            return cls.BEGINNER
        elif score < 0.6:
            return cls.INTERMEDIATE
        elif score < 0.8:
            return cls.ADVANCED
        else:
            return cls.EXPERT


@dataclass
class PerformanceRecord:
    """Record of a user's performance on a task/exercise"""
    task_id: str
    task_type: str  # "exercise", "troubleshooting", "operation"
    competency_id: str
    success: bool
    score: float  # 0-1
    time_taken_seconds: float
    errors_made: List[str] = field(default_factory=list)
    hints_used: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "competency_id": self.competency_id,
            "success": self.success,
            "score": self.score,
            "time_taken_seconds": self.time_taken_seconds,
            "errors_made": self.errors_made,
            "hints_used": self.hints_used,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class CompetencyState:
    """Current state of a user's competency"""
    competency_id: str
    competency_name: str
    category: str
    
    # Mastery tracking
    mastery_score: float = 0.0  # 0-1
    mastery_level: MasteryLevel = MasteryLevel.NOVICE
    
    # Performance history
    total_attempts: int = 0
    successful_attempts: int = 0
    recent_scores: List[float] = field(default_factory=list)  # Last 10 scores
    
    # Time tracking
    total_time_spent_seconds: float = 0.0
    first_attempt: Optional[datetime] = None
    last_attempt: Optional[datetime] = None
    
    # Learning indicators
    improvement_rate: float = 0.0  # Positive = improving, negative = declining
    consistency: float = 0.0  # How consistent are the scores
    
    def update_from_performance(self, record: PerformanceRecord):
        """Update competency state based on a new performance record"""
        self.total_attempts += 1
        if record.success:
            self.successful_attempts += 1
        
        self.total_time_spent_seconds += record.time_taken_seconds
        
        if self.first_attempt is None:
            self.first_attempt = record.timestamp
        self.last_attempt = record.timestamp
        
        # Update recent scores (keep last 10)
        self.recent_scores.append(record.score)
        if len(self.recent_scores) > 10:
            self.recent_scores = self.recent_scores[-10:]
        
        # Recalculate mastery score using exponential moving average
        # More recent performances have higher weight
        alpha = 0.3  # Smoothing factor
        self.mastery_score = alpha * record.score + (1 - alpha) * self.mastery_score
        
        # Apply bonus for consistency
        if self.total_attempts >= 3:
            self._calculate_consistency()
            if self.consistency > 0.8:
                self.mastery_score = min(1.0, self.mastery_score * 1.05)
        
        # Update mastery level
        self.mastery_level = MasteryLevel.from_score(self.mastery_score)
        
        # Calculate improvement rate (slope of recent scores)
        if len(self.recent_scores) >= 3:
            self._calculate_improvement_rate()
    
    def _calculate_consistency(self):
        """Calculate consistency of recent scores"""
        if len(self.recent_scores) < 2:
            self.consistency = 0.0
            return
        
        mean = sum(self.recent_scores) / len(self.recent_scores)
        variance = sum((s - mean) ** 2 for s in self.recent_scores) / len(self.recent_scores)
        std_dev = math.sqrt(variance)
        
        # Consistency is inverse of coefficient of variation
        if mean > 0:
            cv = std_dev / mean
            self.consistency = max(0.0, 1.0 - cv)
        else:
            self.consistency = 0.0
    
    def _calculate_improvement_rate(self):
        """Calculate improvement rate using linear regression"""
        n = len(self.recent_scores)
        if n < 3:
            self.improvement_rate = 0.0
            return
        
        # Simple linear regression
        x_mean = (n - 1) / 2
        y_mean = sum(self.recent_scores) / n
        
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(self.recent_scores))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator > 0:
            self.improvement_rate = numerator / denominator
        else:
            self.improvement_rate = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "competency_id": self.competency_id,
            "competency_name": self.competency_name,
            "category": self.category,
            "mastery_score": self.mastery_score,
            "mastery_level": self.mastery_level.value,
            "total_attempts": self.total_attempts,
            "successful_attempts": self.successful_attempts,
            "success_rate": self.successful_attempts / self.total_attempts if self.total_attempts > 0 else 0,
            "recent_scores": self.recent_scores,
            "total_time_spent_seconds": self.total_time_spent_seconds,
            "first_attempt": self.first_attempt.isoformat() if self.first_attempt else None,
            "last_attempt": self.last_attempt.isoformat() if self.last_attempt else None,
            "improvement_rate": self.improvement_rate,
            "consistency": self.consistency,
        }


class UserCompetencyTracker:
    """
    Tracks and manages user competencies.
    
    Provides:
    - Competency state tracking
    - Performance recording
    - Mastery level calculations
    - Knowledge gap identification
    - Learning recommendations
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self._competencies: Dict[str, CompetencyState] = {}
        self._performance_history: List[PerformanceRecord] = []
        self._error_patterns: Dict[str, int] = {}  # error_type -> count
        self._created_at = datetime.utcnow()
        self._last_activity = datetime.utcnow()
        
        # Initialize base competencies
        self._initialize_competencies()
    
    def _initialize_competencies(self):
        """Initialize with base competency definitions"""
        base_competencies = [
            ("comp_safety_basics", "Safety Basics", "safety"),
            ("comp_cobot_operation", "Cobot Basic Operation", "robotics"),
            ("comp_plc_basics", "PLC Fundamentals", "automation"),
            ("comp_troubleshooting", "Basic Troubleshooting", "maintenance"),
            ("comp_pid_tuning", "PID Controller Tuning", "control"),
            ("comp_urscript", "URScript Programming", "robotics"),
            ("comp_ladder_logic", "Ladder Logic Programming", "automation"),
            ("comp_vision_systems", "Vision System Integration", "vision"),
            ("comp_process_quality", "Process Quality Control", "quality"),
            ("comp_maintenance", "Preventive Maintenance", "maintenance"),
        ]
        
        for comp_id, comp_name, category in base_competencies:
            self._competencies[comp_id] = CompetencyState(
                competency_id=comp_id,
                competency_name=comp_name,
                category=category
            )
    
    # ==========================================
    # Performance Recording
    # ==========================================
    
    def record_performance(
        self,
        task_id: str,
        task_type: str,
        competency_id: str,
        success: bool,
        score: float,
        time_taken_seconds: float,
        errors_made: Optional[List[str]] = None,
        hints_used: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PerformanceRecord:
        """Record a performance event"""
        record = PerformanceRecord(
            task_id=task_id,
            task_type=task_type,
            competency_id=competency_id,
            success=success,
            score=min(1.0, max(0.0, score)),
            time_taken_seconds=time_taken_seconds,
            errors_made=errors_made or [],
            hints_used=hints_used,
            metadata=metadata or {},
        )
        
        self._performance_history.append(record)
        self._last_activity = datetime.utcnow()
        
        # Update competency state
        if competency_id in self._competencies:
            self._competencies[competency_id].update_from_performance(record)
        
        # Track error patterns
        for error in record.errors_made:
            self._error_patterns[error] = self._error_patterns.get(error, 0) + 1
        
        return record
    
    def record_troubleshooting_attempt(
        self,
        station: int,
        error_type: str,
        resolution_success: bool,
        time_to_resolve: float,
        steps_taken: List[str],
        hints_requested: int = 0
    ) -> PerformanceRecord:
        """Convenience method for recording troubleshooting attempts"""
        # Calculate score based on success, time, and hints
        base_score = 1.0 if resolution_success else 0.3
        time_penalty = min(0.3, time_to_resolve / 600)  # Max 10 min penalty
        hint_penalty = hints_requested * 0.1
        
        score = max(0.1, base_score - time_penalty - hint_penalty)
        
        return self.record_performance(
            task_id=f"ts_{station}_{error_type}_{datetime.utcnow().timestamp()}",
            task_type="troubleshooting",
            competency_id="comp_troubleshooting",
            success=resolution_success,
            score=score,
            time_taken_seconds=time_to_resolve,
            errors_made=[error_type] if not resolution_success else [],
            hints_used=hints_requested,
            metadata={
                "station": station,
                "error_type": error_type,
                "steps_taken": steps_taken,
            }
        )
    
    # ==========================================
    # Competency Queries
    # ==========================================
    
    def get_competency(self, competency_id: str) -> Optional[CompetencyState]:
        """Get a specific competency state"""
        return self._competencies.get(competency_id)
    
    def get_all_competencies(self) -> Dict[str, CompetencyState]:
        """Get all competency states"""
        return self._competencies.copy()
    
    def get_competencies_by_category(self, category: str) -> List[CompetencyState]:
        """Get competencies in a specific category"""
        return [c for c in self._competencies.values() if c.category == category]
    
    def get_mastery_level(self, competency_id: str) -> Optional[MasteryLevel]:
        """Get the mastery level for a competency"""
        comp = self._competencies.get(competency_id)
        return comp.mastery_level if comp else None
    
    def get_overall_level(self) -> MasteryLevel:
        """Calculate overall user level based on all competencies"""
        if not self._competencies:
            return MasteryLevel.NOVICE
        
        total_score = sum(c.mastery_score for c in self._competencies.values())
        avg_score = total_score / len(self._competencies)
        return MasteryLevel.from_score(avg_score)
    
    # ==========================================
    # Knowledge Gap Analysis
    # ==========================================
    
    def identify_knowledge_gaps(self) -> List[Tuple[CompetencyState, str]]:
        """
        Identify knowledge gaps based on performance.
        Returns list of (competency, reason) tuples.
        """
        gaps = []
        
        for comp in self._competencies.values():
            reasons = []
            
            # Low mastery with attempts
            if comp.total_attempts >= 3 and comp.mastery_score < 0.4:
                reasons.append("low_mastery_after_practice")
            
            # Declining performance
            if comp.improvement_rate < -0.1:
                reasons.append("declining_performance")
            
            # High error rate
            if comp.total_attempts >= 3:
                success_rate = comp.successful_attempts / comp.total_attempts
                if success_rate < 0.5:
                    reasons.append("high_failure_rate")
            
            # Inconsistent performance
            if comp.total_attempts >= 5 and comp.consistency < 0.5:
                reasons.append("inconsistent_performance")
            
            # No recent practice (> 7 days)
            if comp.last_attempt:
                days_since_practice = (datetime.utcnow() - comp.last_attempt).days
                if days_since_practice > 7 and comp.mastery_score < 0.8:
                    reasons.append("needs_refresher")
            
            if reasons:
                gaps.append((comp, ", ".join(reasons)))
        
        # Sort by mastery score (lowest first)
        gaps.sort(key=lambda x: x[0].mastery_score)
        return gaps
    
    def get_frequent_errors(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Get most frequent error types"""
        sorted_errors = sorted(
            self._error_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_errors[:limit]
    
    # ==========================================
    # Learning Recommendations
    # ==========================================
    
    def recommend_focus_areas(self, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Recommend areas to focus on based on current state.
        Returns prioritized list of recommendations.
        """
        recommendations = []
        gaps = self.identify_knowledge_gaps()
        
        for comp, reasons in gaps[:limit]:
            rec = {
                "competency_id": comp.competency_id,
                "competency_name": comp.competency_name,
                "category": comp.category,
                "current_level": comp.mastery_level.value,
                "mastery_score": comp.mastery_score,
                "reasons": reasons.split(", "),
                "priority": "high" if comp.mastery_score < 0.3 else "medium",
            }
            
            # Add specific recommendations based on reason
            if "declining_performance" in reasons:
                rec["suggestion"] = "Review fundamental concepts and practice basic exercises"
            elif "inconsistent_performance" in reasons:
                rec["suggestion"] = "Focus on building consistent habits through regular practice"
            elif "needs_refresher" in reasons:
                rec["suggestion"] = "Complete a quick refresher exercise to reinforce learning"
            else:
                rec["suggestion"] = "Work through progressive exercises to build mastery"
            
            recommendations.append(rec)
        
        return recommendations
    
    def get_next_challenge_level(self, competency_id: str) -> str:
        """
        Determine appropriate difficulty for next challenge.
        Returns: "easy", "medium", "hard", or "expert"
        """
        comp = self._competencies.get(competency_id)
        if not comp:
            return "easy"
        
        # Consider mastery, improvement rate, and consistency
        base_level = comp.mastery_level
        
        # If improving rapidly, increase challenge
        if comp.improvement_rate > 0.1:
            if base_level == MasteryLevel.BEGINNER:
                return "medium"
            elif base_level == MasteryLevel.INTERMEDIATE:
                return "hard"
            elif base_level in [MasteryLevel.ADVANCED, MasteryLevel.EXPERT]:
                return "expert"
        
        # If declining or inconsistent, decrease challenge
        if comp.improvement_rate < -0.1 or comp.consistency < 0.5:
            if base_level == MasteryLevel.EXPERT:
                return "hard"
            elif base_level == MasteryLevel.ADVANCED:
                return "medium"
            else:
                return "easy"
        
        # Normal progression
        level_to_difficulty = {
            MasteryLevel.NOVICE: "easy",
            MasteryLevel.BEGINNER: "easy",
            MasteryLevel.INTERMEDIATE: "medium",
            MasteryLevel.ADVANCED: "hard",
            MasteryLevel.EXPERT: "expert",
        }
        return level_to_difficulty.get(base_level, "medium")
    
    # ==========================================
    # Serialization
    # ==========================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize tracker state"""
        return {
            "user_id": self.user_id,
            "created_at": self._created_at.isoformat(),
            "last_activity": self._last_activity.isoformat(),
            "overall_level": self.get_overall_level().value,
            "competencies": {k: v.to_dict() for k, v in self._competencies.items()},
            "performance_count": len(self._performance_history),
            "error_patterns": self._error_patterns,
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a brief summary of user progress"""
        active_competencies = [c for c in self._competencies.values() if c.total_attempts > 0]
        
        return {
            "user_id": self.user_id,
            "overall_level": self.get_overall_level().value,
            "active_competencies": len(active_competencies),
            "total_competencies": len(self._competencies),
            "total_practice_sessions": len(self._performance_history),
            "knowledge_gaps": len(self.identify_knowledge_gaps()),
            "most_common_errors": self.get_frequent_errors(3),
            "last_activity": self._last_activity.isoformat(),
        }


# ==========================================
# User Tracker Management
# ==========================================

_user_trackers: Dict[str, UserCompetencyTracker] = {}


def get_user_tracker(user_id: str) -> UserCompetencyTracker:
    """Get or create a tracker for a user"""
    if user_id not in _user_trackers:
        _user_trackers[user_id] = UserCompetencyTracker(user_id)
    return _user_trackers[user_id]


def reset_user_tracker(user_id: str):
    """Reset a user's tracker"""
    if user_id in _user_trackers:
        del _user_trackers[user_id]
