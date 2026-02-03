"""
memory - Shared Memory System (Knowledge Graph)

This module provides the knowledge management backbone for the multi-agent system:
- Knowledge Graph for connecting domain entities and relationships
- User Competency Tracking for monitoring learning progress
- Decision logging for traceability
- Helper functions for easy integration with workers

Components:
- knowledge_graph: Domain knowledge representation and querying
- user_competency: User skill tracking and learning recommendations
- knowledge_helpers: Easy-to-use functions for workers
"""

from .knowledge_graph import (
    KnowledgeGraph,
    get_knowledge_graph,
    reset_knowledge_graph,
    Entity,
    Relation,
    EntityType,
    RelationType,
)

from .user_competency import (
    UserCompetencyTracker,
    get_user_tracker,
    reset_user_tracker,
    CompetencyState,
    PerformanceRecord,
    MasteryLevel,
)

from .knowledge_helpers import (
    # Troubleshooting
    find_solution_for_error,
    get_station_knowledge,
    get_terminology,
    log_agent_decision,
    # Pedagogical
    get_user_level,
    get_adaptive_support_level,
    get_hint_for_user,
    record_user_performance,
    get_learning_recommendations,
    get_learning_path,
    # Quick access
    get_kg_stats,
    search_knowledge,
)

__all__ = [
    # Knowledge Graph
    "KnowledgeGraph",
    "get_knowledge_graph",
    "reset_knowledge_graph",
    "Entity",
    "Relation",
    "EntityType",
    "RelationType",
    # User Competency
    "UserCompetencyTracker",
    "get_user_tracker",
    "reset_user_tracker",
    "CompetencyState",
    "PerformanceRecord",
    "MasteryLevel",
    # Helpers - Troubleshooting
    "find_solution_for_error",
    "get_station_knowledge",
    "get_terminology",
    "log_agent_decision",
    # Helpers - Pedagogical
    "get_user_level",
    "get_adaptive_support_level",
    "get_hint_for_user",
    "record_user_performance",
    "get_learning_recommendations",
    "get_learning_path",
    # Helpers - Quick access
    "get_kg_stats",
    "search_knowledge",
]
