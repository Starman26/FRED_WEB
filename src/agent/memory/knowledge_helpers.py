"""
knowledge_helpers.py - Helper functions to use Knowledge Graph in workers

Provides easy-to-use functions for:
- Looking up solutions for errors
- Finding related information
- Getting recommendations
- Logging decisions
"""
from typing import Dict, List, Any, Optional, Tuple

from src.agent.memory.knowledge_graph import (
    get_knowledge_graph,
    EntityType,
    RelationType,
)
from src.agent.memory.user_competency import (
    get_user_tracker,
    MasteryLevel,
)
from src.agent.pedagogy.scaffolding import (
    get_scaffolder,
    ScaffoldingContext,
)


# ==========================================
# TROUBLESHOOTING HELPERS
# ==========================================

def find_solution_for_error(error_description: str) -> Dict[str, Any]:
    """
    Find solutions for an error based on description.
    
    Args:
        error_description: Text describing the error (e.g., "PLC not connecting")
    
    Returns:
        Dict with matched errors and their solutions
    """
    kg = get_knowledge_graph()
    
    # Find related errors
    related_errors = kg.get_related_errors(error_description)
    
    results = {
        "query": error_description,
        "matches": [],
    }
    
    for error_entity, relevance in related_errors[:3]:  # Top 3 matches
        solutions = kg.get_solutions_for_error(error_entity.id)
        
        match = {
            "error_id": error_entity.id,
            "error_name": error_entity.name,
            "relevance": relevance,
            "causes": error_entity.properties.get("causes", []),
            "solutions": [
                {
                    "id": sol.id,
                    "action": sol.name,
                    "details": sol.properties,
                }
                for sol in solutions
            ],
        }
        results["matches"].append(match)
    
    return results


def get_station_knowledge(station_number: int) -> Dict[str, Any]:
    """
    Get all knowledge related to a station.
    
    Args:
        station_number: Station number (1-6)
    
    Returns:
        Dict with station info, equipment, and common issues
    """
    kg = get_knowledge_graph()
    station_id = f"station_{station_number}"
    
    station = kg.get_entity(station_id)
    if not station:
        return {"error": f"Station {station_number} not found"}
    
    # Get related equipment (cobots, PLCs connected to this station)
    related_equipment = []
    for rel in kg.get_relations(target_id=station_id, relation_type=RelationType.BELONGS_TO):
        equipment = kg.get_entity(rel.source_id)
        if equipment:
            related_equipment.append({
                "id": equipment.id,
                "name": equipment.name,
                "type": equipment.type.value,
                "properties": equipment.properties,
            })
    
    return {
        "station_id": station_id,
        "name": station.name,
        "description": station.properties.get("description", ""),
        "equipment_list": station.properties.get("equipment", []),
        "operations": station.properties.get("operations", []),
        "connected_equipment": related_equipment,
    }


def get_terminology(term: str) -> Optional[str]:
    """
    Look up a term in the knowledge graph.
    
    Args:
        term: Term to look up
    
    Returns:
        Definition string or None
    """
    kg = get_knowledge_graph()
    
    # Search for term in terminology entities
    term_lower = term.lower().replace(" ", "_")
    term_id = f"term_{term_lower}"
    
    entity = kg.get_entity(term_id)
    if entity:
        return entity.properties.get("definition", entity.name)
    
    # Fuzzy search in all terminology
    for entity in kg.get_entities_by_type(EntityType.PARAMETER):
        if entity.properties.get("type") == "terminology":
            if term.lower() in entity.name.lower():
                return f"{entity.name}: {entity.properties.get('definition', '')}"
    
    return None


def log_agent_decision(
    decision_id: str,
    decision_type: str,
    context: Dict[str, Any],
    outcome: str,
    explanation: str,
    confidence: float = 0.8
) -> None:
    """
    Log an agent decision for traceability.
    
    Args:
        decision_id: Unique ID for this decision
        decision_type: Type (e.g., "start_cobot", "diagnose_error")
        context: Context dict with relevant info
        outcome: What happened
        explanation: Why this decision was made
        confidence: Confidence level 0-1
    """
    kg = get_knowledge_graph()
    kg.log_decision(
        decision_id=decision_id,
        decision_type=decision_type,
        context=context,
        outcome=outcome,
        explanation=explanation,
        confidence=confidence
    )


# ==========================================
# PEDAGOGICAL HELPERS
# ==========================================

def get_user_level(user_id: str, competency_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get user's competency level.
    
    Args:
        user_id: User identifier
        competency_id: Specific competency to check (or None for overall)
    
    Returns:
        Dict with level info
    """
    tracker = get_user_tracker(user_id)
    
    if competency_id:
        comp = tracker.get_competency(competency_id)
        if comp:
            return {
                "competency_id": competency_id,
                "mastery_level": comp.mastery_level.value,
                "mastery_score": comp.mastery_score,
                "total_attempts": comp.total_attempts,
                "improvement_rate": comp.improvement_rate,
            }
        return {"error": f"Competency {competency_id} not found"}
    
    return tracker.get_summary()


def get_adaptive_support_level(
    user_id: str,
    competency_id: str,
    task_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Determine how much support/detail to provide.
    
    Args:
        user_id: User identifier
        competency_id: Competency being used
        task_context: Optional context (errors made, time spent, etc.)
    
    Returns:
        Support level: "full", "high", "medium", "low", "none"
    """
    tracker = get_user_tracker(user_id)
    scaffolder = get_scaffolder()
    
    context = None
    if task_context:
        context = ScaffoldingContext(
            user_id=user_id,
            task_type=task_context.get("task_type", "general"),
            competency_id=competency_id,
            errors_in_task=task_context.get("errors", []),
            hints_given=task_context.get("hints_used", []),
            time_on_task_seconds=task_context.get("time_spent", 0),
        )
    
    support = scaffolder.determine_support_level(tracker, competency_id, context)
    return support.value


def get_hint_for_user(
    user_id: str,
    competency_id: str,
    hints_already_given: List[str] = None,
    error_type: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get next appropriate hint for the user.
    
    Args:
        user_id: User identifier
        competency_id: Competency area
        hints_already_given: List of hint IDs already shown
        error_type: Specific error if applicable
    
    Returns:
        Hint dict or None
    """
    scaffolder = get_scaffolder()
    
    context = ScaffoldingContext(
        user_id=user_id,
        task_type="troubleshooting" if error_type else "learning",
        competency_id=competency_id,
        hints_given=hints_already_given or [],
    )
    
    hint = scaffolder.get_next_hint(competency_id, context, error_type)
    
    if hint:
        return hint.to_dict()
    return None


def record_user_performance(
    user_id: str,
    task_id: str,
    task_type: str,
    competency_id: str,
    success: bool,
    score: float,
    time_seconds: float,
    errors: List[str] = None,
    hints_used: int = 0
) -> Dict[str, Any]:
    """
    Record a user's performance on a task.
    
    Args:
        user_id: User identifier
        task_id: Unique task ID
        task_type: Type of task
        competency_id: Competency being assessed
        success: Did they succeed?
        score: Score 0-1
        time_seconds: Time taken
        errors: List of error types made
        hints_used: Number of hints used
    
    Returns:
        Updated competency summary
    """
    tracker = get_user_tracker(user_id)
    
    record = tracker.record_performance(
        task_id=task_id,
        task_type=task_type,
        competency_id=competency_id,
        success=success,
        score=score,
        time_taken_seconds=time_seconds,
        errors_made=errors or [],
        hints_used=hints_used
    )
    
    # Return updated state
    comp = tracker.get_competency(competency_id)
    return {
        "recorded": True,
        "new_mastery_level": comp.mastery_level.value if comp else "unknown",
        "new_mastery_score": comp.mastery_score if comp else 0,
        "improvement_rate": comp.improvement_rate if comp else 0,
    }


def get_learning_recommendations(user_id: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Get learning recommendations based on knowledge gaps.
    
    Args:
        user_id: User identifier
        limit: Max recommendations
    
    Returns:
        List of recommendation dicts
    """
    tracker = get_user_tracker(user_id)
    return tracker.recommend_focus_areas(limit)


def get_learning_path(
    user_id: str,
    target_competency: str
) -> List[Dict[str, Any]]:
    """
    Get recommended learning path to a target competency.
    
    Args:
        user_id: User identifier
        target_competency: Competency ID to achieve
    
    Returns:
        Ordered list of competencies to learn
    """
    tracker = get_user_tracker(user_id)
    kg = get_knowledge_graph()
    
    # Get current competencies (those with intermediate+ mastery)
    current = set()
    for comp_id, comp in tracker.get_all_competencies().items():
        if comp.mastery_score >= 0.5:
            current.add(comp_id)
    
    path = kg.recommend_learning_path(target_competency, current)
    
    return [
        {
            "competency_id": entity.id,
            "name": entity.name,
            "level": entity.properties.get("level", 1),
            "category": entity.properties.get("category", "general"),
        }
        for entity in path
    ]


# ==========================================
# QUICK ACCESS
# ==========================================

def get_kg_stats() -> Dict[str, Any]:
    """Get knowledge graph statistics"""
    kg = get_knowledge_graph()
    return kg.get_stats()


def search_knowledge(query: str, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Search the knowledge graph.
    
    Args:
        query: Search text
        entity_type: Optional filter by type
    
    Returns:
        List of matching entities
    """
    kg = get_knowledge_graph()
    query_lower = query.lower()
    
    results = []
    
    for entity in kg._entities.values():
        # Filter by type if specified
        if entity_type and entity.type.value != entity_type:
            continue
        
        # Search in name and properties
        searchable = f"{entity.name} {str(entity.properties)}".lower()
        if query_lower in searchable:
            results.append({
                "id": entity.id,
                "type": entity.type.value,
                "name": entity.name,
                "properties": entity.properties,
            })
    
    return results[:10]  # Limit results
