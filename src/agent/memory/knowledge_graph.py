"""
knowledge_graph.py - Shared Memory System (Simplified Knowledge Graph)

Estructura ontológica que conecta:
- Parámetros de proceso <-> Resultados de calidad
- Errores comunes <-> Soluciones documentadas
- Competencias del usuario <-> Módulos de aprendizaje
- Historial de decisiones <-> Explicaciones generadas

Este módulo implementa un Knowledge Graph en memoria con persistencia opcional
a Supabase para mantener relaciones entre entidades del sistema.
"""
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class EntityType(Enum):
    """Tipos de entidades en el Knowledge Graph"""
    # Equipamiento
    STATION = "station"
    COBOT = "cobot"
    PLC = "plc"
    SENSOR = "sensor"
    
    # Proceso
    PARAMETER = "parameter"
    QUALITY_RESULT = "quality_result"
    ROUTINE = "routine"
    
    # Problemas y soluciones
    ERROR = "error"
    SYMPTOM = "symptom"
    SOLUTION = "solution"
    MAINTENANCE_ACTION = "maintenance_action"
    
    # Aprendizaje
    COMPETENCY = "competency"
    LEARNING_MODULE = "learning_module"
    EXERCISE = "exercise"
    
    # Historial
    DECISION = "decision"
    EXPLANATION = "explanation"
    USER_ACTION = "user_action"


class RelationType(Enum):
    """Tipos de relaciones entre entidades"""
    # Equipamiento
    BELONGS_TO = "belongs_to"           # sensor -> station
    CONTROLS = "controls"               # plc -> cobot
    MONITORS = "monitors"               # sensor -> parameter
    
    # Proceso
    AFFECTS = "affects"                 # parameter -> quality_result
    PRODUCES = "produces"               # routine -> quality_result
    REQUIRES = "requires"               # routine -> parameter
    
    # Problemas
    CAUSES = "causes"                   # error -> symptom
    INDICATES = "indicates"             # symptom -> error
    SOLVES = "solves"                   # solution -> error
    PREVENTS = "prevents"               # maintenance_action -> error
    
    # Aprendizaje
    TEACHES = "teaches"                 # learning_module -> competency
    ASSESSES = "assesses"               # exercise -> competency
    PREREQUISITE_OF = "prerequisite_of" # competency -> competency
    RECOMMENDED_FOR = "recommended_for" # exercise -> user (based on errors)
    
    # Historial
    TRIGGERED_BY = "triggered_by"       # decision -> user_action
    EXPLAINED_BY = "explained_by"       # decision -> explanation
    RESULTED_IN = "resulted_in"         # decision -> quality_result


@dataclass
class Entity:
    """Una entidad en el Knowledge Graph"""
    id: str
    type: EntityType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class Relation:
    """Una relación entre dos entidades"""
    source_id: str
    target_id: str
    type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0  # Strength/confidence of relation
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type.value,
            "properties": self.properties,
            "weight": self.weight,
            "created_at": self.created_at.isoformat(),
        }


class KnowledgeGraph:
    """
    In-memory Knowledge Graph with optional persistence.
    
    Provides:
    - Entity and relation management
    - Graph traversal and querying
    - Pattern matching for recommendations
    - Integration with learning system
    """
    
    def __init__(self):
        self._entities: Dict[str, Entity] = {}
        self._relations: List[Relation] = []
        self._adjacency: Dict[str, Set[str]] = {}  # source_id -> set of target_ids
        self._reverse_adjacency: Dict[str, Set[str]] = {}  # target_id -> set of source_ids
        
        # Initialize with base knowledge
        self._initialize_base_knowledge()
    
    def _initialize_base_knowledge(self):
        """Initialize the graph with base domain knowledge"""
        # Cargar conocimiento del laboratorio si está disponible
        self._load_lab_knowledge()
        
        # Common errors and solutions (fallback si no hay lab_knowledge)
        if not self.get_entities_by_type(EntityType.ERROR):
            self._load_default_errors()
        
        # Competencies (siempre se cargan)
        self._load_competencies()
    
    def _load_lab_knowledge(self):
        """Load knowledge from lab_knowledge.py into the graph"""
        try:
            # Import usando path relativo para evitar circular imports
            import importlib.util
            import os
            
            # Construir path al archivo
            current_dir = os.path.dirname(os.path.abspath(__file__))
            lab_knowledge_path = os.path.join(current_dir, '..', 'knowledge', 'lab_knowledge.py')
            lab_knowledge_path = os.path.normpath(lab_knowledge_path)
            
            if not os.path.exists(lab_knowledge_path):
                print(f"Warning: lab_knowledge.py not found at {lab_knowledge_path}")
                return
            
            # Cargar módulo dinámicamente
            spec = importlib.util.spec_from_file_location("lab_knowledge", lab_knowledge_path)
            lab_knowledge = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(lab_knowledge)
            
            # Extraer datos
            STATIONS = getattr(lab_knowledge, 'STATIONS', {})
            ROBOTS = getattr(lab_knowledge, 'ROBOTS', {})
            EQUIPMENT_INFO = getattr(lab_knowledge, 'EQUIPMENT_INFO', {})
            COMMON_ERRORS = getattr(lab_knowledge, 'COMMON_ERRORS', {})
            TERMINOLOGY = getattr(lab_knowledge, 'TERMINOLOGY', {})
            
            # 1. Cargar ESTACIONES
            for station_num, station_info in STATIONS.items():
                station_id = f"station_{station_num}"
                self.add_entity(Entity(
                    id=station_id,
                    type=EntityType.STATION,
                    name=f"Station {station_num}: {station_info.get('nombre', '')}",
                    properties={
                        "station_number": station_num,
                        "description": station_info.get("descripcion", ""),
                        "equipment": station_info.get("equipos", []),
                        "operations": station_info.get("operaciones", []),
                    }
                ))
            
            # 2. Cargar ROBOTS/COBOTS
            for robot_name, robot_info in ROBOTS.items():
                robot_id = f"cobot_{robot_name.lower()}"
                self.add_entity(Entity(
                    id=robot_id,
                    type=EntityType.COBOT,
                    name=robot_info.get("nombre_completo", robot_name),
                    properties={
                        "model": robot_info.get("modelo", ""),
                        "manufacturer": robot_info.get("fabricante", ""),
                        "location": robot_info.get("ubicacion", ""),
                        "capabilities": robot_info.get("capacidades", []),
                        "routines": robot_info.get("rutinas_disponibles", {}),
                        "description": robot_info.get("descripcion", ""),
                    }
                ))
                
                # Conectar robot a su estación si se menciona
                location = robot_info.get("ubicacion", "").lower()
                for i in range(1, 7):
                    if f"estación {i}" in location or f"station {i}" in location:
                        self.add_relation(Relation(
                            source_id=robot_id,
                            target_id=f"station_{i}",
                            type=RelationType.BELONGS_TO
                        ))
                        break
            
            # 3. Cargar ERRORES COMUNES con sus soluciones
            for error_code, error_info in COMMON_ERRORS.items():
                error_id = f"err_{error_code.lower()}"
                self.add_entity(Entity(
                    id=error_id,
                    type=EntityType.ERROR,
                    name=error_info.get("descripcion", error_code),
                    properties={
                        "code": error_code,
                        "causes": error_info.get("causas", []),
                        "category": self._infer_error_category(error_code),
                    }
                ))
                
                # Crear solución asociada
                solution_id = f"sol_{error_code.lower()}"
                self.add_entity(Entity(
                    id=solution_id,
                    type=EntityType.SOLUTION,
                    name=error_info.get("solucion", ""),
                    properties={
                        "target_error": error_id,
                        "causes_addressed": error_info.get("causas", []),
                    }
                ))
                
                # Conectar solución con error
                self.add_relation(Relation(
                    source_id=solution_id,
                    target_id=error_id,
                    type=RelationType.SOLVES,
                    weight=0.9
                ))
            
            # 4. Cargar TERMINOLOGÍA como entidades de conocimiento
            for term, definition in TERMINOLOGY.items():
                term_id = f"term_{term.lower().replace(' ', '_')}"
                self.add_entity(Entity(
                    id=term_id,
                    type=EntityType.PARAMETER,  # Usamos PARAMETER para términos
                    name=term,
                    properties={
                        "definition": definition,
                        "type": "terminology",
                    }
                ))
            
            # 5. Cargar EQUIPMENT_INFO
            for equip_type, equip_info in EQUIPMENT_INFO.items():
                equip_id = f"equip_{equip_type.lower().replace(' ', '_')}"
                entity_type = EntityType.PLC if "PLC" in equip_type.upper() else EntityType.COBOT if "Cobot" in equip_type else EntityType.SENSOR
                
                self.add_entity(Entity(
                    id=equip_id,
                    type=entity_type,
                    name=equip_type,
                    properties={
                        "manufacturer": equip_info.get("fabricante", ""),
                        "models": equip_info.get("modelos", []),
                        "software": equip_info.get("software", ""),
                        "protocols": equip_info.get("protocolos", []),
                        "description": equip_info.get("descripcion", equip_info.get("funcion", "")),
                    }
                ))
                
        except Exception as e:
            print(f"Warning: Could not load lab_knowledge: {e}")
    
    def _infer_error_category(self, error_code: str) -> str:
        """Infer error category from error code"""
        code_upper = error_code.upper()
        if "DOOR" in code_upper or "ESTOP" in code_upper or "SAFETY" in code_upper:
            return "safety"
        elif "CONN" in code_upper or "TIMEOUT" in code_upper:
            return "communication"
        elif "COBOT" in code_upper or "ROBOT" in code_upper:
            return "operation"
        elif "TEMP" in code_upper or "SENSOR" in code_upper:
            return "process"
        return "general"
    
    def _load_default_errors(self):
        """Load default errors if lab_knowledge is not available"""
        common_errors = [
            ("err_door_open", "Safety Door Open", {"severity": "blocker", "category": "safety"}),
            ("err_estop", "Emergency Stop Active", {"severity": "critical", "category": "safety"}),
            ("err_plc_disconnect", "PLC Disconnected", {"severity": "high", "category": "communication"}),
            ("err_cobot_fault", "Cobot Protective Stop", {"severity": "medium", "category": "operation"}),
            ("err_sensor_timeout", "Sensor Timeout", {"severity": "medium", "category": "communication"}),
            ("err_temp_out_of_range", "Temperature Out of Range", {"severity": "high", "category": "process"}),
        ]
        
        for err_id, err_name, props in common_errors:
            self.add_entity(Entity(
                id=err_id,
                type=EntityType.ERROR,
                name=err_name,
                properties=props
            ))
        
        # Solutions for common errors
        solutions = [
            ("sol_close_doors", "Close all safety doors", "err_door_open"),
            ("sol_reset_estop", "Reset emergency stop button", "err_estop"),
            ("sol_reconnect_plc", "Reconnect PLC and verify network", "err_plc_disconnect"),
            ("sol_clear_fault", "Clear fault and restart cobot", "err_cobot_fault"),
            ("sol_check_sensor", "Check sensor wiring and replace if needed", "err_sensor_timeout"),
            ("sol_adjust_pid", "Adjust PID parameters for temperature control", "err_temp_out_of_range"),
        ]
        
        for sol_id, sol_name, error_id in solutions:
            self.add_entity(Entity(
                id=sol_id,
                type=EntityType.SOLUTION,
                name=sol_name,
                properties={"target_error": error_id}
            ))
            self.add_relation(Relation(
                source_id=sol_id,
                target_id=error_id,
                type=RelationType.SOLVES,
                weight=0.9
            ))
    
    def _load_competencies(self):
        competencies = [
            ("comp_safety_basics", "Safety Basics", {"level": 1, "category": "safety"}),
            ("comp_cobot_operation", "Cobot Basic Operation", {"level": 1, "category": "robotics"}),
            ("comp_plc_basics", "PLC Fundamentals", {"level": 1, "category": "automation"}),
            ("comp_troubleshooting", "Basic Troubleshooting", {"level": 2, "category": "maintenance"}),
            ("comp_pid_tuning", "PID Controller Tuning", {"level": 2, "category": "control"}),
            ("comp_urscript", "URScript Programming", {"level": 3, "category": "robotics"}),
            ("comp_ladder_logic", "Ladder Logic Programming", {"level": 2, "category": "automation"}),
            ("comp_vision_systems", "Vision System Integration", {"level": 3, "category": "vision"}),
        ]
        
        for comp_id, comp_name, props in competencies:
            self.add_entity(Entity(
                id=comp_id,
                type=EntityType.COMPETENCY,
                name=comp_name,
                properties=props
            ))
        
        # Competency prerequisites
        prerequisites = [
            ("comp_troubleshooting", "comp_safety_basics"),
            ("comp_troubleshooting", "comp_cobot_operation"),
            ("comp_pid_tuning", "comp_plc_basics"),
            ("comp_urscript", "comp_cobot_operation"),
            ("comp_ladder_logic", "comp_plc_basics"),
            ("comp_vision_systems", "comp_cobot_operation"),
        ]
        
        for comp_id, prereq_id in prerequisites:
            self.add_relation(Relation(
                source_id=prereq_id,
                target_id=comp_id,
                type=RelationType.PREREQUISITE_OF
            ))
    
    # ==========================================
    # Entity Management
    # ==========================================
    
    def add_entity(self, entity: Entity) -> bool:
        """Add an entity to the graph"""
        if entity.id in self._entities:
            return False
        self._entities[entity.id] = entity
        self._adjacency[entity.id] = set()
        self._reverse_adjacency[entity.id] = set()
        return True
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID"""
        return self._entities.get(entity_id)
    
    def update_entity(self, entity_id: str, properties: Dict[str, Any]) -> bool:
        """Update entity properties"""
        if entity_id not in self._entities:
            return False
        entity = self._entities[entity_id]
        entity.properties.update(properties)
        entity.updated_at = datetime.utcnow()
        return True
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of a specific type"""
        return [e for e in self._entities.values() if e.type == entity_type]
    
    # ==========================================
    # Relation Management
    # ==========================================
    
    def add_relation(self, relation: Relation) -> bool:
        """Add a relation between two entities"""
        if relation.source_id not in self._entities or relation.target_id not in self._entities:
            return False
        
        self._relations.append(relation)
        self._adjacency[relation.source_id].add(relation.target_id)
        self._reverse_adjacency[relation.target_id].add(relation.source_id)
        return True
    
    def get_relations(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relation_type: Optional[RelationType] = None
    ) -> List[Relation]:
        """Query relations with optional filters"""
        results = []
        for rel in self._relations:
            if source_id and rel.source_id != source_id:
                continue
            if target_id and rel.target_id != target_id:
                continue
            if relation_type and rel.type != relation_type:
                continue
            results.append(rel)
        return results
    
    # ==========================================
    # Graph Traversal
    # ==========================================
    
    def get_neighbors(self, entity_id: str, direction: str = "outgoing") -> List[Entity]:
        """Get neighboring entities"""
        if direction == "outgoing":
            neighbor_ids = self._adjacency.get(entity_id, set())
        elif direction == "incoming":
            neighbor_ids = self._reverse_adjacency.get(entity_id, set())
        else:  # both
            neighbor_ids = self._adjacency.get(entity_id, set()) | self._reverse_adjacency.get(entity_id, set())
        
        return [self._entities[nid] for nid in neighbor_ids if nid in self._entities]
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5
    ) -> Optional[List[str]]:
        """Find a path between two entities using BFS"""
        if source_id not in self._entities or target_id not in self._entities:
            return None
        
        if source_id == target_id:
            return [source_id]
        
        visited = {source_id}
        queue = [(source_id, [source_id])]
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            for neighbor_id in self._adjacency.get(current, set()):
                if neighbor_id == target_id:
                    return path + [neighbor_id]
                
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))
        
        return None
    
    # ==========================================
    # Domain-Specific Queries
    # ==========================================
    
    def get_solutions_for_error(self, error_id: str) -> List[Entity]:
        """Get all solutions for a specific error"""
        solutions = []
        for rel in self._relations:
            if rel.target_id == error_id and rel.type == RelationType.SOLVES:
                solution = self.get_entity(rel.source_id)
                if solution:
                    solutions.append(solution)
        return solutions
    
    def get_prerequisites_for_competency(self, competency_id: str) -> List[Entity]:
        """Get prerequisite competencies for a given competency"""
        prerequisites = []
        for rel in self._relations:
            if rel.target_id == competency_id and rel.type == RelationType.PREREQUISITE_OF:
                prereq = self.get_entity(rel.source_id)
                if prereq:
                    prerequisites.append(prereq)
        return prerequisites
    
    def recommend_learning_path(
        self,
        target_competency_id: str,
        current_competencies: Set[str]
    ) -> List[Entity]:
        """
        Recommend a learning path to achieve a target competency.
        Returns ordered list of competencies to learn.
        """
        if target_competency_id in current_competencies:
            return []
        
        # Get all prerequisites recursively
        def get_all_prerequisites(comp_id: str, visited: Set[str]) -> List[str]:
            if comp_id in visited:
                return []
            visited.add(comp_id)
            
            prereqs = []
            for prereq in self.get_prerequisites_for_competency(comp_id):
                if prereq.id not in current_competencies:
                    prereqs.extend(get_all_prerequisites(prereq.id, visited))
                    prereqs.append(prereq.id)
            return prereqs
        
        path_ids = get_all_prerequisites(target_competency_id, set())
        path_ids.append(target_competency_id)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_path = []
        for pid in path_ids:
            if pid not in seen and pid not in current_competencies:
                seen.add(pid)
                unique_path.append(pid)
        
        return [self.get_entity(pid) for pid in unique_path if self.get_entity(pid)]
    
    def get_related_errors(self, symptom_description: str) -> List[Tuple[Entity, float]]:
        """
        Find errors related to a symptom description.
        Returns list of (error, relevance_score) tuples.
        """
        # Simple keyword matching - can be enhanced with embeddings
        keywords = symptom_description.lower().split()
        
        errors = self.get_entities_by_type(EntityType.ERROR)
        scored_errors = []
        
        for error in errors:
            score = 0.0
            error_text = f"{error.name} {json.dumps(error.properties)}".lower()
            
            for keyword in keywords:
                if keyword in error_text:
                    score += 1.0
            
            if score > 0:
                scored_errors.append((error, score / len(keywords)))
        
        # Sort by score descending
        scored_errors.sort(key=lambda x: x[1], reverse=True)
        return scored_errors[:5]
    
    # ==========================================
    # Decision Logging
    # ==========================================
    
    def log_decision(
        self,
        decision_id: str,
        decision_type: str,
        context: Dict[str, Any],
        outcome: str,
        explanation: str,
        confidence: float
    ) -> Entity:
        """Log a system decision for traceability"""
        decision = Entity(
            id=decision_id,
            type=EntityType.DECISION,
            name=f"Decision: {decision_type}",
            properties={
                "decision_type": decision_type,
                "context": context,
                "outcome": outcome,
                "confidence": confidence,
            }
        )
        self.add_entity(decision)
        
        # Add explanation
        explanation_entity = Entity(
            id=f"exp_{decision_id}",
            type=EntityType.EXPLANATION,
            name=f"Explanation for {decision_id}",
            properties={"text": explanation}
        )
        self.add_entity(explanation_entity)
        
        self.add_relation(Relation(
            source_id=decision_id,
            target_id=f"exp_{decision_id}",
            type=RelationType.EXPLAINED_BY
        ))
        
        return decision
    
    # ==========================================
    # Serialization
    # ==========================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the knowledge graph"""
        return {
            "entities": {eid: e.to_dict() for eid, e in self._entities.items()},
            "relations": [r.to_dict() for r in self._relations],
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        entity_counts = {}
        for entity in self._entities.values():
            entity_type = entity.type.value
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        relation_counts = {}
        for relation in self._relations:
            rel_type = relation.type.value
            relation_counts[rel_type] = relation_counts.get(rel_type, 0) + 1
        
        return {
            "total_entities": len(self._entities),
            "total_relations": len(self._relations),
            "entity_counts": entity_counts,
            "relation_counts": relation_counts,
        }


# Singleton instance
_knowledge_graph: Optional[KnowledgeGraph] = None


def get_knowledge_graph() -> KnowledgeGraph:
    """Get the singleton KnowledgeGraph instance"""
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeGraph()
    return _knowledge_graph


def reset_knowledge_graph():
    """Reset the knowledge graph (useful for testing)"""
    global _knowledge_graph
    _knowledge_graph = None
