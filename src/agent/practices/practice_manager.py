"""
practice_manager.py - Gestión de sesiones de práctica

Maneja:
- Cargar prácticas desde YAML
- Crear/recuperar sesiones de usuario
- Tracking de progreso
- Generación de resúmenes finales
"""
import os
import json
import uuid
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from .practice_schema import (
    Practice,
    PracticeSession,
    PracticeStep,
    PracticeSection,
    StepType,
    load_practice_from_file,
)

# Intentar importar Supabase
try:
    from src.agent.services import supabase_client
    SUPABASE_AVAILABLE = supabase_client is not None
except ImportError:
    SUPABASE_AVAILABLE = False
    supabase_client = None


class PracticeManager:
    """
    Gestiona prácticas y sesiones de usuario.
    
    Uso:
        manager = PracticeManager()
        practice = manager.get_practice("abb_pick_and_place")
        session = manager.get_or_create_session(user_id, practice.id)
        
        # Usuario completa un paso
        manager.complete_step(session, "step_1", response="MoveJ", score=1.0)
        
        # Al finalizar
        summary = manager.complete_practice(session)
    """
    
    def __init__(self, practices_dir: Optional[str] = None):
        """
        Args:
            practices_dir: Directorio donde están los YAML de prácticas
        """
        if practices_dir is None:
            # Directorio por defecto: mismo que este archivo
            practices_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.practices_dir = practices_dir
        self._practices_cache: Dict[str, Practice] = {}
        self._sessions_cache: Dict[str, PracticeSession] = {}
        
        # Cargar prácticas disponibles
        self._load_available_practices()
    
    def _load_available_practices(self):
        """Carga todas las prácticas YAML del directorio"""
        if not os.path.exists(self.practices_dir):
            return
        
        for filename in os.listdir(self.practices_dir):
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                filepath = os.path.join(self.practices_dir, filename)
                try:
                    practice = load_practice_from_file(filepath)
                    self._practices_cache[practice.id] = practice
                except Exception as e:
                    print(f"Warning: Could not load practice from {filename}: {e}")
    
    def get_available_practices(self) -> List[Dict[str, Any]]:
        """Retorna lista de prácticas disponibles"""
        return [
            {
                "id": p.id,
                "title": p.title,
                "description": p.description,
                "difficulty": p.difficulty_level,
                "estimated_time": p.estimated_total_time_minutes,
                "prerequisites": p.prerequisites,
            }
            for p in self._practices_cache.values()
        ]
    
    def get_practice(self, practice_id: str) -> Optional[Practice]:
        """Obtiene una práctica por ID"""
        return self._practices_cache.get(practice_id)
    
    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================
    
    def get_or_create_session(
        self, 
        user_id: str, 
        practice_id: str
    ) -> Optional[PracticeSession]:
        """
        Obtiene sesión activa o crea una nueva.
        Busca primero en Supabase, luego en cache local.
        """
        practice = self.get_practice(practice_id)
        if not practice:
            return None
        
        # Intentar cargar de Supabase
        if SUPABASE_AVAILABLE:
            session = self._load_session_from_db(user_id, practice_id)
            if session:
                return session
        
        # Buscar en cache local
        cache_key = f"{user_id}:{practice_id}"
        if cache_key in self._sessions_cache:
            return self._sessions_cache[cache_key]
        
        # Crear nueva sesión
        session = PracticeSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            practice_id=practice_id,
            current_section_id=practice.sections[0].id if practice.sections else None,
            current_step_id=practice.sections[0].steps[0].id if practice.sections and practice.sections[0].steps else None,
            started_at=datetime.utcnow().isoformat(),
        )
        
        # Guardar en DB y cache
        self._save_session_to_db(session)
        self._sessions_cache[cache_key] = session
        
        return session
    
    def _load_session_from_db(
        self, 
        user_id: str, 
        practice_id: str
    ) -> Optional[PracticeSession]:
        """Carga sesión activa desde Supabase"""
        if not SUPABASE_AVAILABLE:
            return None
        
        try:
            result = supabase_client.table("lab_practice_sessions").select("*").eq(
                "user_id", user_id
            ).eq(
                "practice_id", practice_id
            ).eq(
                "status", "in_progress"
            ).order(
                "started_at", desc=True
            ).limit(1).execute()
            
            if result.data and len(result.data) > 0:
                row = result.data[0]
                return PracticeSession(
                    session_id=row["id"],
                    user_id=row["user_id"],
                    practice_id=row["practice_id"],
                    current_section_id=row.get("current_section_id"),
                    current_step_id=row.get("current_step_id"),
                    completed_step_ids=row.get("completed_steps", []),
                    responses=row.get("responses", {}),
                    scores=row.get("scores", {}),
                    started_at=row.get("started_at"),
                    last_activity_at=row.get("last_activity_at"),
                    hints_requested=row.get("hints_requested", 0),
                    mistakes_made=row.get("mistakes_made", 0),
                )
        except Exception as e:
            print(f"Error loading session from DB: {e}")
        
        return None
    
    def _save_session_to_db(self, session: PracticeSession) -> bool:
        """Guarda sesión en Supabase"""
        if not SUPABASE_AVAILABLE:
            return False
        
        try:
            data = {
                "id": session.session_id,
                "user_id": session.user_id,
                "practice_id": session.practice_id,
                "current_section_id": session.current_section_id,
                "current_step_id": session.current_step_id,
                "completed_steps": session.completed_step_ids,
                "responses": session.responses,
                "scores": session.scores,
                "hints_requested": session.hints_requested,
                "mistakes_made": session.mistakes_made,
                "last_activity_at": datetime.utcnow().isoformat(),
            }
            
            supabase_client.table("lab_practice_sessions").upsert(data).execute()
            return True
        except Exception as e:
            print(f"Error saving session to DB: {e}")
            return False
    
    # =========================================================================
    # STEP PROGRESSION
    # =========================================================================
    
    def get_current_step(self, session: PracticeSession) -> Optional[Tuple[PracticeSection, PracticeStep]]:
        """Obtiene la sección y paso actual"""
        practice = self.get_practice(session.practice_id)
        if not practice or not session.current_step_id:
            return None
        
        step = practice.get_step_by_id(session.current_step_id)
        section = practice.get_section_for_step(session.current_step_id)
        
        if step and section:
            return (section, step)
        return None
    
    def complete_step(
        self,
        session: PracticeSession,
        step_id: str,
        response: Any = None,
        score: float = 1.0,
        time_seconds: float = 0
    ) -> Optional[PracticeStep]:
        """
        Marca un paso como completado y avanza al siguiente.
        
        Returns:
            El siguiente paso, o None si la práctica terminó
        """
        practice = self.get_practice(session.practice_id)
        if not practice:
            return None
        
        # Marcar completado
        session.mark_step_complete(step_id, response, score)
        session.time_per_step[step_id] = time_seconds
        
        # Obtener siguiente paso
        next_step = practice.get_next_step(step_id)
        
        if next_step:
            session.current_step_id = next_step.id
            session.current_section_id = practice.get_section_for_step(next_step.id).id
        
        # Guardar progreso
        self._save_session_to_db(session)
        
        return next_step
    
    def record_hint_request(self, session: PracticeSession):
        """Registra que el usuario pidió una pista"""
        session.hints_requested += 1
        self._save_session_to_db(session)
    
    def record_mistake(self, session: PracticeSession):
        """Registra un error del usuario"""
        session.mistakes_made += 1
        self._save_session_to_db(session)
    
    # =========================================================================
    # PRACTICE COMPLETION
    # =========================================================================
    
    def complete_practice(
        self,
        session: PracticeSession,
        additional_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Completa la práctica y genera resumen final.
        
        Returns:
            Resumen con fortalezas, áreas de mejora, recomendaciones
        """
        practice = self.get_practice(session.practice_id)
        if not practice:
            return {"error": "Practice not found"}
        
        # Calcular métricas
        total_steps = practice.get_total_steps()
        completed_steps = len(session.completed_step_ids)
        completion_rate = completed_steps / total_steps if total_steps > 0 else 0
        avg_score = session.get_overall_score()
        
        # Analizar respuestas para identificar patrones
        strengths = []
        areas_to_improve = []
        
        # Analizar por sección
        for section in practice.sections:
            section_scores = []
            for step in section.steps:
                if step.id in session.scores:
                    section_scores.append(session.scores[step.id])
            
            if section_scores:
                section_avg = sum(section_scores) / len(section_scores)
                if section_avg >= 0.8:
                    strengths.append(section.title)
                elif section_avg < 0.6:
                    areas_to_improve.append(section.title)
        
        # Generar recomendaciones basadas en métricas
        recommendations = []
        
        if session.hints_requested > 5:
            recommendations.append("Consider reviewing the theoretical concepts before the next practice")
        
        if session.mistakes_made > 3:
            recommendations.append("Practice the jogging exercises more to build confidence")
        
        if avg_score < 0.7:
            recommendations.append("Review the sections marked as 'areas to improve' before attempting advanced practices")
        elif avg_score >= 0.9:
            recommendations.append("Excellent work! You're ready for more advanced practices")
        
        # Construir resumen
        summary = {
            "practice_id": practice.id,
            "practice_title": practice.title,
            "completion_rate": round(completion_rate * 100, 1),
            "average_score": round(avg_score * 100, 1),
            "total_time_minutes": sum(session.time_per_step.values()) / 60,
            "hints_requested": session.hints_requested,
            "mistakes_made": session.mistakes_made,
            "strengths": strengths if strengths else ["Completed all steps"],
            "areas_to_improve": areas_to_improve if areas_to_improve else ["None identified"],
            "recommendations": recommendations,
            "passed": avg_score >= practice.passing_score,
            "completed_at": datetime.utcnow().isoformat(),
            "additional_notes": additional_notes,
        }
        
        # Guardar en sesión
        session.final_summary = summary
        session.completed_at = datetime.utcnow().isoformat()
        
        # Actualizar en DB
        if SUPABASE_AVAILABLE:
            try:
                supabase_client.table("lab_practice_sessions").update({
                    "status": "completed",
                    "completed_at": session.completed_at,
                    "final_summary": summary,
                    "total_score": avg_score,
                }).eq("id", session.session_id).execute()
            except Exception as e:
                print(f"Error updating session completion: {e}")
        
        return summary
    
    def get_user_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Obtiene historial de prácticas del usuario"""
        if not SUPABASE_AVAILABLE:
            return []
        
        try:
            result = supabase_client.table("lab_practice_sessions").select(
                "practice_id, status, total_score, completed_at, final_summary"
            ).eq(
                "user_id", user_id
            ).order(
                "started_at", desc=True
            ).execute()
            
            return result.data if result.data else []
        except Exception as e:
            print(f"Error fetching user history: {e}")
            return []
    
    # =========================================================================
    # CONTEXT GENERATION FOR TUTOR
    # =========================================================================
    
    def get_step_context_for_tutor(
        self,
        session: PracticeSession,
        include_hints: bool = False
    ) -> str:
        """
        Genera contexto del paso actual para el tutor.
        
        Returns:
            String con toda la información que el tutor necesita
        """
        practice = self.get_practice(session.practice_id)
        if not practice:
            return ""
        
        current = self.get_current_step(session)
        if not current:
            return ""
        
        section, step = current
        
        # Construir contexto
        context_parts = [
            f"## PRÁCTICA ACTIVA: {practice.title}",
            f"**Sección:** {section.title}",
            f"**Paso {session.completed_step_ids.index(step.id) + 1 if step.id in session.completed_step_ids else len(session.completed_step_ids) + 1} de {practice.get_total_steps()}**",
            "",
            f"### {step.title}",
            f"**Tipo:** {step.step_type.value}",
            "",
            step.content,
        ]
        
        # Agregar detalles si es paso crítico
        if step.is_critical:
            context_parts.append("")
            context_parts.append(f"⚠️ **PASO CRÍTICO** - El usuario DEBE confirmar antes de continuar")
        
        if step.safety_warning:
            context_parts.append("")
            context_parts.append(f"🛑 **ADVERTENCIA DE SEGURIDAD:** {step.safety_warning}")
        
        # Agregar preguntas si hay
        if step.questions:
            context_parts.append("")
            context_parts.append("### Preguntas de Comprensión:")
            for q in step.questions:
                context_parts.append(f"- {q.question}")
                if q.options:
                    for opt in q.options:
                        context_parts.append(f"  - {opt}")
        
        # Agregar hints si se solicitan
        if include_hints and step.tips:
            context_parts.append("")
            context_parts.append("### Tips para el usuario:")
            for tip in step.tips:
                context_parts.append(f"- {tip}")
        
        # Agregar errores comunes si el usuario ha cometido errores
        if session.mistakes_made > 0 and step.common_mistakes:
            context_parts.append("")
            context_parts.append("### Errores comunes a evitar:")
            for mistake in step.common_mistakes:
                context_parts.append(f"- {mistake}")
        
        # Progreso
        progress = practice.get_progress_percentage(session.completed_step_ids)
        context_parts.append("")
        context_parts.append(f"---")
        context_parts.append(f"**Progreso:** {progress:.0f}% completado")
        
        return "\n".join(context_parts)
    
    def format_step_for_display(self, session: PracticeSession) -> Dict[str, Any]:
        """Formatea el paso actual para mostrar en UI"""
        practice = self.get_practice(session.practice_id)
        if not practice:
            return {}
        
        current = self.get_current_step(session)
        if not current:
            return {}
        
        section, step = current
        
        return {
            "practice_title": practice.title,
            "section_title": section.title,
            "step_title": step.title,
            "step_type": step.step_type.value,
            "content": step.content,
            "is_critical": step.is_critical,
            "safety_warning": step.safety_warning,
            "image_ref": step.image_ref,
            "has_questions": len(step.questions) > 0,
            "questions": [q.to_dict() for q in step.questions],
            "progress_percent": practice.get_progress_percentage(session.completed_step_ids),
            "current_step_number": len(session.completed_step_ids) + 1,
            "total_steps": practice.get_total_steps(),
        }


# Singleton instance
_practice_manager: Optional[PracticeManager] = None


def get_practice_manager() -> PracticeManager:
    """Obtiene instancia singleton del PracticeManager"""
    global _practice_manager
    if _practice_manager is None:
        _practice_manager = PracticeManager()
    return _practice_manager


def reset_practice_manager():
    """Reset para testing"""
    global _practice_manager
    _practice_manager = None
