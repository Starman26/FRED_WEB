"""
ChatService - Lógica de negocio para streaming de chat
"""
import asyncio
import logging
from typing import Dict, AsyncGenerator

from fastapi import HTTPException
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.api.models import StreamEvent, EventType
from src.api.services.thread_manager import ThreadManager
from src.api.utils.helpers import (
    extract_message_content,
    extract_questions_from_event,
    extract_worker_content,
    format_event_display,
    parse_questions,
    format_answers_for_agent,
)


logger = logging.getLogger(__name__)

# Grafo singleton
_graph = None


def _get_graph():
    """Carga el grafo LangGraph."""
    global _graph
    if _graph is None:
        try:
            from src.agent.graph import graph
            _graph = graph
            logger.info("Grafo cargado correctamente")
        except ImportError as e:
            logger.error(f"Error al cargar el grafo: {e}")
            raise HTTPException(status_code=500, detail=f"Error loading agent: {e}")
    return _graph


class ChatService:
    """Servicio para manejar streaming de chat con el agente."""
    
    # Nodos del grafo para extraer eventos
    LOG_NODES = [
        "bootstrap", "plan", "route", "synthesize", "chat",
        "research", "tutor", "troubleshooting", "summarizer"
    ]
    
    MESSAGE_NODES = [
        "plan", "route", "synthesize", "chat", "tutor",
        "troubleshooting", "summarizer", "research"
    ]
    
    async def stream_chat(
        self,
        query: str,
        thread_id: str,
        user_name: str,
        thread_manager: ThreadManager
    ) -> AsyncGenerator[str, None]:
        """
        Genera eventos para streaming de respuesta.
        """
        graph = _get_graph()
        config = {"configurable": {"thread_id": thread_id}}
        
        payload = {
            "messages": [HumanMessage(content=query)],
            "user_name": user_name,
            "window_count": 0,
            "rolling_summary": "",
        }
        
        all_graph_events = []
        interrupted = False
        extracted_questions = []
        
        try:
            # Evento inicial
            yield StreamEvent(
                type=EventType.log,
                data={"message": "Procesando consulta...", "node": "START"}
            ).to_sse()
            
            # Stream del grafo
            for event in graph.stream(payload, config=config, stream_mode="updates"):
                all_graph_events.append(event)
                
                # Detectar interrupt
                if isinstance(event, dict) and "__interrupt__" in event:
                    interrupted = True
                    for prev_event in all_graph_events:
                        questions = extract_questions_from_event(prev_event)
                        if questions:
                            extracted_questions = questions
                            break
                    break
                
                for node_name in self.LOG_NODES:
                    if node_name in event and isinstance(event[node_name], dict):
                        node_events = event[node_name].get("events", [])
                        for evt in node_events:
                            formatted = format_event_display(evt)
                            yield StreamEvent(
                                type=EventType.log,
                                data=formatted
                            ).to_sse()
                        
                        # Extraer preguntas si existen
                        questions = event[node_name].get("clarification_questions", [])
                        if questions:
                            extracted_questions = questions
                
                await asyncio.sleep(0.01)
            
            # Manejar interrupt con preguntas
            if interrupted and extracted_questions:
                for event in self._handle_interrupt(
                    thread_id, extracted_questions, all_graph_events, thread_manager
                ):
                    yield event
            else:
                for event in self._handle_completion(all_graph_events):
                    yield event
            
            # Evento de finalización
            yield StreamEvent(
                type=EventType.done,
                data={"thread_id": thread_id}
            ).to_sse()
            
        except Exception as e:
            logger.error(f"Error en streaming: {e}")
            yield StreamEvent(
                type=EventType.error,
                data={"error": str(e)}
            ).to_sse()
    
    def _handle_interrupt(
        self,
        thread_id: str,
        questions: list,
        events: list,
        thread_manager: ThreadManager
    ):
        """Maneja un interrupt con preguntas HITL."""
        worker_content = extract_worker_content(events)
        thread_manager.set_pending_questions(thread_id, questions, worker_content)
        
        if worker_content:
            yield StreamEvent(
                type=EventType.message,
                data={"content": worker_content, "role": "assistant"}
            ).to_sse()
        
        parsed_questions = parse_questions(questions)
        yield StreamEvent(
            type=EventType.questions,
            data={
                "questions": [q.model_dump() for q in parsed_questions],
                "message": "Necesito más información para ayudarte mejor."
            }
        ).to_sse()
    
    def _handle_completion(self, events: list):
        """Extrae y envía mensajes finales."""
        messages = []
        
        for event in events:
            for node_name in self.MESSAGE_NODES:
                if node_name in event:
                    node_data = event[node_name]
                    if isinstance(node_data, dict) and "messages" in node_data:
                        node_messages = node_data["messages"]
                        if isinstance(node_messages, list):
                            for msg in node_messages:
                                content = extract_message_content(msg)
                                if content and content not in messages:
                                    messages.append(content)
                        else:
                            content = extract_message_content(node_messages)
                            if content and content not in messages:
                                messages.append(content)
        
        for msg in messages:
            yield StreamEvent(
                type=EventType.message,
                data={"content": msg, "role": "assistant"}
            ).to_sse()
    
    async def stream_resume(
        self,
        thread_id: str,
        answers: Dict[str, str],
        thread_manager: ThreadManager
    ) -> AsyncGenerator[str, None]:
        """Genera eventos para responder preguntas (resume)."""
        graph = _get_graph()
        config = {"configurable": {"thread_id": thread_id}}
        
        questions = thread_manager.get_pending_questions(thread_id)
        
        if not questions:
            yield StreamEvent(
                type=EventType.error,
                data={"error": "No hay preguntas pendientes para este thread"}
            ).to_sse()
            return
        
        formatted_answers = format_answers_for_agent(questions, answers)
        thread_manager.clear_pending_questions(thread_id)
        
        try:
            yield StreamEvent(
                type=EventType.log,
                data={"message": "Procesando respuestas...", "node": "RESUME"}
            ).to_sse()
            
            payload = Command(resume=formatted_answers)
            all_graph_events = []
            
            for event in graph.stream(payload, config=config, stream_mode="updates"):
                all_graph_events.append(event)
                
                for node_name in self.LOG_NODES:
                    if node_name in event and isinstance(event[node_name], dict):
                        node_events = event[node_name].get("events", [])
                        for evt in node_events:
                            formatted = format_event_display(evt)
                            yield StreamEvent(
                                type=EventType.log,
                                data=formatted
                            ).to_sse()
                
                await asyncio.sleep(0.01)
            
            # Extraer y enviar mensajes
            for event in self._handle_completion(all_graph_events):
                yield event
            
            yield StreamEvent(
                type=EventType.done,
                data={"thread_id": thread_id}
            ).to_sse()
            
        except Exception as e:
            logger.error(f"Error en resume: {e}")
            yield StreamEvent(
                type=EventType.error,
                data={"error": str(e)}
            ).to_sse()


# Singleton
_chat_service = ChatService()


def get_chat_service() -> ChatService:
    """Dependency injection para FastAPI."""
    return _chat_service
