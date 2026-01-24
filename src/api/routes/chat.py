import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from src.api.models import ChatRequest, ResumeRequest, ThreadStatus
from src.api.services import (
    ThreadManager, 
    get_thread_manager,
    ChatService,
    get_chat_service,
)
from src.api.utils import parse_questions


router = APIRouter(prefix="/chat", tags=["Chat"])
logger = logging.getLogger(__name__)


@router.post("/stream", summary="Chat con streaming SSE")
async def chat_stream(
    request: ChatRequest,
    thread_manager: ThreadManager = Depends(get_thread_manager),
    chat_service: ChatService = Depends(get_chat_service),
):
    """
    Inicia una conversación con el agente usando Server-Sent Events.
    """
    thread_id = request.thread_id or f"api-{uuid.uuid4()}"
    logger.info(f"[chat/stream] thread={thread_id}")
    
    return StreamingResponse(
        chat_service.stream_chat(
            query=request.query,
            thread_id=thread_id,
            user_name=request.user_name or "Usuario",
            thread_manager=thread_manager
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Thread-ID": thread_id
        }
    )


@router.post("/resume/stream", summary="Responder preguntas con streaming SSE")
async def resume_stream(
    request: ResumeRequest,
    thread_manager: ThreadManager = Depends(get_thread_manager),
    chat_service: ChatService = Depends(get_chat_service),
):
    """
    Responde a las preguntas del agente y continúa la conversación.
    
    Usar después de recibir un evento `questions` en el stream.
    """
    if not thread_manager.has_pending_questions(request.thread_id):
        raise HTTPException(
            status_code=400,
            detail="No hay preguntas pendientes para este thread"
        )
    
    logger.info(f"[chat/resume] thread={request.thread_id}, answers={len(request.answers)}")
    
    return StreamingResponse(
        chat_service.stream_resume(
            thread_id=request.thread_id,
            answers=request.answers,
            thread_manager=thread_manager
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Thread-ID": request.thread_id
        }
    )


@router.get("/thread/{thread_id}/status", response_model=ThreadStatus, summary="Estado del thread")
async def get_thread_status(
    thread_id: str,
    thread_manager: ThreadManager = Depends(get_thread_manager),
):
    """Obtiene el estado de un thread (útil para reconexiones)."""
    has_questions = thread_manager.has_pending_questions(thread_id)
    questions = None
    
    if has_questions:
        raw_questions = thread_manager.get_pending_questions(thread_id)
        questions = parse_questions(raw_questions)
    
    return ThreadStatus(
        thread_id=thread_id,
        has_pending_questions=has_questions,
        questions=questions
    )
