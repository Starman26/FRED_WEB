class ThreadManager:
    """
    Gestiona estado de threads para Human-in-the-Loop
    """

    def __init__(self):
        self._pending_questions: dict = {}
        self._worker_content: dict = {}
    
    def set_pending_questions(
        self, 
        thread_id: str, 
        questions: list, 
        worker_content: str = ""
    ) -> None:
        """Guarda preguntas pendientes para un thread."""
        self._pending_questions[thread_id] = questions
        self._worker_content[thread_id] = worker_content
    
    def get_pending_questions(self, thread_id: str) -> list:
        """Obtiene preguntas pendientes de un thread."""
        return self._pending_questions.get(thread_id, [])
    
    def get_worker_content(self, thread_id: str) -> str:
        """Obtiene contenido del worker previo a las preguntas."""
        return self._worker_content.get(thread_id, "")
    
    def has_pending_questions(self, thread_id: str) -> bool:
        """Verifica si hay preguntas pendientes."""
        return thread_id in self._pending_questions
    
    def clear_pending_questions(self, thread_id: str) -> None:
        """Limpia estado de un thread."""
        self._pending_questions.pop(thread_id, None)
        self._worker_content.pop(thread_id, None)


# Singleton
_thread_manager = ThreadManager()


def get_thread_manager() -> ThreadManager:
    """Dependency injection para FastAPI."""
    return _thread_manager
