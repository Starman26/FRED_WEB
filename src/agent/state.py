"""
state.py - Definición del estado compartido del agente con soporte para orchestration multi-step

CAMBIOS PRINCIPALES vs versión anterior:
1. Añadido soporte para orchestration_plan (lista de workers a ejecutar)
2. Añadido worker_outputs acumulativo (historial de outputs)
3. Añadido pending_context (contexto para el siguiente worker)
4. Añadido soporte para human-in-the-loop
5. Campos consistentes (troubleshooting_result, no troubleshooter_result)
"""
from typing import TypedDict, Annotated, Sequence, Any, List, Dict, Optional
from langchain_core.messages import BaseMessage
import operator


def merge_worker_outputs(left: List[Dict], right: List[Dict]) -> List[Dict]:
    """
    Reducer personalizado para worker_outputs.
    
    REGLAS:
    - Si right es EXACTAMENTE [] (lista vacía), RESETEAR (devolver [])
    - Si right es None, mantener left
    - Si right tiene elementos, concatenar sin duplicados
    """
    # RESET: Si se pasa lista vacía explícitamente, limpiar todo
    if right is not None and len(right) == 0:
        return []
    
    # Mantener left si right es None
    if right is None:
        return left or []
    
    if not left:
        return right or []
    
    # Usar task_id para evitar duplicados
    existing_ids = {o.get("task_id") for o in left if o.get("task_id")}
    merged = list(left)
    
    for output in right:
        task_id = output.get("task_id")
        if task_id and task_id not in existing_ids:
            merged.append(output)
            existing_ids.add(task_id)
        elif not task_id:
            merged.append(output)
    
    return merged


def merge_dicts(left: Dict, right: Dict) -> Dict:
    """
    Reducer para mergear dicts (pending_context).
    
    REGLAS:
    - Si right es EXACTAMENTE {} (dict vacío), RESETEAR
    - Si right es None, mantener left
    - Si right tiene keys, hacer merge
    """
    # RESET: Si se pasa dict vacío explícitamente, limpiar
    if right is not None and len(right) == 0:
        return {}
    
    # Mantener left si right es None
    if right is None:
        return left or {}
    
    if not left:
        return right or {}
    
    return {**left, **right}


class AgentState(TypedDict):
    """
    Estado compartido entre todos los nodos del grafo.
    
    IMPORTANTE:
    - messages: obligatorio, se acumula con operator.add
    - events: canal de logs, se acumula con operator.add
    - worker_outputs: historial de outputs de workers, se acumula
    - El resto se inicializa en bootstrap
    
    CAMPOS POR CATEGORÍA:
    
    1. MENSAJES Y EVENTOS
       - messages: Historial de mensajes de la conversación
       - events: Logs estilo Manus para debugging/UI
    
    2. CONTROL DE FLUJO
       - next: Próximo nodo a ejecutar
       - done: Si True, termina el grafo
    
    3. ORCHESTRATION MULTI-STEP
       - orchestration_plan: Cola de workers a ejecutar ["research", "tutor"]
       - current_step: Índice actual en el plan
       - worker_outputs: Historial de outputs de todos los workers
       - pending_context: Contexto acumulado para el siguiente worker
    
    4. HUMAN-IN-THE-LOOP
       - needs_human_input: Flag para pausar y pedir input
       - clarification_questions: Preguntas para el usuario
    
    5. MEMORIA
       - rolling_summary: Resumen acumulado de la conversación
       - window_count: Contador de mensajes en la ventana actual
       - loaded_memory: Preferencias del usuario cargadas
    
    6. METADATA
       - task_type: Tipo de tarea actual
       - user_name: Nombre del usuario
       - customer_id: ID del cliente (para verificación)
    
    7. RESULTADOS DE WORKERS (compatibilidad legacy)
       - research_result: JSON string del research worker
       - tutor_result: JSON string del tutor worker
       - troubleshooting_result: JSON string del troubleshooting worker
       - summarizer_result: JSON string del summarizer worker
    
    8. HERRAMIENTAS (inyectadas por bootstrap)
       - supabase: Cliente de Supabase
       - embeddings: Modelo de embeddings
    """
    
    # ==========================================
    # 1. MENSAJES Y EVENTOS (Obligatorios)
    # ==========================================
    messages: Annotated[Sequence[Any], operator.add]
    events: Annotated[List[Dict[str, Any]], operator.add]
    
    # ==========================================
    # 2. CONTROL DE FLUJO
    # ==========================================
    next: str  # Próximo nodo: "research", "tutor", "troubleshooting", "summarizer", "END"
    done: bool  # Si True, el grafo termina
    
    # ==========================================
    # 3. ORCHESTRATION MULTI-STEP (NUEVO)
    # ==========================================
    orchestration_plan: List[str]  # Cola de workers: ["research", "troubleshooting", "tutor"]
    current_step: int  # Índice actual en orchestration_plan
    worker_outputs: Annotated[List[Dict[str, Any]], merge_worker_outputs]  # Historial acumulado
    pending_context: Annotated[Dict[str, Any], merge_dicts]  # Contexto para siguiente worker
    
    # ==========================================
    # 4. HUMAN-IN-THE-LOOP (NUEVO)
    # ==========================================
    needs_human_input: bool  # True si hay que pausar para input del usuario
    clarification_questions: List[str]  # Preguntas para el usuario
    
    # ==========================================
    # 4.5. ANÁLISIS DE INTENCIÓN (NUEVO)
    # ==========================================
    intent_analysis: Dict[str, Any]  # Resultado del análisis de intención del usuario
    
    # ==========================================
    # 5. MEMORIA
    # ==========================================
    rolling_summary: str  # Resumen acumulado de la conversación
    window_count: int  # Contador de mensajes en la ventana actual
    loaded_memory: str  # Preferencias del usuario cargadas desde store
    
    # ==========================================
    # 6. METADATA
    # ==========================================
    task_type: str  # Tipo de tarea: 'tutor', 'troubleshooting', 'research', 'summarizer'
    user_name: str  # Nombre del usuario
    customer_id: Optional[str]  # ID del cliente para verificación
    learning_style: Dict[str, Any]  # Estilo de aprendizaje del usuario (JSONB del profile)
    
    # ==========================================
    # 7. RESULTADOS DE WORKERS (Legacy/Compatibilidad)
    # ==========================================
    research_result: Any  # JSON string del research worker
    tutor_result: Any  # JSON string del tutor worker
    troubleshooting_result: Any  # JSON string del troubleshooting worker (NOTA: sin "er")
    summarizer_result: Any  # JSON string del summarizer worker
    
    # ==========================================
    # 8. HERRAMIENTAS - REMOVIDO
    # ==========================================
    # NOTA: supabase y embeddings ya NO están en el state porque no son serializables.
    # Usar src.agent.services.get_supabase() y get_embeddings() en su lugar.


# ==========================================
# VALORES POR DEFECTO para bootstrap
# ==========================================
STATE_DEFAULTS: Dict[str, Any] = {
    # Control de flujo
    "next": "supervisor",
    "done": False,
    
    # Orchestration
    "orchestration_plan": [],
    "current_step": 0,
    "worker_outputs": [],
    "pending_context": {},
    
    # Human-in-the-loop
    "needs_human_input": False,
    "clarification_questions": [],
    
    # Intent analysis
    "intent_analysis": {},
    
    # Memoria
    "rolling_summary": "",
    "window_count": 0,
    "loaded_memory": "",
    
    # Metadata
    "task_type": "",
    "user_name": "Usuario",
    "customer_id": None,
    "learning_style": {},
    
    # Resultados legacy
    "research_result": None,
    "tutor_result": None,
    "troubleshooting_result": None,
    "summarizer_result": None,
    
    # Events (se inicializa vacío)
    "events": [],
}


def get_state_defaults() -> Dict[str, Any]:
    """Retorna una copia de los valores por defecto del state"""
    return STATE_DEFAULTS.copy()


def validate_state(state: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Valida que el state tenga la estructura correcta.
    
    Returns:
        (is_valid, error_message)
    """
    # Campo obligatorio: messages
    if "messages" not in state:
        return False, "Campo obligatorio faltante: messages"
    
    if not isinstance(state.get("messages"), (list, tuple)):
        return False, "messages debe ser list o tuple"
    
    return True, None
