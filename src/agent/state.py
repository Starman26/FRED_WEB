"""
state.py 
Definición del estado compartido del agente 
con soporte para orchestration multi-step

"""
from typing import TypedDict, Annotated, Sequence, Any, List, Dict, Optional
from langchain_core.messages import BaseMessage
import operator


# Usar esto en vez de [] cuando se quiera limpiar los outputs.
RESET_WORKER_OUTPUTS = "__RESET_WORKER_OUTPUTS__"


def merge_worker_outputs(left: List[Dict], right: Any) -> List[Dict]:
    """
    Reducer personalizado para worker_outputs.

    REGLAS:
    - Si right es RESET_WORKER_OUTPUTS sentinel → RESETEAR (devolver [])
    - Si right es None o [] → mantener left (sin cambios)
    - Si right tiene elementos → concatenar sin duplicados por task_id
    """
    # RESET explícito con sentinel
    if right == RESET_WORKER_OUTPUTS:
        return []

    # Sin cambios: None o lista vacía
    if right is None:
        return left or []
    if isinstance(right, list) and len(right) == 0:
        return left or []

    if not left:
        return right if isinstance(right, list) else []

    if not isinstance(right, list):
        return left

    # Merge sin duplicados por task_id
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
    _route_count: int  # Anti-loop counter for adaptive_router (must be in TypedDict or LangGraph drops it)
    
    # ==========================================
    # 4. HUMAN-IN-THE-LOOP (NUEVO) - Con soporte Wizard
    # ==========================================
    needs_human_input: bool  # True si hay que pausar para input del usuario
    clarification_questions: List[Any]  # Preguntas (strings o ClarificationQuestion dicts)
    follow_up_suggestions: List[str]  # Sugerencias de seguimiento generadas por workers
    # NOTA: El estado del wizard se almacena en pending_context["wizard_state"]
    # pending_context también puede contener:
    #   - "question_set": QuestionSet serializado para iniciar wizard
    #   - "wizard_responses": Respuestas estructuradas del wizard
    #   - "wizard_completed": True si el wizard se completó
    #   - "wizard_cancelled": True si el usuario canceló
    
    # ==========================================
    # 4.5. PLANIFICACIÓN Y ANÁLISIS
    # ==========================================
    intent_analysis: Dict[str, Any]  # Resultado del análisis de intención del usuario
    plan_reasoning: str  # Chain-of-thought reasoning del planner
    planner_method: str  # "fast" (regex) o "llm" (chain-of-thought)
    
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
    user_id: Optional[str]  # UUID del usuario (de profiles)
    team: str  # Nombre del equipo activo del usuario
    interaction_mode: str  # Modo de interacción: 'chat', 'code', 'agent', 'voice'
    llm_model: str  # Modelo LLM seleccionado por el usuario (e.g., 'claude-sonnet-4-20250514')
    auth_user_id: Optional[str]  # UUID del usuario autenticado (para RLS filtering en analysis)
    team_id: Optional[str]  # UUID del equipo activo del usuario
    customer_id: Optional[str]  # ID del cliente para verificación
    token_usage: Annotated[int, operator.add]  # Total tokens consumed (accumulates across workers)
    image_attachments: List[Dict[str, Any]]  # Multimodal image blocks from user (kept out of messages to avoid .lower() crashes)

    # ==========================================
    # 6b. PRACTICE / AUTOMATION MODE
    # ==========================================
    automation_id: Optional[str]  # ID de la automatización activa (lab.automations)
    automation_md_content: str  # Guion markdown de la automatización
    automation_step: int  # Paso actual en el guion
    automation_type: str  # Tipo: 'practice', 'automation', etc.
    automation_context: str  # Contexto acumulado de la sesión (observaciones del agente)
    practice_status: str  # Status from Supabase: "in_progress", "completed", "paused"
    practice_chunks: List[Dict[str, Any]]  # Multi-message chunks for SSE streaming (tool execution flow)
    last_tool_step: int  # Step number where tool was last executed (prevents re-execution)
    user_profile_md: str  # Perfil del usuario en markdown (public.user_agent_profiles)
    robot_ids: List[str]  # IDs de robots seleccionados en la UI (e.g., ["xarm-201", "xarm-202"])
    robot_state: Annotated[Dict[str, Any], merge_dicts]  # Telemetría de robots desde lab.robot_state

    # === BITL (Bridge-in-the-Loop) fields ===
    bridge_report: Optional[dict]              # Último reporte recibido del bridge
    practice_session_active: bool              # Si hay practice session BITL activa
    current_practice_step: int                 # Índice del paso actual (0-based)
    total_practice_steps: int                  # Total de pasos en la rutina
    practice_results: list                     # Resultados por paso: [{"step": 0, "passed": True, "score": 0.95, ...}]
    practice_expected_steps: list              # Pasos esperados de la rutina (parseados del automation content)
    target_robot_id: Optional[str]             # Robot principal de la practice session

    # ==========================================
    # 6d. TOOL EXECUTION & DEVICES
    # ==========================================
    tool_execution_log: Annotated[List[Dict[str, Any]], operator.add]  # Historial de tool calls con resultados y tiempos
    active_devices: Dict[str, Any]  # Snapshot de dispositivos conectados / bridge metadata (se sobrescribe)

    # ==========================================
    # 6c. STREAMING CALLBACK (not serializable, injected per-request)
    # ==========================================
    _stream_session_id: Optional[str]  # Session ID para lookup de callback en api_server registry

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
    "_route_count": 0,
    
    # Human-in-the-loop
    "needs_human_input": False,
    "clarification_questions": [],
    "follow_up_suggestions": [],
    
    # Planning & analysis
    "intent_analysis": {},
    "plan_reasoning": "",
    "planner_method": "",
    
    # Memoria
    "rolling_summary": "",
    "window_count": 0,
    "loaded_memory": "",
    
    # Metadata
    "task_type": "",
    "user_name": "Usuario",
    "user_id": None,
    "team": "",
    "interaction_mode": "chat",
    "llm_model": "",
    "auth_user_id": None,
    "team_id": None,
    "customer_id": None,
    "token_usage": 0,
    "image_attachments": [],

    # Practice / Automation mode
    "automation_id": None,
    "automation_md_content": "",
    "automation_step": 1,
    "automation_type": "",
    "automation_context": "",
    "practice_status": "in_progress",
    "practice_chunks": [],
    "last_tool_step": 0,
    "user_profile_md": "",
    "robot_ids": [],
    "robot_state": {},

    # BITL (Bridge-in-the-Loop)
    "bridge_report": None,
    "practice_session_active": False,
    "current_practice_step": 0,
    "total_practice_steps": 0,
    "practice_results": [],
    "practice_expected_steps": [],
    "target_robot_id": None,

    # Tool execution & devices
    "tool_execution_log": [],
    "active_devices": {},

    # Streaming callback
    "_stream_session_id": None,

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
