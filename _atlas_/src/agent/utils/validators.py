"""
validators.py - Validadores de inputs para el agente

CAMBIOS PRINCIPALES:
1. Validación de research_result agregada
2. Validación de whitelist (agent_tables)
3. Sanitización mejorada de SQL
4. Logging estructurado
"""

import re
from typing import Tuple, Optional, List
from .logger import logger


class InputValidator:
    """Valida y sanitiza inputs del usuario"""
    
    # Límites
    MAX_MESSAGE_LENGTH = 5000  # caracteres
    MAX_USER_NAME_LENGTH = 100
    MIN_MESSAGE_LENGTH = 1
    
    # Patrones peligrosos (inyección básica)
    DANGEROUS_PATTERNS = [
        r"(?i)(drop\s+table|delete\s+from|insert\s+into|update\s+set)",  # SQL
        r"(?i)(eval|exec|__import__|system)",  # Python
        r"<script|javascript:|onerror=|onclick=",  # XSS
    ]
    
    @staticmethod
    def validate_user_message(message: str) -> Tuple[bool, str, Optional[str]]:
        """
        Valida un mensaje del usuario.
        
        Returns:
            (is_valid, sanitized_message, error_reason)
        """
        if not isinstance(message, str):
            logger.validation_error("user_message", str(type(message)), "No es string")
            return False, "", "El mensaje debe ser texto"
        
        # Trim whitespace
        message = message.strip()
        
        # Verificar longitud
        if len(message) < InputValidator.MIN_MESSAGE_LENGTH:
            logger.validation_error("user_message", message, "Mensaje vacío")
            return False, "", "El mensaje no puede estar vacío"
        
        if len(message) > InputValidator.MAX_MESSAGE_LENGTH:
            logger.validation_error("user_message", message[:50], f"Excede {InputValidator.MAX_MESSAGE_LENGTH} chars")
            return False, "", f"El mensaje es demasiado largo (máx {InputValidator.MAX_MESSAGE_LENGTH} caracteres)"
        
        # Verificar patrones peligrosos
        for pattern in InputValidator.DANGEROUS_PATTERNS:
            if re.search(pattern, message):
                logger.validation_error("user_message", message[:50], f"Patrón peligroso detectado: {pattern}")
                return False, "", "El mensaje contiene patrones no permitidos"
        
        return True, message, None
    
    @staticmethod
    def validate_user_name(name: str) -> Tuple[bool, str, Optional[str]]:
        """
        Valida nombre del usuario.
        
        Returns:
            (is_valid, sanitized_name, error_reason)
        """
        if not isinstance(name, str):
            logger.validation_error("user_name", str(type(name)), "No es string")
            return False, "", "El nombre debe ser texto"
        
        name = name.strip()
        
        if len(name) < 1:
            logger.validation_error("user_name", name, "Nombre vacío")
            return False, "", "El nombre no puede estar vacío"
        
        if len(name) > InputValidator.MAX_USER_NAME_LENGTH:
            logger.validation_error("user_name", name[:50], f"Excede {InputValidator.MAX_USER_NAME_LENGTH} chars")
            return False, "", f"El nombre es demasiado largo (máx {InputValidator.MAX_USER_NAME_LENGTH} caracteres)"
        
        # Sanitizar: solo alfanuméricos, espacios, guiones, puntos
        sanitized = re.sub(r"[^a-zA-Z0-9\s\-\.]", "", name).strip()
        
        if not sanitized:
            logger.validation_error("user_name", name, "Nombre contiene solo caracteres inválidos")
            return False, "", "El nombre contiene caracteres no permitidos"
        
        return True, sanitized, None
    
    @staticmethod
    def validate_task_type(task_type: str) -> Tuple[bool, str, Optional[str]]:
        """
        Valida tipo de tarea.
        
        Returns:
            (is_valid, task_type, error_reason)
        """
        valid_types = {"tutor", "troubleshooting", "summarizer", "research"}
        
        if not isinstance(task_type, str):
            return False, "", "task_type debe ser string"
        
        task_type = task_type.strip().lower()
        
        if task_type not in valid_types:
            logger.validation_error("task_type", task_type, f"Debe ser uno de: {valid_types}")
            return False, "", f"task_type debe ser uno de: {valid_types}"
        
        return True, task_type, None
    
    @staticmethod
    def validate_window_count(window_count: int) -> Tuple[bool, int, Optional[str]]:
        """
        Valida window_count.
        
        Returns:
            (is_valid, window_count, error_reason)
        """
        if not isinstance(window_count, int):
            logger.validation_error("window_count", str(type(window_count)), "No es int")
            return False, 0, "window_count debe ser entero"
        
        if window_count < 0:
            logger.validation_error("window_count", str(window_count), "Negativo")
            return False, 0, "window_count no puede ser negativo"
        
        return True, window_count, None


class SQLValidator:
    """Valida consultas SQL contra whitelist"""
    
    # Operaciones permitidas por defecto
    ALLOWED_OPERATIONS = {"SELECT"}
    
    # Palabras clave peligrosas
    DANGEROUS_KEYWORDS = {
        "DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", 
        "TRUNCATE", "GRANT", "REVOKE", "EXECUTE", "EXEC"
    }
    
    @staticmethod
    def validate_query(query: str, allowed_tables: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Valida una consulta SQL contra la whitelist de tablas.
        
        Args:
            query: Consulta SQL a validar
            allowed_tables: Lista de tablas permitidas
        
        Returns:
            (is_valid, error_reason)
        """
        if not isinstance(query, str):
            return False, "La consulta debe ser texto"
        
        query_upper = query.upper().strip()
        
        # Verificar que sea SELECT
        if not query_upper.startswith("SELECT"):
            return False, "Solo se permiten consultas SELECT"
        
        # Verificar palabras clave peligrosas
        for keyword in SQLValidator.DANGEROUS_KEYWORDS:
            # Buscar la palabra como token completo
            if re.search(rf"\b{keyword}\b", query_upper):
                logger.validation_error("sql_query", query[:50], f"Palabra clave peligrosa: {keyword}")
                return False, f"Operación no permitida: {keyword}"
        
        # Extraer tablas mencionadas en la consulta
        # Patrón simple para FROM y JOIN
        table_pattern = r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
        mentioned_tables = re.findall(table_pattern, query, re.IGNORECASE)
        
        # Verificar que todas las tablas estén en whitelist
        allowed_set = set(t.lower() for t in allowed_tables)
        for table in mentioned_tables:
            if table.lower() not in allowed_set:
                logger.validation_error("sql_query", query[:50], f"Tabla no permitida: {table}")
                return False, f"Tabla no permitida: {table}"
        
        return True, None
    
    @staticmethod
    async def get_allowed_tables(supabase) -> List[str]:
        """
        Obtiene la lista de tablas permitidas desde agent_tables.
        
        Args:
            supabase: Cliente de Supabase
        
        Returns:
            Lista de nombres de tablas permitidas
        """
        try:
            resp = supabase.table("agent_tables").select("table_name").eq("role", "agent").execute()
            return [r["table_name"] for r in (resp.data or [])]
        except Exception as e:
            logger.error("sql_validator", f"Error obteniendo whitelist: {e}")
            # Fallback: tablas seguras por defecto
            return ["documents", "document_chunks"]


def validate_state_before_llm(state: dict) -> Tuple[bool, Optional[str]]:
    """
    Valida y NORMALIZA el estado antes de pasar a LLM.
    
    IMPORTANTE: En lugar de fallar si faltan campos, inicializa con valores por defecto.
    Esto es crítico porque los campos son NotRequired en el TypedDict.
    
    Returns:
        (is_valid, error_reason)
    """
    if not isinstance(state, dict):
        return False, "State no es dict"
    
    # ========================================
    # 1. Validar campo obligatorio: messages
    # ========================================
    if "messages" not in state:
        logger.validation_error("state", "messages", "Campo obligatorio faltante")
        return False, "Campo obligatorio faltante: messages"
    
    if not isinstance(state.get("messages"), (list, tuple)):
        logger.validation_error("state", "messages", "No es list/tuple")
        return False, "messages debe ser list o tuple"
    
    # ========================================
    # 2. Inicializar campos opcionales con defaults
    # ========================================
    
    # user_name: default "Usuario"
    if "user_name" not in state or not isinstance(state.get("user_name"), str):
        state["user_name"] = "Usuario"
    
    # window_count: default 0
    if "window_count" not in state or not isinstance(state.get("window_count"), int):
        state["window_count"] = 0
    
    # done: default False
    if "done" not in state or not isinstance(state.get("done"), bool):
        state["done"] = False
    
    # next: default "tutor"
    if "next" not in state or not isinstance(state.get("next"), str):
        state["next"] = "tutor"
    
    # rolling_summary: default ""
    if "rolling_summary" not in state or not isinstance(state.get("rolling_summary"), str):
        state["rolling_summary"] = ""
    
    # task_type: default ""
    if "task_type" not in state or not isinstance(state.get("task_type"), str):
        state["task_type"] = ""
    
    # tutor_result: default ""
    if "tutor_result" not in state or not isinstance(state.get("tutor_result"), str):
        state["tutor_result"] = ""
    
    # troubleshooter_result: default ""
    if "troubleshooter_result" not in state or not isinstance(state.get("troubleshooter_result"), str):
        state["troubleshooter_result"] = ""
    
    # summarizer_result: default ""
    if "summarizer_result" not in state or not isinstance(state.get("summarizer_result"), str):
        state["summarizer_result"] = ""
    
    # research_result: default "" (NUEVO)
    if "research_result" not in state or not isinstance(state.get("research_result"), str):
        state["research_result"] = ""
    
    return True, None


def validate_research_result(result: str) -> Tuple[bool, dict, Optional[str]]:
    """
    Valida el resultado del research_node.
    
    Args:
        result: JSON string del resultado
    
    Returns:
        (is_valid, parsed_result, error_reason)
    """
    import json
    
    if not isinstance(result, str):
        return False, {}, "research_result debe ser string"
    
    if not result.strip():
        return False, {}, "research_result está vacío"
    
    try:
        parsed = json.loads(result)
    except json.JSONDecodeError as e:
        logger.validation_error("research_result", result[:50], f"JSON inválido: {e}")
        return False, {}, f"JSON inválido: {e}"
    
    # Validar campos requeridos
    required_fields = ["answer", "sources", "strategy_used", "confidence_score"]
    for field in required_fields:
        if field not in parsed:
            return False, parsed, f"Campo faltante: {field}"
    
    # Validar tipos
    if not isinstance(parsed.get("answer"), str):
        return False, parsed, "answer debe ser string"
    
    if not isinstance(parsed.get("sources"), list):
        return False, parsed, "sources debe ser lista"
    
    if not isinstance(parsed.get("strategy_used"), str):
        return False, parsed, "strategy_used debe ser string"
    
    if not isinstance(parsed.get("confidence_score"), (int, float)):
        return False, parsed, "confidence_score debe ser número"
    
    return True, parsed, None