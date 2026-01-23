"""
verify_info.py - Nodo de verificación de identidad del usuario

Este nodo verifica la identidad del usuario antes de acceder a datos sensibles.
Similar al patrón del repo de referencia (FareedKhan-dev/Multi-Agent-AI-System).

Se activa cuando:
1. El usuario solicita información personal (invoices, historial, etc.)
2. No hay customer_id en el state

Flujo:
1. Extraer identificador del mensaje (ID, email, teléfono)
2. Verificar en base de datos
3. Si válido: guardar customer_id y continuar
4. Si inválido: pedir información vía human-in-the-loop
"""
import os
import re
from typing import Dict, Any, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.agent.state import AgentState
from src.agent.services import get_supabase
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_read, event_report, event_error


# ============================================
# SCHEMA PARA EXTRACCIÓN DE IDENTIFICADOR
# ============================================

class UserIdentifier(BaseModel):
    """Schema para extraer identificador del usuario"""
    identifier: str = Field(
        default="",
        description="Identificador del cliente: puede ser customer_id, email, o teléfono"
    )
    identifier_type: str = Field(
        default="unknown",
        description="Tipo de identificador: 'id', 'email', 'phone', 'unknown'"
    )


# ============================================
# PROMPTS
# ============================================

EXTRACT_IDENTIFIER_PROMPT = """Eres un extractor de información de identificación.

Tu tarea es extraer el identificador del cliente del mensaje.
El identificador puede ser:
- Un ID numérico (ej: "mi ID es 42", "customer_id: 123")
- Un email (ej: "mi correo es user@example.com")
- Un teléfono (ej: "mi número es +52 55 1234 5678")

Si el usuario NO ha proporcionado ningún identificador, retorna string vacío.

Responde en JSON:
{
  "identifier": "el identificador extraído o ''",
  "identifier_type": "id|email|phone|unknown"
}
"""

VERIFICATION_REQUEST_PROMPT = """Para acceder a información personal (historial de compras, facturas, preferencias guardadas), 
necesito verificar tu cuenta.

Por favor proporciona uno de los siguientes:
• Tu ID de cliente
• Tu email registrado
• Tu número de teléfono

Esto me permitirá acceder a tu información de forma segura."""


# ============================================
# FUNCIONES DE LOOKUP EN BASE DE DATOS
# ============================================

def lookup_customer_by_id(supabase, customer_id: str) -> Optional[Dict[str, Any]]:
    """Busca cliente por ID en Supabase"""
    try:
        response = supabase.table("profiles").select("*").eq("id", customer_id).execute()
        if response.data:
            return response.data[0]
    except Exception as e:
        logger.error("verify_info", f"Error buscando por ID: {e}")
    return None


def lookup_customer_by_email(supabase, email: str) -> Optional[Dict[str, Any]]:
    """Busca cliente por email en Supabase"""
    try:
        response = supabase.table("profiles").select("*").eq("email", email).execute()
        if response.data:
            return response.data[0]
    except Exception as e:
        logger.error("verify_info", f"Error buscando por email: {e}")
    return None


def lookup_customer_by_phone(supabase, phone: str) -> Optional[Dict[str, Any]]:
    """Busca cliente por teléfono en Supabase"""
    try:
        # Normalizar teléfono (quitar espacios, guiones)
        normalized_phone = re.sub(r"[\s\-\(\)]", "", phone)
        
        response = supabase.table("profiles").select("*").eq("phone", normalized_phone).execute()
        if response.data:
            return response.data[0]
        
        # Intentar sin el prefijo +
        if normalized_phone.startswith("+"):
            response = supabase.table("profiles").select("*").eq("phone", normalized_phone[1:]).execute()
            if response.data:
                return response.data[0]
    except Exception as e:
        logger.error("verify_info", f"Error buscando por teléfono: {e}")
    return None


def get_customer_id_from_identifier(
    supabase,
    identifier: str,
    identifier_type: str = "unknown"
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Busca el customer_id usando el identificador proporcionado.
    
    Returns:
        (customer_id, customer_data) o (None, None) si no se encuentra
    """
    if not identifier:
        return None, None
    
    identifier = identifier.strip()
    
    # Si no tenemos Supabase, usar fallback
    if supabase is None:
        logger.warning("verify_info", "Supabase no disponible, usando verificación simulada")
        # Fallback: aceptar cualquier ID numérico
        if identifier.isdigit():
            return identifier, {"id": identifier, "name": "Usuario"}
        return None, None
    
    customer = None
    
    # Intentar según el tipo detectado
    if identifier_type == "id" or identifier.isdigit():
        customer = lookup_customer_by_id(supabase, identifier)
    elif identifier_type == "email" or "@" in identifier:
        customer = lookup_customer_by_email(supabase, identifier)
    elif identifier_type == "phone" or identifier.startswith("+"):
        customer = lookup_customer_by_phone(supabase, identifier)
    else:
        # Intentar todos los métodos
        if identifier.isdigit():
            customer = lookup_customer_by_id(supabase, identifier)
        if not customer and "@" in identifier:
            customer = lookup_customer_by_email(supabase, identifier)
        if not customer and (identifier.startswith("+") or re.match(r"^\d{10,}$", identifier)):
            customer = lookup_customer_by_phone(supabase, identifier)
    
    if customer:
        return str(customer.get("id")), customer
    
    return None, None


# ============================================
# NODO PRINCIPAL
# ============================================

def verify_info_node(state: AgentState) -> Dict[str, Any]:
    """
    Nodo de verificación de identidad del usuario.
    
    Flujo:
    1. Si ya hay customer_id → Continuar al siguiente nodo
    2. Si no hay customer_id:
       a. Intentar extraer identificador del último mensaje
       b. Si se encuentra → Verificar en BD
       c. Si es válido → Guardar customer_id y continuar
       d. Si no es válido → Pedir información (human-in-the-loop)
    
    Returns:
        Dict con actualizaciones al state
    """
    logger.node_start("verify_info", {
        "has_customer_id": state.get("customer_id") is not None
    })
    
    events = [event_read("verify_info", "Verificando identidad del usuario...")]
    
    # ==========================================
    # Caso 1: Ya verificado
    # ==========================================
    if state.get("customer_id"):
        logger.info("verify_info", f"Usuario ya verificado: {state['customer_id']}")
        return {
            "next": "plan",  # Continuar al orchestrator
            "events": events + [event_report("verify_info", "✅ Usuario ya verificado")],
        }
    
    # ==========================================
    # Caso 2: Intentar extraer identificador
    # ==========================================
    messages = state.get("messages", [])
    last_message = ""
    
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_message = m.content
            break
        if isinstance(m, dict) and m.get("role") in ("user", "human"):
            last_message = m.get("content", "")
            break
    
    if not last_message:
        logger.warning("verify_info", "No hay mensaje del usuario")
        return {
            "needs_human_input": True,
            "clarification_questions": [VERIFICATION_REQUEST_PROMPT],
            "next": "human_input",
            "events": events + [event_error("verify_info", "Sin mensaje del usuario")],
        }
    
    # ==========================================
    # Usar LLM para extraer identificador
    # ==========================================
    model_name = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-20250514")
    
    try:
        if "claude" in model_name.lower():
            llm = ChatAnthropic(model=model_name, temperature=0)
        else:
            llm = ChatOpenAI(model=model_name, temperature=0)
        
        # Usar structured output
        structured_llm = llm.with_structured_output(UserIdentifier)
        
        parsed = structured_llm.invoke([
            SystemMessage(content=EXTRACT_IDENTIFIER_PROMPT),
            HumanMessage(content=last_message)
        ])
        
        identifier = parsed.identifier.strip()
        identifier_type = parsed.identifier_type
        
        logger.info("verify_info", f"Identificador extraído: {identifier} ({identifier_type})")
        
    except Exception as e:
        logger.error("verify_info", f"Error extrayendo identificador: {e}")
        identifier = ""
        identifier_type = "unknown"
        
        # Intentar extracción simple con regex
        # Buscar números que parezcan IDs
        id_match = re.search(r"\b(\d{1,10})\b", last_message)
        if id_match:
            identifier = id_match.group(1)
            identifier_type = "id"
        
        # Buscar emails
        email_match = re.search(r"\b[\w.-]+@[\w.-]+\.\w+\b", last_message)
        if email_match:
            identifier = email_match.group(0)
            identifier_type = "email"
        
        # Buscar teléfonos
        phone_match = re.search(r"\+?\d[\d\s\-\(\)]{8,}", last_message)
        if phone_match:
            identifier = phone_match.group(0)
            identifier_type = "phone"
    
    # ==========================================
    # Verificar en base de datos
    # ==========================================
    if identifier:
        supabase = get_supabase()
        customer_id, customer_data = get_customer_id_from_identifier(
            supabase, identifier, identifier_type
        )
        
        if customer_id:
            # ¡Verificación exitosa!
            user_name = customer_data.get("full_name", customer_data.get("name", state.get("user_name", "Usuario")))
            learning_style = customer_data.get("learning_style", {})

            confirmation_msg = f"✅ ¡Cuenta verificada! Hola {user_name} (ID: {customer_id})."

            logger.node_end("verify_info", {"status": "verified", "customer_id": customer_id, "has_learning_style": bool(learning_style)})

            return {
                "customer_id": customer_id,
                "user_name": user_name,
                "learning_style": learning_style,
                "messages": [AIMessage(content=confirmation_msg)],
                "next": "plan",  # Continuar al orchestrator
                "events": events + [event_report("verify_info", f"Verificado: {customer_id}")],
            }
        else:
            # Identificador no encontrado en BD
            logger.info("verify_info", f"Identificador '{identifier}' no encontrado en BD")
    
    # ==========================================
    # Caso 3: No se pudo verificar → Pedir información
    # ==========================================
    logger.info("verify_info", "No se pudo verificar, solicitando información")
    
    return {
        "messages": [AIMessage(content=VERIFICATION_REQUEST_PROMPT)],
        "needs_human_input": True,
        "clarification_questions": [
            "¿Cuál es tu ID de cliente?",
            "¿O prefieres proporcionar tu email registrado?",
        ],
        "next": "human_input",
        "events": events + [event_report("verify_info", "Solicitando verificación")],
    }


def needs_verification(state: AgentState, message: str) -> bool:
    """
    Helper para determinar si una solicitud necesita verificación.
    
    Args:
        state: Estado actual
        message: Mensaje del usuario
        
    Returns:
        True si necesita verificación
    """
    # Si ya está verificado, no necesita
    if state.get("customer_id"):
        return False
    
    # Keywords que indican necesidad de datos personales
    personal_keywords = [
        "mi factura", "mi invoice", "mis compras", "mi historial",
        "mi cuenta", "mis datos", "mi perfil", "mis preferencias",
        "refund", "reembolso", "devolución",
    ]
    
    message_lower = message.lower()
    return any(kw in message_lower for kw in personal_keywords)
