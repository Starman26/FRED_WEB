"""
Aplicación Streamlit para el agente multiagente ATLAS

Features:
- Human-in-the-Loop con tarjetas de preguntas estructuradas
- Opciones múltiples con botones/radio
- Máximo 3 preguntas por interacción
- Flujo tipo wizard (una pregunta a la vez)
"""

import os
import sys
import json
import logging
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.types import Command
from dotenv import load_dotenv

# ============================================
# CONFIGURACIÓN
# ============================================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="ATLAS Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para tarjetas de preguntas
st.markdown("""
<style>
.question-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    border-left: 4px solid #4da6ff;
}
.question-title {
    color: #4da6ff;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 8px;
}
.question-text {
    color: #ffffff;
    font-size: 16px;
    font-weight: 500;
    margin-bottom: 15px;
}
.option-button {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 5px 0;
    cursor: pointer;
    transition: all 0.2s;
}
.option-button:hover {
    background: rgba(77, 166, 255, 0.3);
    border-color: #4da6ff;
}
.option-id {
    color: #4da6ff;
    font-weight: 600;
    margin-right: 10px;
}
.option-label {
    color: #ffffff;
}
.option-desc {
    color: rgba(255,255,255,0.6);
    font-size: 12px;
    margin-left: 25px;
}
</style>
""", unsafe_allow_html=True)


# ============================================
# LOG HANDLER
# ============================================

class StreamlitLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if "raw_logs" not in st.session_state:
                st.session_state.raw_logs = []
            st.session_state.raw_logs.append(msg)
            st.session_state.raw_logs = st.session_state.raw_logs[-500:]
        except Exception:
            pass

def install_log_handler_once():
    if "log_handler_installed" in st.session_state:
        return
    handler = StreamlitLogHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.INFO)
    st.session_state.log_handler_installed = True


# ============================================
# FUNCIONES AUXILIARES
# ============================================

def load_graph():
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from src.agent.graph import graph
        logger.info("✅ Grafo cargado correctamente")
        return graph
    except ImportError as e:
        logger.error(f"❌ Error al cargar el grafo: {str(e)}")
        st.error(f"Error al cargar el grafo: {str(e)}")
        st.stop()


def initialize_session_state():
    defaults = {
        "messages": [],
        "user_name": "Usuario",
        "window_count": 0,
        "rolling_summary": "",
        "is_loading": False,
        "pending_interrupt": None,
        "raw_logs": [],
        "pending_questions": [],  # Preguntas estructuradas pendientes
        "pending_content": "",  # Contenido del worker antes de las preguntas
        "current_question_idx": 0,  # Índice de pregunta actual
        "question_answers": {},  # Respuestas acumuladas
    }
    
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default
    
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"streamlit-{uuid.uuid4()}"
        logger.info(f"🆔 Nueva sesión: {st.session_state.thread_id}")
    
    install_log_handler_once()


def get_langgraph_config() -> Dict[str, Any]:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def extract_message_content(msg: Any) -> Optional[str]:
    if isinstance(msg, (AIMessage, HumanMessage)):
        return msg.content if hasattr(msg, "content") else None
    elif isinstance(msg, dict):
        return msg.get("content")
    elif isinstance(msg, str):
        return msg
    elif hasattr(msg, "content"):
        return msg.content
    return None


def get_event_icon(event: Dict) -> str:
    type_icons = {
        "read": "📖", "plan": "🧠", "execute": "⚙️",
        "report": "📊", "route": "🔀", "error": "❌", "done": "✅",
    }
    return type_icons.get(event.get("type", "").lower(), "•")


def format_event_display(event: Dict) -> str:
    icon = get_event_icon(event)
    node = event.get("source", event.get("node", "?")).upper()
    message = event.get("content", event.get("message", ""))
    return f"{icon} **{node}**: {message}"


def extract_messages_from_events(events: List[Dict]) -> tuple[List[str], Dict]:
    messages = []
    final_state = {}
    node_names = ["plan", "route", "synthesize", "chat", "tutor",
                  "troubleshooting", "summarizer", "research", "human_input"]

    for event in events:
        if isinstance(event, dict):
            for node_name in node_names:
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
            final_state = event

    return messages, final_state


def extract_questions_from_event(event: Dict) -> List[Dict]:
    """Extrae preguntas estructuradas de un evento de interrupt"""
    # Buscar en troubleshooting o cualquier nodo que tenga clarification_questions
    for node_name in ["troubleshooting", "research", "tutor"]:
        if node_name in event and isinstance(event[node_name], dict):
            questions = event[node_name].get("clarification_questions", [])
            if questions:
                return questions
    return []


# ============================================
# COMPONENTES DE UI PARA PREGUNTAS
# ============================================

def render_question_card(question: Dict, question_num: int) -> Optional[str]:
    """
    Renderiza una tarjeta de pregunta con opciones.
    
    Returns:
        La respuesta seleccionada o None si no se ha respondido
    """
    q_id = question.get("id", f"q{question_num}")
    q_text = question.get("question", "")
    q_type = question.get("type", "choice")
    options = question.get("options", [])
    placeholder = question.get("placeholder", "Escribe tu respuesta...")
    
    st.markdown(f"""
    <div class="question-card">
        <div class="question-title">📋 Pregunta {question_num}</div>
        <div class="question-text">{q_text}</div>
    </div>
    """, unsafe_allow_html=True)
    
    answer = None
    
    if q_type == "choice" and options:
        # Crear opciones para radio buttons
        option_labels = []
        option_values = {}
        
        for opt in options:
            opt_id = opt.get("id", "")
            opt_label = opt.get("label", "")
            opt_desc = opt.get("description", "")
            
            display = f"{opt_id}) {opt_label}"
            if opt_desc:
                display += f" — _{opt_desc}_"
            
            option_labels.append(display)
            option_values[display] = {"id": opt_id, "label": opt_label}
        
        selected = st.radio(
            "Selecciona una opción:",
            option_labels,
            key=f"radio_{q_id}",
            label_visibility="collapsed"
        )
        
        if selected and selected in option_values:
            answer = option_values[selected]["label"]
            
            # Si es "Otro", mostrar campo de texto
            if option_values[selected]["id"] == "other":
                other_text = st.text_input(
                    "Especifica:",
                    key=f"other_{q_id}",
                    placeholder="Escribe aquí..."
                )
                if other_text:
                    answer = other_text
    
    elif q_type == "boolean":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Sí", key=f"yes_{q_id}", use_container_width=True):
                answer = "Sí"
        with col2:
            if st.button("❌ No", key=f"no_{q_id}", use_container_width=True):
                answer = "No"
    
    elif q_type == "text":
        answer = st.text_area(
            "Tu respuesta:",
            key=f"text_{q_id}",
            placeholder=placeholder,
            height=100,
            label_visibility="collapsed"
        )
    
    return answer


def render_questions_wizard():
    """
    Renderiza las preguntas una por una (wizard).
    Retorna True si todas las preguntas fueron respondidas.
    """
    questions = st.session_state.pending_questions
    current_idx = st.session_state.current_question_idx
    answers = st.session_state.question_answers
    
    if not questions:
        return True
    
    total = len(questions)
    
    # Progress bar
    progress = (current_idx) / total
    st.progress(progress, text=f"Pregunta {current_idx + 1} de {total}")
    
    # Mostrar pregunta actual
    if current_idx < total:
        current_q = questions[current_idx]
        q_id = current_q.get("id", f"q{current_idx}")
        
        answer = render_question_card(current_q, current_idx + 1)
        
        # Botón para confirmar respuesta
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("✓ Confirmar respuesta", key=f"confirm_{q_id}", 
                        use_container_width=True, type="primary",
                        disabled=not answer):
                if answer:
                    # Guardar respuesta
                    answers[q_id] = answer
                    st.session_state.question_answers = answers
                    
                    # Avanzar a siguiente pregunta
                    st.session_state.current_question_idx = current_idx + 1
                    st.rerun()
        
        # Mostrar respuestas anteriores
        if answers:
            with st.expander("📝 Respuestas anteriores", expanded=False):
                for i, q in enumerate(questions[:current_idx]):
                    qid = q.get("id", f"q{i}")
                    if qid in answers:
                        st.markdown(f"**{q.get('question', '')}**")
                        st.markdown(f"→ _{answers[qid]}_")
                        st.divider()
        
        return False
    
    return True


def format_answers_for_agent(questions: List[Dict], answers: Dict) -> str:
    """Formatea las respuestas para enviar al agente"""
    lines = []
    for q in questions:
        q_id = q.get("id", "")
        q_text = q.get("question", "")
        answer = answers.get(q_id, "No respondida")
        lines.append(f"- {q_text}: {answer}")
    return "\n".join(lines)


# ============================================
# INTERFAZ PRINCIPAL
# ============================================

def main():
    initialize_session_state()
    graph = load_graph()

    st.title("🤖 ATLAS - Agente Multiagente")
    st.markdown("Sistema inteligente con orchestration multi-step")

    # ============================================
    # SIDEBAR
    # ============================================
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        user_name = st.text_input(
            "Nombre de usuario",
            value=st.session_state.user_name,
            help="Tu nombre para personalizar respuestas",
        )
        st.session_state.user_name = user_name

        st.divider()
        st.subheader("🆔 Sesión")
        st.code(st.session_state.thread_id[:20] + "...", language=None)

        if st.button("🔄 Nueva conversación"):
            for key in ["thread_id", "messages", "window_count", "rolling_summary",
                       "pending_interrupt", "raw_logs", "pending_questions",
                       "pending_content", "current_question_idx", "question_answers"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        st.divider()
        st.subheader("📊 Estado")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mensajes", len(st.session_state.messages))
        with col2:
            pending = len(st.session_state.pending_questions)
            st.metric("Preguntas", pending)

        if st.button("🗑️ Limpiar historial"):
            st.session_state.messages = []
            st.session_state.pending_questions = []
            st.session_state.pending_content = ""
            st.session_state.question_answers = {}
            st.session_state.current_question_idx = 0
            st.rerun()

        st.divider()
        with st.expander("🧾 Logs (debug)"):
            raw = st.session_state.get("raw_logs", [])
            st.code("\n".join(raw[-50:]) if raw else "— sin logs —", language=None)

    # ============================================
    # ÁREA PRINCIPAL
    # ============================================
    st.subheader("💬 Conversación")

    # Mostrar historial de mensajes
    chat_container = st.container(height=500)
    with chat_container:
        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]
            msg_type = message.get("type", "text")

            if msg_type == "separator":
                st.divider()
            elif msg_type == "log":
                st.markdown(content)
            elif msg_type == "announcement":
                with st.chat_message("assistant"):
                    st.info(content)
            else:
                with st.chat_message(role):
                    st.write(content)

    # ============================================
    # MODO PREGUNTAS (Human-in-the-Loop)
    # ============================================
    if st.session_state.pending_questions:
        st.divider()
        
        # Mostrar el contenido del worker ANTES de las preguntas
        if st.session_state.get("pending_content"):
            with st.chat_message("assistant"):
                st.markdown(st.session_state.pending_content)
        
        st.subheader("📋 Necesito más información")
        
        all_answered = render_questions_wizard()
        
        if all_answered:
            # Todas las preguntas respondidas, enviar al agente
            formatted_answers = format_answers_for_agent(
                st.session_state.pending_questions,
                st.session_state.question_answers
            )
            
            # Mostrar resumen de respuestas
            st.success("✅ ¡Gracias! Procesando tu información...")
            
            with st.expander("Ver respuestas enviadas"):
                st.markdown(formatted_answers)
            
            # Enviar como Command(resume)
            try:
                config = get_langgraph_config()
                payload = Command(resume=formatted_answers)
                
                all_events = []
                all_graph_events = []
                
                for event in graph.stream(payload, config=config, stream_mode="updates"):
                    all_graph_events.append(event)
                    
                    # Extraer eventos de log
                    for node_name in ["bootstrap", "plan", "route", "synthesize", "chat",
                                     "research", "tutor", "troubleshooting", "summarizer"]:
                        if node_name in event and isinstance(event[node_name], dict):
                            node_events = event[node_name].get("events", [])
                            all_events.extend(node_events)
                
                # Guardar logs
                for evt in all_events:
                    st.session_state.messages.append({
                        "role": "system",
                        "content": format_event_display(evt),
                        "type": "log",
                    })
                
                # Extraer mensajes finales
                messages, _ = extract_messages_from_events(all_graph_events)
                for msg in messages:
                    if msg:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": msg,
                            "type": "text",
                        })
                
                # Limpiar estado de preguntas
                st.session_state.pending_questions = []
                st.session_state.pending_content = ""  # Limpiar contenido también
                st.session_state.question_answers = {}
                st.session_state.current_question_idx = 0
                st.session_state.pending_interrupt = None
                
                logger.info("✅ Respuestas procesadas")
                
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                st.error(f"❌ Error: {str(e)}")
            
            st.rerun()
        
        return  # No mostrar input normal mientras hay preguntas pendientes

    # ============================================
    # INPUT NORMAL
    # ============================================
    st.divider()
    user_input = st.chat_input(
        "Escribe tu mensaje...",
        disabled=st.session_state.is_loading,
    )

    if user_input:
        st.session_state.is_loading = True
        
        # Añadir mensaje del usuario
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "type": "text",
        })

        try:
            config = get_langgraph_config()
            payload = {
                "messages": [HumanMessage(content=user_input)],
                "user_name": st.session_state.user_name,
                "window_count": st.session_state.window_count,
                "rolling_summary": st.session_state.rolling_summary,
            }

            all_events = []
            all_graph_events = []
            interrupted = False
            extracted_questions = []

            with st.spinner("🔄 Procesando..."):
                for event in graph.stream(payload, config=config, stream_mode="updates"):
                    all_graph_events.append(event)

                    # Detectar interrupt
                    if isinstance(event, dict) and "__interrupt__" in event:
                        interrupted = True
                        st.session_state.pending_interrupt = True
                        
                        # Buscar preguntas estructuradas en los eventos anteriores
                        for prev_event in all_graph_events:
                            questions = extract_questions_from_event(prev_event)
                            if questions:
                                extracted_questions = questions
                                break
                        
                        break

                    # Extraer eventos de log
                    for node_name in ["bootstrap", "plan", "route", "synthesize", "chat",
                                     "research", "tutor", "troubleshooting", "summarizer"]:
                        if node_name in event and isinstance(event[node_name], dict):
                            node_events = event[node_name].get("events", [])
                            all_events.extend(node_events)
                            
                            # Extraer preguntas si existen
                            questions = event[node_name].get("clarification_questions", [])
                            if questions:
                                extracted_questions = questions

            # Guardar logs
            if all_events:
                st.session_state.messages.append({"role": "system", "content": "", "type": "separator"})
                for evt in all_events:
                    st.session_state.messages.append({
                        "role": "system",
                        "content": format_event_display(evt),
                        "type": "log",
                    })

            if interrupted and extracted_questions:
                # Extraer el contenido del worker que generó las preguntas
                worker_content = ""
                for event in all_graph_events:
                    for node_name in ["troubleshooting", "research", "tutor", "chat"]:
                        if node_name in event and isinstance(event[node_name], dict):
                            # Buscar en worker_outputs
                            worker_outputs = event[node_name].get("worker_outputs", [])
                            for wo in worker_outputs:
                                if isinstance(wo, dict) and wo.get("content"):
                                    worker_content = wo.get("content", "")
                                    break
                            if worker_content:
                                break
                    if worker_content:
                        break
                
                # Activar modo preguntas
                st.session_state.pending_questions = extracted_questions
                st.session_state.pending_content = worker_content  # NUEVO: Guardar contenido
                st.session_state.current_question_idx = 0
                st.session_state.question_answers = {}
                logger.info(f"📋 {len(extracted_questions)} preguntas pendientes")
            else:
                # Extraer mensajes finales
                messages, _ = extract_messages_from_events(all_graph_events)
                for msg in messages:
                    if msg:
                        existing = [m.get("content") for m in st.session_state.messages if m.get("type") == "text"]
                        if msg not in existing:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": msg,
                                "type": "text",
                            })
                
                logger.info("✅ Procesamiento completado")

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            st.error(f"❌ Error: {str(e)}")

        finally:
            st.session_state.is_loading = False
            st.rerun()


if __name__ == "__main__":
    main()
