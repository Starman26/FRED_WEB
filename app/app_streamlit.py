"""
Aplicación Streamlit para el desarrollo y prueba 
del sistema multi agente (Nombre por confirmar.

Features:
- Human-in-the-Loop con tarjetas de preguntas estructuradas
- Opciones múltiples con botones/radio
- Máximo 3 preguntas por interacción
- Flujo tipo wizard (una pregunta a la vez)
- Diseño profesional con tonos grises medios
- Animación de carga en eventos en tiempo real
- Sugerencias de seguimiento generadas por el agente

Contirbuciones generales:
- Manejo avanzado de estado en session_state
- Integración con LangGraph y LangChain

"""

# ============================================
# IMPORTS
# ============================================

import re
import os
import sys
import json
import logging
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime
import html as html_escape
import textwrap

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.types import Command
from dotenv import load_dotenv

# ============================================
# CONFIGURACIÓN
# ============================================

load_dotenv()

# Regex para detectar si un mensaje es solo un tag HTML
_TAG_ONLY = re.compile(r"^\s*</?[\w\-]+(?:\s+[^>]*)?>\s*$")

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración de la página Streamlit

st.set_page_config(
    page_title="SENTINEL Multi-Agent System",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded"
)

#============================================
# Estilos CSS personalizados
#============================================

st.markdown("""
<style>
/* =========================================
   PALETA DE COLORES 
   ========================================= */
:root {
    --gray-900: #0f0f0f;
    --gray-850: #1a1a1a;
    --gray-800: #2a2a2a;
    --gray-700: #3a3a3a;
    --gray-600: #4a4a4a;
    --gray-500: #5a5a5a;
    --gray-400: #7a7a7a;
    --gray-300: #9a9a9a;
    --gray-200: #b0b0b0;
    --gray-100: #d0d0d0;
}

/* =========================================
   DETALLES DEL CONTENEDOR PRINCIPAL Y SIDEBAR
   ========================================= */
.main { background-color: #141414; }
[data-testid="stAppViewContainer"] { background-color: #141414; }
[data-testid="stHeader"] { background-color: #141414; }
section[data-testid="stSidebar"] + div { background-color: #141414; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: #2a2a2a; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #d0d0d0 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: #b0b0b0; }

/* =========================================
   TARJETAS DE PREGUNTAS 
   ========================================= */
.question-card {
    background: #3a3a3a;
    border-radius: 4px;
    padding: 24px;
    margin: 16px 0;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.question-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3);
    border-left-color: #cecece;
}

.question-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: #3a3a3a;
    transform: translateX(-100%);
}

.question-card:hover::before {
    animation: shimmer 1.5s ease-in-out;
}

@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.question-title {
    color: #d0d0d0;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.question-title::before {
    font-size: 12px;
    opacity: 0.8;
}

.question-text {
    color: #ffffff;
    font-size: 14px;
    font-weight: 400;
    margin-bottom: 20px;
    line-height: 1.5;
    padding: 8px 0;
}

.question-progress {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
    padding: 8px 0;
}

.progress-bar {
    flex-grow: 1;
    height: 4px;
    background-color: #2a2a2a;
    border-radius: 2px;
    overflow: hidden;
    margin: 0 12px;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #7a7a7a, #9a9a9a);
    border-radius: 2px;
    transition: width 0.6s ease;
}

            
/* =========================================
   CONTROLES Y BOTONES
   ========================================= */
                        
.stButton > button { background-color: #3a3a3a; color: #d0d0d0; border: 1px solid #7a7a7a; border-radius: 8px; transition: all 0.2s; }
.stButton > button:hover { background-color: #4a4a4a; border-color: #9a9a9a; }
.stSelectbox, .stTextInput, .stTextArea { background-color: #2a2a2a; color: #d0d0d0; }
.stSelectbox [data-testid="stSelectboxOption"], .stTextInput input, .stTextArea textarea { background-color: #3a3a3a !important; color: #d0d0d0 !important; }

/* Chat y Logs */
.stChatMessage { background-color: #2a2a2a; border-radius: 8px; padding: 12px; margin-bottom: 8px; border: 1px solid #3a3a3a; }
[data-testid="stChatInput"] { background-color: #1a1a1a !important; }
[data-testid="stChatInput"] textarea { background-color: #2a2a2a !important; color: #d0d0d0 !important; }
[data-testid="stBottomBlockContainer"] { background-color: #141414 !important; }

/* =========================================
   ANIMACION TIMELINE 
   ========================================= */

/* Entrada suave de los items */
@keyframes slideUpFade {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

.timeline-row {
    display: flex;
    align-items: flex-start;
    margin: 0;
    padding-bottom: 0;
    animation: slideUpFade 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

/* Columna de la línea (izquierda) */
.timeline-left {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 24px;
    margin-right: 14px;
    padding-top: 6px;
}

/* El punto (Dot) estático */
.timeline-dot {
    width: 15px;
    height: 15px;
    background-color: #5a5a5a;
    border-radius: 50%;
    border: 2px solid #2a2a2a;
    box-shadow: 0 0 0 1px #5a5a5a;
    z-index: 2;
}

/* El punto activo (Pulsing) */
.timeline-dot.active {
    background-color: #d0d0d0;
    box-shadow: 0 0 0 0 rgba(200, 200, 200, 0.7);
    animation: pulse-gray 2s infinite;
    border-color: #d0d0d0;
}

@keyframes pulse-gray {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(180, 180, 180, 0.5); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(180, 180, 180, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(180, 180, 180, 0); }
}

/* La línea conectora vertical */
.timeline-line {
    width: 1px;
    background-color: #3a3a3a;
    flex-grow: 1;
    min-height: 45px;
    margin-top: -2px;
    z-index: 1;
}

.timeline-line.hidden {
    display: none;
}

/* Contenido de texto */
.timeline-content {
    padding-top: 2px;
    padding-bottom: 16px;
    display: flex;
    flex-direction: column;
}

.timeline-node {
    font-family: 'SF Mono', 'Roboto Mono', monospace;
    font-size: 11px;
    color: #7a7a7a;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 2px;
}

.timeline-msg {
    font-size: 14px;
    color: #c0c0c0;
    line-height: 1.4;
}
@keyframes tw-reveal {
  from { width: 0; }
  to { width: 100%; }
}

@keyframes tw-caret {
  0%, 100% { border-color: transparent; }
  50% { border-color: rgba(208,208,208,0.55); }
}

.typewriter{
  display: inline-block;
  max-width: 100%;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;

  width: 0;
  border-right: 0 !important;
  animation: tw-reveal 0.65s steps(30, end) forwards !important; /* sin tw-caret */

  animation-delay: var(--t-delay, 0s);
}

.timeline-msg.typewriter{
  color: #c0c0c0;
  font-size: 14px;
}

.timeline-node.typewriter{
  color: #7a7a7a;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}

/* =========================================
   SUGERENCIAS DE SEGUIMIENTO
   ========================================= */
.suggestions-container {
    margin-top: 24px;
    padding-top: 20px;
    border-top: 1px solid #2a2a2a;
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: 20px;
}

.suggestions-title {
    color: #aeaeae;
    font-size: 14px;
    font-weight: 400;
    margin-bottom: 12px;
    margin-left: 4px;
    text-transform: none;
    letter-spacing: 0.2px;
    text-align: left;
}
/* =========================================
   BOTONES DE SUGERENCIAS
   ========================================= */

div[data-testid="stVerticalBlock"] button[kind="secondary"] {
  width: 100% !important;
  background: transparent !important;
  border: 0 !important;
  border-top: 1px solid #262626 !important;
  border-radius: 0 !important;
  padding: 12px 6px !important;
  color: #b5b5b5 !important;
  font-size: 15px !important;
  font-weight: 400;
  
  display: flex !important;
  justify-content: flex-start !important;
  
  /* IMPORTANTE: Necesario para posicionar la flecha absoluta respecto al botón */
  position: relative !important; 
  overflow: hidden !important; /* Opcional: evita desbordes si la animación es muy larga */
}

div[data-testid="stVerticalBlock"] button[kind="secondary"] > div {
  width: 100% !important;
  display: flex !important;
  justify-content: flex-start !important;
}

div[data-testid="stVerticalBlock"] button[kind="secondary"] p {
  width: 100% !important;
  margin: 0 !important;
  text-align: left !important;
  color: #b5b5b5 !important;
  font-size: 14px !important;
  transition: color 0.2s ease; /* Suaviza el cambio de color del texto */
}

/* Estado Hover del Botón */
div[data-testid="stVerticalBlock"] button[kind="secondary"]:hover {
  background: #171717 !important;
  color: #d0d0d0 !important;
}

div[data-testid="stVerticalBlock"] button[kind="secondary"]:hover p {
  color: #d0d0d0 !important;
}

/* =========================================
   Flecha a la derecha del botón
   ========================================= */

/* Crear la flecha invisible por defecto */
div[data-testid="stVerticalBlock"] button[kind="secondary"]::after {
  content: "→";             
  position: absolute;       
  right: 20px;              
  top: 50%;                 
  transform: translateY(-50%);
  
  font-size: 18px;
  color: #d0d0d0;           
  opacity: 0;               
  transition: all 0.3s ease; 
}

/* Hacer visible la flecha al hacer Hover y moverla */
div[data-testid="stVerticalBlock"] button[kind="secondary"]:hover::after {
  opacity: 1;              
  right: 10px;              


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

def normalize_markdown(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\r\n", "\n")
    # Si NO trae code fences, quitamos indent común (evita que todo se vuelva "code block")
    if "```" not in s:
        s = textwrap.dedent(s)
    return s.strip("\n")

def render_assistant_markdown(content: str, label: str = "SENTINEL"):
    st.markdown(sentinel_badge_html(label), unsafe_allow_html=True)
    st.markdown(normalize_markdown(content))
def should_skip_typewriter(content: str) -> bool:
    """Evita animar outputs con markdown pesado (code fences/tablas)."""
    if not content:
        return True
    s = str(content)
    if "```" in s:
        return True
    if "\n|" in s and "|---" in s:
        return True
    return False

def render_assistant_typed_html(content: str, label: str = "SENTINEL", delay_s: float = 0.0):
    badge = sentinel_badge_html(label)
    body = text_to_safe_html(content)

    # pequeño jitter para forzar restart del CSS animation
    jitter = (uuid.uuid4().int % 97) / 100000  # 0.0 a 0.00096
    dur = 0.55 + jitter

    html = f"""
    <div class="agent-wrap">
      {badge}
      <div class="agent-row">
        <div class="agent-bubble">
          <div class="typewriter-multi" style="--t-delay:{delay_s}s; --tw-dur:{dur}s;">{body}</div>
        </div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ============================================
# FUNCIONES AUXILIARES
# ============================================

def load_graph():
    """
    Carga el grafo con un checkpointer persistente en session_state.
    """
    if "langgraph_instance" in st.session_state:
        return st.session_state.langgraph_instance
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import StateGraph, END
        from src.agent.state import AgentState
        from src.agent.bootstrap import bootstrap_node
        from src.agent.orchestrator import orchestrator_plan_node, orchestrator_route_node, synthesize_node
        from src.agent.nodes.human_input import human_input_node
        from src.agent.nodes.intent_analyzer import intent_analyzer_node
        from src.agent.workers.chat_node import chat_node
        from src.agent.workers.research_node import research_node
        from src.agent.workers.tutor_node import tutor_node
        from src.agent.workers.troubleshooter_node import troubleshooter_node
        from src.agent.workers.summarizer_node import summarizer_node
        
        if "langgraph_checkpointer" not in st.session_state:
            st.session_state.langgraph_checkpointer = MemorySaver()
            logger.info("Checkpointer creado")
        
        workflow = StateGraph(AgentState)
        
        workflow.add_node("bootstrap", bootstrap_node)
        workflow.add_node("intent_analyzer", intent_analyzer_node)
        workflow.add_node("plan", orchestrator_plan_node)
        workflow.add_node("route", orchestrator_route_node)
        workflow.add_node("synthesize", synthesize_node)
        workflow.add_node("human_input", human_input_node)
        workflow.add_node("chat", chat_node)
        workflow.add_node("research", research_node)
        workflow.add_node("tutor", tutor_node)
        workflow.add_node("troubleshooting", troubleshooter_node)
        workflow.add_node("summarizer", summarizer_node)
        
        workflow.set_entry_point("bootstrap")
        workflow.add_edge("bootstrap", "intent_analyzer")
        workflow.add_edge("intent_analyzer", "plan")
        
        def route_from_plan(state):
            next_node = state.get("next", "chat")
            valid = {"chat", "research", "tutor", "troubleshooting", "summarizer"}
            if next_node in valid:
                return next_node
            elif next_node in ("END", "__end__", "end"):
                return "END"
            return "chat"
        
        def route_from_orchestrator(state):
            if state.get("needs_human_input"):
                return "human_input"
            next_node = state.get("next", "END")
            valid = {"chat", "research", "tutor", "troubleshooting", "summarizer", "synthesize", "human_input", "END"}
            return next_node if next_node in valid else "END"
        
        def route_after_human_input(state):
            if state.get("orchestration_plan"):
                return "route"
            return "plan"
        
        workflow.add_conditional_edges("plan", route_from_plan, {
            "chat": "chat", "research": "research", "tutor": "tutor", 
            "troubleshooting": "troubleshooting", "summarizer": "summarizer", "END": END
        })
        
        for worker in ["chat", "research", "tutor", "troubleshooting", "summarizer"]:
            workflow.add_edge(worker, "route")
        
        workflow.add_conditional_edges("route", route_from_orchestrator, {
            "chat": "chat", "research": "research", "tutor": "tutor", 
            "troubleshooting": "troubleshooting", "summarizer": "summarizer", 
            "synthesize": "synthesize", "human_input": "human_input", "END": END
        })
        
        workflow.add_edge("synthesize", END)
        workflow.add_conditional_edges("human_input", route_after_human_input, {
            "route": "route", "plan": "plan"
        })
        
        graph = workflow.compile(checkpointer=st.session_state.langgraph_checkpointer)
        
        st.session_state.langgraph_instance = graph
        logger.info("Grafo cargado con checkpointer persistente")
        return graph
        
    except Exception as e:
        logger.error(f"Error al cargar el grafo: {str(e)}")
        st.error(f"Error al cargar el grafo: {str(e)}")
        st.stop()


def initialize_session_state():
    defaults = {
        "messages": [],
        "user_name": "User",
        "learning_style": {
            "type": "visual",
            "pace": "medium",
            "depth": "intermediate",
            "examples": "practical"
        },
        "window_count": 0,
        "rolling_summary": "",
        "is_loading": False,
        "pending_interrupt": None,
        "raw_logs": [],
        "pending_questions": [],
        "pending_content": "",
        "current_question_idx": 0,
        "question_answers": {},
        "follow_up_suggestions": [], 
    }
    
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default
    
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"streamlit-{uuid.uuid4()}"
        logger.info(f"Nueva sesión: {st.session_state.thread_id}")
    
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

def clean_message_content(content: str) -> str | None:
    if not content:
        return None

    s = str(content).strip()
    if s.lower() in {"none", "null", ""}:
        return None

    # Si el mensaje ES un tag suelto (</div>, <br/>, <p>, etc.), lo tiramos
    if _TAG_ONLY.match(s):
        return None

    # Si viene HTML inline (tu UI), lo tiramos
    if "<div style=" in s or "<span style=" in s:
        return None

    wizard_indicators = [
        "Pregunta 1: ¿",
        "Pregunta 2: ¿",
        "Pregunta 3: ¿",
        "Respuestas del usuario:\n",
        "Por favor, proporciona la información solicitada",
        "Necesito saber qué estación presenta problemas y qué tipo de fallo",
        '"question_set"',
        '"wizard_mode"',
        "- Estación 1\n- Estación 2\n- Estación 3",
        "- PLC: Controlador lógico programable\n",
        "- Cobot: Robot colaborativo\n",
    ]
    if any(ind in s for ind in wizard_indicators):
        return None

    if len(s) < 10:
        return None

    return s

def sentinel_badge_html(label: str = "SENTINEL") -> str:
    eye_svg = """
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z"></path>
      <path d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z"></path>
    </svg>
    """
    return f"""
    <div class="agent-badge">
      <span class="agent-badge-icon">{eye_svg}</span>
      <span>{html_escape.escape(str(label))}</span>
    </div>
    """

def text_to_safe_html(content: str) -> str:
    """
    Convierte texto a HTML seguro, soportando bullets tipo '- '.
    (Sin markdown complejo; mantiene estilo como en tu screenshot.)
    """
    if content is None:
        return ""

    raw = str(content).strip()
    if raw in ("None", "null", ""):
        return ""

    esc = raw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    lines = esc.splitlines()

    html_parts = []
    in_list = False

    def close_list():
        nonlocal in_list
        if in_list:
            html_parts.append("</ul>")
            in_list = False

    for line in lines:
        stripped = line.strip()

        is_bullet = stripped.startswith("- ") or stripped.startswith("* ")
        if is_bullet:
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            html_parts.append(f"<li>{stripped[2:].strip()}</li>")
            continue

        close_list()

        if stripped == "":
            html_parts.append("<br/>")
        else:
            html_parts.append(f"<div>{stripped}</div>")

    close_list()
    return "".join(html_parts)

def render_sentinel_message_html(content: str) -> str:
    badge = sentinel_badge_html("SENTINEL")
    body = text_to_safe_html(content)
    return f"""
    <div class="agent-wrap">
      {badge}
      <div class="agent-row">
        <div class="agent-bubble">{body}</div>
      </div>
    </div>
    """
def format_event_display(event: Dict, is_last: bool = False, is_loading: bool = False) -> str:
    node = event.get("source", event.get("node", "?")).upper()
    message = event.get("content", event.get("message", ""))

    message = html_escape.escape(str(message))
    node = html_escape.escape(str(node))

    dot_class = "timeline-dot active" if (is_loading and is_last) else "timeline-dot"
    line_class = "timeline-line hidden" if is_last else "timeline-line"

    tw = " typewriter" if (is_loading and is_last) else ""

    return f"""
    <div class="timeline-row">
        <div class="timeline-left">
            <div class="{dot_class}"></div>
            <div class="{line_class}"></div>
        </div>
        <div class="timeline-content">
            <div class="timeline-node{tw}" style="--t-delay:0s;">{node}</div>
            <div class="timeline-msg{tw}" style="--t-delay:0.06s;">{message}</div>
        </div>
    </div>
    """


def extract_messages_from_events(events: List[Dict]) -> tuple[List[str], Dict]:
    messages = []
    seen = set()
    final_state = {}

    node_names = ["plan","route","synthesize","chat","tutor","troubleshooting","summarizer","research","human_input"]

    for event in events:
        if not isinstance(event, dict):
            continue

        for node_name in node_names:
            if node_name not in event:
                continue

            node_data = event[node_name]
            if isinstance(node_data, dict):
                final_state.update(node_data)

                msgs = node_data.get("messages", [])
                if isinstance(msgs, list):
                    for m in msgs:
                        c = extract_message_content(m)
                        c = clean_message_content(c) if c else None
                        if not c:
                            continue

                        key = c.strip()
                        if key in seen:
                            continue
                        seen.add(key)
                        messages.append(key)

    return messages, final_state



def extract_suggestions_from_events(events: List[Dict]) -> List[str]:
    """Extrae las sugerencias de seguimiento de los eventos del grafo"""
    suggestions = []
    node_names = ["chat", "tutor", "troubleshooting", "research", "summarizer"]
    
    for event in events:
        if isinstance(event, dict):
            for node_name in node_names:
                if node_name in event:
                    node_data = event[node_name]
                    if isinstance(node_data, dict):
                        # Buscar sugerencias directamente en el nodo
                        node_suggestions = node_data.get("follow_up_suggestions", [])
                        if node_suggestions:
                            suggestions = node_suggestions
    
    return suggestions[:3]  # Máximo 3 sugerencias


def extract_questions_from_event(event: Dict) -> List[Dict]:
    """Extrae preguntas estructuradas del evento"""
    if not isinstance(event, dict):
        return []
    
    for node_name in ["human_input", "troubleshooting", "tutor"]:
        if node_name in event:
            node_data = event[node_name]
            if isinstance(node_data, dict):
                questions = node_data.get("clarification_questions", [])
                if questions:
                    return questions
    return []


def render_suggestions(suggestions: List[str]):
    if not suggestions:
        return

    st.markdown("""
    <div class="suggestions-container">
        <div class="suggestions-title">Suggested follow-up questions</div>
    </div>
    """, unsafe_allow_html=True)

    for i, suggestion in enumerate(suggestions):
        if isinstance(suggestion, dict):
            text = suggestion.get("text", suggestion.get("question", str(suggestion)))
        else:
            text = str(suggestion)

        if st.button(
            text,
            key=f"suggestion_{i}_{hash(text)}",
            type="secondary",
            use_container_width=True,
        ):
            st.session_state.pending_suggestion = text
            st.rerun()


def render_questions_wizard() -> bool:
    """
    Renderiza el wizard de preguntas usando componentes nativos de Streamlit.
    Retorna True si todas las preguntas han sido respondidas.
    """
    if not st.session_state.pending_questions:
        return False

    questions = st.session_state.pending_questions
    current_idx = st.session_state.current_question_idx

    if current_idx >= len(questions):
        return True

    current_q = questions[current_idx]
    total = len(questions)
    progress = ((current_idx) / total) * 100

    with st.container():
        st.markdown(f"""
        <div class="question-card">
            <div class="question-title">QUESTION {current_idx + 1} OF {total}</div>
            <div class="question-text">{html_escape.escape(current_q.get("question", current_q.get("text", "")))}</div>
            <div class="question-progress">
                <span style="color: #9a9a9a; font-size: 12px;">Progress</span>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {progress}%"></div>
                </div>
                <span style="color: #9a9a9a; font-size: 12px;">{current_idx + 1}/{total}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    question_id = current_q.get("id", f"q{current_idx}")
    options = current_q.get("options", current_q.get("choices", []))
    option_data = []

    if options and len(options) > 0:
        first_opt = options[0]
        if isinstance(first_opt, dict):
            for opt in options:
                label = opt.get("label", opt.get("text", opt.get("name", "")))
                value = opt.get("value", opt.get("id", label))
                option_data.append((label, value))
        else:
            option_data = [(opt, opt) for opt in options]

    if option_data:
        cols = st.columns(2)
        for i, (label, value) in enumerate(option_data):
            col_idx = i % 2
            with cols[col_idx]:
                if st.button(
                    label,
                    key=f"opt_{question_id}_{i}",
                    use_container_width=True,
                    type="primary" if i == 0 else "secondary"
                ):
                    st.session_state.question_answers[question_id] = value
                    st.session_state.current_question_idx += 1
                    st.rerun()
    else:
        user_answer = st.text_area(
            "Your response:",
            key=f"q_{question_id}",
            height=80,
            placeholder="Type your answer here...",
            label_visibility="visible"
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Continue", key=f"next_{question_id}", use_container_width=True):
                if user_answer:
                    st.session_state.question_answers[question_id] = user_answer
                    st.session_state.current_question_idx += 1
                    st.rerun()
                else:
                    st.warning("Please provide an answer before continuing.")

        with col2:
            if st.button("Skip question", key=f"skip_{question_id}", use_container_width=True):
                st.session_state.question_answers[question_id] = "Not answered"
                st.session_state.current_question_idx += 1
                st.rerun()

    return False


def format_answers_for_agent(questions: List[Dict], answers: Dict) -> str:
    """Formatea las respuestas para enviar al agente"""
    formatted = "User responses:\n\n"
    
    for q in questions:
        q_id = q.get("id", "")
        q_text = q.get("question", "")
        answer = answers.get(q_id, "Not answered")
        formatted += f"**{q_text}**\n{answer}\n\n"
    
    return formatted


# ============================================
# MAIN
# ============================================

def main():
    initialize_session_state()
    graph = load_graph()
    
    # ============================================
    # SIDEBAR
    # ============================================
    with st.sidebar:
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] .stTextInput input {
                background-color: #2b2b2b !important;
                color: #ffffff !important;
                border: 1px solid #4a4a4a;
            }
            [data-testid="stSidebar"] div[data-baseweb="select"] > div {
                background-color: #2b2b2b !important;
                color: #ffffff !important;
                border: 1px solid #4a4a4a;
            }
            [data-testid="stSidebar"] .stCode {
                background-color: #2b2b2b !important;
            }
            [data-testid="stSidebar"] .stCode pre {
                background-color: #2b2b2b !important;
            }
            /* =========================================
                AGENT BRAND (SENTINEL HEADER)
                ========================================= */
                .agent-wrap {
                    margin: 12px 0;
                }

                .agent-badge {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin: 0 0 8px 0;
                    color: #d0d0d0;
                    font-size: 12px;
                    font-weight: 700;
                    letter-spacing: 0.9px;
                    text-transform: uppercase;
                    opacity: 0.95;
                }

                .agent-badge-icon {
                    width: 28px;
                    height: 28px;
                    border-radius: 8px;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                }

                .agent-badge-icon svg {
                    width: 16px;
                    height: 16px;
                    fill: none;
                    stroke: #d0d0d0;
                    stroke-width: 2;
                }

                .agent-row {
                    display: flex;
                    justify-content: flex-start;
                }

                .agent-bubble {
                    background: transparent;
                    border: 0;
                    border-radius: 0;
                    padding: 0;
                    max-width: 78%;
                    color: #e6e6e6;
                    word-wrap: break-word;
                    line-height: 1.55;
                }


                .agent-bubble ul {
                    margin: 10px 0 0 18px;
                    padding: 0;
                }

                .agent-bubble li {
                    margin: 6px 0;
                }
                /* ===== TYPEWRITER MULTILINE (GLOBAL) ===== */
                @keyframes tw-clip-reveal {
                from { clip-path: inset(0 100% 0 0); }
                to   { clip-path: inset(0 0 0 0); }
                }
                @keyframes tw-caret-blink {
                0%, 100% { opacity: 0; }
                50% { opacity: 1; }
                }

                .typewriter-multi{
                display:block;
                white-space:pre-wrap;
                overflow:hidden;
                position:relative;

                clip-path: inset(0 100% 0 0);
                animation-name: tw-clip-reveal;
                animation-duration: var(--tw-dur, 0.55s);
                animation-timing-function: ease;
                animation-fill-mode: forwards;
                animation-delay: var(--t-delay, 0s);
                }



                /* =========================================
                FIX: typewriter en timeline (evita saltos raros)
                ========================================= */
                .timeline-node.typewriter,
                .timeline-msg.typewriter{
                display:inline-block;
                max-width:100%;
                overflow:hidden;
                white-space:nowrap;
                text-overflow:ellipsis;
                }

                /* =========================================
                TYPEWRITER MULTILÍNEA PARA OUTPUT DEL AGENTE
                (sin pop, sin background, sin borde)
                ========================================= */
                @keyframes tw-clip-reveal {
                from { clip-path: inset(0 100% 0 0); }
                to   { clip-path: inset(0 0 0 0); }
                }

                @keyframes tw-caret-blink {
                0%, 100% { opacity: 0; }
                50% { opacity: 1; }
                }

                .typewriter-multi{
                display:block;
                white-space:pre-wrap;     /* respeta saltos de línea */
                overflow:hidden;
                position:relative;

                clip-path: inset(0 100% 0 0);
                animation: tw-clip-reveal 0.55s ease forwards;
                animation-delay: var(--t-delay, 0s);
                }



            </style>
            """,
            unsafe_allow_html=True
        )

        st.title("SENTINEL")
        
        st.markdown(
            "**Multi-Agent Operations System**\n\n"
            "Technical diagnostics and laboratory operations interface."
        )
        
        st.divider()
        
        st.subheader("User Profile")
        user_name = st.text_input(
            "Display Name",
            value=st.session_state.user_name,
            placeholder="Enter your name"
        )
        st.session_state.user_name = user_name

        st.subheader("Cognitive Settings")

        current_style = st.session_state.learning_style
        if isinstance(current_style, str):
            current_style = {
                "type": "visual",
                "pace": "medium",
                "depth": "intermediate",
                "examples": "practical"
            }

        learning_type = st.selectbox(
            "Learning Modality",
            options=["visual", "auditory", "kinesthetic", "reading"],
            index=["visual", "auditory", "kinesthetic", "reading"].index(current_style.get("type", "visual")),
        )

        pace = st.selectbox(
            "Interaction Pace",
            options=["slow", "medium", "fast"],
            index=["slow", "medium", "fast"].index(current_style.get("pace", "medium")),
        )

        depth = st.selectbox(
            "Technical Depth",
            options=["basic", "intermediate", "advanced"],
            index=["basic", "intermediate", "advanced"].index(current_style.get("depth", "intermediate")),
        )

        examples = st.selectbox(
            "Example Orientation",
            options=["theoretical", "practical", "both"],
            index=["theoretical", "practical", "both"].index(current_style.get("examples", "practical")),
        )

        st.session_state.learning_style = {
            "type": learning_type,
            "pace": pace,
            "depth": depth,
            "examples": examples
        }

        st.divider()

        st.subheader("Session Management")
        st.caption("Active Thread ID:")
        st.code(st.session_state.thread_id[:20] + "...", language=None)

        if st.button("Reset System State", type="primary", use_container_width=True):
            keys_to_reset = [
                "thread_id", "messages", "window_count", "rolling_summary",
                "pending_interrupt", "raw_logs", "pending_questions",
                "pending_content", "current_question_idx", "question_answers",
                "langgraph_checkpointer", "langgraph_instance", "follow_up_suggestions"
            ]
            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_questions = []
            st.session_state.pending_content = ""
            st.session_state.question_answers = {}
            st.session_state.current_question_idx = 0
            st.session_state.follow_up_suggestions = []
            st.rerun()

        st.divider()

        st.subheader("System Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Messages", len(st.session_state.messages))
        with col2:
            pending = len(st.session_state.get("pending_questions", []))
            st.metric("Pending Queries", pending)

        with st.expander("System Logs (Debug)"):
            raw = st.session_state.get("raw_logs", [])
            st.code("\n".join(raw[-50:]) if raw else "No logs available", language="text")

    # ============================================
    # ÁREA PRINCIPAL
    # ============================================
    
    chat_container = st.container()
    with chat_container:
        for msg_idx, message in enumerate(st.session_state.messages):
            role = message["role"]
            content = message["content"]
            msg_type = message.get("type", "text")

            if msg_type == "separator":
                st.divider()
            elif msg_type == "log":
                st.markdown(content, unsafe_allow_html=True)
            elif msg_type == "logs_group":
                with st.expander("Show Sentinel Thoughts", expanded=False):
                    st.markdown(content, unsafe_allow_html=True)


            elif msg_type == "announcement":
                    if message.get("animate") and not should_skip_typewriter(content):
                        render_assistant_typed_html(content, "SENTINEL", delay_s=0.0)
                        message["animate"] = False
                    else:
                        render_assistant_markdown(content, "SENTINEL")


            elif role == "user":
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-end; margin: 12px 0;">
                    <div style="background-color: #3a3a3a; color: #e0e0e0; padding: 12px 16px; border-radius: 18px 18px 4px 18px; max-width: 70%; word-wrap: break-word;">
                        {html_escape.escape(str(content))}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                    if message.get("animate") and not should_skip_typewriter(content):
                        render_assistant_typed_html(content, "SENTINEL", delay_s=0.0)
                        message["animate"] = False
                    else:
                        render_assistant_markdown(content, "SENTINEL")



        # Mostrar sugerencias después del último mensaje del asistente
        if st.session_state.follow_up_suggestions and not st.session_state.pending_questions:
            render_suggestions(st.session_state.follow_up_suggestions)

    # ============================================
    # MODO PREGUNTAS (Human-in-the-Loop)
    # ============================================
    if st.session_state.pending_questions:
        st.divider()
        
        pending_content = st.session_state.get("pending_content", "")
        
        if pending_content in [None, "None", "null", ""]:
            pending_content = ""
        
        is_just_questions = not pending_content or any(indicator in str(pending_content) for indicator in [
            "Paso 1 de 3",
            "[●○○]",
            "[○○○]", 
            "Pregunta 1: ¿",
            "Pregunta 2: ¿",
            "Necesito saber qué estación presenta problemas",
            '"question_set"',
            '"wizard_mode"',
        ])
        
        if pending_content and not is_just_questions:
            st.markdown(pending_content)
        
        all_answered = render_questions_wizard()
        
        if all_answered:
            formatted_answers = format_answers_for_agent(
                st.session_state.pending_questions,
                st.session_state.question_answers
            )
            
            st.success("Sentinel is processing your responses...")
            
            with st.expander("View submitted answers"):
                st.markdown(formatted_answers)
            
            try:
                config = get_langgraph_config()
                payload = Command(resume=formatted_answers)
                
                all_events = []
                all_graph_events = []
                
                for event in graph.stream(payload, config=config, stream_mode="updates"):
                    all_graph_events.append(event)
                    
                    for node_name in ["bootstrap", "plan", "route", "synthesize", "chat",
                                     "research", "tutor", "troubleshooting", "summarizer"]:
                        if node_name in event and isinstance(event[node_name], dict):
                            node_events = event[node_name].get("events", [])
                            all_events.extend(node_events)
                
                if all_events:
                    logs_html = ""
                    for idx, evt in enumerate(all_events):
                        is_last = (idx == len(all_events) - 1)
                        logs_html += format_event_display(evt, is_last, is_loading=False)
                    
                    st.session_state.messages.append({
                        "role": "system",
                        "content": logs_html,
                        "type": "logs_group",
                        "animate": True,
                    })
                
                # Extraer sugerencias
                suggestions = extract_suggestions_from_events(all_graph_events)
                # Si no hay sugerencias, agregar defaults post-HITL
                if not suggestions:
                    suggestions = [
                        "Check the status of other stations",
                        "Run a diagnostic on similar equipment",
                        "View the complete lab overview"
                    ]
                st.session_state.follow_up_suggestions = suggestions
                
                messages, _ = extract_messages_from_events(all_graph_events)
                for msg in messages:
                    if msg:
                        cleaned_msg = clean_message_content(msg)
                        if cleaned_msg and not cleaned_msg.startswith('<div'):
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": cleaned_msg,
                                "type": "text",
                                "animate": True,
                            })
                
                st.session_state.pending_questions = []
                st.session_state.pending_content = ""
                st.session_state.question_answers = {}
                st.session_state.current_question_idx = 0
                st.session_state.pending_interrupt = None
                
                logger.info("Respuestas procesadas")
                
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                st.error(f"Error: {str(e)}")
            
            st.rerun()
        
        return

    # ============================================
    # INPUT NORMAL
    # ============================================
    st.divider()
    
    # Manejar sugerencia clickeada
    if "pending_suggestion" in st.session_state and st.session_state.pending_suggestion:
        user_input = st.session_state.pending_suggestion
        st.session_state.pending_suggestion = None
    else:
        user_input = st.chat_input(
            "Type your message...",
            disabled=st.session_state.is_loading,
        )

    if user_input:
        st.session_state.is_loading = True
        st.session_state.follow_up_suggestions = []  # Limpiar sugerencias anteriores
        
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "type": "text",
        })
        
        st.markdown(f"""
        <div style="display:flex;justify-content:flex-end;margin:12px 0;">
            <div style="background-color:#3a3a3a;color:#e0e0e0;padding:12px 16px;border-radius:18px 18px 4px 18px;max-width:70%;word-wrap:break-word;">
                {html_escape.escape(str(user_input))}
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            config = get_langgraph_config()
            payload = {
                "messages": [HumanMessage(content=user_input)],
                "user_name": st.session_state.user_name,
                "learning_style": st.session_state.learning_style,
                "window_count": st.session_state.window_count,
                "rolling_summary": st.session_state.rolling_summary,
            }

            all_events = []
            all_graph_events = []
            interrupted = False
            extracted_questions = []

            events_placeholder = st.empty()
            
            for event in graph.stream(payload, config=config, stream_mode="updates"):
                all_graph_events.append(event)

                if isinstance(event, dict) and "__interrupt__" in event:
                    interrupted = True
                    st.session_state.pending_interrupt = True
                    
                    for prev_event in all_graph_events:
                        questions = extract_questions_from_event(prev_event)
                        if questions:
                            extracted_questions = questions
                            break
                    
                    break

                for node_name in ["bootstrap", "plan", "route", "synthesize", "chat",
                                 "research", "tutor", "troubleshooting", "summarizer"]:
                    if node_name in event and isinstance(event[node_name], dict):
                        node_events = event[node_name].get("events", [])
                        for evt in node_events:
                            all_events.append(evt)
                        
                        questions = event[node_name].get("clarification_questions", [])
                        if questions:
                            extracted_questions = questions
                
                if all_events:
                    events_parts = []
                    for idx, evt in enumerate(all_events):
                        is_last = (idx == len(all_events) - 1)
                        events_parts.append(format_event_display(evt, is_last, is_loading=True))
                    
                    events_html = "".join(events_parts)
                    events_placeholder.markdown(events_html, unsafe_allow_html=True)
            
            events_placeholder.empty()

            if all_events:
                logs_html = ""
                for idx, evt in enumerate(all_events):
                    is_last = (idx == len(all_events) - 1)
                    logs_html += format_event_display(evt, is_last, is_loading=False)
                
                st.session_state.messages.append({
                    "role": "system",
                    "content": logs_html,
                    "type": "logs_group",
                })

            if interrupted and extracted_questions:
                worker_content = ""
                for event in all_graph_events:
                    for node_name in ["chat", "research", "tutor", "troubleshooting", "summarizer"]:
                        if node_name in event and isinstance(event[node_name], dict):
                            msgs = event[node_name].get("messages", [])
                            if msgs:
                                for m in msgs:
                                    content = extract_message_content(m)
                                    if content:
                                        worker_content = content
                                        break
                
                skip_content_indicators = [
                    "Pregunta 1: ¿",
                    "Pregunta 2: ¿",
                    "Pregunta 3: ¿",
                    "Necesito saber qué estación presenta problemas",
                    "Por favor, proporciona la información solicitada",
                    '"question_set"',
                    '"wizard_mode"',
                    "Respuestas del usuario:\n",
                    "- Estación 1\n- Estación 2\n",
                ]
                
                should_skip_content = False
                if not worker_content or str(worker_content).strip() in ["None", "null", ""]:
                    should_skip_content = True
                else:
                    for indicator in skip_content_indicators:
                        if indicator in str(worker_content):
                            should_skip_content = True
                            break
                
                st.session_state.pending_questions = extracted_questions
                st.session_state.pending_content = "" if should_skip_content else worker_content
                st.session_state.current_question_idx = 0
                st.session_state.question_answers = {}
                
                logger.info(f"Preguntas pendientes: {len(extracted_questions)}")
            
            else:
                # Extraer sugerencias de seguimiento
                suggestions = extract_suggestions_from_events(all_graph_events)
                st.session_state.follow_up_suggestions = suggestions
                
                messages, final_state = extract_messages_from_events(all_graph_events)
                for msg in messages:
                    if msg:
                        cleaned_msg = clean_message_content(msg)
                        if cleaned_msg and not cleaned_msg.startswith('<div'):
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": cleaned_msg,
                                "type": "text",
                                "animate": True,
                            })
                
                if final_state:
                    st.session_state.window_count = final_state.get("window_count", st.session_state.window_count)
                    st.session_state.rolling_summary = final_state.get("rolling_summary", st.session_state.rolling_summary)
            
            st.session_state.is_loading = False
            logger.info("Procesamiento completado")

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            st.error(f"Error: {str(e)}")
            st.session_state.is_loading = False
        
        st.rerun()


if __name__ == "__main__":
    main()
