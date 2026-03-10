"""
---------------------------------------------------------------------------
api_server.py
---------------------------------------------------------------------------

SSE Event Types sent to client:
    event: thinking        --> Agent is processing (node started)
    event: node_update     --> Node completed with events
    event: suggestions     --> Follow-up suggestions
    event: questions       --> HITL clarification questions (interrupt)
    event: response        --> Final assistant message
    event: tool_lifecycle  --> Tool execution lifecycle (planned/executing/verifying/completed/failed)
    event: audio_chunk     --> Base64-encoded MP3 audio chunk (voice mode only)
    event: audio_done      --> TTS streaming complete (voice mode only)
    event: tokens          --> Token usage info
    event: error           --> Error occurred
    event: done            --> Stream complete
---------------------------------------------------------------------------
---------------------------------------------------------------------------

"""

import os
import json
import base64
import asyncio
import logging
import threading
from collections import deque
from datetime import datetime
from typing import AsyncGenerator, Dict, Any, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request, Depends, Header, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uuid
from pydantic import BaseModel, Field
import uvicorn

# ── Agent imports ──
from langchain_core.messages import HumanMessage, AIMessage

# ── Config ──
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("api_server")

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:4173,http://localhost:3000,https://orion.ai").split(",")
API_KEY = os.getenv("API_KEY", "")  # Optional: set for auth

# ============================================
# MODELS
# ============================================
class ChatRequest(BaseModel):
    message: Union[str, List[dict]] = Field(
        ...,
        description="Text string or multimodal content blocks "
        '[{"type":"text","text":"..."},{"type":"image_url","image_url":{"url":"data:..."}}]',
    )
    user_id: Optional[str] = None
    user_name: Optional[str] = "Usuario"
    session_id: Optional[str] = None
    interaction_mode: Optional[str] = "chat"
    llm_model: Optional[str] = ""
    automation_id: Optional[str] = None
    automation_md_content: Optional[str] = None
    automation_step: Optional[int] = None
    robot_ids: Optional[List[str]] = None
    equipment_id: Optional[str] = None  
    voice_enabled: bool = False
    voice_id: Optional[str] = None


class ConfirmRequest(BaseModel):
    session_id: str
    answers: Union[dict, list]  
    completed: bool = True
    cancelled: bool = False
    interaction_mode: Optional[str] = "chat"


# ============================================
# APP
# ============================================
app = FastAPI(
    title="ORION Agent API",
    description="SSE streaming API for the orion multi-agent system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Robot bridge registry 
BRIDGE_TOKEN = os.getenv("BRIDGE_TOKEN", "dev-bridge-token")
ROBOT_CONNECTIONS: Dict[str, WebSocket] = {}   # robot_id → WebSocket
ROBOT_METADATA: Dict[str, dict] = {}           # robot_id → {type, model, capabilities, last_heartbeat, ...}
PENDING_COMMANDS: Dict[str, dict] = {}          # cmd_id  → {"event": Event, "result": dict|None}
BRIDGE_ANOMALY_QUEUE: deque = deque(maxlen=100)  # Anomalies from bridge status_update
_main_loop: asyncio.AbstractEventLoop = None

#  Stream callback registry (out-of-state, avoids msgpack serialization) 
_STREAM_CALLBACKS: Dict[str, Any] = {}
_STREAM_CALLBACKS_LOCK = threading.Lock()


def register_stream_callback(session_id: str, callback):
    """Registra un callback de streaming para una sesión."""
    with _STREAM_CALLBACKS_LOCK:
        _STREAM_CALLBACKS[session_id] = callback


def unregister_stream_callback(session_id: str):
    """Elimina el callback de streaming de una sesión."""
    with _STREAM_CALLBACKS_LOCK:
        _STREAM_CALLBACKS.pop(session_id, None)


def get_stream_callback(session_id: str):
    """Obtiene el callback de streaming de una sesión (thread-safe)."""
    with _STREAM_CALLBACKS_LOCK:
        return _STREAM_CALLBACKS.get(session_id)


@app.on_event("startup")
async def _capture_loop():
    """Store reference to the main async event loop so worker threads can schedule coroutines."""
    global _main_loop
    _main_loop = asyncio.get_running_loop()


def get_main_loop() -> asyncio.AbstractEventLoop:
    """Return the main event loop (used by edge_router sync wrappers)."""
    return _main_loop


async def send_robot_command(robot_id: str, command: str, params: dict = None, timeout: float = 10.0) -> dict:
    """Send a command to a connected robot via WebSocket bridge and wait for response."""
    ws = ROBOT_CONNECTIONS.get(robot_id)

    # Priority 2: match by device_type (explicit — edge_router always sets this)
    if not ws and isinstance(params, dict):
        target_type = params.get("_device_type", "")
        if target_type:
            for rid, meta in ROBOT_METADATA.items():
                if meta.get("type") == target_type and rid in ROBOT_CONNECTIONS:
                    logger.info(f"Routing '{command}' to '{rid}' (matched type='{target_type}')")
                    robot_id = rid
                    ws = ROBOT_CONNECTIONS[rid]
                    break

    # Priority 3: match by IP (fallback if type didn't match)
    if not ws:
        search_ip = robot_id if "." in robot_id else ""
        if not search_ip and isinstance(params, dict):
            search_ip = params.get("plc_ip", "") or params.get("ip", "")
        if search_ip:
            for rid, meta in ROBOT_METADATA.items():
                if search_ip in meta.get("ips", []) and rid in ROBOT_CONNECTIONS:
                    logger.info(f"Routing '{command}' to '{rid}' (matched IP {search_ip})")
                    robot_id = rid
                    ws = ROBOT_CONNECTIONS[rid]
                    break

    # Last resort fallback
    if not ws and ROBOT_CONNECTIONS:
        actual_id = next(iter(ROBOT_CONNECTIONS))
        logger.info(f"Robot '{robot_id}' not found, using '{actual_id}'")
        robot_id = actual_id
        ws = ROBOT_CONNECTIONS[actual_id]

    if not ws:
        return {
            "status": "error",
            "error": "No devices connected. Check that the lab bridge is running.",
            "connected_robots": [],
        }

    cmd_id = str(uuid.uuid4())
    event = asyncio.Event()
    PENDING_COMMANDS[cmd_id] = {"event": event, "result": None}

    try:
        await ws.send_json({
            "id": cmd_id,
            "command": command,
            "params": params or {},
        })
        await asyncio.wait_for(event.wait(), timeout=timeout)
        result = PENDING_COMMANDS[cmd_id]["result"]
        return result if result else {"status": "error", "error": "No response received"}
    except asyncio.TimeoutError:
        return {"status": "error", "error": f"Command '{command}' timed out after {timeout}s"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        PENDING_COMMANDS.pop(cmd_id, None)


async def notify_bridge(robot_id: str, message: dict) -> bool:
    """Send a one-way notification to a connected robot bridge. Returns True on success."""
    ws = ROBOT_CONNECTIONS.get(robot_id)
    if not ws:
        logger.warning(f"notify_bridge: robot '{robot_id}' not connected")
        return False
    try:
        await ws.send_json(message)
        return True
    except Exception as e:
        logger.error(f"notify_bridge: failed to send to '{robot_id}': {e}")
        return False


# ── Graph (singleton) ──
_graph = None

def _get_checkpointer():
    """Get persistent checkpointer, fall back to in-memory."""
    db_url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
    if db_url:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            saver = PostgresSaver.from_conn_string(db_url)
            saver.setup()  # Creates checkpoint tables if they don't exist
            logger.info("Using PostgresSaver checkpointer")
            return saver
        except Exception as e:
            logger.warning(f"PostgresSaver failed, using MemorySaver: {e}")
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()


def get_graph():
    global _graph
    if _graph is None:
        from src.agent.graph import create_graph_with_checkpointer
        checkpointer = _get_checkpointer()
        _graph = create_graph_with_checkpointer(checkpointer=checkpointer)
        logger.info(f"Graph compiled from graph.py with {type(checkpointer).__name__} checkpointer")
    return _graph


# Auth For Production (Not implemented Yet)
async def verify_auth(authorization: Optional[str] = Header(None)):
    if API_KEY and API_KEY != "":
        if not authorization or authorization.replace("Bearer ", "") != API_KEY:
            raise HTTPException(401, "Unauthorized")


# Worker Events to watch for — imported from the single source of truth
from src.agent.graph import ALL_NODES as _ALL_NODES_SET
_ALL_NODES = sorted(_ALL_NODES_SET)


# ============================================
# SSE HELPERS
# ============================================
def _extract_text_from_message(message: Union[str, list]) -> str:
    """Extract plain text from a message (str or multimodal blocks)."""
    if isinstance(message, str):
        return message
    parts = []
    for block in message:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return " ".join(parts)


def _extract_image_blocks(message: Union[str, list]) -> List[dict]:
    """
    Extract image blocks from a multimodal message.

    Returns list of image_url dicts ready for LangChain HumanMessage content:
        [{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]
    """
    if isinstance(message, str):
        return []
    images = []
    for block in message:
        if block.get("type") == "image_url":
            url = (block.get("image_url") or {}).get("url", "")
            if url:
                images.append({
                    "type": "image_url",
                    "image_url": {"url": url},
                })
    return images


def sse_event(event_type: str, data: Any) -> str:
    """Format a Server-Sent Event."""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event_type}\ndata: {payload}\n\n"


def extract_events_from_node(event: dict) -> list:
    """Extract run_events from a graph stream event."""
    events = []
    for node_name in _ALL_NODES:
        if node_name in event and isinstance(event[node_name], dict):
            node_data = event[node_name]
            for evt in node_data.get("events", []):
                events.append(evt)
    return events


def _strip_suggestion_block(text: str) -> str:
    """Safety-net: remove any residual ---SUGGESTIONS--- block from response text."""
    if "---SUGGESTIONS---" not in text:
        return text
    return text.split("---SUGGESTIONS---", 1)[0].strip()


def _extract_ai_from_node(node_data: dict) -> Optional[str]:
    """Helper: extract AIMessage content from a node's state update."""
    if not isinstance(node_data, dict):
        return None
    for msg in node_data.get("messages", []):
        if isinstance(msg, AIMessage):
            content = (msg.content or "").strip()
            if content:
                return content
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = (msg.get("content") or "").strip()
            if content:
                return content
    return None


def extract_response(event: dict) -> Optional[str]:
    """Extract final AI message from graph stream event.

    Priority:
    1. synthesize node (normal multi-worker flow)
    2. Any worker node as fallback (safety net for edge cases)

    Only returns substantive responses (>50 chars) from workers to ignore plan announcements.
    """
    # Priority 1: synthesize — the designed final response path
    if "synthesize" in event:
        msg = _extract_ai_from_node(event["synthesize"])
        if msg:
            return _strip_suggestion_block(msg)

    # Priority 2: fallback — any worker that emitted a substantive AIMessage
    # This catches edge cases where synthesize bypass produces empty content
    _WORKER_NODES = ("chat", "tutor", "research", "troubleshooting",
                      "robot_operator", "analysis", "summarizer")
    for node_name in _WORKER_NODES:
        if node_name in event:
            msg = _extract_ai_from_node(event[node_name])
            # Only accept substantive messages (ignore plan announcements like "Voy a buscar...")
            if msg and len(msg) > 50:
                return _strip_suggestion_block(msg)

    return None


def extract_suggestions(event: dict) -> list:
    """Extract follow-up suggestions from state."""
    for node_name in _ALL_NODES:
        if node_name in event and isinstance(event[node_name], dict):
            sugs = event[node_name].get("follow_up_suggestions", [])
            if sugs:
                return sugs
    return []


def extract_questions(event: dict) -> list:
    """Extract HITL clarification questions."""
    for node_name in _ALL_NODES:
        if node_name in event and isinstance(event[node_name], dict):
            qs = event[node_name].get("clarification_questions", [])
            if qs:
                return qs
    return []


def extract_chart_data(event: dict) -> Optional[dict]:
    """Extract chart data from pending_context (set by analysis worker's prepare_chart tool)."""
    for node_name in _ALL_NODES:
        if node_name in event and isinstance(event[node_name], dict):
            pc = event[node_name].get("pending_context")
            if isinstance(pc, dict) and "chart_data" in pc:
                return pc["chart_data"]
    return None


def extract_practice_data(graph, config: dict) -> Optional[dict]:
    """Read practice_update and practice_chunks from the final graph state.

    Returns a flat dict with practice_completed, automation_step, step_completed,
    and optionally practice_chunks (for multi-message tool execution flow).
    Returns None if not a practice session.
    """
    try:
        final_state = graph.get_state(config)
        if not final_state or not hasattr(final_state, "values"):
            return None
        vals = final_state.values

        # Find practice_update inside the last worker output
        practice_update = None
        worker_outputs = vals.get("worker_outputs", [])
        for output in reversed(worker_outputs):
            if isinstance(output, dict) and "practice_update" in output:
                practice_update = output["practice_update"]
                break

        # Include practice_chunks if present (used by practice AND troubleshoot modes)
        chunks = vals.get("practice_chunks", [])

        if not practice_update:
            # No practice_update — but if there are chunks (e.g. troubleshoot mode), return them
            if chunks:
                return {"practice_chunks": chunks}
            return None

        result = {
            "practice_completed": practice_update.get("practice_completed", False),
            "automation_step": practice_update.get("step", 0),
            "step_completed": practice_update.get("step_completed", False),
        }

        if chunks:
            result["practice_chunks"] = chunks

        return result
    except Exception as e:
        print(f"[SSE] extract_practice_data error: {e}", flush=True)
        return None


# ============================================
# TTS STREAMING (voice mode)
# ============================================
async def stream_tts_chunks(text: str, voice_service, voice_id: str = None) -> AsyncGenerator[str, None]:
    """
    Stream ElevenLabs TTS as SSE audio_chunk events.

    Runs synchronous ElevenLabs streaming in a thread executor,
    yielding base64-encoded MP3 chunks as SSE events.

    Graceful degradation: if TTS fails, yields audio_done with
    error info. The client already has the text from event: response.
    """
    # Validate text before attempting TTS
    if not text or not text.strip():
        print("[TTS] stream_tts_chunks called with EMPTY text!", flush=True)
        yield sse_event("audio_done", {"error": "Empty text", "format": "mp3"})
        return

    print(f"[TTS] stream_tts_chunks START: {len(text)} chars, preview: {text[:80]}...", flush=True)

    loop = asyncio.get_running_loop()
    chunk_queue: asyncio.Queue = asyncio.Queue()

    def _run_tts():
        """Run ElevenLabs TTS in a sync thread, putting chunks via call_soon_threadsafe."""
        try:
            print(f"[TTS-THREAD] Calling voice_service.stream_tts(), text_len={len(text)}, voice_id={getattr(voice_service, 'voice_id', '?')}, model_id={getattr(voice_service, 'model_id', '?')}", flush=True)

            index = 0
            for audio_bytes in voice_service.stream_tts(text, voice_id=voice_id):
                if index == 0:
                    print(f"[TTS-THREAD] First chunk received: {len(audio_bytes)} bytes", flush=True)
                item = {
                    "type": "chunk",
                    "data": base64.b64encode(audio_bytes).decode("ascii"),
                    "index": index,
                    "size": len(audio_bytes),
                }
                loop.call_soon_threadsafe(chunk_queue.put_nowait, item)
                index += 1

            if index == 0:
                print("[TTS-THREAD] ElevenLabs returned 0 audio fragments!", flush=True)
            else:
                print(f"[TTS-THREAD] ElevenLabs stream complete: {index} chunks", flush=True)

            loop.call_soon_threadsafe(
                chunk_queue.put_nowait,
                {"type": "done", "total_chunks": index},
            )
        except Exception as e:
            print(f"[TTS-THREAD] EXCEPTION: {type(e).__name__}: {e}", flush=True)
            loop.call_soon_threadsafe(
                chunk_queue.put_nowait,
                {"type": "error", "message": f"{type(e).__name__}: {e}"},
            )

    tts_task = loop.run_in_executor(None, _run_tts)

    while True:
        try:
            item = await asyncio.wait_for(chunk_queue.get(), timeout=30)
        except asyncio.TimeoutError:
            logger.warning("TTS chunk timeout after 30s")
            yield sse_event("audio_done", {"error": "TTS timeout", "format": "mp3"})
            break

        if item["type"] == "chunk":
            print(f"[TTS] Emitting audio_chunk index={item['index']}, b64_len={len(item['data'])}", flush=True)
            yield sse_event("audio_chunk", {
                "chunk": item["data"],
                "index": item["index"],
            })
        elif item["type"] == "done":
            print(f"[TTS] Emitting audio_done, total_chunks={item['total_chunks']}", flush=True)
            yield sse_event("audio_done", {
                "total_chunks": item["total_chunks"],
                "format": "mp3",
            })
            break
        elif item["type"] == "error":
            print(f"[TTS] Emitting audio_done with ERROR: {item['message']}", flush=True)
            yield sse_event("audio_done", {
                "error": item["message"],
                "format": "mp3",
            })
            break

    await tts_task


# ============================================
# STREAMING ENDPOINT
# ============================================
@app.post("/api/chat")
async def chat_stream(req: ChatRequest, auth=Depends(verify_auth)):
    """
    Main chat endpoint with SSE streaming.
    
    Client usage (Next.js):
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: 'hello', user_id: '...' })
        });
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const text = decoder.decode(value);
            // Parse SSE events from text
        }
    """
    graph = get_graph()
    
    session_id = req.session_id or f"session-{req.user_id or 'anon'}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    
    config = {
        "configurable": {
            "thread_id": session_id,
        }
    }
    
    # Split text and images: text goes in HumanMessage (safe for .lower()),
    # images go in image_attachments (used only by LLM-facing nodes).
    message_text = _extract_text_from_message(req.message)
    image_blocks = _extract_image_blocks(req.message)

    # Resolve team_id from user profile (needed for RLS filtering in analysis worker)
    team_id = None
    if req.user_id:
        try:
            from src.agent.services import get_supabase
            sb = get_supabase()
            if sb:
                profile = sb.table("profiles").select("active_team_id").eq("auth_user_id", req.user_id).maybe_single().execute()
                if profile and profile.data:
                    team_id = profile.data.get("active_team_id")
        except Exception as e:
            logger.warning(f"Could not fetch team_id for user {req.user_id}: {e}")

    # ── Load conversation history from Supabase for session continuity ──
    MAX_HISTORY_MESSAGES = 10
    MAX_MESSAGE_CHARS = 1000

    prior_messages = []
    if req.session_id:
        try:
            from src.agent.services import get_supabase
            sb = get_supabase()
            if sb:
                result = sb.schema("chat").from_("messages") \
                    .select("sender, content") \
                    .eq("session_id", req.session_id) \
                    .order("created_at", desc=True) \
                    .limit(MAX_HISTORY_MESSAGES + 1) \
                    .execute()
                if result and result.data:
                    rows = list(reversed(result.data))  # Back to chronological
                    for msg in rows:
                        content = (msg.get("content") or "").strip()
                        if not content:
                            continue
                        # Truncate long messages
                        if len(content) > MAX_MESSAGE_CHARS:
                            content = content[:MAX_MESSAGE_CHARS] + "..."
                        if msg["sender"] == "user":
                            prior_messages.append(HumanMessage(content=content))
                        elif msg["sender"] == "ai":
                            prior_messages.append(AIMessage(content=content))
                    # Remove last if it duplicates current message
                    if prior_messages and isinstance(prior_messages[-1], HumanMessage):
                        if prior_messages[-1].content == message_text[:MAX_MESSAGE_CHARS]:
                            prior_messages = prior_messages[:-1]
                    logger.info(f"Loaded {len(prior_messages)} prior messages for session {req.session_id}")
        except Exception as e:
            logger.warning(f"Could not load session history: {e}")

    payload = {
        "messages": prior_messages + [HumanMessage(content=message_text)],
        "user_name": req.user_name,
        "user_id": req.user_id,
        "auth_user_id": req.user_id,
        "team_id": team_id,
        "interaction_mode": req.interaction_mode or "chat",
        "llm_model": req.llm_model or "",
        "image_attachments": image_blocks,
    }
    if req.robot_ids:
        payload["robot_ids"] = req.robot_ids
    if req.automation_id:
        payload["automation_id"] = req.automation_id
    if req.automation_md_content:
        payload["automation_md_content"] = req.automation_md_content
    if req.automation_step is not None:
        payload["automation_step"] = req.automation_step
    if req.equipment_id:
        equipment_context = {"equipment_id": req.equipment_id}
        # Resolve equipment details for the whole session
        try:
            from src.agent.services import get_supabase
            sb = get_supabase()
            if sb:
                eq_result = sb.schema("lab").from_("equipment_profiles") \
                    .select("name, brand, model, type, ip_address, description") \
                    .eq("id", req.equipment_id) \
                    .maybe_single() \
                    .execute()
                if eq_result and eq_result.data:
                    eq = eq_result.data
                    equipment_context.update({
                        "equipment_name": eq.get("name", ""),
                        "equipment_brand": eq.get("brand", ""),
                        "equipment_model": eq.get("model", ""),
                        "equipment_type": eq.get("type", ""),
                        "equipment_ip": eq.get("ip_address", ""),
                        "equipment_description": eq.get("description", ""),
                    })

                # Also get linked manual document IDs
                doc_result = sb.schema("lab").from_("equipment_documents") \
                    .select("document_id") \
                    .eq("equipment_id", req.equipment_id) \
                    .execute()
                if doc_result and doc_result.data:
                    equipment_context["equipment_doc_ids"] = [
                        str(d["document_id"]) for d in doc_result.data
                    ]
        except Exception as e:
            logger.warning(f"Could not resolve equipment details: {e}")

        payload["pending_context"] = equipment_context
    
    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events from LangGraph stream."""
        try:
            # Send session info
            yield sse_event("session", {"session_id": session_id})
            # Feedback inmediato al frontend
            yield sse_event("thinking", {"node": "start", "message": "Procesando tu mensaje..."})

            final_response = None
            all_suggestions = []
            chart_payload = None
            interrupted = False
            interrupt_payload = None

            # Narration rate limiter — max 6 per request, dedup by content prefix
            _emitted_narrations = set()
            _narration_count = 0
            _MAX_NARRATIONS = 6

            def _try_emit_narration(content: str, source: str, phase: str) -> Optional[str]:
                nonlocal _narration_count
                if _narration_count >= _MAX_NARRATIONS:
                    return None
                key = content[:80]
                if key in _emitted_narrations:
                    return None
                _emitted_narrations.add(key)
                _narration_count += 1
                return sse_event("narration", {"content": content, "source": source, "phase": phase})
            
            # Run graph in a thread (LangGraph is sync)
            loop = asyncio.get_running_loop()
            event_queue: asyncio.Queue = asyncio.Queue()

            # Real-time streaming callback for troubleshoot mode
            def emit_practice_chunk(chunk_data: dict):
                """Called by troubleshooter_node to stream chunks in real-time."""
                sse_payload = {"__practice_chunk__": True, **chunk_data}
                loop.call_soon_threadsafe(event_queue.put_nowait, sse_payload)

            def emit_tool_lifecycle(event_data: dict):
                """Called by ToolExecutor to stream tool lifecycle events in real-time."""
                sse_payload = {"__tool_lifecycle__": True, **event_data}
                loop.call_soon_threadsafe(event_queue.put_nowait, sse_payload)

            # Registrar stream callback para TODOS los modos (excepto voice, que solo usa audio)
            is_voice = (req.interaction_mode or "").lower() == "voice"
            if not is_voice:
                register_stream_callback(session_id, emit_practice_chunk)
                payload["_stream_session_id"] = session_id

            def run_graph_with_queue():
                """Run graph and put events in queue (thread-safe via call_soon_threadsafe)."""
                try:
                    for event in graph.stream(payload, config=config, stream_mode="updates"):
                        loop.call_soon_threadsafe(event_queue.put_nowait, event)
                    loop.call_soon_threadsafe(event_queue.put_nowait, None)  # Signal done
                except Exception as e:
                    loop.call_soon_threadsafe(event_queue.put_nowait, {"__error__": str(e)})
                    loop.call_soon_threadsafe(event_queue.put_nowait, None)

            # Run in thread
            thread = loop.run_in_executor(None, run_graph_with_queue)
            
            # Consume events from queue
            while True:
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=120)
                except asyncio.TimeoutError:
                    yield sse_event("error", {"message": "Timeout waiting for agent response"})
                    break
                
                if event is None:
                    break  # Graph finished

                # Handle real-time streaming chunks from any worker
                if isinstance(event, dict) and event.get("__practice_chunk__"):
                    chunk_event = {k: v for k, v in event.items() if k != "__practice_chunk__"}
                    yield sse_event("practice_chunk", chunk_event)
                    # Partial chunks contain full CoT text — goes to final response, not narrations.
                    # Only tool_status chunks are short enough for narration (emitted via node events).
                    continue

                # Handle tool lifecycle events from ToolExecutor
                if isinstance(event, dict) and event.get("__tool_lifecycle__"):
                    lifecycle_event = {k: v for k, v in event.items() if k != "__tool_lifecycle__"}
                    yield sse_event("tool_lifecycle", lifecycle_event)
                    continue

                if "__error__" in event:
                    yield sse_event("error", {"message": event["__error__"]})
                    break

                # Check for interrupt (LangGraph emits __interrupt__ as tuple)
                if isinstance(event, dict) and "__interrupt__" in event:
                    interrupted = True
                    int_data = event.get("__interrupt__", ())
                    # LangGraph 1.x: int_data is a tuple of Interrupt objects
                    if isinstance(int_data, (list, tuple)):
                        for item in int_data:
                            if hasattr(item, 'value') and item.value:
                                interrupt_payload = item.value
                                print(f"[SSE] Interrupt captured: type={type(interrupt_payload).__name__}, keys={list(interrupt_payload.keys()) if isinstance(interrupt_payload, dict) else 'N/A'}", flush=True)
                    elif isinstance(int_data, dict):
                        interrupt_payload = int_data
                    continue
                
                # Extract and send run events (thinking/processing indicators)
                node_events = extract_events_from_node(event)
                is_voice = (req.interaction_mode or "").lower() == "voice"
                for evt in node_events:
                    yield sse_event("node_update", evt)
                    # Re-emit narration events as SSE narration (rate-limited, deduped)
                    if not is_voice and evt.get("type") == "narration":
                        narr = _try_emit_narration(evt.get("content", ""), evt.get("source", ""), evt.get("phase", "thinking"))
                        if narr:
                            yield narr

                # Si hay un event de planning, emitir thinking con el plan
                if any(evt.get("type") == "plan" for evt in node_events):
                    plan_evt = next((e for e in node_events if e.get("type") == "plan"), None)
                    if plan_evt:
                        yield sse_event("thinking", {
                            "node": "planner",
                            "message": plan_evt.get("content", "Planificando..."),
                        })

                # Extract suggestions
                sugs = extract_suggestions(event)
                if sugs:
                    all_suggestions = sugs
                
                # Extract chart data (from analysis worker's prepare_chart)
                cd = extract_chart_data(event)
                if cd:
                    chart_payload = cd

                # Extract response
                response = extract_response(event)
                if response:
                    final_response = response
                    print(f"[SSE] final_response captured: {len(response)} chars, preview: {response[:80]}...", flush=True)

            # Wait for thread to finish
            await thread
            print(f"[SSE] Graph done. final_response={'YES' if final_response else 'NONE'}, interrupted={interrupted}", flush=True)

            # Send final data
            if interrupted:
                print(f"[SSE] HITL: interrupted={interrupted}, has_payload={interrupt_payload is not None}, payload_type={type(interrupt_payload).__name__ if interrupt_payload else 'None'}", flush=True)
                if interrupt_payload:
                    if isinstance(interrupt_payload, dict):
                        q_count = len(interrupt_payload.get("questions", []))
                        print(f"[SSE] Emitting 'questions' event: title={interrupt_payload.get('title', '?')}, questions={q_count}", flush=True)
                        yield sse_event("questions", {
                            **interrupt_payload,
                            "session_id": session_id,
                        })
                    else:
                        print(f"[SSE] Emitting legacy 'questions' event: {str(interrupt_payload)[:100]}", flush=True)
                        yield sse_event("questions", {
                            "questions": [],
                            "prompt": str(interrupt_payload),
                            "session_id": session_id,
                        })
                else:
                    print("[SSE] WARNING: interrupted=True but no payload captured!", flush=True)
            
            if all_suggestions:
                yield sse_event("suggestions", {"suggestions": all_suggestions})
            
            if final_response:
                response_data = {
                    "content": final_response,
                    "session_id": session_id,
                }
                # Inject practice data if available
                practice_data = extract_practice_data(graph, config)
                if practice_data:
                    # Send practice_chunks as separate SSE events BEFORE the final response
                    chunks = practice_data.pop("practice_chunks", [])
                    if chunks:
                        print(f"[SSE] Sending {len(chunks)} practice_chunks", flush=True)
                        for chunk in chunks:
                            yield sse_event("practice_chunk", chunk)
                            if chunk.get("type") == "tool_status" and chunk.get("status") == "executing":
                                await asyncio.sleep(0.5)
                        response_data["chunks_sent"] = True

                    response_data.update(practice_data)
                    print(f"[SSE] Practice data injected: {practice_data}", flush=True)
                yield sse_event("response", response_data)

            # ── Chart data (analysis worker) ──
            if chart_payload:
                yield sse_event("chart", chart_payload)

            # ── TTS streaming for voice mode ──
            mode = (req.interaction_mode or "").lower()
            voice_enabled = req.voice_enabled
            voice_id = req.voice_id or None
            print(f"[TTS] === DECISION POINT === mode='{mode}', voice_enabled={voice_enabled}, final_response={'YES len=' + str(len(final_response)) if final_response else 'NONE'}", flush=True)

            if final_response and (mode == "voice" or voice_enabled):
                from src.agent.services import get_elevenlabs
                voice_service = get_elevenlabs()
                print(f"[TTS] voice_service={'OK type=' + type(voice_service).__name__ if voice_service else 'NONE'}", flush=True)

                if voice_service:
                    print(f"[TTS] Calling stream_tts_chunks with {len(final_response)} chars, voice_id={voice_id}", flush=True)
                    chunk_count = 0
                    async for audio_event in stream_tts_chunks(final_response, voice_service, voice_id=voice_id):
                        chunk_count += 1
                        yield audio_event
                    print(f"[TTS] stream_tts_chunks finished: {chunk_count} SSE events yielded", flush=True)
                else:
                    print("[TTS] ERROR: voice_service is None! Check ELEVENLABS_API_KEY", flush=True)
                    yield sse_event("audio_done", {"error": "ElevenLabs not configured", "format": "mp3"})

            elif (mode == "voice" or voice_enabled) and not final_response:
                print("[TTS] ERROR: voice enabled but final_response is None!", flush=True)
                yield sse_event("audio_done", {"error": "No response text to convert", "format": "mp3"})

            else:
                print(f"[TTS] Skipped: mode='{mode}', voice_enabled={voice_enabled}", flush=True)

            # ── Token deduction + balance (independent flows) ──
            if req.user_id and final_response:
                from src.agent.utils.token_manager import deduct_tokens, check_balance

                # Get real token count from graph state
                real_tokens = 0
                try:
                    final_state = graph.get_state(config)
                    if final_state and hasattr(final_state, "values"):
                        real_tokens = final_state.values.get("token_usage", 0) or 0
                except Exception as e:
                    logger.warning(f"Could not read graph state for tokens: {e}")

                tokens_to_deduct = real_tokens if real_tokens > 0 else max(len(final_response) // 3, 100)
                token_source = "real" if real_tokens > 0 else "estimated"
                print(f"[TOKENS] user={req.user_id[:8]}... amount={tokens_to_deduct} source={token_source} session_id={session_id}", flush=True)

                # 1) Deduct tokens (best-effort — never blocks response)
                #    deduct_tokens() catches its own exceptions and returns a dict,
                #    so we check the return value, not try/except.
                deduct_result = deduct_tokens(
                    req.user_id,
                    tokens_to_deduct,
                    description="Chat interaction",
                    session_id=session_id,
                )
                deduct_ok = isinstance(deduct_result, dict) and deduct_result.get("success", False)
                print(f"[TOKENS] deduct result: {deduct_result}", flush=True)

                # 2) Balance ALWAYS sent to frontend, even if deduction failed
                try:
                    balance = check_balance(req.user_id)
                    print(f"[TOKENS] check_balance raw: {balance}", flush=True)

                    if balance and isinstance(balance, dict):
                        # Handle both possible RPC response formats
                        remaining = (
                            balance.get("available")
                            or balance.get("remaining")
                            or balance.get("remaining_tokens")
                            or 0
                        )
                        total = (
                            balance.get("total")
                            or balance.get("total_tokens")
                            or 0
                        )
                        yield sse_event("tokens", {
                            "used": tokens_to_deduct if deduct_ok else 0,
                            "source": token_source if deduct_ok else "deduct_failed",
                            "remaining": remaining,
                            "total": total,
                            "has_credits": balance.get("has_credits", True),
                        })
                except Exception as e:
                    print(f"[TOKENS] check_balance FAILED: {e}", flush=True)
                    logger.warning(f"Balance check failed: {e}")
            
            unregister_stream_callback(session_id)
            yield sse_event("done", {"session_id": session_id})

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            unregister_stream_callback(session_id)
            yield sse_event("error", {"message": str(e)})
            yield sse_event("done", {"session_id": session_id})
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@app.post("/api/confirm")
async def confirm_interrupt(req: ConfirmRequest, auth=Depends(verify_auth)):
    """Confirm HITL interrupt with answers."""
    graph = get_graph()
    config = {"configurable": {"thread_id": req.session_id}}
    
    from langgraph.types import Command
    
    async def event_generator():
        try:
            # Build resume data — supports v2 dict and legacy list
            if isinstance(req.answers, dict):
                resume_data = {"answers": req.answers, "completed": req.completed, "cancelled": req.cancelled}
            elif isinstance(req.answers, list):
                resume_data = {"answers": req.answers, "completed": req.completed, "cancelled": req.cancelled}
            else:
                resume_data = req.answers
            resume_payload = Command(resume=resume_data)
            
            logger.info(f"Confirm: resuming session {req.session_id} with answers type={type(req.answers).__name__}")
            # Feedback inmediato al frontend
            yield sse_event("thinking", {"node": "start", "message": "Procesando tu respuesta..."})

            event_queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def run_resume():
                try:
                    for event in graph.stream(resume_payload, config=config, stream_mode="updates"):
                        logger.info(f"Confirm event: {list(event.keys()) if isinstance(event, dict) else type(event)}")
                        loop.call_soon_threadsafe(event_queue.put_nowait, event)
                    loop.call_soon_threadsafe(event_queue.put_nowait, None)
                except Exception as e:
                    logger.error(f"Confirm resume error: {e}", exc_info=True)
                    loop.call_soon_threadsafe(event_queue.put_nowait, {"__error__": str(e)})
                    loop.call_soon_threadsafe(event_queue.put_nowait, None)

            thread = loop.run_in_executor(None, run_resume)
            
            final_response = None
            all_suggestions = []
            
            while True:
                event = await asyncio.wait_for(event_queue.get(), timeout=120)
                if event is None:
                    break
                if "__error__" in event:
                    yield sse_event("error", {"message": event["__error__"]})
                    break
                
                # Check for new interrupt (graph might need more info)
                if isinstance(event, dict) and "__interrupt__" in event:
                    logger.info("Confirm: graph interrupted again")
                    continue
                
                for evt in extract_events_from_node(event):
                    yield sse_event("node_update", evt)
                
                sugs = extract_suggestions(event)
                if sugs:
                    all_suggestions = sugs
                
                response = extract_response(event)
                if response:
                    final_response = response
            
            # Wait for thread
            await thread
            
            if all_suggestions:
                yield sse_event("suggestions", {"suggestions": all_suggestions})
            if final_response:
                response_data = {
                    "content": final_response,
                    "session_id": req.session_id,
                }
                practice_data = extract_practice_data(graph, config)
                if practice_data:
                    chunks = practice_data.pop("practice_chunks", [])
                    if chunks:
                        for chunk in chunks:
                            yield sse_event("practice_chunk", chunk)
                            if chunk.get("type") == "tool_status" and chunk.get("status") == "executing":
                                await asyncio.sleep(0.5)
                        response_data["chunks_sent"] = True
                    response_data.update(practice_data)
                yield sse_event("response", response_data)
            else:
                logger.warning("Confirm: no final response extracted from resumed graph")

            # ── TTS streaming for voice mode ──
            if final_response and (req.interaction_mode or "").lower() == "voice":
                from src.agent.services import get_elevenlabs
                voice_service = get_elevenlabs()
                if voice_service:
                    async for audio_event in stream_tts_chunks(final_response, voice_service):
                        yield audio_event

            yield sse_event("done", {"session_id": req.session_id})
        
        except Exception as e:
            yield sse_event("error", {"message": str(e)})
            yield sse_event("done", {"session_id": req.session_id})
    
    return StreamingResponse(event_generator(), media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/deepgram-token")
async def get_deepgram_token():
    """Return a temporary Deepgram API key for browser STT."""
    import os
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        return JSONResponse({"error": "Deepgram not configured"}, status_code=500)
    # For now, return the key directly. In production, use Deepgram's
    # /keys endpoint to create short-lived tokens.
    return JSONResponse({"key": api_key})


@app.get("/api/health")
async def health():
    """Health check for Azure Container Apps."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/tts-test")
async def tts_test():
    """
    Diagnostic endpoint: tests TTS pipeline independently of the agent graph.
    Call GET /api/tts-test to verify ElevenLabs is working.
    Returns SSE stream with audio chunks for a test phrase.
    """
    from src.agent.services import get_elevenlabs, ServiceRegistry

    # Check service status
    voice_service = get_elevenlabs()
    if not voice_service:
        elevenlabs_error = ServiceRegistry._errors.get("elevenlabs", "No error recorded")
        return {
            "status": "error",
            "message": "ElevenLabs service not available",
            "elevenlabs_error": elevenlabs_error,
            "env_key_set": bool(os.getenv("ELEVENLABS_API_KEY")),
            "env_voice_id": os.getenv("ELEVENLABS_VOICE_ID", "(default)"),
            "env_model_id": os.getenv("ELEVENLABS_MODEL_ID", "(default)"),
        }

    test_text = "Hola, esta es una prueba del sistema de voz. Todo funciona correctamente."

    async def test_generator():
        yield sse_event("info", {
            "voice_id": voice_service.voice_id,
            "model_id": voice_service.model_id,
            "output_format": voice_service.output_format,
            "test_text": test_text,
        })
        async for audio_event in stream_tts_chunks(test_text, voice_service):
            yield audio_event
        yield sse_event("done", {"test": "complete"})

    return StreamingResponse(
        test_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============================================
# ROBOT BRIDGE (WebSocket)
# ============================================
@app.websocket("/ws/robot")
async def ws_robot(ws: WebSocket):
    """WebSocket endpoint for robot bridge clients (Lab PC).

    Protocol:
        1. Client connects and sends auth message: {"token": "...", "robot_id": "xarm-01"}
        2. Server validates and registers the connection.
        3. Server sends commands as: {"id": "<uuid>", "command": "...", "params": {...}}
        4. Client responds with: {"id": "<uuid>", "status": "ok", ...result data}
        5. Client can also push unsolicited: {"type": "status_update", ...}
    """
    await ws.accept()

    # First message must be auth + registration
    try:
        init = await asyncio.wait_for(ws.receive_json(), timeout=10)
    except (asyncio.TimeoutError, Exception):
        await ws.close(code=4001, reason="Auth timeout")
        return

    token = init.get("token", "")
    robot_id = init.get("robot_id", "")

    if token != BRIDGE_TOKEN or not robot_id:
        await ws.close(code=4003, reason="Invalid token or missing robot_id")
        return

    ROBOT_CONNECTIONS[robot_id] = ws

    # Store metadata from init payload
    now_iso = datetime.utcnow().isoformat() + "Z"
    meta = {
        "type": init.get("type", init.get("robot_type", "unknown")),
        "model": init.get("model", ""),
        "protocol": init.get("protocol", "websocket"),
        "capabilities": init.get("capabilities", []),
        "ips": init.get("ips", []),
        "last_heartbeat": now_iso,
        "active_session": init.get("active_session"),
    }
    ROBOT_METADATA[robot_id] = meta

    # Mirror to shared_state for worker access (avoids circular imports)
    try:
        from src.agent.shared_state import register_robot as _register_shared
        _register_shared(robot_id, ws, meta)
    except ImportError:
        pass

    logger.info(f"Robot bridge connected: {robot_id} (type={meta['type']}, model={meta['model']})")
    await ws.send_json({"type": "registered", "robot_id": robot_id})

    try:
        while True:
            data = await ws.receive_json()

            # Update last_heartbeat on any valid message
            if robot_id in ROBOT_METADATA:
                ROBOT_METADATA[robot_id]["last_heartbeat"] = datetime.utcnow().isoformat() + "Z"

            # Handle command responses
            cmd_id = data.get("id")
            if cmd_id and cmd_id in PENDING_COMMANDS:
                PENDING_COMMANDS[cmd_id]["result"] = data
                PENDING_COMMANDS[cmd_id]["event"].set()

            # Handle unsolicited status updates
            if data.get("type") == "status_update":
                logger.info(f"Robot {robot_id} status: {data}")

                # Store latest status in metadata
                if robot_id in ROBOT_METADATA:
                    ROBOT_METADATA[robot_id]["last_status"] = data.get("status", {})

                # Queue anomaly if error detected
                status = data.get("status", {})
                if status.get("error_code") or status.get("state") == "error":
                    BRIDGE_ANOMALY_QUEUE.append({
                        "robot_id": robot_id,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "type": "bridge_alert",
                        "data": data,
                    })
    except WebSocketDisconnect:
        logger.info(f"Robot bridge disconnected: {robot_id}")
    except Exception as e:
        logger.warning(f"Robot bridge error ({robot_id}): {e}")
    finally:
        ROBOT_CONNECTIONS.pop(robot_id, None)
        ROBOT_METADATA.pop(robot_id, None)
        try:
            from src.agent.shared_state import unregister_robot as _unregister_shared
            _unregister_shared(robot_id)
        except ImportError:
            pass


@app.get("/api/robots")
async def list_robots():
    """List currently connected robots with metadata."""
    robots = []
    for rid in ROBOT_CONNECTIONS:
        meta = ROBOT_METADATA.get(rid, {})
        robots.append({
            "robot_id": rid,
            "connected": True,
            "type": meta.get("type", "unknown"),
            "model": meta.get("model", ""),
            "protocol": meta.get("protocol", "websocket"),
            "capabilities": meta.get("capabilities", []),
            "ips": meta.get("ips", []),
            "last_heartbeat": meta.get("last_heartbeat"),
            "active_session": meta.get("active_session"),
        })

    # Summary by type
    types = {}
    for r in robots:
        t = r["type"]
        types[t] = types.get(t, 0) + 1

    return {
        "robots": robots,
        "count": len(robots),
        "by_type": types,
    }


# ============================================
# AMBIENT MONITOR
# ============================================

def _triage_anomalies(anomalies: list) -> dict:
    """Classify anomalies into action categories.

    Returns: {"auto_fix": [...], "diagnose": [...], "notify": [...]}
    """
    auto_fix = []
    diagnose = []
    notify = []

    for a in anomalies:
        severity = a.get("severity", "low")
        atype = a.get("type", "")

        if severity == "critical":
            notify.append(a)
        elif atype == "active_error" and severity in ("warning", "info"):
            auto_fix.append(a)
        elif atype == "station_offline":
            diagnose.append(a)
        else:
            diagnose.append(a)

    return {"auto_fix": auto_fix, "diagnose": diagnose, "notify": notify}


@app.post("/api/monitor/tick")
async def monitor_tick(auth=Depends(verify_auth)):
    """Called by cron every 30-60s. Checks lab state and triages anomalies."""
    connected_devices = {
        rid: ROBOT_METADATA.get(rid, {})
        for rid in ROBOT_CONNECTIONS
    }

    if not connected_devices:
        return {"status": "idle", "reason": "no_devices_connected", "anomalies": 0}

    anomalies = []

    # --- Collect bridge anomalies ---
    bridge_anomalies = list(BRIDGE_ANOMALY_QUEUE)
    BRIDGE_ANOMALY_QUEUE.clear()
    anomalies.extend([{
        "type": "bridge_alert",
        "robot_id": a["robot_id"],
        "severity": "high",
        "message": str(a.get("data", {}).get("status", {})),
        "timestamp": a.get("timestamp"),
    } for a in bridge_anomalies])

    # Lab tools removed — health check via lab schema no longer available

    if not anomalies:
        return {"status": "healthy", "devices": len(connected_devices), "anomalies": 0}

    triaged = _triage_anomalies(anomalies)

    return {
        "status": "anomalies_detected",
        "devices": len(connected_devices),
        "anomalies": len(anomalies),
        "triaged": triaged,
    }


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=True)