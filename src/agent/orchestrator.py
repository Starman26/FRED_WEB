"""
orchestrator.py - Adaptive Router + Synthesizer

Contiene:
- adaptive_router_node: Router inteligente que evalúa outputs de workers y adapta el plan
- synthesize_node: Combina outputs de workers en una respuesta coherente

NOTA: orchestrator_plan_node fue reemplazado por planner_node en src/agent/nodes/planner.py
"""
import os
import json
from typing import Dict, Any, List, Optional
import re
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.agent.contracts.worker_contract import WorkerOutput, EvidenceItem
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_plan, event_route, event_report, event_error


VALID_WORKERS = {"chat", "research", "tutor", "troubleshooting", "summarizer", "robot_operator", "analysis"}


# ============================================
# ADAPTIVE ROUTER (replaces orchestrator_route_node)
# ============================================

def _evaluate_worker_output(
    last_output: Dict[str, Any],
    plan: List[str],
    current_step: int,
) -> Dict[str, Any]:
    """
    Evaluación heurística del output del último worker (0 LLM calls).
    Usa campos del WorkerOutput contract para decidir si adaptar el plan.

    Returns:
        {
            "action": "continue" | "skip_remaining" | "add_worker" | "stop_early",
            "reason": str,
            "modified_plan": Optional[List[str]],
        }
    """
    status = last_output.get("status", "ok")
    confidence = last_output.get("confidence", 0.8)
    content = last_output.get("content", "")
    evidence = last_output.get("evidence", [])
    worker_name = last_output.get("worker", "unknown")
    next_actions = last_output.get("next_actions", [])

    remaining_plan = plan[current_step + 1:]

    # RULE 1: Worker error crítico → ir directo a synthesize con lo que hay
    if status == "error":
        return {
            "action": "stop_early",
            "reason": f"{worker_name} returned error, synthesizing available results",
            "modified_plan": None,
        }

    # RULE 2: Research no encontró evidencia → quitar tutor del plan restante
    # (tutor sin evidencia solo repite knowledge general, no agrega valor)
    if worker_name == "research" and len(evidence) == 0:
        if "tutor" in remaining_plan:
            new_remaining = [w for w in remaining_plan if w != "tutor"]
            if not new_remaining:
                return {
                    "action": "stop_early",
                    "reason": "Research found no evidence; skipping tutor — no context to explain",
                    "modified_plan": None,
                }
            return {
                "action": "skip_remaining",
                "reason": "Research found no evidence; removed tutor from remaining plan",
                "modified_plan": plan[: current_step + 1] + new_remaining,
            }

    # RULE 3: Worker completó con alta confianza y no quedan más workers → stop early
    if (
        status == "ok"
        and confidence >= 0.9
        and len(content) > 100
        and not remaining_plan
    ):
        return {
            "action": "stop_early",
            "reason": f"{worker_name} answered with high confidence ({confidence:.2f}), plan complete",
            "modified_plan": None,
        }

    # RULE 5: Research has gaps mentioning lab entities → add troubleshooting to get real data
    if worker_name == "research" and "troubleshooting" not in remaining_plan:
        extra = last_output.get("extra", {})
        gaps = extra.get("gaps", []) if isinstance(extra, dict) else []
        if gaps:
            import re as _re
            gap_text = " ".join(str(g) for g in gaps).lower()
            if _re.search(r"estaci[oó]n|station|lab|equipo|plc|cobot|puerta|sensor", gap_text):
                return {
                    "action": "add_worker",
                    "reason": f"Research gaps mention lab entities: {gaps[:2]}; adding troubleshooting for real data",
                    "modified_plan": plan[: current_step + 1] + ["troubleshooting"] + remaining_plan,
                }

    # RULE 4: Troubleshooter recomienda tutor via next_actions
    if worker_name == "troubleshooting" and next_actions:
        for action_item in next_actions:
            if isinstance(action_item, dict):
                target = action_item.get("target", "")
                action_type = action_item.get("type", "")
                if action_type == "call_worker" and target == "tutor" and "tutor" not in remaining_plan:
                    return {
                        "action": "add_worker",
                        "reason": f"Troubleshooter recommends tutor: {action_item.get('reason', 'complex issue')}",
                        "modified_plan": plan[: current_step + 1] + remaining_plan + ["tutor"],
                    }

    # Default: continue as planned
    return {
        "action": "continue",
        "reason": f"{worker_name} completed ({status}, confidence={confidence:.2f}), advancing",
        "modified_plan": None,
    }


def adaptive_router_node(state: AgentState) -> Dict[str, Any]:
    """
    Adaptive Router: después de cada worker, evalúa el output y adapta el plan.

    Tres modos:
    1. HEURISTIC: Reglas ligeras usando campos del WorkerOutput (0 LLM calls)
    2. HUMAN-IN-THE-LOOP: Si el worker necesita input del usuario
    3. ANTI-LOOP: Forzar synthesize después de 3+ ciclos de routing
    """
    logger.node_start("adaptive_router", {
        "plan": state.get("orchestration_plan"),
        "current_step": state.get("current_step"),
    })

    plan = state.get("orchestration_plan", [])
    current_step = state.get("current_step", 0)
    worker_outputs = state.get("worker_outputs", [])
    pending_context = state.get("pending_context", {}) or {}
    state_questions = state.get("clarification_questions", [])

    events = [event_route("adaptive_router", "Evaluating worker output...", route="pending")]
    last_output = worker_outputs[-1] if worker_outputs else None
    current_worker_name = plan[current_step] if current_step < len(plan) else None

    # ══════════════════════════════════════════
    # ANTI-LOOP
    # ══════════════════════════════════════════
    route_count = state.get("_route_count", 0) + 1

    if route_count > 3:
        logger.warning("adaptive_router", f"ANTI-LOOP: route called {route_count} times, forcing synthesize")
        return {
            "orchestration_plan": plan,
            "current_step": len(plan),
            "done": True,
            "next": "synthesize",
            "_route_count": 0,
            "events": events + [event_route("adaptive_router", "Anti-loop: forcing synthesize", route="synthesize")],
        }

    # ══════════════════════════════════════════
    # HUMAN-IN-THE-LOOP: User already responded → re-execute worker
    # ══════════════════════════════════════════
    has_user_clarification = bool(pending_context.get("user_clarification"))
    wizard_completed = pending_context.get("wizard_completed", False)
    hitl_consumed = pending_context.get("_hitl_consumed", False)

    if (has_user_clarification or wizard_completed) and not hitl_consumed:
        if current_worker_name:
            updated_outputs = [o for o in worker_outputs if o.get("status") != "needs_context"]
            # Keep user_clarification so the worker can read it, but mark
            # _hitl_consumed so the router won't re-trigger after the worker completes.
            cleaned_context = dict(pending_context)
            cleaned_context["_hitl_consumed"] = True
            return {
                "worker_outputs": updated_outputs,
                "pending_context": cleaned_context,
                "clarification_questions": [],
                "needs_human_input": False,
                "next": current_worker_name,
                "task_type": current_worker_name,
                "_route_count": route_count,
                "events": events + [event_route("adaptive_router", f"Re-executing {current_worker_name} with user input", route=current_worker_name)],
            }

    # ══════════════════════════════════════════
    # HUMAN-IN-THE-LOOP: Worker needs input
    # ══════════════════════════════════════════
    if last_output and last_output.get("status") == "needs_context":
        questions = last_output.get("clarification_questions", []) or state_questions
        logger.info("adaptive_router", "Worker needs input, routing to human_input")
        return {
            "needs_human_input": True,
            "clarification_questions": questions,
            "next": "human_input",
            "_route_count": route_count,
            "events": events + [event_route("adaptive_router", "Requires human input", route="human_input")],
        }

    # ══════════════════════════════════════════
    # ADAPTIVE EVALUATION: inspect worker output quality
    # ══════════════════════════════════════════
    if last_output:
        evaluation = _evaluate_worker_output(last_output, plan, current_step)

        events.append(event_route(
            "adaptive_router",
            f"Eval: {evaluation['reason']}",
            route=evaluation["action"],
        ))

        if evaluation["action"] == "stop_early":
            logger.info("adaptive_router", f"Stop early: {evaluation['reason']}")
            return {
                "orchestration_plan": plan,
                "current_step": len(plan),
                "done": True,
                "next": "synthesize",
                "_route_count": 0,
                "events": events + [event_route("adaptive_router", "Adaptive: early synthesize", route="synthesize")],
            }

        if evaluation["action"] in ("skip_remaining", "add_worker") and evaluation["modified_plan"]:
            plan = evaluation["modified_plan"]
            logger.info("adaptive_router", f"Plan adapted: {plan}")

    # ══════════════════════════════════════════
    # ADVANCE to next step
    # ══════════════════════════════════════════
    next_step = current_step + 1

    if next_step >= len(plan):
        logger.info("adaptive_router", "Plan completed, synthesizing...")
        return {
            "orchestration_plan": plan,
            "current_step": next_step,
            "done": True,
            "next": "synthesize",
            "_route_count": 0,
            "events": events + [event_route("adaptive_router", "Plan completed", route="synthesize")],
        }

    # Pass evidence context to next worker
    for output in worker_outputs:
        evidence = output.get("evidence", [])
        if evidence:
            pending_context["evidence"] = pending_context.get("evidence", []) + evidence

    next_worker = plan[next_step]
    return {
        "orchestration_plan": plan,
        "current_step": next_step,
        "pending_context": pending_context,
        "next": next_worker,
        "task_type": next_worker,
        "_route_count": 0,
        "events": events + [event_route("adaptive_router", f"Next: {next_worker}", route=next_worker)],
    }


# ============================================
# SYNTHESIZE NODE
# ============================================

def _strip_emojis(text: str) -> str:
    """Remove all emojis from text."""
    text = re.sub(r"[🏭📍📝🖥️🤖📡⚠️✅❌🔴🟢🔒🚪▶️⏹️⚡🔍🛡️⛔🔧💡🎯📊📈📉🚀💻🔄🔁⏳⏱️🕐🎉👋🧠💬🆘🛑🔔📢📌🔗📎🗂️📋📄📃🔐🔑⭐🌟💫✨🎁🎊🏆🥇🥈🥉💰💸📞📧📬🌍🌎🌏🔌🔋⚙️🛠️🔩📐📏🧪🧬🔬🔭🩺💊🧯🚒🚑🏗️🏠🏢🏫🏥🏦]", "", text)
    emoji_pattern = re.compile(
        "["
        "😀-🙏"
        "🌀-🗿"
        "🚀-🛿"
        "🇠-🇿"
        "✂-➰"
        "︀-️"
        "‍"
        "☀-⛿"
        "⌀-⏿"
        "‼-㊙"
        "🤀-🧿"
        "🨀-🩯"
        "🩰-🫿"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)
    return text.strip()


def _extract_suggestions(text: str) -> tuple:
    """
    Extract ---SUGGESTIONS--- block from text.

    Returns (clean_text, suggestions_list).
    """
    if "---SUGGESTIONS---" not in text:
        return text, []

    parts = text.split("---SUGGESTIONS---", 1)
    clean = parts[0].strip()
    suggestions = []

    if len(parts) > 1:
        tail = parts[1]
        # Remove the end marker if present
        if "---END_SUGGESTIONS---" in tail:
            tail = tail.split("---END_SUGGESTIONS---")[0]
        for line in tail.strip().splitlines():
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                cleaned = line.lstrip("0123456789.-) ").strip()
                if cleaned:
                    suggestions.append(cleaned)

    return clean, suggestions[:3]


def _persist_practice_updates(state: dict):
    """Persist automation progress and user profile updates to Supabase."""
    try:
        import re
        from src.agent.services import get_supabase
        sb = get_supabase()
        if not sb:
            return

        worker_outputs = state.get("worker_outputs", [])
        if not worker_outputs:
            return
        last_output = worker_outputs[-1] if isinstance(worker_outputs[-1], dict) else {}

        practice_update = last_output.get("practice_update")
        auth_user_id = state.get("auth_user_id") or state.get("user_id")
        if practice_update and state.get("automation_id") and auth_user_id:
            new_step = practice_update.get("step", 0)
            is_done = practice_update.get("practice_completed", False)

            update_data = {
                "current_step": new_step,
                "status": "completed" if is_done else "in_progress",
                "last_active_at": "now()",
            }
            if is_done:
                update_data["completed_at"] = "now()"

            sb.schema("lab").from_("user_automation_progress") \
                .update(update_data) \
                .eq("automation_id", state["automation_id"]) \
                .eq("auth_user_id", auth_user_id) \
                .execute()
            logger.info("orchestrator", f"Practice progress persisted: step={new_step}, completed={is_done}")

        updated_profile = last_output.get("updated_profile_md")
        if updated_profile and state.get("auth_user_id"):
            sb.from_("user_agent_profiles") \
                .update({"profile_md": updated_profile, "updated_at": "now()"}) \
                .eq("auth_user_id", state["auth_user_id"]) \
                .execute()
    except Exception as e:
        logger.warning("orchestrator", f"Persist practice failed: {e}")


def synthesize_node(state: AgentState) -> Dict[str, Any]:
    """Synthesizes worker outputs into a coherent response aligned with the user's question.

    LLM bypass conditions (saves 1 LLM call):
    - Single chat/tutor output without sources
    - Single research output with evidence and confidence >= 0.7 (already self-synthesized)

    Modes:
    - Agent: ultra-concise (1-2 lines, first person)
    - Voice: no markdown, short spoken sentences
    - Chat/Code: structured Markdown
    """
    logger.node_start("synthesize", {"outputs_count": len(state.get("worker_outputs", []))})
    events = [event_report("synthesize", "Combining results...")]

    worker_outputs = state.get("worker_outputs", [])
    interaction_mode = state.get("interaction_mode", "chat").lower()

    # ── Practice mode: bypass synthesis, persist updates ──
    if interaction_mode == "practice":
        _persist_practice_updates(state)
        if worker_outputs:
            last = worker_outputs[-1] if isinstance(worker_outputs[-1], dict) else {}
            content = last.get("content", "")
        else:
            content = ""
        return {
            "messages": [AIMessage(content=content)] if content else [],
            "done": False, "next": "END", "events": events,
        }

    if not worker_outputs:
        return {
            "messages": [AIMessage(content="No se generaron resultados para tu solicitud.")],
            "done": False, "next": "END", "events": events,
        }

    # ── Extract user's original question ──
    user_message = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            user_message = (m.content or "").strip()
            break
        if isinstance(m, dict) and m.get("role", m.get("type")) in ("user", "human"):
            user_message = (m.get("content") or "").strip()
            break

    # ── Collect worker data ──
    content_parts = []
    all_sources = set()

    for output in worker_outputs:
        content = output.get("content", "")
        worker_name = output.get("worker", "unknown")

        if content:
            content_parts.append(f"[{worker_name}]: {content}")

        for ev in output.get("evidence", []):
            title = ev.get("title", "")
            page = ev.get("page", "?")
            if title:
                all_sources.add(f"{title} (Pag. {page})")

    raw_data = "\n\n".join(content_parts)

    # ── Decide if LLM synthesis is needed ──
    worker_names = {o.get("worker", "unknown") for o in worker_outputs}

    # Bypass 1: Lightweight workers (chat, tutor) without sources
    lightweight_workers = {"chat", "tutor"}
    is_lightweight = (
        len(worker_outputs) == 1
        and worker_names.issubset(lightweight_workers)
        and not all_sources
    )

    # Bypass 2: Solo research that already synthesized its answer with evidence
    is_self_sufficient_research = (
        len(worker_outputs) == 1
        and worker_names == {"research"}
        and all_sources
        and worker_outputs[0].get("confidence", 0) >= 0.7
        and len(worker_outputs[0].get("content", "")) > 100
    )

    # Bypass 3: Analysis worker already produces a complete, self-contained response
    is_analysis = (
        len(worker_outputs) == 1
        and worker_names == {"analysis"}
    )

    skip_llm_synthesis = is_lightweight or is_self_sufficient_research or is_analysis

    synth_tokens = 0
    if skip_llm_synthesis:
        combined = content_parts[0].split("]: ", 1)[-1] if content_parts else ""
        bypass_reason = "lightweight" if is_lightweight else "self-sufficient research"
        events.append(event_report("synthesize", f"LLM bypass ({bypass_reason})"))
    else:
        events.append(event_report("synthesize", f"Synthesizing ({', '.join(worker_names)})..."))
        combined, synth_tokens = _synthesize_with_llm(user_message, raw_data, all_sources, state)

    # ── Mode-specific reformatting ──
    if interaction_mode == "agent" and combined:
        combined = _condense_for_agent_mode(combined, state)
    elif interaction_mode == "voice" and combined:
        combined = _condense_for_voice_mode(combined, state)
    elif interaction_mode in ("chat", "code") and combined:
        combined = _format_as_markdown(combined, state)

    # Extract suggestions from text (before stripping emojis/formatting)
    combined, extracted_suggestions = _extract_suggestions(combined)

    # Strip emojis
    combined = _strip_emojis(combined)

    # Use extracted suggestions if available, otherwise fall back to worker suggestions
    # Analysis mode never generates suggestions
    if interaction_mode == "analysis":
        follow_ups = []
    else:
        follow_ups = extracted_suggestions or state.get("follow_up_suggestions", [])

    return {
        "messages": [AIMessage(content=combined)],
        "worker_outputs": [],
        "pending_context": {},
        "clarification_questions": [],
        "orchestration_plan": [],
        "current_step": 0,
        "_route_count": 0,
        "done": False,
        "next": "END",
        "events": events + [event_report("synthesize", "Response synthesized")],
        "follow_up_suggestions": follow_ups,
        "token_usage": synth_tokens,
    }


def _synthesize_with_llm(
    user_question: str,
    raw_data: str,
    sources: set,
    state: Dict,
) -> str:
    """Uses LLM to synthesize multiple worker outputs into a coherent answer."""
    user_name = state.get("user_name", "User")

    sources_text = ""
    if sources:
        sources_text = "\n\nFuentes disponibles:\n" + "\n".join(f"- {s}" for s in sorted(sources))

    prompt = f"""You are SENTINEL, synthesizing information from multiple agents to answer a user's question.

USER'S QUESTION:
{user_question}

DATA COLLECTED BY AGENTS:
{raw_data[:4000]}
{sources_text}

INSTRUCTIONS:
1. **Address EVERY part** of the user's question — if they asked about X AND Y, cover both. Do not ignore any sub-question.
2. **Connect and reason** across data sources: if one agent provides theory and another provides real data, synthesize them into an integrated answer (e.g., "The paper says X, and station 3 currently does Y, so you could apply it by Z").
3. **Directly answer** what the user asked — do not dump raw data
4. Structure your response to match the user's intent:
   - If they asked a yes/no question, lead with the answer
   - If they asked "how", give steps
   - If they asked "what/why", give a clear explanation
   - If they asked for status, give a concise summary
   - If they asked to apply/connect knowledge, provide a concrete proposal with reasoning
5. **Answer-first with highlight**: Always lead with the direct answer using ==text== markers, then provide details in normal text.
   - ==text== for the key answer (yellow highlight)
   - ==+text== for positive/ok status (green), ==-text== for errors (red), ==~text== for warnings (amber), ==?text== for info (blue)
   - Example: "==No, no están cerradas.== En la estación 4 y la 5 tenemos puertas abiertas."
   - Example: "==+El PLC está conectado y operativo.== Última lectura hace 2 minutos, sin errores activos."
   - Example: "==-Error de comunicación con el cobot.== Último heartbeat hace 5 minutos."
   - The user should be able to read ONLY the highlighted text and understand the core answer
6. Always respond in professional Markdown:
   - **## heading** for the main topic/answer
   - **### subheadings** for sub-sections
   - **---** horizontal rule to separate major thematic sections (e.g., before sources, between distinct topics)
   - **Numbered lists** (1. 2. 3.) for sequential steps or procedures
   - **Bullet points** (- ) for unordered items — never mix bullets and numbers in the same list
   - **Markdown tables** for structured data, comparisons, or status reports
   - ==highlight== for direct answers, **Bold** for key terms, `inline code` for technical values, IPs, error codes
   - ```code blocks``` for code
   - Short paragraphs (2-3 lines max)
   - For simple/short answers, skip heavy formatting
7. Include sources at the end (after a --- separator) if available
8. LANGUAGE: Respond in the same language as the user's question
9. NEVER use emojis — no exceptions
10. Be concise but complete — don't omit important findings

User name: {user_name}

Always end your response with exactly 3 follow-up suggestions in this format:
---SUGGESTIONS---
1. [First follow-up question or action the user might want next]
2. [Second suggestion]
3. [Third suggestion]
---END_SUGGESTIONS---

Your synthesized response:"""

    try:
        from src.agent.utils.llm_factory import get_llm, invoke_and_track

        llm = get_llm(state, temperature=0.3, max_tokens=2000)

        response, tokens = invoke_and_track(llm, [
            SystemMessage(content="You are SENTINEL, an AI assistant that synthesizes agent findings into clear, well-structured answers."),
            HumanMessage(content=prompt),
        ], "synthesize")

        result = (response.content or "").strip()

        if len(result) < 10:
            return raw_data.split("]: ", 1)[-1] if "]: " in raw_data else raw_data, tokens

        return result, tokens

    except Exception as e:
        logger.error("synthesize", f"Synthesis LLM error: {e}")
        parts = []
        for line in raw_data.split("\n\n"):
            if "]: " in line:
                parts.append(line.split("]: ", 1)[1])
            else:
                parts.append(line)
        combined = "\n\n---\n\n".join(parts)
        if sources:
            combined += "\n\n---\n**Fuentes consultadas:**\n"
            for s in sorted(sources):
                combined += f"- {s}\n"
        return combined, 0


def _condense_for_agent_mode(content: str, state: Dict) -> str:
    """Reformula respuesta para modo Agent: primera persona, 1-3 oraciones máximo."""
    try:
        from src.agent.utils.llm_factory import get_llm

        llm = get_llm(state, temperature=0, max_tokens=150)

        user_message = ""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "content") and hasattr(msg, "type") and msg.type == "human":
                user_message = msg.content
                break

        prompt = f"""Eres SENTINEL, el agente que controla un laboratorio de manufactura.
Responde en PRIMERA PERSONA como si TÚ fueras el sistema del lab.
Máximo 1-2 oraciones. Sin markdown, sin listas, sin emojis.
Natural y directo, como un operador experto hablando por radio.
Usa ==texto== para resaltar el dato clave de tu respuesta (ej: "==Puertas cerradas.== Todo en orden.").

Pregunta del usuario: {user_message}

Datos técnicos disponibles:
{content[:1500]}

Tu respuesta (primera persona, ultra-concisa):"""

        response = llm.invoke(prompt)
        result = response.content.strip()

        if len(result) > 300:
            result = result[:300].rsplit(".", 1)[0] + "."

        return result

    except Exception as e:
        logger.error("synthesize", f"Error en condensación agent: {e}")
        lines = [
            l.strip()
            for l in content.split("\n")
            if l.strip()
            and not l.strip().startswith("#")
            and not l.strip().startswith("|")
            and not l.strip().startswith("---")
        ]
        return "\n".join(lines[:3]) if lines else content


def _strip_markdown(content: str) -> str:
    """Limpia todo formato markdown/highlight de un texto."""
    content = re.sub(r"==([+\-~?])?(.+?)==", r"\2", content)
    content = re.sub(r"#{1,6}\s*", "", content)
    content = re.sub(r"\*\*(.+?)\*\*", r"\1", content)
    content = re.sub(r"\*(.+?)\*", r"\1", content)
    content = re.sub(r"`(.+?)`", r"\1", content)
    content = re.sub(r"^[-•]\s*", "", content, flags=re.MULTILINE)
    content = re.sub(r"\|.+\|", "", content)
    content = re.sub(r"---+", "", content)
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = re.sub(r"[🏭📍📝🖥️🤖📡⚠️✅❌🔴🟢🔒🚪▶️⏹️⚡🔍🛡️]", "", content)
    return content.strip()


def _condense_for_voice_mode(content: str, state: Dict) -> str:
    """
    Reformula la respuesta para modo Voice: primera persona, ultra-concisa,
    como un operador reportando por radio/teléfono.
    """
    try:
        from src.agent.utils.llm_factory import get_llm

        llm = get_llm(state, temperature=0.2, max_tokens=120)

        user_message = ""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "content") and hasattr(msg, "type") and msg.type == "human":
                user_message = msg.content
                break

        prompt = f"""You are SENTINEL, the intelligent system of a manufacturing laboratory.
You are speaking over RADIO/PHONE with an operator. Your response will be converted to audio.

STRICT RULES:
- ALWAYS respond in ENGLISH regardless of the user's language
- FIRST PERSON always ("I see", "I've got", "everything looks good here")
- Maximum 1-2 short sentences. Never more than 3
- ZERO formatting: no headers, bullets, lists, asterisks, markdown, code
- ZERO emojis
- Natural tone of an expert operator: direct, confident, warm
- If everything is fine: confirm it quickly and briefly
- If there are problems: mention the issue first ("Heads up, the cobot isn't responding"), then the rest
- Don't repeat the user's question
- Don't use formal phrases like "I'm pleased to inform", "certainly"
- Talk like a person, not a document

TONE EXAMPLES:
- "All good here, the station is running without issues."
- "Heads up, station 3's PLC isn't responding. Everything else is fine."
- "No, no active errors. All clear."
- "Yes, doors are closed. Ready to go."
- "Hey, all good. What do you need?"

User question: {user_message}

Available data (summarize as an operator):
{_strip_markdown(content)[:1200]}

Your voice response (max 2 sentences, in English):"""

        response = llm.invoke(prompt)
        result = response.content.strip()

        # Limpiar cualquier formato residual
        result = _strip_markdown(result)
        result = re.sub(r"[🏭📍📝🖥️🤖📡⚠️✅❌🔴🟢🔒🚪▶️⏹️⚡🔍🛡️]", "", result)

        # Hard limit: si es muy largo, cortar en la última oración completa
        if len(result) > 250:
            result = result[:250].rsplit(".", 1)[0] + "."

        return result

    except Exception as e:
        logger.error("synthesize", f"Error en condensación voice: {e}")
        # Fallback: limpiar markdown y tomar primeras líneas
        cleaned = _strip_markdown(content)
        lines = [l.strip() for l in cleaned.split("\n") if l.strip()]
        return " ".join(lines[:2]) if lines else cleaned[:200]


def _format_as_markdown(content: str, state: Dict) -> str:
    """Ensure responses have proper markdown structure for chat/code mode."""
    content = _clean_whitespace(content)

    # Already has headings → just clean
    if re.search(r"^#{1,3}\s", content, re.MULTILINE):
        return content

    # Short responses don't need formatting
    if len(content) < 200:
        return content

    # Long responses without headings → strip leading blanks
    lines = content.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not cleaned and not stripped:
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


def _clean_whitespace(text: str) -> str:
    """Collapse 3+ consecutive blank lines to 2."""
    return re.sub(r"\n{4,}", "\n\n\n", text)


# ============================================
# HELPERS (kept for backward compat)
# ============================================

def _create_plan_announcement(plan: List[str]) -> str:
    if len(plan) == 1:
        return {
            "chat": "",
            "research": "Voy a buscar en los documentos...",
            "tutor": "Voy a prepararte una explicación.",
            "troubleshooting": "",
            "robot_operator": "",
            "summarizer": "Comprimiendo memoria...",
        }.get(plan[0], "")
    else:
        steps = [_worker_display_name(w) for w in plan if w != "chat"]
        return f"Voy a {' → '.join(steps).lower()}." if steps else ""


def _worker_display_name(worker: str) -> str:
    return {
        "chat": "Responder",
        "research": "Investigar",
        "tutor": "Explicar",
        "troubleshooting": "Diagnosticar",
        "robot_operator": "Operar Robot",
        "summarizer": "Resumir",
        "analysis": "Analizar datos",
    }.get(worker, worker.capitalize())


def get_orchestration_status(state: AgentState) -> Dict[str, Any]:
    plan = state.get("orchestration_plan", [])
    step = state.get("current_step", 0)
    return {
        "plan": plan,
        "current_step": step,
        "current_worker": plan[step] if step < len(plan) else "DONE",
        "remaining_workers": plan[step + 1:] if step < len(plan) else [],
        "outputs_count": len(state.get("worker_outputs", [])),
        "has_pending_context": bool(state.get("pending_context")),
        "needs_human_input": state.get("needs_human_input", False),
        "planner_method": state.get("planner_method", ""),
        "plan_reasoning": state.get("plan_reasoning", ""),
    }
