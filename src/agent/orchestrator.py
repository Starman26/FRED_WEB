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

from src.agent.state import AgentState, RESET_WORKER_OUTPUTS
from src.agent.contracts.worker_contract import WorkerOutput, EvidenceItem
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_plan, event_route, event_report, event_error, event_narration
from src.agent.interaction_modes import get_truth_hierarchy, get_shared_rules


VALID_WORKERS = {"chat", "research", "tutor", "troubleshooting", "summarizer", "robot_operator", "analysis", "practice"}

_WORKER_DESC = {
    "research": "Searching technical documents",
    "tutor": "Preparing explanation",
    "troubleshooting": "Running diagnostics",
    "robot_operator": "Controlling robot",
    "analysis": "Analyzing data",
    "chat": "Thinking",
    "summarizer": "Compressing memory",
    "practice": "practice — Guided hands-on session with bridge-in-the-loop (BITL)",
}


# ============================================
# SYNTHESIS PROMPTS
# ============================================

# Shared synthesis base — injected into all synth prompts
_SYNTH_BASE = f"""
{get_truth_hierarchy()}

{get_shared_rules()}

## CONFLICT RESOLUTION
- If real-time diagnostics and documentation disagree, state both and trust the real-time data
- If two workers provide contradictory information, explain the conflict briefly and state which source you trust more
- Direct tool output > generalized explanation
- If conflict remains, say uncertainty clearly — never invent certainty
"""

SYNTH_PROMPT_CHAT = """You are ORION, synthesizing outputs from multiple agents into one final response for the user.

## USER QUESTION
{{user_question}}

## AGENT OUTPUTS
{{raw_data}}
{{sources_text}}

## YOUR JOB
Produce a single response that directly answers the user's question using the most reliable information available.

{base}

## SYNTHESIS RULES
- Cover every important part of the user's request — if they asked about X AND Y, cover both
- Integrate outputs into one coherent answer — do not dump separate worker blocks
- If one source has real-time operational data and another has documentation, prioritize real-time and use docs as supporting context
- Do not repeat the same point in different words
- Do not narrate the internal multi-agent process unless it helps answer the question
- Present only the useful conclusion

## RESPONSE STYLE
Adapt structure to user intent:
- Yes/No question → answer immediately in the first line
- How-to question → concise steps
- What/Why question → direct explanation first, then supporting detail
- Status question → current state first, then relevant issues or next action
- Troubleshooting question → diagnosis first, then recommended actions

## FORMAT
- Lead with the key answer using ==text== markers
- ==+text== confirmed good status | ==-text== confirmed issue | ==~text== warning/partial risk | ==?text== uncertainty
- Use clean Markdown only when it improves readability
- Use short sections, not overly formal reports
- Use numbered steps only for action sequences
- Include sources after --- only if sources are available
- Be concise but complete

User name: {{user_name}}

End with exactly 3 follow-up suggestions:
---SUGGESTIONS---
1. [suggestion]
2. [suggestion]
3. [suggestion]
---END_SUGGESTIONS---

Your synthesized response:""".format(base=_SYNTH_BASE)


SYNTH_PROMPT_AGENT = """You are ORION, the live intelligent operating system of a manufacturing laboratory.

You are not writing a report. You are giving a direct operational answer as the system itself.

## USER QUESTION
{{user_question}}

## AVAILABLE DATA
{{raw_data}}

{base}

## PRIORITY
1. State the most important operational fact first
2. If there is a problem, mention the problem before anything else
3. If everything is normal, confirm normal status quickly
4. If the data is incomplete, say exactly what you cannot confirm
5. Prefer real observed state over interpretation

## STRICT STYLE
- FIRST PERSON only
- Maximum 2 short sentences, 3 only if critical
- No markdown headers, no bullets, no lists, no emojis
- Natural, direct, technical
- No repetition, no filler

## EXAMPLES
Good: "==-PLC on station 2 is offline.== I'm not seeing a valid response from the controller."
Good: "==+All stations are currently healthy.== No active errors detected."
Good: "Moved X +20mm. Current position: (120, 50, 200)."
Bad:  "After reviewing the available information, I can confirm that station 2 appears to be experiencing a connectivity issue."
Bad:  "I have successfully completed the requested operation. The new position is..."
Bad:  "Certainly! Based on the data provided..."

## NEVER
- Never say "Certainly", "I'd be happy to", "Based on the information provided"
- Never explain why you did something unless the user asks "why"
- Never enumerate multiple options — pick the best one
- Never use greetings or pleasantries

Use ==text== to highlight only the most important fact.

Your response:""".format(base=_SYNTH_BASE)


SYNTH_PROMPT_VOICE = """You are ORION, the intelligent operating system of a manufacturing laboratory.
Your response will be spoken aloud to an operator over voice.

## USER QUESTION
{{user_question}}

## AVAILABLE DATA
{{raw_data}}

{base}

## PRIMARY GOAL
Give a spoken response that is immediately understandable, operationally useful, and easy to hear once.

## SPEAKING PRIORITY
1. Say the most important fact first
2. If there is a problem, say the problem first
3. If everything is fine, confirm that quickly
4. If action is needed, mention only the next best action
5. If uncertain, say exactly what is not confirmed

## STRICT RULES
- FIRST PERSON only
- Maximum 2 short sentences, 3 only if absolutely necessary
- No markdown, no bullets, no lists, no code formatting
- No emojis, no special characters
- No long numbers unless essential
- No repeating the user's question
- No formal writing, no report tone

## VOICE STYLE
- Sound like an expert operator on radio
- Direct, calm, clear
- Short spoken phrasing over written phrasing

## EXAMPLES
Good: "Station 4 is offline right now. Check the PLC network link first."
Good: "All clear. No active errors on any station."
Good: "Done. Robot moved to home position."
Bad:  "After reviewing the available information, I can say that station 4 appears to be experiencing a connectivity-related issue that may require further diagnosis."
Bad:  "Here's what I found: first, the PLC is showing... second, the network..."
Bad:  "I'm pleased to inform you that all systems are operational."

## NEVER
- Never use markdown formatting of any kind
- Never stack multiple details in one sentence
- Never use formal or report-style language
- Never enumerate options — give one clear answer

Your voice response:""".format(base=_SYNTH_BASE)


# ============================================
# ADAPTIVE ROUTER
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

    # RULE 1: Worker error crítico → synthesize con lo que hay
    if status == "error":
        return {
            "action": "stop_early",
            "reason": f"{worker_name} returned error, synthesizing available results",
            "modified_plan": None,
        }

    # RULE 2: Research sin evidencia → quitar tutor del plan
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

    # RULE 3: Alta confianza y plan completo → stop early
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

    # RULE 4: Research tiene gaps sobre entidades del lab → agregar troubleshooting
    if worker_name == "research" and "troubleshooting" not in remaining_plan:
        extra = last_output.get("extra", {})
        gaps = extra.get("gaps", []) if isinstance(extra, dict) else []
        if gaps:
            gap_text = " ".join(str(g) for g in gaps).lower()
            if re.search(r"estaci[oó]n|station|lab|equipo|plc|cobot|puerta|sensor", gap_text):
                return {
                    "action": "add_worker",
                    "reason": f"Research gaps mention lab entities: {gaps[:2]}; adding troubleshooting for real data",
                    "modified_plan": plan[: current_step + 1] + ["troubleshooting"] + remaining_plan,
                }

    # RULE 5: Troubleshooter recomienda tutor via next_actions
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

    # Default: continue
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
    _raw_context = state.get("pending_context", {}) or {}
    pending_context = dict(_raw_context)
    if "evidence" in pending_context:
        pending_context["evidence"] = list(pending_context["evidence"])
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
    # ADAPTIVE EVALUATION
    # ══════════════════════════════════════════
    if last_output:
        evaluation = _evaluate_worker_output(last_output, plan, current_step)

        events.append(event_route(
            "adaptive_router",
            f"Eval: {evaluation['reason']}",
            route=evaluation["action"],
        ))

        if evaluation["action"] != "continue":
            events.append(event_narration("router", evaluation["reason"], phase="routing"))

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
    accumulated_evidence = list(pending_context.get("evidence", []))
    for output in worker_outputs:
        evidence = output.get("evidence", [])
        if evidence:
            accumulated_evidence.extend(evidence)
    if accumulated_evidence:
        pending_context["evidence"] = accumulated_evidence

    next_worker = plan[next_step]
    worker_desc = _WORKER_DESC.get(next_worker, next_worker)
    events.append(event_narration("router", f"{worker_desc}...", phase="transition"))

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
    # Known emojis used in the codebase
    text = re.sub(r"[🏭📍📝🖥️🤖📡⚠️✅❌🔴🟢🔒🚪▶️⏹️⚡🔍🛡️⛔🔧💡🎯📊📈📉🚀💻🔄🔁⏳⏱️🕐🎉👋🧠💬🆘🛑🔔📢📌🔗📎🗂️📋📄📃🔐🔑⭐🌟💫✨🎁🎊🏆🥇🥈🥉💰💸📞📧📬🌍🌎🌏🔌🔋⚙️🛠️🔩📐📏🧪🧬🔬🔭🩺💊🧯🚒🚑🏗️🏠🏢🏫🏥🏦]", "", text)
    # Broad unicode emoji ranges
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


def _strip_markdown(content: str) -> str:
    """Remove all markdown/highlight formatting from text."""
    content = re.sub(r"==([+\-~?])?(.+?)==", r"\2", content)
    content = re.sub(r"#{1,6}\s*", "", content)
    content = re.sub(r"\*\*(.+?)\*\*", r"\1", content)
    content = re.sub(r"\*(.+?)\*", r"\1", content)
    content = re.sub(r"`(.+?)`", r"\1", content)
    content = re.sub(r"^[-•]\s*", "", content, flags=re.MULTILINE)
    content = re.sub(r"\|.+\|", "", content)
    content = re.sub(r"---+", "", content)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return _strip_emojis(content)


def _extract_suggestions(text: str) -> tuple:
    """Extract ---SUGGESTIONS--- block from text. Returns (clean_text, suggestions_list)."""
    if "---SUGGESTIONS---" not in text:
        return text, []

    parts = text.split("---SUGGESTIONS---", 1)
    clean = parts[0].strip()
    suggestions = []

    if len(parts) > 1:
        tail = parts[1]
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


def _save_diagnostic_history(state: dict, response_text: str):
    """Save completed diagnosis to diagnostic_history for cross-session learning."""
    try:
        from src.agent.services import get_supabase
        sb = get_supabase()
        if not sb:
            return

        worker_outputs = state.get("worker_outputs", [])
        if not worker_outputs:
            return

        # Solo guardar si hubo troubleshooting, analysis, o research
        workers_used = [o.get("worker", "") for o in worker_outputs if isinstance(o, dict)]
        if not any(w in ("troubleshooting", "analysis", "research") for w in workers_used):
            return

        # Extraer query del usuario
        user_query = ""
        for m in reversed(state.get("messages", [])):
            if hasattr(m, "type") and m.type == "human":
                user_query = (m.content or "")[:500]
                break
            if isinstance(m, dict) and m.get("role") in ("human", "user"):
                user_query = (m.get("content") or "")[:500]
                break

        if not user_query:
            return

        # --- Extraer equipment_type ---
        # Fuente 1: intent_analysis
        intent = state.get("intent_analysis", {})
        entities = intent.get("entities", {}) if isinstance(intent, dict) else {}
        equipment_type = entities.get("equipment")

        # Fuente 2: worker outputs extras
        for output in worker_outputs:
            if not isinstance(output, dict):
                continue
            extra = output.get("extra", {})
            if isinstance(extra, dict):
                if not equipment_type:
                    equipment_type = extra.get("equipment_type")

        # Fuente 3: pending_context
        pending = state.get("pending_context", {}) or {}
        if not equipment_type:
            detection = pending.get("detection", {})
            equipment_type = detection.get("equipment") if isinstance(detection, dict) else pending.get("detected_equipment")

        # --- Extraer tools_used desde tool_execution_log ---
        tool_log = state.get("tool_execution_log", [])
        tools_used = list(set(
            entry.get("tool", "") for entry in tool_log if isinstance(entry, dict) and entry.get("tool")
        ))

        # Fallback: extraer de worker outputs si tool_execution_log está vacío
        if not tools_used:
            for output in worker_outputs:
                if not isinstance(output, dict):
                    continue
                extra = output.get("extra", {})
                if isinstance(extra, dict) and extra.get("web_search_used"):
                    tools_used.append("web_search")
                if output.get("worker") == "troubleshooting":
                    tools_used.append("troubleshooting_tools")

        # --- Extraer actions_taken ---
        actions_taken = [
            {
                "tool": entry.get("tool"),
                "success": entry.get("success"),
                "verified": entry.get("verified"),
                "phase": entry.get("phase"),
                "duration_ms": entry.get("duration_ms"),
            }
            for entry in tool_log
            if isinstance(entry, dict) and entry.get("tool")
        ]

        # --- Extraer duration_ms total ---
        total_duration = 0
        for output in worker_outputs:
            if isinstance(output, dict):
                meta = output.get("metadata", {})
                if isinstance(meta, dict):
                    total_duration += meta.get("processing_time_ms", 0)

        # --- Extraer evidence sources ---
        evidence_sources = []
        for output in worker_outputs:
            if not isinstance(output, dict):
                continue
            for ev in output.get("evidence", []):
                if isinstance(ev, dict):
                    evidence_sources.append({
                        "title": ev.get("title"),
                        "page": ev.get("page"),
                        "type": (ev.get("metadata") or {}).get("source_type", "internal"),
                    })

        # --- Build record ---
        record = {
            "session_id": state.get("_stream_session_id", ""),
            "user_id": state.get("auth_user_id") or state.get("user_id"),
            "team_id": state.get("team_id"),
            "user_query": user_query,
            "equipment_type": equipment_type,
            "diagnosis": response_text[:2000],
            "actions_taken": actions_taken if actions_taken else [],
            "tools_used": tools_used if tools_used else [],
            "evidence_sources": evidence_sources if evidence_sources else [],
            "tokens_used": state.get("token_usage", 0),
            "duration_ms": total_duration if total_duration > 0 else None,
            "workers_used": workers_used,
        }

        # Remove None values (Supabase rejects explicit nulls for some columns)
        record = {k: v for k, v in record.items() if v is not None}

        sb.schema("lab").from_("diagnostic_history").insert(record).execute()
        logger.info("orchestrator", f"Diagnostic saved: equipment={equipment_type}, station={station_number}, tools={len(tools_used)}, actions={len(actions_taken)}")

    except Exception as e:
        logger.warning("orchestrator", f"Failed to save diagnostic history: {e}")


def synthesize_node(state: AgentState) -> Dict[str, Any]:
    """Synthesizes worker outputs into a coherent response.

    LLM bypass conditions (saves 1 LLM call):
    - Single chat/tutor output without sources
    - Single research output with evidence and confidence >= 0.7
    - Single analysis output (already self-contained)

    Modes: agent, voice, chat/code, practice
    """
    logger.node_start("synthesize", {"outputs_count": len(state.get("worker_outputs", []))})
    events = [event_report("synthesize", "Combining results...")]

    worker_outputs = state.get("worker_outputs", [])
    interaction_mode = state.get("interaction_mode", "chat").lower()

    # - Practice mode: bypass synthesis, persist updates -
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

    # - Extract user's original question -
    user_message = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            user_message = (m.content or "").strip()
            break
        if isinstance(m, dict) and m.get("role", m.get("type")) in ("user", "human"):
            user_message = (m.get("content") or "").strip()
            break

    # - Collect worker data -
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

    # - Decide if LLM synthesis is needed -
    worker_names = {o.get("worker", "unknown") for o in worker_outputs}

    # Bypass: Lightweight workers (chat, tutor) without sources
    is_lightweight = (
        len(worker_outputs) == 1
        and worker_names.issubset({"chat", "tutor"})
        and not all_sources
    )

    # Bypass: Solo research that already synthesized with evidence
    is_self_sufficient_research = (
        len(worker_outputs) == 1
        and worker_names == {"research"}
        and all_sources
        and worker_outputs[0].get("confidence", 0) >= 0.7
        and len(worker_outputs[0].get("content", "")) > 100
    )

    # Bypass: Analysis worker (already produces complete response)
    is_analysis = (
        len(worker_outputs) == 1
        and worker_names == {"analysis"}
    )

    # Troubleshooting-only: format but don't rewrite (user saw investigation in real-time)
    is_troubleshoot_only = (
        len(worker_outputs) == 1
        and worker_names == {"troubleshooting"}
    )

    skip_llm_synthesis = is_lightweight or is_self_sufficient_research or is_analysis

    synth_tokens = 0
    if is_troubleshoot_only:
        # Skip LLM synthesis entirely — troubleshooter already formatted
        ts_content = ""
        for out in worker_outputs:
            if isinstance(out, dict):
                ts_content = out.get("content", "")
            elif hasattr(out, "content"):
                ts_content = out.content or ""
        combined = ts_content
        events.append(event_report("synthesize", "LLM bypass (troubleshoot pass-through)"))
    elif skip_llm_synthesis:
        combined = content_parts[0].split("]: ", 1)[-1] if content_parts else ""
        bypass_reason = (
            "lightweight" if is_lightweight
            else "self-sufficient research" if is_self_sufficient_research
            else "analysis"
        )
        events.append(event_report("synthesize", f"LLM bypass ({bypass_reason})"))
    else:
        events.append(event_report("synthesize", f"Synthesizing ({', '.join(worker_names)})..."))
        combined, synth_tokens = _synthesize_with_llm(
            user_message, raw_data, all_sources, state,
        )

    # - Mode-specific post-processing -
    if interaction_mode in ("chat", "code") and combined:
        combined = _format_as_markdown(combined)

    # Extract suggestions before cleanup
    combined, extracted_suggestions = _extract_suggestions(combined)
    combined = _strip_emojis(combined)

    # Suggestions: analysis never generates them
    if interaction_mode == "analysis":
        follow_ups = []
    else:
        follow_ups = extracted_suggestions or state.get("follow_up_suggestions", [])

    # Save diagnostic to history for cross-session learning
    if interaction_mode not in ("practice",) and combined:
        _save_diagnostic_history(state, combined)

    base_return = {
        "worker_outputs": RESET_WORKER_OUTPUTS,
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

    # Always emit the final formatted message
    base_return["messages"] = [AIMessage(content=combined)]
    return base_return


def _synthesize_with_llm(
    user_question: str,
    raw_data: str,
    sources: set,
    state: Dict,
) -> tuple:
    """Uses LLM to synthesize worker outputs. Returns (text, token_count)."""
    user_name = state.get("user_name", "User")
    interaction_mode = state.get("interaction_mode", "chat").lower()

    sources_text = ""
    if sources:
        sources_text = "\nFuentes disponibles:\n" + "\n".join(f"- {s}" for s in sorted(sources))

    # - Select prompt and params by mode -
    if interaction_mode == "agent":
        prompt = SYNTH_PROMPT_AGENT.format(
            user_question=user_question,
            raw_data=raw_data[:1500],
        )
        system_msg = "You are ORION, the live operating system of a manufacturing laboratory. Follow the response contract exactly."
        max_tokens = 200
        temperature = 0.0

    elif interaction_mode == "voice":
        prompt = SYNTH_PROMPT_VOICE.format(
            user_question=user_question,
            raw_data=_strip_markdown(raw_data)[:1200],
        )
        system_msg = "You are ORION speaking over voice radio. Your response will be converted to audio. Be immediately understandable in a single listen."
        max_tokens = 150
        temperature = 0.2

    else:
        prompt = SYNTH_PROMPT_CHAT.format(
            user_question=user_question,
            raw_data=raw_data[:4000],
            sources_text=sources_text,
            user_name=user_name,
        )
        system_msg = "You are ORION, synthesizing multi-agent outputs into one coherent response. Follow the synthesis contract exactly."
        max_tokens = 2000
        temperature = 0.3

    try:
        from src.agent.utils.llm_factory import get_llm, invoke_and_track

        llm = get_llm(state, temperature=temperature, max_tokens=max_tokens)
        response, tokens = invoke_and_track(llm, [
            SystemMessage(content=system_msg),
            HumanMessage(content=prompt),
        ], "synthesize")

        result = (response.content or "").strip()

        # Post-processing for voice/agent
        if interaction_mode in ("agent", "voice"):
            result = _strip_markdown(result)
            result = _strip_emojis(result)
            if len(result) > 300:
                result = result[:300].rsplit(".", 1)[0] + "."

        if len(result) < 10:
            return raw_data.split("]: ", 1)[-1] if "]: " in raw_data else raw_data, tokens

        return result, tokens

    except Exception as e:
        logger.error("synthesize", f"Synthesis LLM error: {e}")
        # Fallback: concatenate worker outputs
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


# ============================================
# FORMAT HELPERS
# ============================================

def _format_as_markdown(content: str) -> str:
    """Ensure responses have proper markdown structure for chat/code mode."""
    content = re.sub(r"\n{4,}", "\n\n\n", content)

    # Already has headings → just clean
    if re.search(r"^#{1,3}\s", content, re.MULTILINE):
        return content

    # Short responses don't need formatting
    if len(content) < 200:
        return content

    # Strip leading blank lines
    lines = content.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not cleaned and not stripped:
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


# ============================================
# PUBLIC HELPERS
# ============================================

def get_orchestration_status(state: AgentState) -> Dict[str, Any]:
    """Get current orchestration status for debugging/monitoring."""
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