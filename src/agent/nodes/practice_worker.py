"""
practice_worker.py

Bridge-in-the-Loop (BITL) practice worker. Guides students step-by-step
through automation routines, interrupting for bridge reports after each action.
LangGraph re-executes the node on each Command(resume=...).
"""
import json
import asyncio
from typing import Dict, Any, List, Optional

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from src.agent.state import AgentState
from src.agent.utils.device_comparator import compare_device
from src.agent.utils.logger import logger
from src.agent.utils.run_events import event_execute, event_report, event_narration


def practice_worker_node(state: AgentState) -> dict:
    """Practice worker with BITL. Runs once per bridge resume."""
    logger.node_start("practice", {"step": state.get("current_practice_step", 0)})

    if not state.get("practice_session_active"):
        return _handle_setup(state)

    return _handle_step_evaluation(state)


def _handle_setup(state: AgentState) -> dict:
    """First invocation: parse routine, notify bridge, emit first instruction."""
    steps = parse_automation_steps(
        state.get("automation_md_content", ""),
        state.get("automation_id", ""),
    )

    if not steps:
        logger.warning("practice", "No steps found in automation content")
        return {
            "messages": [AIMessage(content="No pude encontrar los pasos de la rutina. Verifica que la automatización tenga pasos definidos.")],
            "practice_session_active": False,
            "events": [event_report("practice", "No steps found in automation")],
        }

    robot_id = determine_target_robot(state)

    notify_bridge(robot_id, {
        "type": "practice_start",
        "session_id": state.get("_stream_session_id", ""),
        "automation_id": state.get("automation_id", ""),
        "robot_id": robot_id,
        "total_steps": len(steps),
    })

    first_step = steps[0]
    _emit_instruction(state, first_step, 0, len(steps))

    logger.info("practice", f"Practice session started: {len(steps)} steps, robot={robot_id}")

    interrupt_value = interrupt({
        "type": "awaiting_bridge",
        "step_index": 0,
        "instruction": first_step.get("description", ""),
        "expected": first_step.get("expected", {}),
        "robot_id": robot_id,
        "timeout_seconds": first_step.get("timeout", 120),
    })

    return {
        "practice_session_active": True,
        "current_practice_step": 0,
        "total_practice_steps": len(steps),
        "practice_expected_steps": steps,
        "target_robot_id": robot_id,
        "bridge_report": interrupt_value if isinstance(interrupt_value, dict) else None,
        "events": [event_execute("practice", f"Practice started: {len(steps)} steps")],
    }


def _handle_step_evaluation(state: AgentState) -> dict:
    """Evaluate current step against bridge_report, then advance or finish."""
    bridge_data = state.get("bridge_report")
    if not bridge_data:
        logger.warning("practice", "No bridge report received")
        return {
            "messages": [AIMessage(content="No recibí reporte del bridge. Verifica la conexión.")],
            "events": [event_report("practice", "No bridge report")],
        }

    current_step = state.get("current_practice_step", 0)
    steps = state.get("practice_expected_steps", [])
    robot_id = state.get("target_robot_id", "unknown")

    if current_step >= len(steps):
        return {
            "messages": [AIMessage(content="La sesión de práctica ya terminó.")],
            "practice_session_active": False,
            "events": [event_report("practice", "Session already completed")],
        }

    device_type = bridge_data.get("device_type", "xarm")
    step_data = steps[current_step]
    expected = step_data.get("expected", {})
    tolerance = step_data.get("tolerance", {})

    evaluation = compare_device(device_type, bridge_data.get("action_result", {}), expected, tolerance)

    feedback = _generate_feedback(
        step=step_data,
        evaluation=evaluation,
        bridge_data=bridge_data,
        step_number=current_step + 1,
        total_steps=len(steps),
        state=state,
    )

    step_result = {
        "step": current_step,
        "passed": evaluation["passed"],
        "score": evaluation["score"],
        "errors": evaluation["errors"],
        "details": evaluation["details"],
        "feedback": feedback,
    }
    practice_results = list(state.get("practice_results", [])) + [step_result]

    _emit_step_evaluation(state, step_result)

    logger.info("practice", f"Step {current_step} evaluated: passed={evaluation['passed']}, score={evaluation['score']}")

    next_step = current_step + 1

    if next_step >= len(steps):
        return _handle_session_complete(state, steps, practice_results, robot_id)

    next_step_data = steps[next_step]
    _emit_instruction(state, next_step_data, next_step, len(steps))

    notify_bridge(robot_id, {
        "type": "practice_step",
        "session_id": state.get("_stream_session_id", ""),
        "step_index": next_step,
        "expected": next_step_data.get("expected", {}),
    })

    interrupt_value = interrupt({
        "type": "awaiting_bridge",
        "step_index": next_step,
        "instruction": next_step_data.get("description", ""),
        "expected": next_step_data.get("expected", {}),
        "robot_id": robot_id,
        "timeout_seconds": next_step_data.get("timeout", 120),
    })

    return {
        "current_practice_step": next_step,
        "practice_results": practice_results,
        "bridge_report": interrupt_value if isinstance(interrupt_value, dict) else None,
        "events": [
            event_narration("practice", feedback, phase="evaluation"),
            event_execute("practice", f"Step {next_step + 1}/{len(steps)}: {next_step_data.get('description', '')[:80]}"),
        ],
    }


def _handle_session_complete(state: dict, steps: list, practice_results: list, robot_id: str) -> dict:
    """Generate final summary and close the practice session."""
    summary = generate_practice_summary(steps, practice_results, state)

    notify_bridge(robot_id, {
        "type": "practice_end",
        "session_id": state.get("_stream_session_id", ""),
        "robot_id": robot_id,
    })

    logger.info("practice", f"Practice completed: {summary['passed']}/{summary['total_steps']} passed, score={summary['overall_score']}")

    return {
        "messages": [AIMessage(content=summary["narrative"])],
        "practice_session_active": False,
        "practice_results": practice_results,
        "practice_update": {
            "practice_completed": True,
            "step": len(steps),
            "step_completed": True,
        },
        "events": [
            event_narration("practice", "Evaluacion completa", phase="done"),
            event_report("practice", f"Session complete: {summary['passed']}/{summary['total_steps']} passed"),
        ],
    }


def parse_automation_steps(md_content: str, automation_id: str) -> list:
    """Parse steps from markdown or fetch from Supabase as fallback."""
    steps = []

    if md_content:
        steps = _parse_steps_from_markdown(md_content)

    if not steps and automation_id:
        steps = _fetch_steps_from_supabase(automation_id)

    return steps


def _parse_steps_from_markdown(md_content: str) -> list:
    """Parse step blocks (### Paso N / ### Step N) from markdown."""
    import re

    steps = []
    step_pattern = re.compile(
        r"###\s+(?:Paso|Step)\s+(\d+)[:\s]*(.+?)(?=\n###\s+(?:Paso|Step)\s+\d+|\Z)",
        re.DOTALL | re.IGNORECASE,
    )

    for match in step_pattern.finditer(md_content):
        step_num = int(match.group(1))
        block = match.group(2).strip()
        description_line = block.split("\n")[0].strip()

        step = {
            "step_index": step_num - 1,
            "description": description_line,
            "expected": {},
            "tolerance": {},
            "timeout": 120,
            "hints": [],
            "max_retries": 2,
        }

        expected_match = re.search(r"\*\*(?:Esperado|Expected)[:\s]*\*\*\s*```json\s*(.+?)```", block, re.DOTALL)
        if expected_match:
            try:
                step["expected"] = json.loads(expected_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        tol_match = re.search(r"\*\*(?:Tolerancia|Tolerance)[:\s]*\*\*\s*```json\s*(.+?)```", block, re.DOTALL)
        if tol_match:
            try:
                step["tolerance"] = json.loads(tol_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        timeout_match = re.search(r"\*\*Timeout[:\s]*\*\*\s*(\d+)", block)
        if timeout_match:
            step["timeout"] = int(timeout_match.group(1))

        hints_match = re.search(r"\*\*(?:Hints|Pistas)[:\s]*\*\*\s*(.+?)(?:\n\*\*|\Z)", block, re.DOTALL)
        if hints_match:
            hints_text = hints_match.group(1).strip()
            step["hints"] = [h.strip().lstrip("- ") for h in hints_text.split("\n") if h.strip()]

        steps.append(step)

    steps.sort(key=lambda s: s["step_index"])
    return steps


def _fetch_steps_from_supabase(automation_id: str) -> list:
    """Fetch automation steps from Supabase as fallback."""
    try:
        from src.agent.services import get_supabase
        sb = get_supabase()
        if not sb:
            return []

        result = sb.schema("lab").from_("automation_steps") \
            .select("step_index, description, expected, tolerance, timeout, hints, max_retries") \
            .eq("automation_id", automation_id) \
            .order("step_index") \
            .execute()

        if result.data:
            return result.data
    except Exception as e:
        logger.warning("practice", f"Failed to fetch steps from Supabase: {e}")

    return []


def determine_target_robot(state: dict) -> str:
    """Pick target robot: explicit robot_ids > single connected > first available."""
    robot_ids = state.get("robot_ids", [])
    if robot_ids:
        return robot_ids[0]

    from src.agent.shared_state import ROBOT_CONNECTIONS
    if len(ROBOT_CONNECTIONS) == 1:
        return list(ROBOT_CONNECTIONS.keys())[0]

    if ROBOT_CONNECTIONS:
        return list(ROBOT_CONNECTIONS.keys())[0]

    return "unknown"


def _emit_instruction(state: dict, step: dict, step_index: int, total_steps: int):
    """Emit step instruction to student via stream callback."""
    session_id = state.get("_stream_session_id", "")
    if not session_id:
        return
    try:
        from src.api_server import get_stream_callback
        callback = get_stream_callback(session_id)
        if callback:
            callback({
                "type": "instruction",
                "step_index": step_index,
                "description": step.get("description", ""),
                "hints": step.get("hints", []),
                "total_steps": total_steps,
            })
    except ImportError:
        logger.debug("practice", "Cannot import get_stream_callback (not in server context)")


def _emit_step_evaluation(state: dict, result: dict):
    """Emit evaluation result via stream callback."""
    session_id = state.get("_stream_session_id", "")
    if not session_id:
        return
    try:
        from src.api_server import get_stream_callback
        callback = get_stream_callback(session_id)
        if callback:
            callback({
                "type": "step_evaluation",
                **result,
            })
    except ImportError:
        logger.debug("practice", "Cannot import get_stream_callback (not in server context)")


def _generate_feedback(
    step: dict,
    evaluation: dict,
    bridge_data: dict,
    step_number: int,
    total_steps: int,
    state: dict,
) -> str:
    """Generate personalized feedback via LLM, with static fallback."""
    try:
        from src.agent.utils.llm_factory import get_llm

        llm = get_llm(state, temperature=0.7, max_tokens=300)

        prompt = f"""Eres un instructor de robotica evaluando a un estudiante.

Paso {step_number} de {total_steps}: {step.get('description', '')}

Resultado de la evaluacion:
- Aprobado: {'Si' if evaluation['passed'] else 'No'}
- Score: {evaluation['score']:.0%}
- Errores: {', '.join(evaluation['errors']) if evaluation['errors'] else 'Ninguno'}
- Detalles: {json.dumps(evaluation['details'], ensure_ascii=False)}

Datos del robot despues de la accion:
{json.dumps(bridge_data.get('action_result', {}), ensure_ascii=False)[:500]}

Genera un feedback breve (2-3 oraciones) para el estudiante:
- Si aprobo: felicita y explica brevemente por que estuvo bien
- Si no aprobo: explica que fallo de forma constructiva y da un tip concreto
- Se motivador pero preciso tecnicamente
- Responde en espanol"""

        response = llm.invoke(prompt)
        return response.content.strip()

    except Exception as e:
        logger.warning("practice", f"LLM feedback failed, using fallback: {e}")
        if evaluation["passed"]:
            return f"Paso {step_number} completado correctamente (score: {evaluation['score']:.0%})."
        else:
            errors_str = "; ".join(evaluation["errors"][:3])
            return f"Paso {step_number} no aprobado. Errores: {errors_str}. Intentalo de nuevo."


def generate_practice_summary(steps: list, results: list, state: dict = None) -> dict:
    """Generate final practice session summary with narrative."""
    passed = sum(1 for r in results if r.get("passed"))
    failed = len(results) - passed
    avg_score = sum(r.get("score", 0) for r in results) / len(results) if results else 0

    narrative = _generate_summary_narrative(steps, results, passed, failed, avg_score, state)

    return {
        "total_steps": len(steps),
        "passed": passed,
        "failed": failed,
        "overall_score": round(avg_score, 3),
        "step_results": results,
        "narrative": narrative,
    }


def _generate_summary_narrative(
    steps: list, results: list, passed: int, failed: int, avg_score: float, state: dict = None
) -> str:
    """Generate summary narrative via LLM, with static fallback."""
    try:
        if state:
            from src.agent.utils.llm_factory import get_llm

            llm = get_llm(state, temperature=0.7, max_tokens=500)

            failed_details = []
            for r in results:
                if not r.get("passed"):
                    step_desc = steps[r["step"]]["description"] if r["step"] < len(steps) else f"Paso {r['step'] + 1}"
                    failed_details.append(f"- Paso {r['step'] + 1} ({step_desc}): {', '.join(r.get('errors', [])[:2])}")

            prompt = f"""Genera un resumen final de una sesion de practica de robotica.

Resultados:
- Pasos completados: {passed} de {len(steps)}
- Score promedio: {avg_score:.0%}
- Pasos fallidos:
{chr(10).join(failed_details) if failed_details else '  Ninguno'}

Genera un resumen motivacional (3-5 oraciones):
1. Felicita si fue bien, anima si hubo fallos
2. Menciona areas de mejora concretas si aplica
3. Sugiere que practicar para mejorar
4. Responde en espanol, sin emojis"""

            response = llm.invoke(prompt)
            return response.content.strip()
    except Exception as e:
        logger.warning("practice", f"LLM summary failed: {e}")

    narrative = f"Completaste {passed} de {len(steps)} pasos correctamente (score: {avg_score:.0%})."
    if failed > 0:
        failed_steps = [r for r in results if not r.get("passed")]
        narrative += f" Revisa los pasos: {', '.join(str(r['step'] + 1) for r in failed_steps)}."
    return narrative


def notify_bridge(robot_id: str, message: dict):
    """Send message to bridge. Uses run_coroutine_threadsafe from LangGraph's sync thread."""
    try:
        from src.agent.shared_state import ROBOT_CONNECTIONS
        ws = ROBOT_CONNECTIONS.get(robot_id)
        if not ws:
            logger.debug("practice", f"Cannot notify bridge: {robot_id} not connected")
            return

        try:
            from src.api_server import get_main_loop
            loop = get_main_loop()
        except ImportError:
            loop = None

        if loop and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(ws.send_json(message), loop)
            future.result(timeout=5)
        else:
            asyncio.run(ws.send_json(message))

        logger.debug("practice", f"Notified bridge {robot_id}: {message.get('type', '?')}")

    except Exception as e:
        logger.warning("practice", f"Failed to notify bridge {robot_id}: {e}")
