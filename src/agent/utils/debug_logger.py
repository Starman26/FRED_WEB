"""
debug_logger.py - Logger detallado para debugging del grafo
"""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from functools import wraps


class DebugLogger:
    """Logger detallado para debugging del grafo LangGraph"""
    
    CRITICAL_FIELDS = [
        "next", "done", "current_step", "orchestration_plan", "worker_outputs",
        "pending_context", "needs_human_input", "clarification_questions", "task_type",
        "research_result", "tutor_result", "troubleshooting_result",
    ]
    
    def __init__(self, enabled: bool = True, verbose: bool = False):
        self.enabled = enabled
        self.verbose = verbose
        self._snapshots: List[Dict] = []
    
    def log_state_snapshot(self, node: str, phase: str, state: Dict[str, Any], extra: Optional[Dict] = None):
        """Imprime snapshot del estado antes/después de un nodo."""
        if not self.enabled:
            return
        
        timestamp = datetime.now().isoformat()
        snapshot = {"timestamp": timestamp, "node": node, "phase": phase, "state": {}}
        
        for field in self.CRITICAL_FIELDS:
            value = state.get(field)
            if value is not None:
                snapshot["state"][field] = self._format_value(value)
        
        if extra:
            snapshot["extra"] = extra
        
        self._snapshots.append(snapshot)
        
        print(f"\n{'='*60}")
        print(f"[{phase}] Node: {node} @ {timestamp}")
        print(f"{'='*60}")
        for field, value in snapshot["state"].items():
            print(f"  {field}: {value}")
        if extra:
            print(f"  --- Extra ---")
            for k, v in extra.items():
                print(f"  {k}: {v}")
        print(f"{'='*60}\n")
    
    def log_tool_output(self, tool_name: str, input_data: Any, output_data: Any):
        if not self.enabled:
            return
        print(f"\n[TOOL] {tool_name}")
        print(f"  Input: {self._format_value(input_data, max_len=100)}")
        if isinstance(output_data, dict):
            print(f"  Output keys: {list(output_data.keys())}")
        elif isinstance(output_data, tuple):
            print(f"  Output tuple length: {len(output_data)}")
        else:
            print(f"  Output type: {type(output_data).__name__}")
    
    def log_worker_output(self, worker: str, output: Dict):
        if not self.enabled:
            return
        print(f"\n[WORKER OUTPUT] {worker}")
        print(f"  Status: {output.get('status', 'unknown')}, Confidence: {output.get('confidence', 'N/A')}")
        print(f"  Evidence count: {len(output.get('evidence', []))}")
    
    def log_orchestration_step(self, plan: List[str], current_step: int, next_worker: str):
        if not self.enabled:
            return
        print(f"\n[ORCHESTRATION] Step {current_step}/{len(plan)}")
        for i, worker in enumerate(plan):
            marker = "→" if i == current_step else " "
            status = "✓" if i < current_step else ("●" if i == current_step else "○")
            print(f"  {marker} {status} {worker}")
        print(f"  Next: {next_worker}")
    
    def verify_evidence_propagation(self, state: Dict) -> Dict[str, Any]:
        result = {
            "has_worker_outputs": len(state.get("worker_outputs", [])) > 0,
            "has_research_output": False,
            "evidence_count": 0,
            "has_pending_context": bool(state.get("pending_context")),
            "pending_evidence_count": len(state.get("pending_context", {}).get("evidence", [])),
            "issues": []
        }
        
        for output in state.get("worker_outputs", []):
            if output.get("worker") == "research":
                result["has_research_output"] = True
                result["evidence_count"] = len(output.get("evidence", []))
        
        if result["has_research_output"] and result["evidence_count"] == 0:
            result["issues"].append("Research output existe pero sin evidencia")
        if result["evidence_count"] > 0 and result["pending_evidence_count"] == 0:
            result["issues"].append("Evidencia en worker_outputs pero no en pending_context")
        
        if self.enabled:
            print(f"\n[EVIDENCE CHECK]")
            for k, v in result.items():
                if k != "issues":
                    print(f"  {k}: {v}")
            if result["issues"]:
                print(f"  ISSUES:")
                for issue in result["issues"]:
                    print(f"    ⚠️ {issue}")
        return result
    
    def get_snapshots(self) -> List[Dict]:
        return self._snapshots
    
    def clear_snapshots(self):
        self._snapshots = []
    
    def _format_value(self, value: Any, max_len: int = 200) -> str:
        if value is None:
            return "None"
        if isinstance(value, str):
            return f"{value[:max_len]}...[truncated]" if len(value) > max_len and not self.verbose else value
        if isinstance(value, list):
            return f"[{len(value)} items]" if len(value) > 3 and not self.verbose else str(value)[:max_len]
        if isinstance(value, dict):
            return f"{{...{len(value.keys())} keys}}" if len(value.keys()) > 5 and not self.verbose else str(value)[:max_len]
        return str(value)[:max_len]


def with_debug_logging(node_name: str, debug_logger: Optional[DebugLogger] = None):
    """Decorator que añade logging antes/después de cada nodo."""
    def decorator(func):
        @wraps(func)
        def wrapper(state, *args, **kwargs):
            logger = debug_logger or DebugLogger()
            logger.log_state_snapshot(node_name, "BEFORE", state)
            try:
                result = func(state, *args, **kwargs)
                if isinstance(result, dict):
                    logger.log_state_snapshot(node_name, "AFTER", {**state, **result}, extra={"return_keys": list(result.keys())})
                    for output in result.get("worker_outputs", []):
                        logger.log_worker_output(node_name, output)
                return result
            except Exception as e:
                print(f"\n[ERROR] {node_name}: {e}")
                raise
        return wrapper
    return decorator


debug = DebugLogger(enabled=True, verbose=False)

def enable_debug(verbose: bool = False):
    global debug
    debug = DebugLogger(enabled=True, verbose=verbose)

def disable_debug():
    global debug
    debug = DebugLogger(enabled=False)
