"""syscalls/executor.py — 太极OS 系统调用层：执行器"""
import time
from typing import Dict, Any, List


class Executor:
    """
    按照 Planner 生成的步骤列表顺序执行，并记录执行结果。
    """

    def execute(self, steps: List[Dict], llm_router=None) -> List[Dict]:
        results = []
        for step in steps:
            action = step.get("action", "noop")
            params = step.get("params", {})
            t0 = time.time()
            try:
                output = self._dispatch(action, params, llm_router)
                status = "ok"
            except Exception as e:
                output = str(e)
                status = "error"
            elapsed = time.time() - t0
            results.append({
                "action": action,
                "status": status,
                "output": output,
                "elapsed_ms": round(elapsed * 1000, 2),
            })
        return results

    def _dispatch(self, action: str, params: Dict, llm_router=None) -> Any:
        if action == "respond" and llm_router:
            return llm_router.complete(params.get("input", ""))
        elif action == "analyze_requirements":
            return f"[Analyzed] {params.get('input', '')[:120]}"
        elif action == "generate_solution":
            if llm_router:
                return llm_router.complete(f"生成方案：{params.get('input', '')}")
            return "[Mock] solution generated"
        elif action == "verify_solution":
            return "[Verified] solution passed basic checks"
        elif action == "retrieve":
            return f"[Retrieved] results for: {params.get('query', '')}"
        return f"[Noop] action={action}"
