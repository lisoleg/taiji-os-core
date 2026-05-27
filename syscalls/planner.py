"""syscalls/planner.py — 太极OS 系统调用层：规划器"""
from typing import List, Dict


class Planner:
    """
    将用户意图分解为有序的执行步骤列表。
    """

    def plan(self, intent: str, context: Dict) -> List[Dict]:
        """
        返回步骤列表，每步包含 action / params。
        当前为规则基础实现，可替换为 LLM 驱动的规划。
        """
        steps = []
        tokens = intent.lower().split()

        if any(kw in tokens for kw in ("设计", "实现", "开发", "design", "implement")):
            steps.append({"action": "analyze_requirements", "params": {"input": intent}})
            steps.append({"action": "generate_solution", "params": {"input": intent}})
            steps.append({"action": "verify_solution", "params": {}})
        elif any(kw in tokens for kw in ("查询", "搜索", "find", "search", "query")):
            steps.append({"action": "retrieve", "params": {"query": intent}})
        else:
            steps.append({"action": "respond", "params": {"input": intent}})

        return steps
