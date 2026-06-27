"""SkillEngine — 从成功工作流执行中自动创建可复用 Skill。

检测重复步骤模式，将成功链路固化可复用的 Skill 模板，
使得 DynamicPlanner 可以直接引用这些 Skill 而无需重新规划。
"""

from __future__ import annotations

from typing import Any

from engine.experience.store import ExperienceStore
from engine.workflow.types import ExecutionContext, WorkflowPlan


class SkillEngine:
    """技能引擎 — 评估、创建、建议可复用的 Skill。"""

    def __init__(self, store: ExperienceStore):
        self._store = store

    def evaluate(self, plan: WorkflowPlan, ctx: ExecutionContext) -> dict | None:
        """评估执行结果是否值得创建为可复用 Skill。"""
        completed = [s for s in plan.steps if s.status and s.status.name == "COMPLETED"]
        if len(completed) < 2:
            return None

        # 检查是否已有相似 Skill
        existing = self._store.get_all_skills()
        step_signature = self._steps_signature(completed)
        for sk in existing:
            if sk.get("step_signature") == step_signature:
                sk["success_count"] = sk.get("success_count", 1) + 1
                self._store.save_skill(sk)
                return None

        # 创建新 Skill
        name = self._generate_name(plan, completed)
        skill = self.create_skill(
            name=name,
            description=f"从工作流 {plan.workflow_id} 自动提取：{plan.goal[:120]}",
            steps=[
                {
                    "type": s.type.value if s.type else "claude_skill",
                    "skill_name": s.skill_name or "",
                    "prompt_template": s.prompt_template,
                    "params": s.params,
                }
                for s in completed
            ],
        )
        skill["step_signature"] = step_signature
        return skill

    def create_skill(self, name: str, description: str, steps: list[dict]) -> dict:
        """从步骤列表创建可复用 Skill。"""
        step_signature = "|".join(
            f'{s.get("type", "")}:{s.get("skill_name", "")}'
            for s in steps
        )
        return {
            "name": name,
            "description": description[:200],
            "step_signature": step_signature,
            "created_at": __import__("datetime").datetime.now().isoformat(),
            "steps": steps,
            "success_count": 1,
            "avg_duration": 0,
            "trigger_keywords": self._extract_keywords(name),
        }

    def suggest_skills(self, user_input: str) -> list[dict]:
        """根据用户输入建议可复用的 Skill。"""
        skills = self._store.get_all_skills()
        if not skills:
            return []

        input_lower = user_input.lower()
        scored: list[tuple[int, dict]] = []
        for sk in skills:
            score = 0
            for kw in sk.get("trigger_keywords", []):
                if kw.lower() in input_lower:
                    score += 10
            text = f'{sk.get("name", "")} {sk.get("description", "")}'.lower()
            for word in input_lower.split():
                if len(word) > 2 and word in text:
                    score += 2
            if score > 0:
                scored.append((score, sk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:3]]

    def format_prompt(self, user_input: str, max_skills: int = 3) -> str:
        """将相关 Skill 格式化为 LLM prompt。"""
        matched = self.suggest_skills(user_input)[:max_skills]
        if not matched:
            return ""

        lines = ["以下是从历史成功执行中提取的可复用工作流模板："]
        for sk in matched:
            desc = sk.get("description", "")[:100]
            count = sk.get("success_count", 1)
            steps_desc = " → ".join(
                s.get("skill_name", s.get("type", "?"))[:20]
                for s in sk.get("steps", [])
            )
            lines.append(f"- `{sk['name']}` ({count}次成功): {desc}")
            lines.append(f"  步骤: {steps_desc}")
        return "\n".join(lines)

    def prune_skills(self, max_skills: int = 50) -> int:
        """裁减 Skill 到容量限制。"""
        return self._store.prune_skills(max_skills)

    @staticmethod
    def _steps_signature(steps: list) -> str:
        return "|".join(
            f"{s.type.value if s.type else ''}:{s.skill_name or ''}"
            for s in steps
        )

    @staticmethod
    def _generate_name(plan: WorkflowPlan, steps: list) -> str:
        """从计划生成 Skill 名称。"""
        goal = plan.goal or ""
        words = goal.split()[:3]
        name = "-".join(words).lower() if words else "auto-skill"
        name = "".join(c for c in name if c.isalnum() or c in "-_")
        return name[:40] or f"auto-skill-{plan.workflow_id[-6:]}"

    @staticmethod
    def _extract_keywords(name: str) -> list[str]:
        """从名称中提取触发关键词。"""
        return name.replace("-", " ").split()
