"""LLM 驱动的动态工作流规划器。

替代原有的关键词匹配 IntentResolver，使用 LLM 根据用户输入
和可用技能/工具列表动态生成 WorkflowPlan。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

from server.config import settings
from engine.llm_factory import build_workflow_llm, extract_text_content, handle_api_error, retry_llm_call
from engine.skills.registry import get_skill_registry
from engine.workflow.types import (
    StepType,
    WorkflowPlan,
    WorkflowStep,
)


def _load_available_tools() -> list[dict[str, str]]:
    """从 registry 加载可用的 tool 列表。"""
    try:
        from engine.registry import get_all_tools
        tools = get_all_tools()
        return [
            {"name": t.name, "description": getattr(t, "description", "")}
            for t in tools
        ]
    except Exception:
        return []


def _format_mcp_tools() -> str:
    """从 MCPToolManager 获取可用工具列表。"""
    try:
        from engine.workflow.mcp_integration import get_mcp_manager
        manager = get_mcp_manager()
        if not manager.is_initialized:
            return "暂无可用 MCP 工具（管理器未初始化）。"
        tools = manager.get_all_tools()
        if not tools:
            return "暂无可用 MCP 工具。"
        lines = []
        for t in tools:
            server = t.get("server", "?")
            desc = t.get("description", "")[:120]
            lines.append(f"- `{t['name']}` ({server})：{desc}")
        return "\n".join(lines)
    except ImportError:
        return "暂无可用 MCP 工具（模块未加载）。"
    except Exception as e:
        return f"MCP 工具加载失败: {e}"


# ── LLM Prompt 模板 ─────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """你是一位工作流规划专家。你的任务是将用户的需求分解为一个可执行的 DAG 工作流计划。

## 核心规则（必须遵守）

### 1. Skill 选择优先级
对于模型适配/验证场景，必须遵循以下优先级选择 Skill：
- **第一优先**: `verify-agent` — 验证模型在昇腾 NPU 上的适配状态
- **第二优先**: `adapt-agent` — 执行模型适配（仅当验证发现问题时才触发）
- **第三优先**: `optimizer-agent` — **仅在用户明确要求性能调优/优化时使用**
- **禁止**: 未经用户要求不得自动添加优化步骤

### 2. ai4s 系列 Skill 使用限制
- `ai4s-main`、`ai4s-basic`、`ai4s-perf-tuning`、`ai4s-precision-alignment`、`ai4s-profiling`
- **仅当用户明确提及 "AI4S"、"AI for Science"、"科学AI"、"分子动力学"、"气象" 等科学计算场景时才能使用**
- 对于通用 LLM 模型（Qwen、DeepSeek、LLaMA 等），**严禁**使用 ai4s 系列 skill

### 3. 通用模型 / 小模型适配
- 对于 CV 模型、NLP 小模型、非 LLM 的通用模型，优先使用 model-agent 系列 skills
- model-agent 提供了面向多领域的通用适配能力

### 4. 工作流长度控制
- 如果用户仅要求"适配"，不要添加验证、优化、部署等额外步骤
- 仅根据用户明确提出的需求生成对应步骤
- 默认情况下一个简单的模型适配只需要 1-3 个步骤

## 可用 Skill（Claude Code Skills）
每个 Skill 是一个 AI 助手，可以执行复杂的多步骤任务。

{skills_section}

## 可用 Tool（LangChain 工具）
{tools_section}

## 可用 MCP Tool（外部 Agent）
MCP 工具通过 MCP 协议连接到外部 Agent 服务，用于特定场景。

{mcp_tools_section}

## 历史经验参考

{experience_section}

## 可用 Skill 模板

{skill_section}

## 工作流模式说明

你可以组合使用以下步骤类型：

1. **claude_skill** — 调用一个 Claude Code Skill 执行复杂任务。适合验证/适配等需要多步操作的场景。
2. **llm_tool** — 调用一个 LangChain 工具执行单一操作。适合查询/搜索等简单场景。
3. **mcp_tool** — 调用外部 MCP Agent 工具。当任务需要特定外部能力（如算子开发、Profiling 分析）时使用。需在 params 中指定 mcp_server。
4. **verify** — 验证前一个步骤的输出是否正确。将前一步的结果发给另一个 Skill 或 LLM 做交叉验证。
5. **condition** — 条件分支。根据前一步结果决定后续步骤。
6. **parallel** — 并行执行多个子步骤（用 parallel_steps 定义）。适合无依赖的独立任务。
7. **loop** — 循环执行一个步骤直到满足条件。

## mcp_tool 使用指引

当任务涉及以下场景时，优先使用对应的 MCP 工具：
- **缺算子 / 算子开发** → 使用 CANNBot 的 ascendc-op-develop、pypto-op-develop 等
- **性能调优 / Profiling** → 使用 CANNBot 的 ops-profiling 或 ms-agent 的 profiling_analyze（仅当用户明确要求）
- **模型推理优化** → 使用 CANNBot 的 model-infer-optimize 系列（仅当用户明确要求）
- **通用分析查询** → 使用 ms-agent 的通用 Agent 能力

mcp_tool 步骤示例：
```json
{{
  "id": "step_2",
  "type": "mcp_tool",
  "name": "算子开发",
  "skill_name": "ascendc-op-develop",
  "prompt_template": "为 Qwen3-8B 开发 ...",
  "depends_on": ["step_1"],
  "params": {{"mcp_server": "cannbot"}}
}}
```

## 执行规则

- 如果步骤 A 需要步骤 B 的输出，将 A 的 depends_on 设为 [B.id]
- 无依赖的步骤可以并行执行，放在 parallel 类型中
- 验证步骤应紧跟在被验证的步骤之后
- prompt_template 中可以用 {{ctx.step_id}} 引用前一步的输出
- mcp_tool 步骤必须设置 params.mcp_server 指定目标服务器

## 输出格式

你必须返回严格的 JSON（不要 markdown 代码块），格式如下：
{{
  "goal": "简短描述理解到的用户目标",
  "steps": [
    {{
      "id": "step_1",
      "type": "claude_skill",
      "name": "步骤名称",
      "description": "步骤说明",
      "skill_name": "对应的 skill 或 tool 名称",
      "prompt_template": "执行提示词",
      "depends_on": [],
      "params": {{}}
    }}
  ],
  "max_concurrency": 3,
  "verify_enabled": true
}}

## 要求
1. 步骤数量控制在 1-4 个（除非用户明确要求更多）
2. 每个步骤必须有明确的执行目标
3. 模型适配优先用 verify-agent 验证、adapt-agent 适配
4. 简单查询用 llm_tool 类型
5. 需要外部 Agent 能力时用 mcp_tool 类型
6. 如果有多个独立任务，用 parallel 并行执行
7. **不要自动添加优化步骤** — optimizer-agent 仅当用户明确要求性能调优时使用
8. **不要对通用 LLM 模型使用 ai4s 系列 skill**
"""


def _format_skills() -> str:
    reg = get_skill_registry()
    return reg.format_prompt()


def _format_tools() -> str:
    tools = _load_available_tools()
    if not tools:
        return "无可用工具。"
    lines = []
    for t in tools:
        desc = t.get("description", "")[:120]
        lines.append(f"- `{t['name']}`：{desc}")
    return "\n".join(lines)


# ── 规划器 ──────────────────────────────────────────────────────────

class DynamicPlanner:
    """LLM 驱动的工作流规划器。"""

    def __init__(self, llm=None, experience_enabled: bool = True):
        self._llm = llm or build_workflow_llm()
        self._skills_registry = get_skill_registry()
        self._experience_enabled = experience_enabled and settings.hermes_enabled
        self._memory_engine = None
        self._skill_engine = None
        if self._experience_enabled:
            try:
                from engine.experience.memory import MemoryEngine
                from engine.experience.skill import SkillEngine
                from engine.experience.store import ExperienceStore
                store = ExperienceStore()
                self._memory_engine = MemoryEngine(store)
                self._skill_engine = SkillEngine(store)
            except Exception:
                self._experience_enabled = False

    def _call_llm_with_retry(self, messages: list[dict]) -> dict | None:
        """调用 LLM 并重试，返回解析后的 JSON dict。"""
        try:
            response = retry_llm_call(max_retries=2)(self._llm.invoke)(messages)
            raw = extract_text_content(response.content) if hasattr(response, "content") else str(response)
            return self._parse_response(raw)
        except Exception as e:
            handle_api_error(e)
            return None

    async def _call_llm_with_retry_async(self, messages: list[dict]) -> dict | None:
        """异步版本，不阻塞事件循环。"""
        import asyncio
        try:
            response = await asyncio.to_thread(
                retry_llm_call(max_retries=2)(self._llm.invoke), messages
            )
            raw = response.content if hasattr(response, "content") else str(response)
            return self._parse_response(raw)
        except Exception as e:
            handle_api_error(e)
            return None

    def plan(self, user_input: str, context: dict | None = None) -> WorkflowPlan:
        """根据用户输入生成工作流计划（同步版本，可能在事件循环中阻塞）。"""
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            skills_section=_format_skills(),
            tools_section=_format_tools(),
            mcp_tools_section=_format_mcp_tools(),
            experience_section=self._format_experience(user_input),
            skill_section=self._format_reusable_skills(user_input),
        )

        user_prompt = (
            f"用户需求：{user_input}\n\n"
            f"额外上下文：{json.dumps(context or {}, ensure_ascii=False)}\n\n"
            "请分析用户需求并生成一个可执行的工作流计划。"
        )

        plan_dict = self._call_llm_with_retry([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

        if not plan_dict or "steps" not in plan_dict:
            return self._fallback_plan(user_input)

        plan = self._build_plan(plan_dict, user_input)

        # ---- Post-processing: enforce skill rules ----
        plan = self._enforce_skill_rules(plan, user_input)
        # -----------------------------------------------

        self._validate_plan(plan)
        logger.info(
            "Workflow plan created: intent=%.60s, steps=%d, goal=%.60s",
            user_input, len(plan.steps), plan.goal,
        )
        return plan

    async def aplan(self, user_input: str, context: dict | None = None) -> WorkflowPlan:
        """根据用户输入生成工作流计划（异步版本，不阻塞事件循环）。"""
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            skills_section=_format_skills(),
            tools_section=_format_tools(),
            mcp_tools_section=_format_mcp_tools(),
            experience_section=self._format_experience(user_input),
            skill_section=self._format_reusable_skills(user_input),
        )

        user_prompt = (
            f"用户需求：{user_input}\n\n"
            f"额外上下文：{json.dumps(context or {}, ensure_ascii=False)}\n\n"
            "请分析用户需求并生成一个可执行的工作流计划。"
        )

        plan_dict = await self._call_llm_with_retry_async([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

        if not plan_dict or "steps" not in plan_dict:
            return self._fallback_plan(user_input)

        plan = self._build_plan(plan_dict, user_input)

        # ---- Post-processing: enforce skill rules ----
        plan = self._enforce_skill_rules(plan, user_input)
        # -----------------------------------------------

        self._validate_plan(plan)
        logger.info(
            "Workflow plan created: intent=%.60s, steps=%d, goal=%.60s",
            user_input, len(plan.steps), plan.goal,
        )
        return plan

    def replan(
        self,
        original_input: str,
        failed_step: WorkflowStep,
        error: str,
        context: dict | None = None,
        tier2_expanded: dict[str, str] | None = None,
    ) -> WorkflowPlan:
        """在步骤失败时重新规划剩余步骤。

        Args:
            tier2_expanded: 原始 plan 中已引用的 Tier 2 技能展开内容
                           {name: expanded_text}，传递给 prompt 供 LLM 参考。
        """
        # 如有已展开的 Tier 2 技能，附加到 skills_section
        skills_section = _format_skills()
        if tier2_expanded:
            extra = []
            for name, text in sorted(tier2_expanded.items()):
                short = text[:150].replace("\n", " ")
                extra.append(f"- `{name}`: {short}")
            skills_section += "\n\n### Tier 2 已展开技能（上一步引用）\n" + "\n".join(extra)

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            skills_section=skills_section,
            tools_section=_format_tools(),
            mcp_tools_section=_format_mcp_tools(),
            experience_section=self._format_experience(original_input),
            skill_section=self._format_reusable_skills(original_input),
        )
        user_prompt = f"""\
错误：{error}

额外上下文：{json.dumps(context or {}, ensure_ascii=False)}

请重新规划后续步骤来处理这个失败。可以重试、使用替代方案，或跳过。\
"""

        plan_dict = self._call_llm_with_retry([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

        if not plan_dict or "steps" not in plan_dict:
            return self._fallback_plan(original_input)

        plan = self._build_plan(plan_dict, original_input)
        self._validate_plan(plan)
        return plan

    def _parse_response(self, raw: str) -> dict | None:
        """从 LLM 响应中解析 JSON。"""
        # 尝试直接 parse
        text = raw.strip()
        # 去除可能的 markdown 代码块包裹
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试提取 JSON 对象
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _build_plan(self, plan_dict: dict, user_input: str) -> WorkflowPlan:
        """从 LLM 返回的 dict 构建 WorkflowPlan。"""
        steps = []
        for s in plan_dict.get("steps", []):
            step_type_str = s.get("type", "claude_skill")
            try:
                step_type = StepType(step_type_str)
            except ValueError:
                step_type = StepType.CLAUDE_SKILL

            parallel_sub_steps = None
            if step_type == StepType.PARALLEL and "parallel_steps" in s:
                parallel_sub_steps = [
                    WorkflowStep(
                        id=ps.get("id", f"ps-{i}"),
                        type=StepType(ps.get("type", "claude_skill")),
                        name=ps.get("name", ""),
                        description=ps.get("description", ""),
                        skill_name=ps.get("skill_name"),
                        prompt_template=ps.get("prompt_template", ""),
                        params=ps.get("params", {}),
                    )
                    for i, ps in enumerate(s["parallel_steps"])
                ]

            step = WorkflowStep(
                id=s.get("id", f"step_{len(steps) + 1}"),
                type=step_type,
                name=s.get("name", ""),
                description=s.get("description", ""),
                skill_name=s.get("skill_name"),
                prompt_template=s.get("prompt_template", ""),
                depends_on=s.get("depends_on", []),
                params=s.get("params", {}),
                max_retries=s.get("max_retries", 2),
                timeout=s.get("timeout", 600),
                condition_expression=s.get("condition_expression"),
                parallel_steps=parallel_sub_steps,
                loop_max_iterations=s.get("loop_max_iterations", 5),
                loop_condition=s.get("loop_condition"),
            )
            steps.append(step)

        return WorkflowPlan(
            intent=user_input,
            goal=plan_dict.get("goal", ""),
            steps=steps,
            max_concurrency=plan_dict.get("max_concurrency", 3),
            verify_enabled=plan_dict.get("verify_enabled", True),
        )

    def _enforce_skill_rules(self, plan: WorkflowPlan, user_input: str) -> WorkflowPlan:
        """Post-processing: enforce skill priority and trigger rules."""
        lower = user_input.lower()

        # Check if user explicitly wants AI4S
        ai4s_keywords = ["ai4s", "ai for science", "科学ai", "科学计算",
                         "分子动力学", "气象", "气候", "蛋白质",
                         "molecular", "weather", "climate", "protein"]
        wants_ai4s = any(k in lower for k in ai4s_keywords)

        # Check if user explicitly wants optimization
        opt_keywords = ["优化", "optimize", "调优", "性能", "performance",
                        "加速", "speedup", "profile", "profiling", "基准"]
        wants_optimization = any(k in lower for k in opt_keywords)

        filtered_steps = []
        for step in plan.steps:
            skill = (step.skill_name or "").lower()

            # Rule 1: Remove ai4s skills if not explicitly requested
            if skill.startswith("ai4s") and not wants_ai4s:
                logger.info(f"[Planner] Removing ai4s step: {step.id} ({step.name}) - user didn't request AI4S")
                continue

            # Rule 2: Remove optimizer steps if not explicitly requested
            if ("optimizer" in skill or "optim" in skill) and not wants_optimization:
                logger.info(f"[Planner] Removing optimizer step: {step.id} ({step.name}) - user didn't request optimization")
                continue

            # Rule 3: For simple adaptation without optimization, keep only verify + adapt
            filtered_steps.append(step)

        if len(filtered_steps) != len(plan.steps):
            plan.steps = filtered_steps
            # Fix dependency references
            valid_ids = {s.id for s in plan.steps}
            for s in plan.steps:
                s.depends_on = [d for d in s.depends_on if d in valid_ids]

        return plan

    def _validate_plan(self, plan: WorkflowPlan) -> None:
        """校验计划的合法性。"""
        step_ids = {s.id for s in plan.steps}

        for step in plan.steps:
            # 检查依赖是否存在
            for dep in step.depends_on:
                if dep not in step_ids:
                    # 自动修正：移除不存在的依赖
                    step.depends_on = [d for d in step.depends_on if d in step_ids]

            # 检查 skill 是否存在
            if step.type == StepType.CLAUDE_SKILL and step.skill_name:
                if not self._skills_registry.get(step.skill_name):
                    # skill 不在注册表中也容忍（可能由外部处理）
                    pass

        # 检查是否有环
        if self._has_cycle(plan):
            # 简单处理：将依赖全部清空，改为串行
            for i, step in enumerate(plan.steps):
                if i > 0:
                    step.depends_on = [plan.steps[i - 1].id]
                else:
                    step.depends_on = []

    def _has_cycle(self, plan: WorkflowPlan) -> bool:
        """检测 DAG 是否有环（DFS）。"""
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def _dfs(step_id: str) -> bool:
            if step_id in rec_stack:
                return True
            if step_id in visited:
                return False
            visited.add(step_id)
            rec_stack.add(step_id)
            step = plan.get_step(step_id)
            if step:
                for dep in step.depends_on:
                    if _dfs(dep):
                        return True
            rec_stack.discard(step_id)
            return False

        for step in plan.steps:
            if _dfs(step.id):
                return True
        return False

    def _fallback_plan(self, user_input: str) -> WorkflowPlan:
        """LLM 规划失败时的回退方案：直接生成一个调用 Claude 的简单计划。"""
        return WorkflowPlan(
            intent=user_input,
            goal=f"处理用户请求：{user_input}",
            steps=[
                WorkflowStep(
                    id="step_1",
                    type=StepType.CLAUDE_SKILL,
                    name="通用处理",
                    description="直接通过 Claude 处理用户需求",
                    skill_name="",
                    prompt_template=user_input,
                    depends_on=[],
                )
            ],
            max_concurrency=1,
            verify_enabled=False,
        )

    def _format_experience(self, user_input: str) -> str:
        """注入相关历史经验到 system prompt。"""
        if not self._experience_enabled or not self._memory_engine:
            return "暂无历史经验。"
        try:
            return self._memory_engine.format_prompt(user_input)
        except Exception:
            return "暂无历史经验。"

    def _format_reusable_skills(self, user_input: str) -> str:
        """注入可复用的 Skill。"""
        if not self._experience_enabled or not self._skill_engine:
            return ""
        try:
            return self._skill_engine.format_prompt(user_input)
        except Exception:
            return ""


# ── 兼容层（保持 IntentResolver 风格的接口） ───────────────────────────

class IntentResolver:
    """兼容旧接口 IntentResolver — 内部使用 DynamicPlanner。"""

    def __init__(self):
        self._planner = DynamicPlanner()

    def resolve(self, user_input: str) -> IntentMatchAdapter:
        plan = self._planner.plan(user_input)
        return IntentMatchAdapter(plan)

    @staticmethod
    def _resolve_model_path(short_name: str) -> str:
        return short_name

    def get_skill_chain(self, intent_match) -> list[dict]:
        return []

    def format_plan(self, intent_match) -> str:
        return ""


class IntentMatchAdapter:
    """兼容旧 IntentMatch 接口的适配器。"""

    def __init__(self, plan: WorkflowPlan):
        self.plan = plan
        self.confidence = 0.85 if len(plan.steps) > 0 else 0.3
        self.model_name = ""
        self.chain = [s.skill_name or "" for s in plan.steps if s.skill_name]
        self.reasoning = plan.goal
