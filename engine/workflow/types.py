from __future__ import annotations
"""Dynamic Workflow — 数据类型定义。"""


import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepType(str, Enum):
    """工作流步骤类型"""
    CLAUDE_SKILL = "claude_skill"       # 调用 Claude Code Skill
    LLM_TOOL = "llm_tool"               # 调用 LangChain tool
    VERIFY = "verify"                   # 验证步骤
    CONDITION = "condition"             # 条件分支
    LOOP = "loop"                       # 循环
    PARALLEL = "parallel"               # 并行容器（包含多个子步骤）
    CUSTOM = "custom"                   # 自定义
    MCP_TOOL = "mcp_tool"               # 调用 MCP 外部 Agent 工具


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class WorkflowStep:
    """工作流中的单个步骤"""
    id: str
    type: StepType
    name: str
    description: str = ""
    skill_name: str | None = None       # 对应 Claude Skill 或 tool 名称
    prompt_template: str = ""           # 含 {ctx.key} 占位符，执行前渲染
    depends_on: list[str] = field(default_factory=list)  # 依赖的 step id 列表
    params: dict[str, Any] = field(default_factory=dict)
    max_retries: int = 2
    timeout: int = 600                  # 秒
    condition_expression: str | None = None  # CONDITION 步骤的判断表达式
    parallel_steps: list[WorkflowStep] | None = None  # PARALLEL 类型的子步骤
    loop_max_iterations: int = 5
    loop_condition: str | None = None   # LOOP 步骤的终止条件

    # 运行时状态
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: str | None = None
    attempts: int = 0
    started_at: float | None = None
    completed_at: float | None = None


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class WorkflowPlan:
    """LLM 生成的完整工作流计划"""
    workflow_id: str = field(default_factory=lambda: f"wf-{uuid.uuid4().hex[:12]}")
    intent: str = ""                     # 原始用户输入
    goal: str = ""                       # LLM 理解的执行目标
    steps: list[WorkflowStep] = field(default_factory=list)
    max_concurrency: int = 3
    verify_enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_step(self, step_id: str) -> WorkflowStep | None:
        for s in self.steps:
            if s.id == step_id:
                return s
        return None

    def get_ready_steps(self, completed: set[str]) -> list[WorkflowStep]:
        """获取所有依赖已满足的可执行步骤"""
        return [
            s for s in self.steps
            if s.status == StepStatus.PENDING
            and all(dep in completed for dep in s.depends_on)
        ]

    def is_complete(self) -> bool:
        return all(s.status == StepStatus.COMPLETED for s in self.steps)

    def has_failed(self) -> bool:
        return any(s.status == StepStatus.FAILED for s in self.steps)


@dataclass
class ExecutionContext:
    """步骤间共享的执行上下文"""
    workflow_id: str
    variables: dict[str, Any] = field(default_factory=dict)       # step_id.output → value
    step_results: dict[str, Any] = field(default_factory=dict)    # step_id → result
    errors: dict[str, str] = field(default_factory=dict)          # step_id → error
    metadata: dict[str, Any] = field(default_factory=dict)

    def set_step_result(self, step_id: str, result: Any) -> None:
        self.step_results[step_id] = result
        self.variables[step_id] = result
        self.variables[f"{step_id}.output"] = result

    def get(self, key: str, default: Any = None) -> Any:
        if key.startswith("ctx."):
            key = key[4:]
        if key in self.variables:
            return self.variables[key]
        if key in self.step_results:
            return self.step_results[key]
        return default

    def render_prompt(self, template: str) -> str:
        """渲染 prompt_template 中的 {ctx.key} 占位符"""
        import re
        def _replace(m: re.Match) -> str:
            var_name = m.group(1)
            val = self.get(var_name, "")
            return str(val) if val is not None else m.group(0)
        return re.sub(r"\{ctx\.(\w+(?:\.\w+)*)\}", _replace, template)
