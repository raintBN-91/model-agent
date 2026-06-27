"""Dynamic Workflow Engine — LLM 驱动的动态工作流引擎。

替代原有的关键词匹配 IntentResolver，提供：
- LLM-driven 工作流规划
- DAG 执行（并行 + 串行混合）
- 交叉验证
- 状态持久化与恢复
- Brainstorming 多轮意图澄清
"""

from engine.workflow.types import (
    ExecutionContext,
    StepStatus,
    StepType,
    WorkflowPlan,
    WorkflowStep,
)

__all__ = [
    "DynamicPlanner",
    "IntentResolver",
    "WorkflowExecutor",
    "WorkflowPlan",
    "WorkflowStep",
    "ExecutionContext",
    "StepType",
    "StepStatus",
    "Checkpointer",
    "BrainstormingSession",
    "BrainstormState",
    "BrainstormQuestion",
    "BrainstormProposal",
    "BrainstormDesignDoc",
    "assess_ambiguity",
    "MCPToolManager",
    "MCPConnectionConfig",
    "get_mcp_manager",
    "init_mcp_manager",
    "shutdown_mcp_manager",
]


_PKG = "engine.workflow"
_IMPORT_CACHE: dict[str, object] = {}


def _lazy_import(name: str):
    """延迟导入避免触发深层依赖链（带缓存）。"""
    if name not in _IMPORT_CACHE:
        import importlib
        _IMPORT_CACHE[name] = importlib.import_module(f"{_PKG}.{name}")
    return _IMPORT_CACHE[name]


def DynamicPlanner(*args, **kwargs):
    return _lazy_import("planner").DynamicPlanner(*args, **kwargs)


def IntentResolver(*args, **kwargs):
    return _lazy_import("planner").IntentResolver(*args, **kwargs)


def WorkflowExecutor(*args, **kwargs):
    return _lazy_import("executor").WorkflowExecutor(*args, **kwargs)


def Checkpointer(*args, **kwargs):
    return _lazy_import("checkpointer").Checkpointer(*args, **kwargs)


def BrainstormingSession(*args, **kwargs):
    return _lazy_import("brainstorming").BrainstormingSession(*args, **kwargs)


def BrainstormState(*args, **kwargs):
    return _lazy_import("brainstorming").BrainstormState(*args, **kwargs)


def BrainstormQuestion(*args, **kwargs):
    return _lazy_import("brainstorming").BrainstormQuestion(*args, **kwargs)


def BrainstormProposal(*args, **kwargs):
    return _lazy_import("brainstorming").BrainstormProposal(*args, **kwargs)


def BrainstormDesignDoc(*args, **kwargs):
    return _lazy_import("brainstorming").BrainstormDesignDoc(*args, **kwargs)


def assess_ambiguity(*args, **kwargs):
    return _lazy_import("brainstorming").assess_ambiguity(*args, **kwargs)


def MCPToolManager(*args, **kwargs):
    return _lazy_import("mcp_integration").MCPToolManager(*args, **kwargs)


def MCPConnectionConfig(*args, **kwargs):
    return _lazy_import("mcp_integration").MCPConnectionConfig(*args, **kwargs)


def get_mcp_manager(*args, **kwargs):
    return _lazy_import("mcp_integration").get_mcp_manager(*args, **kwargs)


def init_mcp_manager(*args, **kwargs):
    return _lazy_import("mcp_integration").init_mcp_manager(*args, **kwargs)


def shutdown_mcp_manager(*args, **kwargs):
    return _lazy_import("mcp_integration").shutdown_mcp_manager(*args, **kwargs)
