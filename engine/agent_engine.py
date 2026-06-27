from __future__ import annotations
"""LangChain Agent 构建。"""


import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic

from server.config import settings
from engine.llm_factory import build_llm
from engine.registry import get_all_tools, get_tool_names
from commands.registry import CommandRegistry, TriggerMode

# 全局共享的内存检查点（进程内共享，重启丢失）
_memory_saver = MemorySaver()


def _load_skills_registry() -> dict[str, str]:
    """扫描 skills/ 目录下的 SKILL.md，提取 frontmatter 中的 name 与 description。"""
    skills_dir = Path(__file__).resolve().parents[1] / "skills"
    registry: dict[str, str] = {}
    if not skills_dir.is_dir():
        return registry
    for skill_md in skills_dir.rglob("SKILL.md"):
        text = skill_md.read_text(encoding="utf-8")
        if not text.startswith("---"):
            continue
        end = text.find("---", 3)
        if end == -1:
            continue
        front = text[3:end]
        name: str | None = None
        desc_lines: list[str] = []
        in_desc = False
        for line in front.splitlines():
            stripped = line.strip()
            if stripped.startswith("name:"):
                name = stripped.split(":", 1)[1].strip()
                in_desc = False
            elif stripped.startswith("description:"):
                raw = stripped.split(":", 1)[1].strip()
                desc_lines = [raw]
                in_desc = True
            elif in_desc and (stripped.startswith("-") or stripped.startswith(">")):
                desc_lines.append(stripped.lstrip("- >").strip())
            elif in_desc and stripped:
                in_desc = False
        if name:
            registry[name] = " ".join(desc_lines) or ""
    return registry


def _build_skill_recommendation_prompt() -> str:
    """基于 skills/ 目录生成 system prompt 中用于推荐 skill 的段落。"""
    registry = _load_skills_registry()
    if not registry:
        return ""
    lines = [
        "\n## 可用 Skill 速查与推荐规则",
        "你应主动判断用户需求是否匹配以下某个 Skill。若匹配，**直接回复一条以 `/claude` 开头的可复制提示词**，让用户复制后输入即可触发对应 Skill 完成需求。",
    ]
    for name, desc in sorted(registry.items()):
        lines.append(f'- **{name}**：{desc} → 推荐回复格式：`/claude {name} [具体任务描述]`')
    lines.append(
        "\n判断优先级：1) 若用户提到具体 Skill 名称，直接给出对应 /claude 命令；"
        "2) 若需求明显属于验证/适配/优化/量化/迁移等范畴，给出最匹配的 Skill；"
        "3) 若不确定，可提示用户 `可用 /claude [skill名] 来触发，或用 list_claude_skills 查看列表`。"
    )
    return "\n".join(lines)


def _build_command_prompt(cmd_registry: CommandRegistry) -> str:
    """根据命令注册表生成 system prompt 中的命令说明段落。"""
    return cmd_registry.build_system_prompt_addon()


def build_agent(
    *,
    streaming: bool = False,
    cmd_registry: CommandRegistry | None = None,
    user_system_prompt: str = "",
):
    """构建 LangChain ReAct Agent。"""
    llm: ChatAnthropic = build_llm(streaming=streaming)
    tools = get_all_tools()
    tool_names = ", ".join(get_tool_names())
    skill_prompt = _build_skill_recommendation_prompt()
    command_prompt = _build_command_prompt(cmd_registry) if cmd_registry else ""

    system_prompt = (
        f"你是 **Model Agent**（模型全流程助手），已绑定工具。当前可用工具名称为：{tool_names}。\n"
        "search / verify / quantify / adapt / optimizer 类需求须通过对应工具完成，勿编造执行结果。\n"
        "当用户询问某模型能否在昇腾设备上运行、或查询特定框架/硬件组合时，**必须**先调用 check_model_ascend_support 工具；"
        "该工具支持多字段过滤：model_name（模糊匹配）、adapter_framework、adapter_hardware、training_or_inference、model_vendor、adapter_status，可组合使用；"
        "根据工具返回结果回答，勿编造结论；回复中**必须**原样列出工具返回的每条「链接」字段，不得省略。\n"
        "当用户提到 doc-agent、一键小白文档等需求时，你必须通过工具调用完成；\n"
        "回答可活泼简短，但勿编造工具输出。"
        f"{skill_prompt}"
        f"{command_prompt}"
    )
    if user_system_prompt:
        system_prompt += "\n\n" + user_system_prompt

    logger.info("Agent built: model=%s, tools=%d, streaming=%s", llm.model, len(tools), streaming)
    return create_react_agent(llm, tools, prompt=system_prompt, checkpointer=_memory_saver)
