"""Workflow Planner — 意图解析与工作流计划生成器。

从自然语言 Prompt 中提取意图，输出结构化工作流计划，并映射到可执行的 Skill。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class WorkflowType(Enum):
    MODEL_VERIFY = "model_verify"
    MODEL_LIST = "model_list"
    MODEL_BENCHMARK = "model_benchmark"
    MODEL_DEPLOY = "model_deploy"
    FULL_PIPELINE = "full_pipeline"
    CUSTOM = "custom"


@dataclass
class IntentMatch:
    workflow_type: WorkflowType
    confidence: float
    model_name: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    chain: list[str] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_type": self.workflow_type.value,
            "confidence": self.confidence,
            "model_name": self.model_name,
            "params": self.params,
            "chain": self.chain,
            "reasoning": self.reasoning,
        }


class IntentResolver:
    MODEL_KEYWORDS = [
        "qwen", "llama", "deepseek", "chatglm", "gemma", "mistral",
        "phi", "baichuan", "internlm", "yi", "mixtral", "moe",
    ]

    MODEL_NAME_MAP = {
        "qwen3": "Qwen/Qwen3-1.7B",
        "qwen3.5": "Qwen/Qwen3.5-0.8B",
        "qwen3-0": "Qwen/Qwen3.5-0.8B",
        "qwen3-0.8b": "Qwen/Qwen3.5-0.8B",
        "qwen2.5": "Qwen/Qwen2.5-0.5B",
        "qwen2": "Qwen/Qwen2-0.5B",
        "qwen2-0": "Qwen/Qwen2-0.5B",
        "qwen": "Qwen/Qwen2.5-0.5B",
        "llama3": "meta-llama/Meta-Llama-3-8B",
        "llama2": "meta-llama/Llama-2-7b-chat-hf",
        "llama": "meta-llama/Llama-2-7b-chat-hf",
        "deepseek": "deepseek-ai/DeepSeek-Coder-1.3B",
        "deepseekv3": "deepseek-ai/DeepSeek-V3",
        "chatglm": "ZhipuAI/chatglm3-6b",
        "gemma": "google/gemma-2b",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "phi": "microsoft/phi-2",
        "yi": "01ai/Yi-1.5-6B",
    }

    ACTION_PATTERNS = {
        "verify": ["verify", "验证", "test", "check", "适配"],
        "benchmark": ["benchmark", "性能", "throughput", "latency", "test performance", "优化", "调优"],
        "list": ["list", "排行", "rank", "top", "热门", "popular", "leaderboard"],
        "deploy": ["deploy", "部署", "serve", "start", "上线"],
        "document": ["doc", "文档", "write", "生成文档", "documentation", "报告"],
        "query": ["query", "查询", "find", "search", "look up", "搜索"],
        "full": ["full", "complete", "端到端", "end to end", "端到端流程", "整个流程"],
    }

    SKILL_TRIGGERS = {
        "ascend-model-verification": ["verify", "验证", "test", "check", "适配", "完整验证", "full verify", "complete verification"],
        "doc-agent": ["doc", "文档", "documentation", "write doc", "生成文档", "report"],
        "vllm-ascend-model-adapter": ["adapter", "adaptation", "适配器", "model adapter", "迁移"],
        "vllm-ascend-performance-optimization": ["optimize", "优化", "performance tuning", "throughput优化", "latency优化", "性能调优"],
        "ascend-profiling": ["profiling", "性能分析", "profiler", "torch_npu profiler", "算子耗时"],
        "ascend-optimization": ["ascend optimization", "torch_npu", "CPU_AFFINITY", "TASK_QUEUE", "torch优化"],
        "ascend-affinity-operator": ["affinity", "亲和算子", "fusion operator", "融合算子"],
        "vector-triton-ascend-ops-optimizer": ["triton", "vector", "算子优化", "triton kernel", "Triton优化"],
        "quantify-agent": ["quantize", "quantization", "量化", "w8a8", "w4a16"],
        "deploy-agent": ["deploy", "部署", "serve", "start", "启动服务"],
        "ascend-history-to-skill": ["history", "historical", "历史", "记录"],
        "search-agent": ["search", "查询", "find model", "find", "search model", "搜索", "搜索模型", "找模型", "查找", "找一下"],
        "repo-reader": ["repo", "repository", "readme", "仓库", "读取"],
        "adapter-check-principle": ["adapter check", "adapter principle", "适配器检查", "适配原则"],
        "hardware-check-principle": ["hardware check", "hardware principle", "硬件检查", "硬件原则"],
        "model-series-vendor-detector": ["model series", "vendor detector", "模型系列", "厂商检测"],
        "ai4s-basic": ["ai4s", "ai for science", "科学AI"],
        "tf-to-pytorch": ["tf2pytorch", "tensorflow to pytorch", "tf迁移", "tensorflow转换"],
        "npu-adapter-reviewer": ["npu adapter", "adapter reviewer", "NPU适配", "适配审查"],
        "torch-npu-optimization": ["torch-npu", "torch优化", "npu optimization"],
        "npu-basic-migrate": ["npu migrate", "basic migrate", "基础迁移"],
        "esm2-npu": ["esm2", "esm", "protein", "蛋白质"],
        "Boltz2": ["boltz2", "boltz"],
        "GENERator": ["generator", "dna"],
        "boltzgen": ["boltzgen"],
        "diffsbdd": ["diffsbdd", "diffusion"],
        "oligoformer": ["oligoformer", "oligo"],
        "proteinbert": ["proteinbert", "protein"],
    }

    def __init__(self):
        self._skill_chain_templates: dict[WorkflowType, list[str]] = {
            WorkflowType.MODEL_VERIFY: ["ascend-model-verification"],
            WorkflowType.MODEL_LIST: ["search-agent"],
            WorkflowType.MODEL_BENCHMARK: ["vllm-ascend-performance-optimization"],
            WorkflowType.MODEL_DEPLOY: ["ascend-model-verification", "deploy-agent"],
            WorkflowType.FULL_PIPELINE: [
                "search-agent",
                "ascend-model-verification",
                "vllm-ascend-performance-optimization",
                "doc-agent",
            ],
            WorkflowType.CUSTOM: [],
        }

    # 定义多动作组合时的语义优先级（数值小的排前面）
    _ACTION_PRIORITY = {
        "query": 1,
        "list": 1,
        "verify": 2,
        "benchmark": 3,
        "deploy": 4,
        "document": 5,
        "full": 0,
    }

    def resolve(self, user_input: str) -> IntentMatch:
        user_lower = user_input.lower()

        if self._matches_full_pipeline(user_lower):
            return self._build_full_pipeline(user_input, user_lower)

        # 收集所有匹配的 action，而非匹配第一个就返回
        matched_actions = []
        for workflow_action, keywords in self.ACTION_PATTERNS.items():
            if any(kw in user_lower for kw in keywords):
                matched_actions.append(workflow_action)

        if matched_actions:
            if len(matched_actions) == 1:
                return self._build_match(matched_actions[0], user_input, user_lower)
            return self._build_multi_match(matched_actions, user_input, user_lower)

        return self._build_custom_match(user_input, user_lower)

    def _matches_full_pipeline(self, user_lower: str) -> bool:
        full_keywords = [
            "full pipeline", "端到端", "end to end", "整个流程",
            "complete process", "从模型查询到文档",
        ]
        return any(kw in user_lower for kw in full_keywords)

    def _build_full_pipeline(self, user_input: str, user_lower: str) -> IntentMatch:
        model_name = self._extract_model_name(user_lower)
        chain = ["awesome-llm-models", "small-model-verify", "model-benchmark", "doc-agent"]
        return IntentMatch(
            workflow_type=WorkflowType.FULL_PIPELINE,
            confidence=0.95,
            model_name=model_name,
            params={"model": model_name} if model_name else {},
            chain=chain,
            reasoning=f"Full pipeline detected{' for ' + model_name if model_name else ''}: query → verify → benchmark → document",
        )

    def _build_match(self, action: str, user_input: str, user_lower: str) -> IntentMatch:
        wf_map = {
            "verify": WorkflowType.MODEL_VERIFY,
            "benchmark": WorkflowType.MODEL_BENCHMARK,
            "list": WorkflowType.MODEL_LIST,
            "deploy": WorkflowType.MODEL_DEPLOY,
            "full": WorkflowType.FULL_PIPELINE,
            "query": WorkflowType.MODEL_LIST,
            "document": WorkflowType.MODEL_VERIFY,
        }
        workflow_type = wf_map.get(action, WorkflowType.CUSTOM)
        model_name = self._extract_model_name(user_lower)
        chain = self._skill_chain_templates.get(workflow_type, []).copy()

        if action == "document":
            chain.append("doc-agent")
        elif action == "verify" and any(
            kw in user_lower for kw in ["文档", "doc", "documentation", "write doc", "生成文档", "撰写"]
        ):
            chain.append("doc-agent")

        return IntentMatch(
            workflow_type=workflow_type,
            confidence=0.8,
            model_name=model_name,
            params={"model": model_name} if model_name else {},
            chain=chain,
            reasoning=f"Action '{action}' detected → chain: {' → '.join(chain) if chain else 'custom'}",
        )

    def _build_multi_match(self, actions: list[str], user_input: str, user_lower: str) -> IntentMatch:
        """当输入同时匹配多个 action 时，合并生成一条组合工作流链。"""
        wf_map = {
            "verify": WorkflowType.MODEL_VERIFY,
            "benchmark": WorkflowType.MODEL_BENCHMARK,
            "list": WorkflowType.MODEL_LIST,
            "deploy": WorkflowType.MODEL_DEPLOY,
            "full": WorkflowType.FULL_PIPELINE,
            "query": WorkflowType.MODEL_LIST,
            "document": WorkflowType.MODEL_VERIFY,
        }
        model_name = self._extract_model_name(user_lower)

        # 按语义优先级排序，确保合理的执行顺序（如先验证后优化）
        sorted_actions = sorted(
            actions,
            key=lambda a: self._ACTION_PRIORITY.get(a, 99),
        )

        combined_chain: list[str] = []
        for action in sorted_actions:
            wf_type = wf_map.get(action, WorkflowType.CUSTOM)
            chain = self._skill_chain_templates.get(wf_type, []).copy()

            if action == "document":
                chain.append("doc-agent")
            elif action == "verify" and any(
                kw in user_lower for kw in ["文档", "doc", "documentation", "write doc", "生成文档", "撰写"]
            ):
                chain.append("doc-agent")

            combined_chain.extend(chain)

        # 去重，同时保持顺序
        unique_chain = list(dict.fromkeys(combined_chain))

        return IntentMatch(
            workflow_type=WorkflowType.CUSTOM,
            confidence=0.85,
            model_name=model_name,
            params={"model": model_name} if model_name else {},
            chain=unique_chain,
            reasoning=(
                f"Multiple actions detected: {', '.join(sorted_actions)} "
                f"→ chain: {' → '.join(unique_chain) if unique_chain else 'custom'}"
            ),
        )

    def _build_custom_match(self, user_input: str, user_lower: str) -> IntentMatch:
        triggered_skills = []
        for skill_name, keywords in self.SKILL_TRIGGERS.items():
            if any(kw in user_lower for kw in keywords):
                triggered_skills.append(skill_name)

        if not triggered_skills:
            fallback_skills = self._search_fallback_skills(user_lower)
            if fallback_skills:
                return IntentMatch(
                    workflow_type=WorkflowType.CUSTOM,
                    confidence=0.5,
                    chain=fallback_skills,
                    params={},
                    reasoning=f"Fallback skills from ascend-model-agent: {' → '.join(fallback_skills)}",
                )
            return IntentMatch(
                workflow_type=WorkflowType.CUSTOM,
                confidence=0.3,
                chain=[],
                params={},
                reasoning="No specific workflow detected",
            )

        unique_chain = list(dict.fromkeys(triggered_skills))
        return IntentMatch(
            workflow_type=WorkflowType.CUSTOM,
            confidence=0.7,
            chain=unique_chain,
            params={},
            reasoning=f"Custom chain from keyword detection: {' → '.join(unique_chain)}",
        )

    def _search_fallback_skills(self, user_lower: str) -> list[str]:
        fallback_dir = Path(__file__).parent / "fallback-skills"
        if not fallback_dir.exists():
            return []

        fallback_keywords = [
            "ascendc", "cann", "triton", "算子", "kernel", "operator",
            "model", "模型", "推理", "inference", "profiling", "性能分析",
            "migrate", "migration", "迁移", "deploy", "deployment", "部署",
            "quantize", "quantization", "量化", "verify", "verification", "验证", "精度",
            "adapter", "适配", "adaptation", "document", "文档", "doc",
            "ai4s", "science", "科学", "megatron", "mindspore", "mindspeed",
            "docker", "container", "debug", "调试", "test", "测试", "ut",
            "coverage", "覆盖率", "install", "安装", "setup", "driver", "驱动",
            "faqs", "faq", "issue", "python", "pytorch", "torch",
            "ascend", "optimization", "优化",
        ]

        scored_skills = []
        for skill_path in fallback_dir.iterdir():
            if not skill_path.is_dir():
                continue
            skill_md = skill_path / "SKILL.md"
            if not skill_md.exists():
                continue

            skill_name = skill_path.name.lower()
            content = skill_md.read_text(encoding="utf-8").lower()

            kw_in_user = [kw for kw in fallback_keywords if kw in user_lower]
            name_score = sum(1 for kw in kw_in_user if kw in skill_name)
            content_score = sum(1 for kw in kw_in_user if kw in content)
            total_score = name_score * 2 + content_score

            if total_score > 0:
                scored_skills.append((skill_path.name, total_score, name_score))

        scored_skills.sort(key=lambda x: (-x[1], -x[2]))
        return [s[0] for s in scored_skills[:5]]

    def _extract_model_name(self, user_lower: str) -> str:
        patterns = [
            r"(qwen[0-9]*(?:\.[0-9])?(?:-[a-z0-9]+)?)",
            r"(llama[0-9]*(?:\.[0-9])?(?:b)?)",
            r"(deepseek[_-]?v?[0-9]+)",
            r"(chatglm[0-9]*)",
            r"(gemma[_-]?[0-9]+[a-z]?)",
            r"(mistral[_-]?[0-9]*)",
            r"(phi[_-]?[0-9]+)",
            r"(baichuan[_-]?[0-9]+)",
            r"(internlm[_-]?[0-9]+)",
            r"([a-z0-9]+(?:[_-][a-z0-9]+)*-[0-9]+[a-z]?b?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, user_lower, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return ""

    def _resolve_model_path(self, short_name: str) -> str:
        if not short_name:
            return ""
        key = short_name.lower()
        return self.MODEL_NAME_MAP.get(key, short_name)

    def get_skill_chain(self, intent: IntentMatch) -> list[dict[str, Any]]:
        chain = []
        for i, skill_name in enumerate(intent.chain):
            step: dict[str, Any] = {
                "step": i + 1,
                "skill": skill_name,
                "params": {},
                "depends_on": [],
            }

            if i > 0:
                prev_skill = intent.chain[i - 1]
                step["depends_on"].append(prev_skill)

            resolved_model = self._resolve_model_path(intent.model_name)
            if skill_name == "search-agent":
                step["params"] = {"model_query": resolved_model} if resolved_model else {}
            elif skill_name == "ascend-model-verification":
                step["params"] = {
                    "model_path": resolved_model,
                    "model_name": resolved_model.split("/")[-1] if resolved_model else "",
                }
            elif skill_name == "vllm-ascend-performance-optimization":
                step["params"] = {"model": resolved_model}
            elif skill_name == "doc-agent":
                step["params"] = {
                    "model": resolved_model,
                    "purpose": "model verification report",
                }
            elif skill_name == "deploy-agent":
                step["params"] = {"model": resolved_model} if resolved_model else {}
            elif skill_name == "quantify-agent":
                step["params"] = {"model": resolved_model} if resolved_model else {}
            elif skill_name == "vllm-ascend-model-adapter":
                step["params"] = {"model": resolved_model} if resolved_model else {}
            elif skill_name in ["ascend-profiling", "ascend-optimization", "ascend-affinity-operator",
                                "vector-triton-ascend-ops-optimizer", "ai4s-basic", "tf-to-pytorch",
                                "npu-adapter-reviewer", "torch-npu-optimization", "npu-basic-migrate"]:
                step["params"] = {}
            elif skill_name in ["esm2-npu", "Boltz2", "GENERator", "boltzgen", "diffsbdd", "oligoformer", "proteinbert"]:
                step["params"] = {"model": resolved_model} if resolved_model else {}
            elif skill_name == "repo-reader":
                step["params"] = {"repo_url": resolved_model} if resolved_model else {}
            elif skill_name in ["adapter-check-principle", "hardware-check-principle",
                                "model-series-vendor-detector", "ascend-history-to-skill"]:
                step["params"] = {}

            chain.append(step)

        return chain

    def format_plan(self, intent: IntentMatch) -> str:
        lines = [
            f"## Workflow Plan",
            f"",
            f"**Type**: `{intent.workflow_type.value}`",
            f"**Confidence**: {intent.confidence:.0%}",
            f"**Model**: `{intent.model_name or 'N/A'}`",
            f"**Reasoning**: {intent.reasoning}",
            f"",
            f"## Execution Chain ({len(intent.chain)} steps)",
            f"",
        ]

        chain = self.get_skill_chain(intent)
        for step in chain:
            deps = ", ".join(step["depends_on"]) or "none"
            lines.append(f"  {step['step']}. `{step['skill']}` (deps: {deps})")

            params_str = json.dumps(step["params"], indent=4, ensure_ascii=False)
            for p_line in params_str.split("\n"):
                lines.append(f"     {p_line}")

        return "\n".join(lines)


# ── Claude Code Skill 映射与 Prompt 构建 ─────────────────────────────────────

SKILL_TO_CLAUDE_MAP: dict[str, str] = {
    "ascend-model-verification": "verify-agent",
    "vllm-ascend-performance-optimization": "optimizer-agent",
    "quantify-agent": "quantify-agent",
    "vllm-ascend-model-adapter": "adapt-agent",
    "ascend-optimization": "ascend-optimization",
    "ascend-affinity-operator": "ascend-affinity-operator",
    "ai4s-basic": "ai4s-basic",
    "ascend-history-to-skill": "ascend-history-to-skill",
    "npu-basic-migrate": "npu-basic-migrate",
}

# 非 Claude Code skill，需要特殊处理或回退到 Agent
NON_CLAUDE_SKILLS = {"search-agent", "doc-agent", "deploy-agent"}


def map_skill_to_claude(skill_name: str) -> str | None:
    """将 workflow_planner 的 chain skill 名称映射为 Claude Code canonical skill 名称。"""
    return SKILL_TO_CLAUDE_MAP.get(skill_name)


def build_claude_prompt(skill_name: str, intent: IntentMatch) -> str:
    """为高置信度工作流中的指定 skill 构建 Claude Code prompt。

    Args:
        skill_name: workflow_planner chain 中的 skill 名称（如 ascend-model-verification）。
        intent: IntentResolver 解析出的意图结果。
    """
    claude_skill = map_skill_to_claude(skill_name) or skill_name
    resolved_model = IntentResolver()._resolve_model_path(intent.model_name)
    model_display = (
        resolved_model.split("/")[-1]
        if resolved_model
        else (intent.model_name or "未指定")
    )

    # 从 intent 的 execution chain 中提取该 skill 对应的参数
    params: dict[str, Any] = {}
    for step in IntentResolver().get_skill_chain(intent):
        if step["skill"] == skill_name:
            params = step.get("params", {})
            break

    lines = [
        f"请使用 {claude_skill} skill 来完成以下任务。",
        "",
        f"**模型**: {model_display}",
    ]
    if params:
        lines.append(f"**参数**: {json.dumps(params, ensure_ascii=False)}")
    lines.append("")
    lines.append(f"**用户原始需求**: {intent.reasoning}")
    lines.append("")
    lines.append("请直接开始执行，无需询问确认。")

    return "\n".join(lines)


def filter_executable_chain(intent: IntentMatch) -> list[dict[str, Any]]:
    """过滤出高置信度意图中可直接通过 Claude Code 执行的 skill 步骤。

    返回列表元素格式: {"skill": str, "claude_skill": str, "prompt": str}
    """
    executable: list[dict[str, Any]] = []
    for skill_name in intent.chain:
        if skill_name in NON_CLAUDE_SKILLS:
            continue
        claude_skill = map_skill_to_claude(skill_name)
        if claude_skill is None:
            continue
        prompt = build_claude_prompt(skill_name, intent)
        executable.append({
            "skill": skill_name,
            "claude_skill": claude_skill,
            "prompt": prompt,
        })
    return executable
