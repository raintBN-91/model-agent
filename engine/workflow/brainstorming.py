"""Brainstorming Session — LLM 驱动的多轮意图澄清。

在 DynamicPlanner 规划工作流之前，通过多轮对话明确用户需求。
遵循 HARD-GATE 原则：未确认设计前，不执行任何工作流。

   BrainstormState
   ┌──────────┐
   │ INITIAL  │
   └────┬─────┘
        │ LLM 分析用户输入
        v
   ┌──────────┐
   │CLARIFYING│ ◄──── 每轮一个问题，2-3 个选项
   └────┬─────┘
        │ 已收集足够信息
        v
   ┌──────────┐
   │ PROPOSING│ ◄──── 提出 2-3 种方案
   └────┬─────┘
        │ 用户选择方案
        v
   ┌──────────┐
   │ REVIEWING│ ◄──── 展示设计文档
   └────┬─────┘
     ┌──┴───┐
     v      v
  APPROVED REJECTED
     │      └──► CLARIFYING（重新澄清）
     v
  COMPLETED
"""

from __future__ import annotations

import json
import re
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from server.config import settings
from engine.llm_factory import build_workflow_llm, extract_text_content, handle_api_error


# ── 状态枚举 ───────────────────────────────────────────────────────────

class BrainstormState(str, Enum):
    """脑暴会话状态。"""
    INITIAL = "initial"
    CLARIFYING = "clarifying"     # 逐一提问阶段
    PROPOSING = "proposing"       # 提出方案阶段
    REVIEWING = "reviewing"       # 用户审核设计阶段
    APPROVED = "approved"         # 用户已确认设计
    COMPLETED = "completed"       # 脑暴结束，可 transition
    REJECTED = "rejected"         # 用户否决，重新澄清


# ── 数据模型 ───────────────────────────────────────────────────────────

class BrainstormQuestion(BaseModel):
    """向用户提出的问题。"""
    question_id: str = ""
    question: str = ""
    options: list[str] = Field(default_factory=list)
    allow_custom: bool = True
    dimension: str = ""


class BrainstormProposal(BaseModel):
    """执行方案。"""
    title: str = ""
    description: str = ""
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    estimated_effort: str = ""


class BrainstormDesignDoc(BaseModel):
    """经确认的设计文档。"""
    goal: str = ""
    background: str = ""
    platform: str = ""
    model_name: str = ""
    framework: str = ""
    approach: str = ""
    workflow_plan: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    dimensions: dict[str, str] = Field(default_factory=dict)


# ── LLM Response Schema ────────────────────────────────────────────────

LLM_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": ["question", "proposals", "design_doc", "complete"]
        },
        "question": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"}},
                "dimension": {"type": "string"}
            }
        },
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "pros": {"type": "array", "items": {"type": "string"}},
                    "cons": {"type": "array", "items": {"type": "string"}},
                    "estimated_effort": {"type": "string"}
                }
            }
        },
        "design_doc": {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "background": {"type": "string"},
                "platform": {"type": "string"},
                "model_name": {"type": "string"},
                "framework": {"type": "string"},
                "approach": {"type": "string"},
                "workflow_plan": {"type": "array", "items": {"type": "string"}},
                "risks": {"type": "array", "items": {"type": "string"}}
            }
        },
        "assessment": {
            "type": "string",
            "enum": ["clarify_needed", "ready_for_proposals", "ready_for_review", "complete"]
        }
    },
    "required": ["type", "assessment"]
}


# ── BrainstormingSession ───────────────────────────────────────────────

class BrainstormingSession:
    """LLM 驱动的多轮脑暴会话。

    用法::

        session = BrainstormingSession(session_id="xxx")
        question = session.start("验证 Qwen3-8B 在 A2 上的适配")
        # 返回第一个问题

        result = session.answer(question.question_id, "A2")
        # 返回下一个问题 / 方案 / 设计文档
    """

    # ── 系统 Prompt ────────────────────────────────────────────────
    SYSTEM_PROMPT = """你是一个 AI 工作流的 **Brainstorming 助手**。你的职责是在执行工作流之前，
通过对话帮助用户澄清需求。

**重要**：对于模型适配场景，mofix 工作流已内置昇腾 NPU 平台支持，无需向用户询问 target_platform、硬件型号等部署细节（平台已自动设为 Ascend NPU）。你只需要澄清模型相关的信息。**优先引导用户提供 ModelScope 链接**（格式如 https://www.modelscope.cn/models/...），其次是 HuggingFace 链接。如果用户无法提供链接，再询问具体模型名称和规模（如 Qwen3.5-0.8B / 4B / 9B 等）。

## 核心原则

1. **ONE QUESTION AT A TIME**：每次只问一个问题，等待用户回答后再问下一个
2. **MULTIPLE CHOICE FIRST**：优先提供 2-3 个具体选项，让用户选择
3. **HARD-GATE**：在用户确认设计之前，绝不要执行任何操作
4. **上下文感知**：基于已收集的信息，提出有针对性的问题，不要重复问

## 关键维度

根据用户输入，需要澄清的维度包括但不限于：
- `model_name`: 需要处理的模型（优先引导用户提供 ModelScope 链接，其次 HuggingFace 链接，最后手动指定名称和规模）
- `inference_framework`: 推理框架（vLLM-Ascend / SGLang / MindIE / LMDeploy）
- `optimization_goal`: 优化目标（吞吐量 / 延迟 / 显存 / 综合）
- `quantization`: 量化需求（不需要 / W8A16 / INT4 / FP16）
- `accuracy_requirement`: 精度要求
- `use_case`: 使用场景（在线推理 / 离线批处理 / 训练 / 微调）

## 输出格式

必须输出 JSON 格式：

```json
{
    "type": "question | proposals | design_doc | complete",
    "question": {
        "text": "问题内容",
        "options": ["选项1", "选项2", "选项3"],
        "dimension": "model_name"
    },
    "proposals": [
        {
            "title": "方案名称",
            "description": "方案描述",
            "pros": ["优点1"],
            "cons": ["缺点1"],
            "estimated_effort": "预估工作量"
        }
    ],
    "design_doc": {
        "goal": "明确后的目标",
        "background": "背景信息",
        "platform": "目标平台",
        "model_name": "模型名称（优先提供 ModelScope 链接）",
        "framework": "推理框架",
        "approach": "选定方案描述",
        "workflow_plan": ["步骤1", "步骤2"],
        "risks": ["风险1"]
    },
    "assessment": "clarify_needed | ready_for_proposals | ready_for_review | complete"
}
```

## 行为规则

- 只有 type="question" 时，用户才需要回复
- type="proposals" 时，等待用户选择方案
- type="design_doc" 时，等待用户确认或否决
- assessment 指示当前整体进度"""

    def __init__(
        self,
        session_id: str | None = None,
        llm=None,
    ):
        self.session_id = session_id or f"bs-{uuid.uuid4().hex[:12]}"
        self.state = BrainstormState.INITIAL
        self.llm = llm or build_workflow_llm()

        # 用户输入
        self.user_input: str = ""

        # 已澄清的维度
        self.clarified_dimensions: dict[str, str] = {}

        # 已问过的问题（去重）
        self.questions_asked: list[str] = []

        # 方案
        self.proposals: list[BrainstormProposal] = []
        self.selected_proposal_index: int | None = None

        # 设计文档
        self.design_doc: BrainstormDesignDoc | None = None

        # 对话历史
        self.history: list[dict[str, Any]] = []

    # ── 公开 API ───────────────────────────────────────────────────

    def start(self, user_input: str) -> BrainstormQuestion:
        """开始脑暴：LLM 分析用户输入，返回第一个问题或方案。

        Args:
            user_input: 用户的原始输入。

        Returns:
            BrainstormQuestion — 第一个要问用户的问题。
        """
        self.user_input = user_input
        self.state = BrainstormState.CLARIFYING

        response = self._call_llm()
        return self._process_llm_response(response)

    def answer(self, question_id: str, answer: str) -> dict:
        """处理用户对上一个问题的回复。

        Args:
            question_id: 问题 ID（用于历史记录，当前未严格校验）。
            answer: 用户的回答。

        Returns:
            包含 ``type`` 字段的 dict：
            - ``{"type": "question", "question": BrainstormQuestion}``
            - ``{"type": "proposals", "proposals": [BrainstormProposal]}``
            - ``{"type": "design_doc", "design_doc": BrainstormDesignDoc}``
            - ``{"type": "complete", "context": {...}}``
        """
        # 记录回答
        self.history.append({
            "role": "user",
            "type": "answer",
            "question_id": question_id,
            "content": answer,
        })

        # 解析维度
        is_custom = answer.startswith("custom:")
        if is_custom:
            dimension = self.history[-2].get("dimension", "unknown") if len(self.history) >= 2 else "unknown"
            self.clarified_dimensions[dimension] = answer[7:].strip()
        else:
            dimension = self.history[-2].get("dimension", "unknown") if len(self.history) >= 2 else "unknown"
            self.clarified_dimensions[dimension] = answer

        # 根据当前状态处理
        if self.state == BrainstormState.CLARIFYING:
            response = self._call_llm()
            return self._process_llm_response(response)

        elif self.state == BrainstormState.PROPOSING:
            # 用户选择了方案 index
            try:
                idx = int(answer)
                return self._handle_proposal_selection(idx)
            except (ValueError, IndexError):
                # 非法输入，重新返回方案列表
                return {
                    "type": "proposals",
                    "proposals": [p.model_dump() for p in self.proposals],
                    "error": f"请选择 0-{len(self.proposals)-1} 之间的数字",
                }

        elif self.state == BrainstormState.REVIEWING:
            # 不应该通过 answer() 到达这里；使用 approve() / reject()
            return {
                "type": "design_doc",
                "design_doc": self.design_doc.model_dump() if self.design_doc else {},
                "error": "请使用 approve() 确认或 reject() 否决设计",
            }

        return {"type": "complete", "context": self._build_context()}

    def select_proposal(self, index: int) -> dict:
        """用户选择第 *index* 个方案，LLM 生成设计文档。

        Returns:
            包含 design_doc 的 dict。
        """
        return self._handle_proposal_selection(index)

    def approve(self) -> dict:
        """用户确认设计。返回可以在 DynamicPlanner 中使用的上下文。

        Returns:
            ``{"type": "complete", "context": {...}}``
        """
        self.state = BrainstormState.APPROVED

        # 如果设计文档已完成，生成最终 context
        context = self._build_context()
        self.state = BrainstormState.COMPLETED
        return {"type": "complete", "context": context}

    def reject(self, reason: str = "") -> dict:
        """用户否决设计，回到 CLARIFYING 或 PROPOSING。

        Args:
            reason: 用户否决的原因。

        Returns:
            新的问题或方案列表。
        """
        self.state = BrainstormState.REJECTED
        self.history.append({
            "role": "user",
            "type": "reject",
            "content": reason or "用户否决了当前设计",
        })

        # 让 LLM 决定下一步
        response = self._call_llm()
        self.state = BrainstormState.CLARIFYING if response.get("assessment") == "clarify_needed" else BrainstormState.PROPOSING
        return self._process_llm_response(response)

    def to_dict(self) -> dict:
        """序列化为 dict（用于 SSE 传输 / checkpoint）。"""
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "user_input": self.user_input,
            "clarified_dimensions": dict(self.clarified_dimensions),
            "questions_asked": list(self.questions_asked),
            "proposals": [p.model_dump() for p in self.proposals],
            "selected_proposal_index": self.selected_proposal_index,
            "design_doc": self.design_doc.model_dump() if self.design_doc else None,
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, data: dict) -> BrainstormingSession:
        """从 dict 恢复会话。"""
        session = cls(session_id=data.get("session_id"))
        session.state = BrainstormState(data.get("state", "initial"))
        session.user_input = data.get("user_input", "")
        session.clarified_dimensions = data.get("clarified_dimensions", {})
        session.questions_asked = data.get("questions_asked", [])
        session.proposals = [BrainstormProposal(**p) for p in data.get("proposals", [])]
        session.selected_proposal_index = data.get("selected_proposal_index")
        if data.get("design_doc"):
            session.design_doc = BrainstormDesignDoc(**data["design_doc"])
        session.history = list(data.get("history", []))
        return session

    # ── 内部方法 ───────────────────────────────────────────────────

    def _build_llm_messages(self) -> list[dict]:
        """构造发给 LLM 的消息列表。"""
        known = "\n".join(
            f"- {k}: {v}" for k, v in self.clarified_dimensions.items()
        ) if self.clarified_dimensions else "（暂无）"

        pending = self._identify_pending_dimensions()

        # 优先级规则：用户提到了模型系列（如 Qwen）但没有具体版本号 → 优先澄清 model_name
        priority_rule = ""
        lower_input = self.user_input.lower()
        has_model_family = any(k in lower_input for k in ["qwen", "deepseek", "llama", "chatglm", "baichuan", "gpt"])
        has_specific_version = bool(re.search(r'(?:qwen|deepseek|llama|chatglm|baichuan|gpt)\s*-?\s*\d+', lower_input))
        if has_model_family and not has_specific_version and "model_name" not in self.clarified_dimensions:
            priority_rule = "\n**优先级规则**：用户提到了模型系列名但没有具体版本号，请只询问 model_name，引导用户提供模型的 ModelScope 链接（推荐）或 HuggingFace 链接，而非仅列出模型规模。例如：“请提供 Qwen3.5 模型的 ModelScope 链接（推荐）或 HuggingFace 链接，以便我们获取准确的模型权重和配置。” 如果没有链接，再请用户明确具体规模（如 0.8B / 4B / 9B / 27B / 35B）。禁止询问 target_platform 或硬件平台。这些由工作流自动确定。"

        system = self.SYSTEM_PROMPT + f"""

## 当前会话状态

**原始用户输入**: {self.user_input}

**已收集的信息**:
{known}

**待澄清的维度**:
{pending}

**当前状态**: {self.state.value}
**已问问题数**: {len(self.questions_asked)}{priority_rule}
"""

        messages = [{"role": "system", "content": system}]

        # 添加对话历史
        for entry in self.history:
            if entry["role"] == "assistant":
                messages.append({"role": "assistant", "content": json.dumps(entry.get("data", {}), ensure_ascii=False)})
            else:
                messages.append({"role": "user", "content": entry.get("content", "")})

        # 如果当前是 INITIAL 或 CLARIFYING，添加用户原始输入
        if self.state in (BrainstormState.INITIAL, BrainstormState.CLARIFYING) and not self.history:
            messages.append({"role": "user", "content": self.user_input})

        return messages

    def _identify_pending_dimensions(self) -> str:
        """识别还需澄清的维度。"""
        all_dimensions = [
            "model_name", "inference_framework",
            "optimization_goal", "quantization", "accuracy_requirement", "use_case",
        ]
        pending = [d for d in all_dimensions if d not in self.clarified_dimensions]
        return "\n".join(f"- {d}" for d in pending) if pending else "（全部维度已澄清）"

    def _call_llm(self) -> dict:
        """调用 LLM 获取下一步响应。"""
        messages = self._build_llm_messages()

        try:
            response = self.llm.invoke(messages)
            text = extract_text_content(response.content) if hasattr(response, "content") else str(response)
            # 提取 JSON
            return self._parse_json_response(text)
        except Exception as e:
            handle_api_error(e)
            # 降级：返回默认问题
            return self._fallback_response()

    def _parse_json_response(self, text: str) -> dict:
        """从 LLM 响应中提取 JSON。"""
        # 尝试直接解析
        text = text.strip()
        if text.startswith("{"):
            return json.loads(text)
        # 尝试从 ```json ``` 代码块中提取
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start) if "```" in text[start:] else len(text)
            return json.loads(text[start:end].strip())
        if "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start) if "```" in text[start:] else len(text)
            return json.loads(text[start:end].strip())
        raise ValueError(f"无法解析 LLM 响应: {text[:200]}")

    def _fallback_response(self) -> dict:
        """LLM 调用失败时的降级响应。"""
        if self.state == BrainstormState.INITIAL:
            return {
                "type": "question",
                "question": {
                    "text": "请提供需要处理的模型信息？（例如 ModelScope 链接或模型名称）",
                    "options": ["提供 ModelScope 链接", "提供 HuggingFace 链接", "手动指定模型名称和规模"],
                    "dimension": "model_name",
                },
                "assessment": "clarify_needed",
            }
        if "model_name" not in self.clarified_dimensions:
            return {
                "type": "question",
                "question": {
                    "text": "需要处理哪个模型？",
                    "options": ["Qwen3-8B", "Qwen3-14B", "DeepSeek-V3", "DeepSeek-R1"],
                    "dimension": "model_name",
                },
                "assessment": "clarify_needed",
            }
        if "model_name" in self.clarified_dimensions and "inference_framework" not in self.clarified_dimensions:
            return {
                "type": "question",
                "question": {
                    "text": "使用哪个推理框架？",
                    "options": ["vLLM-Ascend", "SGLang", "MindIE", "LMDeploy"],
                    "dimension": "inference_framework",
                },
                "assessment": "clarify_needed",
            }
        # 降级到方案
        return {
            "type": "proposals",
            "proposals": [
                {
                    "title": "标准验证流程",
                    "description": f"验证 {self.clarified_dimensions.get('model_name', '模型')} 的基础适配",
                    "pros": ["流程标准化", "快速执行"],
                    "cons": ["不包含性能优化"],
                    "estimated_effort": "约 1-2 小时",
                },
                {
                    "title": "完整适配+优化",
                    "description": f"适配 {self.clarified_dimensions.get('model_name', '模型')} 并做性能调优",
                    "pros": ["全面覆盖", "性能最优"],
                    "cons": ["耗时较长", "需要多次迭代"],
                    "estimated_effort": "约 4-8 小时",
                },
            ],
            "assessment": "ready_for_proposals",
        }

    def _process_llm_response(self, response: dict) -> dict:
        """处理 LLM 响应，更新状态并返回格式化结果。"""
        response_type = response.get("type", "question")
        assessment = response.get("assessment", "clarify_needed")

        # --- question ---
        if response_type == "question":
            self.state = BrainstormState.CLARIFYING
            q_data = response.get("question", {})
            q = BrainstormQuestion(
                question_id=f"q_{uuid.uuid4().hex[:8]}",
                question=q_data.get("text", ""),
                options=q_data.get("options", []),
                allow_custom=q_data.get("allow_custom", True),
                dimension=q_data.get("dimension", "unknown"),
            )
            self.questions_asked.append(q.question_id)
            self.history.append({
                "role": "assistant",
                "type": "question",
                "question_id": q.question_id,
                "dimension": q.dimension,
                "content": q.question,
                "data": response,
            })
            return {"type": "question", "question": q}

        # --- proposals ---
        if response_type == "proposals":
            self.state = BrainstormState.PROPOSING
            proposals_data = response.get("proposals", [])
            self.proposals = [BrainstormProposal(**p) for p in proposals_data]
            self.history.append({
                "role": "assistant",
                "type": "proposals",
                "content": f"提出 {len(self.proposals)} 个方案",
                "data": response,
            })
            return {
                "type": "proposals",
                "proposals": [p.model_dump() for p in self.proposals],
            }

        # --- design_doc ---
        if response_type == "design_doc":
            self.state = BrainstormState.REVIEWING
            dd_data = response.get("design_doc", {})
            if self.clarified_dimensions:
                dd_data["dimensions"] = dict(self.clarified_dimensions)
            self.design_doc = BrainstormDesignDoc(**dd_data)
            self.history.append({
                "role": "assistant",
                "type": "design_doc",
                "content": "设计文档已生成",
                "data": response,
            })
            return {
                "type": "design_doc",
                "design_doc": self.design_doc.model_dump(),
            }

        # --- complete ---
        if response_type == "complete":
            context = self._build_context()
            self.state = BrainstormState.COMPLETED
            return {"type": "complete", "context": context}

        # 回退
        return {"type": "question", "question": BrainstormQuestion(
            question_id=f"q_{uuid.uuid4().hex[:8]}",
            question="请提供更多信息，以便我能为您制定合适的方案。",
            options=["继续", "重新开始", "结束"],
        )}

    def _handle_proposal_selection(self, index: int) -> dict:
        """用户选择了第 *index* 个方案，生成设计文档。"""
        if index < 0 or index >= len(self.proposals):
            return {
                "type": "proposals",
                "proposals": [p.model_dump() for p in self.proposals],
                "error": f"请选择 0-{len(self.proposals)-1} 之间的数字",
            }

        self.selected_proposal_index = index
        selected = self.proposals[index]

        # 让 LLM 生成设计文档
        self.history.append({
            "role": "user",
            "type": "select_proposal",
            "content": f"选择方案 {index}: {selected.title}",
        })

        response = self._call_llm()
        if response.get("type") == "design_doc":
            return self._process_llm_response(response)

        # LLM 没返回 design_doc，手动构造一个
        dd = BrainstormDesignDoc(
            goal=self.user_input,
            background=f"基于用户需求: {self.user_input}",
            platform="Ascend NPU（内置）",
            model_name=self.clarified_dimensions.get("model_name", ""),
            framework=self.clarified_dimensions.get("inference_framework", ""),
            approach=selected.description,
            workflow_plan=[
                "1. 环境准备（Ascend NPU 内置平台）",
                f"2. {selected.title} 执行",
                "3. 结果验证",
            ],
            risks=["NPU 环境依赖", "模型兼容性"],
            dimensions=dict(self.clarified_dimensions),
        )
        self.design_doc = dd
        self.state = BrainstormState.REVIEWING
        return {"type": "design_doc", "design_doc": dd.model_dump()}

    def _build_context(self) -> dict:
        """构建传递给 DynamicPlanner 的上下文。"""
        context = {
            "session_id": self.session_id,
            "user_input": self.user_input,
            "dimensions": dict(self.clarified_dimensions),
        }
        if self.design_doc:
            context["design_doc"] = self.design_doc.model_dump()
        if self.selected_proposal_index is not None:
            context["selected_proposal"] = self.proposals[self.selected_proposal_index].model_dump()
        return context


# ── 模糊度评估 ─────────────────────────────────────────────────────────

DEFAULT_AMBIGUITY_DIMENSIONS = [
    "model_name",
    "inference_framework",
    "optimization_goal",
    "quantization",
    "use_case",
]


async def assess_ambiguity(user_input: str, llm=None) -> tuple[float, list[str]]:
    """评估用户输入的模糊度。

    判断用户的核心意图是否足够明确以直接执行工作流，
    而非简单地统计"缺失维度"数量。只有核心任务不清晰时才触发澄清。

    Args:
        user_input: 用户原始输入。
        llm: LLM 实例，为空则使用 ``build_workflow_llm()``。

    Returns:
        ``(score: 0.0-1.0, missing_dimensions: list[str])``
        分数越低表示越明确。仅当核心意图不清晰时返回高分。
    """
    if llm is None:
        llm = build_workflow_llm()

    prompt = f"""判断以下用户请求是否足够明确以直接开始执行。

核心原则（重要）：
1. 用户指定了「做什么(动作)」+「带具体规模参数的模型」= 足够明确，不应触发澄清
   - 明确：Qwen3.5-0.8B, Qwen3-14B, DeepSeek-R1-7B, LLaMA-3-8B（含 B 后缀的具体规模）
   - 不明确：Qwen3.5, Qwen3, DeepSeek-V3（仅有系列名/版本号，未指定 0.8B/4B/9B/27B/35B 等具体规模）
2. 缺少平台、框架等参数是正常的，工作流会处理或使用默认值
3. 用户说了部署/优化/验证/适配/测试/量化/迁移等具体动作 = 动作明确
4. 仅提及模型系列名（Qwen、DeepSeek、LLaMA 等）没有具体规模参数 = 对象不明确，missing_dimensions 必须包含 "model_name"
5. 只有"帮我弄一下""帮我测试一下""我想做适配"这类无具体对象的需求 = 模糊，需要澄清
6. 模型系列名后跟数字（如 Qwen3.5、Qwen3、DeepSeek-V3）仅表示系列版本，不代表具体模型规模，仍需澄清具体规模

✅ 明确示例：测试Qwen3.5-0.8B、部署DeepSeek-R1-7B、验证Qwen3-14B的适配、适配LLaMA-3-8B
❌ 模糊示例：测试Qwen模型、帮我适配Qwen3.5、验证DeepSeek-V3、帮我适配Qwen3（未指定具体模型规模）

用户输入: {user_input}

只返回 JSON:
{{"score": 0.xx, "missing_dimensions": ["dim1", "dim2"]}}
如果不需要澄清，missing_dimensions 应为空数组。

**注意**：target_platform（目标平台/硬件）不作为评估维度，因为 mofix 工作流已内置昇腾 NPU 平台支持。只需评估模型相关维度（model_name 等）。优先引导用户提供 ModelScope 链接。
"""

    try:
        import asyncio as _asyncio
        messages = [{"role": "system", "content": "你是一个需求分析助手，只输出 JSON。"},
                    {"role": "user", "content": prompt}]
        response = await _asyncio.to_thread(llm.invoke, messages)
        text = response.content if hasattr(response, "content") else str(response)
        text = text.strip()
        llm_score, llm_missing = 0.0, []
        if text.startswith("{"):
            data = json.loads(text)
            llm_score = float(data.get("score", 0.5))
            llm_missing = list(data.get("missing_dimensions", []))
    except Exception:
        llm_score, llm_missing = 0.0, []

    # 关键词基线检查：确保关键词层面的模糊也不会被漏掉
    kw_score, kw_missing = _keyword_ambiguity_fallback(user_input)

    # 取更保守的结果（更高分 + 更多缺失维度）
    final_score = max(llm_score, kw_score)
    final_missing = list(set(llm_missing) | set(kw_missing))

    return final_score, final_missing


def _keyword_ambiguity_fallback(user_input: str) -> tuple[float, list[str]]:
    """基于关键词的模糊度降级评估。

    只有当核心动作或目标不明确时才返回高分。
    - 有具体动作 + 具体模型（带版本号）= 低模糊度
    - 有具体动作 + 仅模型系列名（无具体规模）= 部分明确，需澄清（优先引导提供 ModelScope 链接）
    - 有动作无模型 = 中等模糊度
    - 无明确动作 = 高模糊度
    """
    lower = user_input.lower()
    missing = []

    # 动作关键词
    action_keywords = [
        "部署", "deploy", "优化", "optimize", "验证", "verify",
        "适配", "adapt", "量化", "quantify", "迁移", "migrate",
        "开发", "develop", "profiling", "性能", "推理", "infer",
        "训练", "train", "微调", "finetune", "评估", "evaluate",
        "测试", "test",
    ]
    has_action = any(k in lower for k in action_keywords)

    # 模型系列关键词
    model_family_keywords = [
        "qwen", "deepseek", "llama", "chatglm", "baichuan", "yi-",
        "bert", "gpt", "model", "模型",
    ]
    has_model_family = any(k in lower for k in model_family_keywords)

    # 判断模型是否带版本号（如 qwen3.5-0.8B、deepseek-r1、Qwen3-8B）
    # Check for specific model SIZE (requires B suffix like 0.8B, 4B, 72B, or dash-number like Qwen3.5-0.8B)
    has_specific_version = bool(re.search(
        r'(?:qwen|deepseek|llama|chatglm|baichuan|gpt|glm|yi)\s*[-.]?\s*\d+\.?\d*[bB]',
        lower,
    ))
    # Also check for known patterns without B suffix: e.g., "Qwen3-14B" type with explicit dash-number-B
    if not has_specific_version:
        has_specific_version = bool(re.search(
            r'\d+[bB]|\d+\.?\d+[bB]',
            lower,
        ))

    # 平台关键词
    platform_keywords = ["a2", "a3", "910b", "ascend", "npu", "昇腾"]

    if has_action and has_specific_version:
        # 有具体动作 + 带版本号的具体模型 → 核心意图明确
        return 0.15, []

    if has_action and has_model_family:
        # 有动作，提及了模型系列但无具体规模 → 只需澄清具体模型规模
        # 平台/硬件由 mofix 工作流自动处理，无需询问用户
        missing.append("model_name")
        score = 0.55
        return score, missing

    if has_action:
        # 有动作但无具体模型 → 部分明确，需澄清模型名称
        missing.append("model_name")
        score = 0.55
        return score, missing

    # 无明确动作 → 模糊
    if not any(k in lower for k in model_family_keywords):
        missing.append("model_name")

    score = min(1.0, 0.5 + len(missing) * 0.15)
    return score, missing
