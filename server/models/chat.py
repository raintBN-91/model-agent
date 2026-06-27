"""OpenAI 兼容的数据模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None

    @field_validator("content")
    @classmethod
    def empty_content_to_none(cls, v: str | None) -> str | None:
        if v is not None and v.strip() == "":
            return None
        return v


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = True
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1, le=131072)
    c_id: str | None = Field(default=None, max_length=128)  # 上下文关联标识，相同 c_id 的请求共享对话状态
    confirmation: str | None = None  # 工具调用确认：confirmed / rejected
    brainstorm: str | None = None  # 脑暴回复格式: "question_id:answer" 或 "question_id:custom:自定义输入"
    trace_id: str | None = None  # trace_id 用于日志关联

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        if not v:
            raise ValueError("messages 不能为空")
        if v[-1].role != "user":
            raise ValueError("messages 最后一条必须是 user")
        return v


class DeltaContent(BaseModel):
    role: str = ""
    content: str = ""
    tool_calls: list[dict] | None = None


class Choice(BaseModel):
    index: int = 0
    delta: DeltaContent = Field(default_factory=DeltaContent)
    finish_reason: str | None = None
    requires_confirmation: bool | None = None
    requires_input: bool | None = None  # 标记需要用户输入（脑暴模式）
    brainstorm_state: str | None = None  # 当前脑暴状态


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[Choice]


class OpenAIError(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str


class OpenAIErrorResponse(BaseModel):
    error: OpenAIError
