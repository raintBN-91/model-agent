from __future__ import annotations
"""Claude Code 历史记录对话轮次提炼模块。

将 Claude Code JSONL 原始消息流提炼成结构化的"一轮对话"（turn），
包含用户问题、助手最终回答、思考过程、工具调用及结果。
"""


from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """一次工具调用记录。"""

    name: str
    input: dict[str, Any] | None
    result: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "input": self.input,
            "result": self.result,
        }


@dataclass
class Turn:
    """一个完整的人机对话轮次。"""

    turn_id: str
    question: str
    started_at: str | None
    completed_at: str | None = None
    is_complete: bool = False
    answer: str = ""
    thinking: list[str] = field(default_factory=list)
    tools: list[ToolCall] = field(default_factory=list)

    # 内部状态，不参与序列化
    known_uuids: set[str] = field(default_factory=set, repr=False)
    _tool_results: dict[str, str] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "question": self.question,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "is_complete": self.is_complete,
            "answer": self.answer,
            "thinking": self.thinking,
            "tools": [t.to_dict() for t in self.tools],
        }


class HistoryExtractor:
    """从 Claude Code JSONL 记录中增量提炼对话轮次。"""

    def __init__(self) -> None:
        self._pending_turns: dict[str, Turn] = {}

    def extract_turns(self, records: list[dict[str, Any]]) -> list[Turn]:
        """输入新增记录，返回已完成的 Turn 列表。"""
        completed_turns: list[Turn] = []

        for record in records:
            msg_type = self._classify(record)

            if msg_type == "human_user":
                # 强制结束所有未完成的旧轮次
                for turn in list(self._pending_turns.values()):
                    turn.is_complete = False
                    completed_turns.append(turn)
                    del self._pending_turns[turn.turn_id]

                turn_id = record.get("uuid", "")
                turn = Turn(
                    turn_id=turn_id,
                    question=self._extract_question(record),
                    started_at=record.get("timestamp"),
                )
                turn.known_uuids.add(turn_id)
                if turn_id:
                    self._pending_turns[turn_id] = turn
                continue

            if msg_type == "tool_result":
                turn = self._find_turn(record)
                if turn is None:
                    continue
                record_uuid = record.get("uuid")
                if record_uuid:
                    turn.known_uuids.add(record_uuid)
                tool_use_id, result = self._extract_tool_result(record)
                if tool_use_id:
                    turn._tool_results[tool_use_id] = result
                    # 尝试给已存在的 tool_call 补全结果
                    for tool in turn.tools:
                        tool_id = tool.input.get("id") if isinstance(tool.input, dict) else None
                        if tool_id == tool_use_id and tool.result is None:
                            tool.result = result
                continue

            if msg_type == "assistant":
                turn = self._find_turn(record)
                if turn is None:
                    continue
                self._attach_assistant_message(turn, record)
                if self._check_complete(record):
                    self._finalize_tools(turn)
                    turn.is_complete = True
                    turn.completed_at = record.get("timestamp")
                    completed_turns.append(turn)
                    if turn.turn_id in self._pending_turns:
                        del self._pending_turns[turn.turn_id]
                continue

            # ignore：不参与内容提取，但如果 parentUuid 属于某轮次，
            # 把自身 uuid 加入 known_uuids，维持后续消息链式查找
            turn = self._find_turn(record)
            if turn is not None:
                record_uuid = record.get("uuid")
                if record_uuid:
                    turn.known_uuids.add(record_uuid)

        return completed_turns

    @staticmethod
    def _classify(record: dict[str, Any]) -> str:
        t = record.get("type")
        if t == "user" and record.get("origin", {}).get("kind") == "human":
            return "human_user"
        if t == "user":
            return "tool_result"
        if t == "assistant":
            return "assistant"
        return "ignore"

    @staticmethod
    def _extract_question(record: dict[str, Any]) -> str:
        content = record.get("message", {}).get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for c in content:
                if isinstance(c, dict):
                    parts.append(c.get("text", ""))
            return "".join(parts)
        return ""

    def _find_turn(self, record: dict[str, Any]) -> Turn | None:
        parent_uuid = record.get("parentUuid")
        record_uuid = record.get("uuid")

        # 优先通过 parentUuid 直接匹配 turn_id
        if parent_uuid and parent_uuid in self._pending_turns:
            return self._pending_turns[parent_uuid]

        # 再遍历 pending turn，匹配 known_uuids
        for turn in self._pending_turns.values():
            if parent_uuid and parent_uuid in turn.known_uuids:
                return turn
            if record_uuid and record_uuid in turn.known_uuids:
                return turn

        return None

    def _attach_assistant_message(self, turn: Turn, record: dict[str, Any]) -> None:
        record_uuid = record.get("uuid")
        if record_uuid:
            turn.known_uuids.add(record_uuid)

        content = record.get("message", {}).get("content", [])
        if isinstance(content, str):
            if content:
                turn.answer += content
            return

        if not isinstance(content, list):
            return

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text", "")
                if text:
                    if turn.answer:
                        turn.answer += "\n"
                    turn.answer += text
            elif block_type == "thinking":
                thinking = block.get("thinking", "")
                if thinking:
                    turn.thinking.append(thinking)
            elif block_type == "tool_use":
                tool_id = block.get("id")
                tool = ToolCall(
                    name=block.get("name", ""),
                    input={"id": tool_id, **(block.get("input") or {})},
                )
                if tool_id and tool_id in turn._tool_results:
                    tool.result = turn._tool_results[tool_id]
                turn.tools.append(tool)

    @staticmethod
    def _check_complete(record: dict[str, Any]) -> bool:
        if record.get("type") == "assistant":
            stop_reason = record.get("message", {}).get("stop_reason")
            if stop_reason == "end_turn":
                # 真实数据中，thinking-only 的 assistant 消息也可能标记 end_turn；
                # 只有包含最终 text 回复的 end_turn 才真正结束本轮。
                content = record.get("message", {}).get("content", [])
                if isinstance(content, list):
                    return any(
                        isinstance(c, dict) and c.get("type") == "text"
                        for c in content
                    )
                return True
        return False

    @staticmethod
    def _extract_tool_result(record: dict[str, Any]) -> tuple[str | None, str]:
        content = record.get("message", {}).get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_use_id = block.get("tool_use_id")
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        result_content = "\n".join(
                            str(c) for c in result_content
                        )
                    return tool_use_id, str(result_content)
        return None, ""

    def _finalize_tools(self, turn: Turn) -> None:
        for tool in turn.tools:
            tool_id = tool.input.get("id") if isinstance(tool.input, dict) else None
            if tool_id and tool.result is None and tool_id in turn._tool_results:
                tool.result = turn._tool_results[tool_id]
