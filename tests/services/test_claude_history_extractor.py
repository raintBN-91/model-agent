"""测试 Claude Code 历史记录对话轮次提炼模块。"""

from __future__ import annotations

import pytest

from app.services.claude_history_extractor import HistoryExtractor


def _human_record(
    text: str,
    uuid: str,
    parent_uuid: str = "",
    timestamp: str = "2026-06-23T13:09:54.017Z",
) -> dict:
    return {
        "type": "user",
        "origin": {"kind": "human"},
        "message": {"role": "user", "content": text},
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "timestamp": timestamp,
    }


def _tool_result_record(
    tool_use_id: str,
    result: str,
    uuid: str,
    parent_uuid: str,
    timestamp: str = "2026-06-23T13:10:01.000Z",
) -> dict:
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result,
                }
            ],
        },
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "timestamp": timestamp,
    }


def _assistant_record(
    content: list,
    stop_reason: str,
    uuid: str,
    parent_uuid: str,
    timestamp: str = "2026-06-23T13:10:05.964Z",
) -> dict:
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": content,
            "stop_reason": stop_reason,
        },
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "timestamp": timestamp,
    }


@pytest.fixture
def extractor() -> HistoryExtractor:
    return HistoryExtractor()


def test_single_simple_turn(extractor: HistoryExtractor):
    """单轮简单对话提炼出 1 个完整 Turn。"""
    records = [
        _human_record("帮我看一下磁盘大小", uuid="u1"),
        _assistant_record(
            [{"type": "text", "text": "当前磁盘使用情况如下..."}],
            stop_reason="end_turn",
            uuid="a1",
            parent_uuid="u1",
        ),
    ]
    turns = extractor.extract_turns(records)
    assert len(turns) == 1
    turn = turns[0]
    assert turn.turn_id == "u1"
    assert turn.question == "帮我看一下磁盘大小"
    assert turn.answer == "当前磁盘使用情况如下..."
    assert turn.is_complete is True
    assert turn.completed_at == "2026-06-23T13:10:05.964Z"


def test_multiple_turns(extractor: HistoryExtractor):
    """多轮连续对话，边界正确。"""
    records = [
        _human_record("问题一", uuid="u1"),
        _assistant_record(
            [{"type": "text", "text": "回答一"}],
            stop_reason="end_turn",
            uuid="a1",
            parent_uuid="u1",
        ),
        _human_record("问题二", uuid="u2"),
        _assistant_record(
            [{"type": "text", "text": "回答二"}],
            stop_reason="end_turn",
            uuid="a2",
            parent_uuid="u2",
        ),
    ]
    turns = extractor.extract_turns(records)
    assert len(turns) == 2
    assert turns[0].question == "问题一"
    assert turns[0].answer == "回答一"
    assert turns[1].question == "问题二"
    assert turns[1].answer == "回答二"


def test_incremental_extraction_across_scans(extractor: HistoryExtractor):
    """一轮跨 4 次扫描，只在最终 end_turn 时输出。"""
    # 第 1 次扫描：human user
    t1 = extractor.extract_turns([_human_record("查磁盘", uuid="u1")])
    assert len(t1) == 0

    # 第 2 次扫描：thinking + tool_use
    t2 = extractor.extract_turns(
        [
            _assistant_record(
                [
                    {"type": "thinking", "thinking": "需要查看磁盘"},
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "Bash",
                        "input": {"command": "df -h"},
                    },
                ],
                stop_reason="tool_use",
                uuid="a1",
                parent_uuid="u1",
                timestamp="2026-06-23T13:10:00.000Z",
            )
        ]
    )
    assert len(t2) == 0

    # 第 3 次扫描：tool_result
    t3 = extractor.extract_turns(
        [_tool_result_record("tool_1", "Filesystem Size 10G", uuid="r1", parent_uuid="a1")]
    )
    assert len(t3) == 0

    # 第 4 次扫描：最终 text end_turn
    t4 = extractor.extract_turns(
        [
            _assistant_record(
                [{"type": "text", "text": "磁盘剩余 5G"}],
                stop_reason="end_turn",
                uuid="a2",
                parent_uuid="r1",
            )
        ]
    )
    assert len(t4) == 1
    turn = t4[0]
    assert turn.is_complete is True
    assert turn.answer == "磁盘剩余 5G"
    assert turn.thinking == ["需要查看磁盘"]
    assert len(turn.tools) == 1
    assert turn.tools[0].name == "Bash"
    assert turn.tools[0].input["command"] == "df -h"
    assert turn.tools[0].result == "Filesystem Size 10G"


def test_consecutive_human_inputs_force_close(extractor: HistoryExtractor):
    """用户连续发多条问题无 assistant 回复，旧轮强制结束。"""
    records = [
        _human_record("问题一", uuid="u1"),
        _human_record("问题二", uuid="u2"),
        _assistant_record(
            [{"type": "text", "text": "回答二"}],
            stop_reason="end_turn",
            uuid="a2",
            parent_uuid="u2",
        ),
    ]
    turns = extractor.extract_turns(records)
    assert len(turns) == 2
    assert turns[0].turn_id == "u1"
    assert turns[0].is_complete is False
    assert turns[0].answer == ""
    assert turns[1].turn_id == "u2"
    assert turns[1].is_complete is True
    assert turns[1].answer == "回答二"


def test_multiple_tool_uses(extractor: HistoryExtractor):
    """assistant 连续多次 tool_use，最终 end_turn 后输出。"""
    records = [
        _human_record("分析日志和配置", uuid="u1"),
        _assistant_record(
            [
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "Read",
                    "input": {"file": "/var/log/app.log"},
                }
            ],
            stop_reason="tool_use",
            uuid="a1",
            parent_uuid="u1",
            timestamp="2026-06-23T13:10:01.000Z",
        ),
        _tool_result_record(
            "tool_1", "log content", uuid="r1", parent_uuid="a1"
        ),
        _assistant_record(
            [
                {
                    "type": "tool_use",
                    "id": "tool_2",
                    "name": "Read",
                    "input": {"file": "/etc/app.conf"},
                }
            ],
            stop_reason="tool_use",
            uuid="a2",
            parent_uuid="r1",
            timestamp="2026-06-23T13:10:02.000Z",
        ),
        _tool_result_record(
            "tool_2", "config content", uuid="r2", parent_uuid="a2"
        ),
        _assistant_record(
            [{"type": "text", "text": "分析完成"}],
            stop_reason="end_turn",
            uuid="a3",
            parent_uuid="r2",
        ),
    ]
    turns = extractor.extract_turns(records)
    assert len(turns) == 1
    turn = turns[0]
    assert turn.is_complete is True
    assert len(turn.tools) == 2
    assert turn.tools[0].name == "Read"
    assert turn.tools[0].result == "log content"
    assert turn.tools[1].name == "Read"
    assert turn.tools[1].result == "config content"


def test_missing_tool_result_keeps_pending(extractor: HistoryExtractor):
    """工具结果缺失，Turn 保持 pending，不输出。"""
    records = [
        _human_record("查磁盘", uuid="u1"),
        _assistant_record(
            [
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "Bash",
                    "input": {"command": "df -h"},
                }
            ],
            stop_reason="tool_use",
            uuid="a1",
            parent_uuid="u1",
        ),
        # 没有 tool_result，也没有最终 end_turn
    ]
    turns = extractor.extract_turns(records)
    assert len(turns) == 0


def test_ignore_non_participant_records(extractor: HistoryExtractor):
    """mode、system 等非参与者记录被忽略。"""
    records = [
        {"type": "mode", "value": "agent"},
        {"type": "system", "content": {"turn_duration": 1000}},
        _human_record("hello", uuid="u1"),
        _assistant_record(
            [{"type": "text", "text": "hi"}],
            stop_reason="end_turn",
            uuid="a1",
            parent_uuid="u1",
        ),
    ]
    turns = extractor.extract_turns(records)
    assert len(turns) == 1
    assert turns[0].question == "hello"
    assert turns[0].answer == "hi"


def test_thinking_only_end_turn_waits_for_text(extractor: HistoryExtractor):
    """真实数据中 thinking-only 的 assistant 消息也可能标记 end_turn，应等待含 text 的消息。"""
    records = [
        _human_record("hi", uuid="u1"),
        _assistant_record(
            [{"type": "thinking", "thinking": "The user said hi."}],
            stop_reason="end_turn",
            uuid="a1",
            parent_uuid="u1",
        ),
        _assistant_record(
            [{"type": "text", "text": "Hi there!"}],
            stop_reason="end_turn",
            uuid="a2",
            parent_uuid="a1",
        ),
    ]
    turns = extractor.extract_turns(records)
    assert len(turns) == 1
    turn = turns[0]
    assert turn.is_complete is True
    assert turn.answer == "Hi there!"
    assert turn.thinking == ["The user said hi."]


def test_tool_result_arrives_before_end_turn(extractor: HistoryExtractor):
    """tool_result 在 tool_use 之后、end_turn 之前到达，结果能正确匹配。"""
    records = [
        _human_record("查磁盘", uuid="u1"),
        _assistant_record(
            [
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "Bash",
                    "input": {"command": "df -h"},
                }
            ],
            stop_reason="tool_use",
            uuid="a1",
            parent_uuid="u1",
        ),
        _tool_result_record("tool_1", "Filesystem Size 10G", uuid="r1", parent_uuid="a1"),
        _assistant_record(
            [{"type": "text", "text": "磁盘剩余 5G"}],
            stop_reason="end_turn",
            uuid="a2",
            parent_uuid="r1",
        ),
    ]
    turns = extractor.extract_turns(records)
    assert len(turns) == 1
    assert turns[0].tools[0].result == "Filesystem Size 10G"
    assert turns[0].answer == "磁盘剩余 5G"
