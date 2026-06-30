"""测试 Claude Code 历史记录监控服务。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.services import claude_history_uploader as uploader
from app.services.claude_history_uploader import HistoryWatcher


@pytest.fixture
def watcher(monkeypatch, tmp_path):
    """创建已启用的 HistoryWatcher，监控临时 projects 根目录。"""
    monkeypatch.setattr(
        "app.services.claude_history_uploader.settings.claude_history_enabled",
        True,
    )
    monkeypatch.setattr(
        "app.services.claude_history_uploader.settings.claude_history_project_dir",
        "",
    )
    monkeypatch.setattr(
        uploader,
        "get_claude_projects_root",
        lambda: tmp_path,
    )
    return HistoryWatcher(interval=3600)


async def _run_scan_once(watcher: HistoryWatcher) -> None:
    """手动触发一次扫描。"""
    await watcher._scan_once()


def _human_record(text: str, uuid: str = "user-uuid") -> dict:
    return {
        "type": "user",
        "origin": {"kind": "human"},
        "message": {"role": "user", "content": text},
        "uuid": uuid,
        "parentUuid": "",
        "timestamp": "2026-06-23T13:09:54.017Z",
    }


def _assistant_record(
    text: str,
    stop_reason: str = "end_turn",
    uuid: str = "assistant-uuid",
    parent_uuid: str = "user-uuid",
) -> dict:
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
        },
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "timestamp": "2026-06-23T13:10:05.964Z",
    }


def test_derive_project_dir_name_linux():
    """测试 Linux 下自动推导项目目录名。"""
    assert uploader._derive_project_dir_name(Path("/home/user/mofix-new")) == "home-user-mofix-new"
    assert uploader._derive_project_dir_name(Path("/root/project")) == "root-project"


def test_get_watched_project_dirs_all(monkeypatch, tmp_path):
    """未配置 project_dir 时监控根目录下所有子目录。"""
    monkeypatch.setattr(
        "app.services.claude_history_uploader.settings.claude_history_project_dir",
        "",
    )
    monkeypatch.setattr(uploader, "get_claude_projects_root", lambda: tmp_path)
    (tmp_path / "project-a").mkdir()
    (tmp_path / "project-b").mkdir()
    (tmp_path / "not-a-dir.txt").write_text("", encoding="utf-8")

    dirs = uploader.get_watched_project_dirs()
    assert sorted(d.name for d in dirs) == ["project-a", "project-b"]


def test_get_watched_project_dirs_filtered(monkeypatch, tmp_path):
    """配置 project_dir 时只监控指定目录。"""
    monkeypatch.setattr(
        "app.services.claude_history_uploader.settings.claude_history_project_dir",
        "project-a",
    )
    monkeypatch.setattr(uploader, "get_claude_projects_root", lambda: tmp_path)
    (tmp_path / "project-a").mkdir()
    (tmp_path / "project-b").mkdir()

    dirs = uploader.get_watched_project_dirs()
    assert [d.name for d in dirs] == ["project-a"]


@pytest.mark.anyio
async def test_first_scan_records_offset_no_upload(watcher, tmp_path):
    """首次扫描只记录偏移量，不触发上传。"""
    project_dir = tmp_path / "project-a"
    project_dir.mkdir()
    history_file = project_dir / "session-1.jsonl"
    history_file.write_text(
        json.dumps(_human_record("hello")) + "\n", encoding="utf-8"
    )

    with patch.object(HistoryWatcher, "_upload_turn", new_callable=AsyncMock) as mock_upload:
        await _run_scan_once(watcher)
        assert watcher._offsets[history_file] == 1
        mock_upload.assert_not_called()


@pytest.mark.anyio
async def test_upload_placeholder_called_for_new_records(watcher, tmp_path):
    """新增完整轮次会触发上传占位函数。"""
    project_dir = tmp_path / "project-a"
    project_dir.mkdir()
    history_file = project_dir / "session-1.jsonl"
    history_file.write_text("", encoding="utf-8")

    # 首次发现空文件，记录偏移量 0
    with patch.object(HistoryWatcher, "_upload_turn", new_callable=AsyncMock) as mock_upload:
        await _run_scan_once(watcher)
        assert watcher._offsets.get(history_file, 0) == 0
        mock_upload.assert_not_called()

    # 追加人类提问
    history_file.write_text(
        json.dumps(_human_record("hello")) + "\n", encoding="utf-8"
    )

    with patch.object(HistoryWatcher, "_upload_turn", new_callable=AsyncMock) as mock_upload:
        await _run_scan_once(watcher)
        assert watcher._offsets[history_file] == 1
        mock_upload.assert_not_called()

    # 追加助手最终回复
    history_file.write_text(
        json.dumps(_human_record("hello")) + "\n"
        + json.dumps(_assistant_record("hi")) + "\n",
        encoding="utf-8",
    )

    with patch.object(HistoryWatcher, "_upload_turn", new_callable=AsyncMock) as mock_upload:
        await _run_scan_once(watcher)
        assert watcher._offsets[history_file] == 2
        mock_upload.assert_called_once()
        call_args = mock_upload.call_args
        assert call_args.args[0] == history_file
        assert call_args.args[1].question == "hello"
        assert call_args.args[1].answer == "hi"
        assert call_args.args[1].is_complete is True


@pytest.mark.anyio
async def test_new_session_first_round_recognized(watcher, tmp_path):
    """新 session 文件首次发现时已含 human 提问，后续 assistant 回复应能完成该轮次。"""
    project_dir = tmp_path / "project-a"
    project_dir.mkdir()
    history_file = project_dir / "session-new.jsonl"

    # 首次发现文件时，文件中已有一条人类提问（模拟 Claude Code 新建会话后立即提问）
    history_file.write_text(
        json.dumps(_human_record("hello", uuid="user-new")) + "\n",
        encoding="utf-8",
    )

    with patch.object(HistoryWatcher, "_upload_turn", new_callable=AsyncMock) as mock_upload:
        await _run_scan_once(watcher)
        assert watcher._offsets[history_file] == 1
        # 首次不输出已完成轮次，但 human 应已进入 pending
        mock_upload.assert_not_called()

    # 追加助手最终回复
    history_file.write_text(
        json.dumps(_human_record("hello", uuid="user-new")) + "\n"
        + json.dumps(
            _assistant_record("hi", uuid="assistant-new", parent_uuid="user-new")
        ) + "\n",
        encoding="utf-8",
    )

    with patch.object(HistoryWatcher, "_upload_turn", new_callable=AsyncMock) as mock_upload:
        await _run_scan_once(watcher)
        assert watcher._offsets[history_file] == 2
        mock_upload.assert_called_once()
        turn = mock_upload.call_args.args[1]
        assert turn.question == "hello"
        assert turn.answer == "hi"
        assert turn.is_complete is True


@pytest.mark.anyio
async def test_monitor_multiple_project_dirs(watcher, tmp_path):
    """同时监控多个项目目录下的新增记录。"""
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    project_a.mkdir()
    project_b.mkdir()

    file_a = project_a / "session-a.jsonl"
    file_b = project_b / "session-b.jsonl"
    file_a.write_text("", encoding="utf-8")
    file_b.write_text("", encoding="utf-8")

    # 首次发现空文件
    await _run_scan_once(watcher)

    # 同时写入人类提问
    file_a.write_text(
        json.dumps(_human_record("hello a", uuid="user-a")) + "\n",
        encoding="utf-8",
    )
    file_b.write_text(
        json.dumps(_human_record("hello b", uuid="user-b")) + "\n",
        encoding="utf-8",
    )

    await _run_scan_once(watcher)
    assert watcher._offsets[file_a] == 1
    assert watcher._offsets[file_b] == 1

    # 同时追加助手最终回复
    file_a.write_text(
        json.dumps(_human_record("hello a", uuid="user-a")) + "\n"
        + json.dumps(
            _assistant_record("hi a", uuid="assistant-a", parent_uuid="user-a")
        ) + "\n",
        encoding="utf-8",
    )
    file_b.write_text(
        json.dumps(_human_record("hello b", uuid="user-b")) + "\n"
        + json.dumps(
            _assistant_record("hi b", uuid="assistant-b", parent_uuid="user-b")
        ) + "\n",
        encoding="utf-8",
    )

    with patch.object(HistoryWatcher, "_upload_turn", new_callable=AsyncMock) as mock_upload:
        await _run_scan_once(watcher)
        assert watcher._offsets[file_a] == 2
        assert watcher._offsets[file_b] == 2
        assert mock_upload.call_count == 2


@pytest.mark.anyio
async def test_incomplete_last_line_is_skipped(watcher, tmp_path):
    """未以换行符结尾的半行不计入偏移量，也不上传。"""
    project_dir = tmp_path / "project-a"
    project_dir.mkdir()
    history_file = project_dir / "session-1.jsonl"
    history_file.write_text("", encoding="utf-8")

    # 首次发现空文件
    await _run_scan_once(watcher)

    # 模拟写入中的半行：第二行没有 \n
    history_file.write_text(
        json.dumps(_human_record("hello")) + "\n"
        + json.dumps({"type": "assistant", "msg": "hi"}),  # 半行
        encoding="utf-8",
    )

    with patch.object(HistoryWatcher, "_upload_turn", new_callable=AsyncMock) as mock_upload:
        await _run_scan_once(watcher)
        # 只有第一行是完整行
        assert watcher._offsets[history_file] == 1
        mock_upload.assert_not_called()

    # 补全第二行
    history_file.write_text(
        json.dumps(_human_record("hello")) + "\n"
        + json.dumps(_assistant_record("hi")) + "\n",
        encoding="utf-8",
    )

    with patch.object(HistoryWatcher, "_upload_turn", new_callable=AsyncMock) as mock_upload:
        await _run_scan_once(watcher)
        assert watcher._offsets[history_file] == 2
        mock_upload.assert_called_once()
        assert mock_upload.call_args.args[1].answer == "hi"


@pytest.mark.anyio
async def test_truncated_file_resets_offset(watcher, tmp_path):
    """文件被截断时重置偏移量。"""
    project_dir = tmp_path / "project-a"
    project_dir.mkdir()
    history_file = project_dir / "session-1.jsonl"

    # 首次发现文件并扫描完整轮次
    history_file.write_text(
        json.dumps(_human_record("hello")) + "\n"
        + json.dumps(_assistant_record("hi")) + "\n",
        encoding="utf-8",
    )
    await _run_scan_once(watcher)
    assert watcher._offsets[history_file] == 2

    # 截断文件
    history_file.write_text(
        json.dumps(_human_record("hello")) + "\n", encoding="utf-8"
    )
    await _run_scan_once(watcher)

    assert watcher._offsets[history_file] == 1


@pytest.mark.anyio
async def test_directory_not_exists(watcher):
    """目录不存在时不崩溃。"""
    watcher._project_dirs = [Path("/nonexistent/claude/projects")]
    with patch.object(HistoryWatcher, "_upload_turn", new_callable=AsyncMock) as mock_upload:
        await _run_scan_once(watcher)
        mock_upload.assert_not_called()
