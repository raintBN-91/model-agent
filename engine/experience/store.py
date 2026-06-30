"""ExperienceStore — Hermes 三层持久化存储。

存储布局:
  ~/.mofix/experience/
  ├── memory.jsonl          # JSONL 时间线记忆
  ├── skills.json           # 可复用 Skill 列表
  ├── insights.jsonl        # Nudge 改进建议
  └── layers/
      ├── model_adapt.json
      ├── error_recovery.json
      ├── user_prefs.json
      └── mcp_optim.json

原子写入复用 checkpointer.py 的 tmp+rename 模式。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _default_data_dir() -> Path:
    return Path.home() / ".mofix" / "experience"


class ExperienceStore:
    """经验存储 — 管理 memory / skills / insights / layers 的持久化。"""

    def __init__(self, data_dir: str | Path | None = None):
        self._dir = Path(data_dir) if data_dir else _default_data_dir()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._layers_dir = self._dir / "layers"
        self._layers_dir.mkdir(exist_ok=True)

        # 文件路径
        self._memory_file = self._dir / "memory.jsonl"
        self._skills_file = self._dir / "skills.json"
        self._insights_file = self._dir / "insights.jsonl"

    # ── 内部工具 ──

    def _atomic_write(self, path: Path, data: Any) -> None:
        """原子写入：先写 tmp 再 rename（使用 os.replace 确保跨平台）。"""
        import os
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(str(tmp_path), str(path))

    def _read_jsonl(self, path: Path) -> list[dict]:
        if not path.is_file():
            return []
        entries: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def _read_json(self, path: Path) -> list[dict]:
        """读取 JSON 数组文件（非 JSONL）。"""
        if not path.is_file():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            return [data] if isinstance(data, dict) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _append_jsonl(self, path: Path, entry: dict) -> None:
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _trim_jsonl(self, path: Path, max_entries: int) -> int:
        entries = self._read_jsonl(path)
        if len(entries) <= max_entries:
            return 0
        removed = len(entries) - max_entries
        trimmed = entries[-max_entries:]
        self._dir.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for e in trimmed:
                f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
        return removed

    @staticmethod
    def _new_id(prefix: str = "mem") -> str:
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── Memory API ──

    def append_memory(self, entry: dict) -> None:
        if "id" not in entry:
            entry["id"] = self._new_id("mem")
        if "timestamp" not in entry:
            entry["timestamp"] = self._now_iso()
        self._append_jsonl(self._memory_file, entry)

    def get_memories(self, limit: int = 50, offset: int = 0) -> list[dict]:
        entries = self._read_jsonl(self._memory_file)
        return entries[offset: offset + limit]

    def search_memories(self, query: str, top_k: int = 10) -> list[dict]:
        """简单关键词匹配检索。"""
        query_lower = query.lower()
        words = set(query_lower.split())
        entries = self._read_jsonl(self._memory_file)

        scored: list[tuple[int, dict]] = []
        for e in entries:
            text = json.dumps(e, ensure_ascii=False).lower()
            score = sum(1 for w in words if w in text)
            if score > 0:
                scored.append((score, e))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def count_memories(self) -> int:
        return len(self._read_jsonl(self._memory_file))

    def trim_memories(self, max_entries: int = 500) -> int:
        return self._trim_jsonl(self._memory_file, max_entries)

    # ── Skills API ──

    def save_skill(self, skill: dict) -> None:
        skills = self._read_json(self._skills_file)
        # 同名替换
        skills = [s for s in skills if s.get("name") != skill.get("name")]
        skills.append(skill)
        self._atomic_write(self._skills_file, skills)

    def get_all_skills(self) -> list[dict]:
        return self._read_json(self._skills_file)

    def get_skill(self, name: str) -> dict | None:
        for s in self.get_all_skills():
            if s.get("name") == name:
                return s
        return None

    def delete_skill(self, name: str) -> bool:
        skills = self._read_json(self._skills_file)
        filtered = [s for s in skills if s.get("name") != name]
        if len(filtered) == len(skills):
            return False
        self._atomic_write(self._skills_file, filtered)
        return True

    def prune_skills(self, max_skills: int = 50) -> int:
        skills = self._read_json(self._skills_file)
        if max_skills <= 0:
            removed = len(skills)
            self._atomic_write(self._skills_file, [])
            return removed
        if len(skills) <= max_skills:
            return 0
        removed = len(skills) - max_skills
        trimmed = skills[-max_skills:]
        self._atomic_write(self._skills_file, trimmed)
        return removed

    # ── Insights API ──

    def append_insight(self, insight: dict) -> None:
        if "id" not in insight:
            insight["id"] = self._new_id("insight")
        if "timestamp" not in insight:
            insight["timestamp"] = self._now_iso()
        if "applied" not in insight:
            insight["applied"] = False
        self._append_jsonl(self._insights_file, insight)

    def get_insights(self, limit: int = 20) -> list[dict]:
        return self._read_jsonl(self._insights_file)[-limit:]

    def get_pending_insights(self) -> list[dict]:
        return [i for i in self._read_jsonl(self._insights_file) if not i.get("applied")]

    def mark_insight_applied(self, insight_id: str) -> None:
        entries = self._read_jsonl(self._insights_file)
        for e in entries:
            if e.get("id") == insight_id:
                e["applied"] = True
                break
        self._dir.mkdir(parents=True, exist_ok=True)
        with self._insights_file.open("w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")

    def trim_insights(self, max_entries: int = 200) -> int:
        return self._trim_jsonl(self._insights_file, max_entries)

    # ── Layers API ──

    def save_layer_data(self, layer: str, data: dict) -> None:
        """保存 Layer 数据（全量覆盖）。"""
        layer_file = self._layers_dir / f"{layer}.json"
        existing = self.load_layer_data(layer)
        if isinstance(existing, dict):
            existing.update(data)
        else:
            existing = data
        self._atomic_write(layer_file, existing)

    def load_layer_data(self, layer: str) -> dict:
        layer_file = self._layers_dir / f"{layer}.json"
        if layer_file.is_file():
            try:
                return json.loads(layer_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def append_layer_array(self, layer: str, entry: dict) -> None:
        """往 Layer 的数组字段追加条目。"""
        layer_file = self._layers_dir / f"{layer}.json"
        data = self.load_layer_data(layer)
        if "entries" not in data:
            data["entries"] = []
        data["entries"].append(entry)
        self._atomic_write(layer_file, data)

    def count_layer_entries(self, layer: str) -> int:
        data = self.load_layer_data(layer)
        return len(data.get("entries", []))
