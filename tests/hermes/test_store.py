"""Test Hermes Experience Store — init, memory counting, insights."""

from __future__ import annotations


class TestExperienceStore:
    def test_store_initialization(self, tmp_path):
        data_dir = tmp_path / "experience"
        from app.tools.experience.store import ExperienceStore
        store = ExperienceStore(data_dir=data_dir)
        assert store is not None
        assert data_dir.is_dir()

    def test_count_memories_empty(self, tmp_path):
        data_dir = tmp_path / "experience"
        from app.tools.experience.store import ExperienceStore
        store = ExperienceStore(data_dir=data_dir)
        assert store.count_memories() == 0

    def test_append_and_count_memories(self, tmp_path):
        data_dir = tmp_path / "experience"
        from app.tools.experience.store import ExperienceStore
        store = ExperienceStore(data_dir=data_dir)
        store.append_memory({"event": "test", "level": "info"})
        assert store.count_memories() == 1

    def test_get_pending_insights_empty(self, tmp_path):
        data_dir = tmp_path / "experience"
        from app.tools.experience.store import ExperienceStore
        store = ExperienceStore(data_dir=data_dir)
        assert store.get_pending_insights() == []

    def test_append_and_retrieve_insights(self, tmp_path):
        data_dir = tmp_path / "experience"
        from app.tools.experience.store import ExperienceStore
        store = ExperienceStore(data_dir=data_dir)
        store.append_insight({"content": "test insight", "category": "improvement"})
        insights = store.get_pending_insights()
        assert len(insights) == 1
        assert insights[0]["content"] == "test insight"

    def test_get_memories(self, tmp_path):
        data_dir = tmp_path / "experience"
        from app.tools.experience.store import ExperienceStore
        store = ExperienceStore(data_dir=data_dir)
        store.append_memory({"event": "first"})
        store.append_memory({"event": "second"})
        memories = store.get_memories()
        assert len(memories) == 2

    def test_save_and_get_skill(self, tmp_path):
        data_dir = tmp_path / "experience"
        from app.tools.experience.store import ExperienceStore
        store = ExperienceStore(data_dir=data_dir)
        store.save_skill({"name": "test-skill", "description": "a test"})
        skill = store.get_skill("test-skill")
        assert skill is not None
        assert skill["description"] == "a test"
