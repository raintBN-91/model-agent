from __future__ import annotations

import json
from pathlib import Path

from engine.skills.registry import SkillRegistry


def test_skill_registry_preserves_tiers_json_counts():
    data = json.loads(Path("skills/tiers.json").read_text(encoding="utf-8"))
    registry = SkillRegistry()
    registry.load()

    assert registry.count["tier1"] == len(data["tier1"])
    assert registry.count["tier2"] == len(data["tier2"])
    assert registry.count["tier3"] == len(data["tier3"])


def test_skill_registry_does_not_promote_tier3_skill_to_tier1():
    registry = SkillRegistry()
    registry.load()

    assert registry.get_tier("ascendc-runtime-debug") == 3
