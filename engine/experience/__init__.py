"""Hermes 自演进引擎 — 经验积累、技能提取、自动优化。"""

from engine.experience.store import ExperienceStore

__all__ = [
    "ExperienceStore",
]


def create_memory_engine(store: ExperienceStore = None):
    """延迟创建 MemoryEngine（避免导入时触发全栈 app 依赖）。"""
    from engine.experience.memory import MemoryEngine
    if store is None:
        store = ExperienceStore()
    return MemoryEngine(store)


def create_skill_engine(store: ExperienceStore = None):
    """延迟创建 SkillEngine。"""
    from engine.experience.skill import SkillEngine
    if store is None:
        store = ExperienceStore()
    return SkillEngine(store)


def create_nudge_engine(store: ExperienceStore = None):
    """延迟创建 NudgeEngine。"""
    from engine.experience.nudge import NudgeEngine
    if store is None:
        store = ExperienceStore()
    return NudgeEngine(store)

