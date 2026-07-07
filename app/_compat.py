"""旧模块路径别名工具。"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


def alias_module(alias: str, target: str) -> ModuleType:
    module = importlib.import_module(target)
    sys.modules[alias] = module
    parent_name, _, child_name = alias.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None and child_name:
        setattr(parent, child_name, module)
    return module

