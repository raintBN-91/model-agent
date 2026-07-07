from app._compat import alias_module

_module = alias_module(__name__, "engine.claude_tools")
globals().update(_module.__dict__)

