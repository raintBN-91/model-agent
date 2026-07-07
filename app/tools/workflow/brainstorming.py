from app._compat import alias_module

_module = alias_module(__name__, "engine.workflow.brainstorming")
globals().update(_module.__dict__)

