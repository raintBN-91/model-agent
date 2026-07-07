from app._compat import alias_module

_module = alias_module(__name__, "engine.experience.store")
globals().update(_module.__dict__)

