from app._compat import alias_module

_module = alias_module(__name__, "engine.doc_tools")
globals().update(_module.__dict__)

