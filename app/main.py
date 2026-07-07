from app._compat import alias_module

_module = alias_module(__name__, "server.main")
globals().update(_module.__dict__)

