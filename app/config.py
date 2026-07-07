from app._compat import alias_module

_module = alias_module(__name__, "server.config")
globals().update(_module.__dict__)

