from app._compat import alias_module

_module = alias_module(__name__, "server.models.chat")
globals().update(_module.__dict__)

