from app._compat import alias_module

_module = alias_module(__name__, "server.services.claude_history_extractor")
globals().update(_module.__dict__)

