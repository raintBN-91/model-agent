"""MoFix Server — unified FastAPI application for Ascend model-agent.

Routes:
    /health, /system  — health checks and system info
    /chat             — agent chat interface
    /config           — configuration management
    /eval             — skill quality evaluation (from ascend-skills-eval)
    /evaluate-repo    — remote repo skill evaluation
    /render-card      — scorecard rendering
"""

from server.main import create_app

__all__ = ["create_app"]
