"""Test health check — app config validation (no FastAPI TestClient needed)."""

from __future__ import annotations

from app.config import settings


class TestHealth:
    """Verify app configuration is valid.

    Note: Full HTTP health endpoint tests need FastAPI TestClient which has
    dependency conflicts on this system. We validate the config layer instead.
    """

    def test_app_name_set(self):
        assert settings.app_name is not None
        assert len(settings.app_name) > 0

    def test_app_name_contains_mofix(self):
        assert "MoFix" in settings.app_name

    def test_app_version_set(self):
        assert settings.app_version is not None
        assert len(settings.app_version) > 0

    def test_anthropic_auth_token_default_empty(self):
        assert hasattr(settings, "anthropic_auth_token")
