"""Test doc-agent tools — root resolution, environment check, manifest parsing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.tools.doc_tools import (
    _detect_git_remote_url,
    _get_doc_agent_url,
    manifest_has_qwen35_27b_entry,
    resolve_doc_agent_root,
)


class TestResolveDocAgentRoot:
    def test_env_var_used(self):
        with patch("os.environ.get", return_value=r"C:\custom\doc-agent"):
            root = resolve_doc_agent_root()
            assert "doc-agent" in str(root)

    def test_env_var_empty_falls_back(self):
        with patch("os.environ.get", return_value=""), patch("pathlib.Path.is_dir", return_value=False):
            root = resolve_doc_agent_root()
            assert root is not None


class TestManifestHasQwen35:
    def test_manifest_not_found(self, tmp_path):
        assert manifest_has_qwen35_27b_entry(tmp_path / "nonexistent.yaml") is False

    def test_manifest_with_entry(self, tmp_path):
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("entries:\n  - model: Qwen3.5-27B\n", encoding="utf-8")
        assert manifest_has_qwen35_27b_entry(manifest) is True

    def test_manifest_without_entry(self, tmp_path):
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("entries: []\n", encoding="utf-8")
        assert manifest_has_qwen35_27b_entry(manifest) is False

    def test_yaml_parse_error_falls_back(self, tmp_path):
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("qwen3.5-27b: [[invalid", encoding="utf-8")
        result = manifest_has_qwen35_27b_entry(manifest)
        assert result is True


class TestDetectGitRemote:
    def test_no_git_dir(self, tmp_path):
        result = _detect_git_remote_url(tmp_path / "nonexistent")
        assert result == ""

    def test_config_with_gitcode_url(self, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        config = git_dir / "config"
        config.write_text(
            '[remote "origin"]\n\turl = https://gitcode.com/user/repo.git\n',
            encoding="utf-8",
        )
        url = _detect_git_remote_url(tmp_path)
        assert "gitcode.com" in url
        assert not url.endswith(".git")


class TestGetDocAgentUrl:
    def test_with_repo_url(self, tmp_path):
        with patch("os.environ.get", return_value="https://gitcode.com/user/repo"):
            doc_root = tmp_path / "doc-agent"
            doc_root.mkdir()
            file_path = doc_root / "models/vllm-ascend/test.md"
            file_path.parent.mkdir(parents=True)
            file_path.write_text("test")
            url = _get_doc_agent_url(doc_root, file_path)
        assert "blob/main" in url
        assert "test.md" in url
