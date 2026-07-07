from __future__ import annotations

from fastapi import HTTPException

from server.api.eval import _pick_repo_file, _render_script_path


def test_render_script_path_finds_integrated_ascend_skills_eval_layout():
    path = _render_script_path()

    assert path.name == "render-card.mjs"
    assert "ascend-skills-eval" in path.parts


def test_pick_repo_file_rejects_absolute_sibling_prefix_escape(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    sibling = tmp_path / "repo_evil"
    sibling.mkdir()
    outside_skill = sibling / "SKILL.md"
    outside_skill.write_text("# outside", encoding="utf-8")

    try:
        _pick_repo_file(repo_dir, str(outside_skill))
    except HTTPException as exc:
        assert exc.status_code == 400
        assert exc.detail == "skill_path is invalid"
    else:
        raise AssertionError("absolute sibling path should be rejected")


def test_pick_repo_file_allows_relative_file_inside_repo(tmp_path):
    repo_dir = tmp_path / "repo"
    nested = repo_dir / "nested"
    nested.mkdir(parents=True)
    skill = nested / "SKILL.md"
    skill.write_text("# inside", encoding="utf-8")

    assert _pick_repo_file(repo_dir, "nested/SKILL.md") == skill.resolve()

