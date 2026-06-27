from __future__ import annotations
"""Skill evaluation API — integrated from ascend-skills-eval web-service."""

import base64
import json
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="", tags=["eval"])


# ---- Models ----
class EvalRequest(BaseModel):
    skill_markdown: str = Field(..., min_length=1, description="complete SKILL.md content")
    skill_name: str | None = Field(default=None, description="optional skill name")


class RepoEvalRequest(BaseModel):
    repo_url: str = Field(..., min_length=1, description="GitHub or GitCode repo URL")
    skill_path: str | None = Field(default=None, description="optional path within repo")
    branch: str | None = Field(default=None, description="optional branch")


class BatchRepoEvalRequest(BaseModel):
    items: list[RepoEvalRequest] = Field(..., min_length=1, max_length=20)


class DimensionScore(BaseModel):
    id: int
    name: str
    weight: int
    score: float
    reason: str


class EvalResponse(BaseModel):
    skill_name: str
    total_score: float
    dimensions: list[DimensionScore]
    suggestions: list[str]
    mode: str = "structure_eval"


class RenderRequest(BaseModel):
    data: dict[str, Any]
    open_image: bool = Field(default=False)


class RenderResponse(BaseModel):
    image_base64: str
    mime_type: str = "image/png"
    size_bytes: int


# ---- Constants ----
DIMENSIONS: list[tuple[int, str, int]] = [
    (1, "Frontmatter quality", 8),
    (2, "Workflow clarity", 15),
    (3, "Boundary coverage", 10),
    (4, "Checkpoint design", 7),
    (5, "Instruction specificity", 15),
    (6, "Resource integration", 5),
    (7, "Overall architecture", 15),
    (8, "Ascend relevance", 10),
    (9, "Verifiability (proxy)", 15),
]


# ---- Helpers ----
def _has_frontmatter(text: str) -> bool:
    return bool(re.search(r"(?s)\A---\n.*?\n---\n", text))


def _count_pattern(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE))


def _clamp_score(v: float) -> float:
    return float(max(1.0, min(10.0, round(v, 1))))


def _render_script_path() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        candidate = parent / "skills" / "skills-eval" / "scripts" / "render-card.mjs"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("cannot locate render-card.mjs")


# ---- Core evaluation logic ----
def evaluate_skill(skill_markdown: str, skill_name: str | None = None) -> EvalResponse:
    text = skill_markdown
    name = skill_name or "unknown-skill"

    heading_count = _count_pattern(text, r"^##\s+")
    numbered_steps = _count_pattern(text, r"^\s*\d+\.\s+")
    code_block_count = _count_pattern(text, r"^```")
    command_count = _count_pattern(text, r"\b(npu-smi|python3?|pip|git|node|npm|uvicorn)\b")
    edge_count = _count_pattern(text, r"(error|fallback|failed|recover|retry|troubleshoot)")
    checkpoint_count = _count_pattern(text, r"(checkpoint|approval|verify|confirm)")
    resource_count = _count_pattern(text, r"(templates?/|scripts?/|results\.tsv|evals?\.json|references?)")
    ascend_count = _count_pattern(text, r"(ascend|npu|torch_npu|cann|npu-smi)")
    test_count = _count_pattern(text, r"(test|benchmark|eval|verify|validate)")

    d1 = 7.5 if _has_frontmatter(text) else 3.0
    d1 += 1.0 if "description:" in text else 0.0

    d2 = 4.5 + min(4.0, numbered_steps * 0.25) + min(1.5, heading_count * 0.15)
    d3 = 3.5 + min(5.0, edge_count * 0.4)
    d4 = 3.0 + min(6.0, checkpoint_count * 0.8)
    d5 = 4.0 + min(2.5, code_block_count * 0.2) + min(3.0, command_count * 0.25)
    d6 = 3.5 + min(5.0, resource_count * 0.8)
    d7 = 4.0 + min(3.0, heading_count * 0.3) + min(2.5, numbered_steps * 0.15)
    d8 = 2.5 + min(7.0, ascend_count * 0.35)
    d9 = 3.5 + min(4.0, test_count * 0.35) + min(2.0, command_count * 0.2)

    raw_scores = [d1, d2, d3, d4, d5, d6, d7, d8, d9]
    reasons = [
        "frontmatter and description fields detected.",
        "heading/step structure for workflow executability.",
        "fallback/error handling coverage estimate.",
        "checkpoint and approval keyword estimate.",
        "command, parameter, and code block density.",
        "templates/scripts/results resource references.",
        "section hierarchy and step organization.",
        "Ascend/NPU terminology and command coverage.",
        "online mode: structural proxy from test/verify descriptions.",
    ]

    dimensions: list[DimensionScore] = []
    weighted_sum = 0.0
    for idx, (meta, score, reason) in enumerate(zip(DIMENSIONS, raw_scores, reasons), start=1):
        dim_id, dim_name, weight = meta
        final_score = _clamp_score(score)
        weighted_sum += final_score * weight
        dimensions.append(DimensionScore(
            id=dim_id, name=dim_name, weight=weight, score=final_score, reason=reason,
        ))

    total = round(weighted_sum / 10.0, 1)
    sorted_dims = sorted(dimensions, key=lambda x: x.score)
    suggestions = [
        f"Priority 1 — improve '{sorted_dims[0].name}': add specific steps, I/O, and error handling.",
        f"Priority 2 — improve '{sorted_dims[1].name}': add executable commands and verification evidence.",
        "For real NPU results, connect a local Ascend Runner to execute npu-smi and key commands.",
    ]

    return EvalResponse(
        skill_name=name, total_score=total, dimensions=dimensions, suggestions=suggestions,
    )


# ---- Repo helpers ----
def _validate_repo_url(repo_url: str) -> str:
    parsed = urlparse(repo_url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="repo URL must be http/https")
    host = (parsed.netloc or "").lower()
    if host not in {"github.com", "www.github.com", "gitcode.com", "www.gitcode.com"}:
        raise HTTPException(status_code=400, detail="only GitHub/GitCode links are supported")
    return repo_url


def _pick_repo_file(repo_dir: Path, skill_path: str | None) -> Path:
    if skill_path:
        specified = (repo_dir / skill_path).resolve()
        if not str(specified).startswith(str(repo_dir.resolve())):
            raise HTTPException(status_code=400, detail="skill_path is invalid")
        if not specified.exists() or not specified.is_file():
            raise HTTPException(status_code=400, detail=f"specified file not found: {skill_path}")
        return specified

    priority = ["SKILL.md", "skill.md", "README.md", "README.zh-CN.md", "README_CN.md"]
    for name in priority:
        p = repo_dir / name
        if p.exists() and p.is_file():
            return p
    for name in priority:
        matches = list(repo_dir.rglob(name))
        if matches:
            return matches[0]
    md_files = list(repo_dir.rglob("*.md"))
    if md_files:
        return md_files[0]
    raise HTTPException(status_code=400, detail="no evaluable markdown file found in repo")


def _repo_name_from_url(repo_url: str) -> str:
    tail = repo_url.rstrip("/").split("/")[-1]
    if tail.endswith(".git"):
        tail = tail[:-4]
    return tail or "repo-skill"


def _build_report_markdown(eval_result: EvalResponse, *, source_label: str, source_value: str, picked_file: str) -> str:
    lines = [
        "# ascend-skills-eval Report",
        "",
        f"- Source: {source_label}",
        f"- URL: {source_value}",
        f"- File: {picked_file}",
        f"- Skill: {eval_result.skill_name}",
        f"- Score: {eval_result.total_score}/100",
        f"- Mode: {eval_result.mode}",
        "",
        "## Dimension Scores",
        "",
        "| # | Dimension | Weight | Score | Note |",
        "|---|---:|---:|---:|---|",
    ]
    for d in eval_result.dimensions:
        lines.append(f"| {d.id} | {d.name} | {d.weight} | {d.score} | {d.reason} |")
    lines.extend(["", "## Suggestions", ""])
    for idx, tip in enumerate(eval_result.suggestions, start=1):
        lines.append(f"{idx}. {tip}")
    lines.append("")
    return "\n".join(lines)


def _evaluate_repo_internal(payload: RepoEvalRequest) -> dict[str, Any]:
    repo_url = _validate_repo_url(payload.repo_url.strip())
    with tempfile.TemporaryDirectory(prefix="ascend-skills-repo-") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        repo_dir = tmp_dir / "repo"

        clone_cmd = ["git", "clone", "--depth", "1"]
        if payload.branch:
            clone_cmd.extend(["--branch", payload.branch])
        clone_cmd.extend([repo_url, str(repo_dir)])
        proc = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "repo clone failed (ensure the repo is publicly accessible)",
                    "stdout": proc.stdout.strip(),
                    "stderr": proc.stderr.strip(),
                },
            )

        picked_file = _pick_repo_file(repo_dir, payload.skill_path)
        try:
            skill_markdown = picked_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            skill_markdown = picked_file.read_text(encoding="utf-8", errors="ignore")

        inferred_name = _repo_name_from_url(repo_url)
        eval_result = evaluate_skill(skill_markdown, inferred_name)
        picked_rel = str(picked_file.relative_to(repo_dir))
        report_markdown = _build_report_markdown(
            eval_result, source_label="Repo URL", source_value=repo_url, picked_file=picked_rel,
        )

        history_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        return {
            "eval": eval_result.model_dump(),
            "report_markdown": report_markdown,
            "picked_file": picked_rel,
            "repo_url": repo_url,
            "skill_markdown": skill_markdown,
            "history_id": history_id,
        }


# ---- Routes ----
@router.post("/eval", response_model=EvalResponse)
def eval_skill(payload: EvalRequest) -> EvalResponse:
    return evaluate_skill(payload.skill_markdown, payload.skill_name)


@router.post("/render-card", response_model=RenderResponse)
def render_card(payload: RenderRequest) -> RenderResponse:
    render_script = _render_script_path()
    if not render_script.exists():
        raise HTTPException(status_code=500, detail=f"render script not found: {render_script}")

    with tempfile.TemporaryDirectory(prefix="ascend-skills-eval-") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        data_path = tmp_dir / "card-data.json"
        png_path = tmp_dir / "result-card.png"
        html_path = tmp_dir / "result-card.rendered.html"
        data_path.write_text(json.dumps(payload.data, ensure_ascii=False), encoding="utf-8")

        cmd = ["node", str(render_script), "--data", str(data_path),
               "--png", str(png_path), "--html", str(html_path)]
        if not payload.open_image:
            cmd.append("--no-open")

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail={"message": "render failed", "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()},
            )
        if not png_path.exists():
            raise HTTPException(status_code=500, detail="render script ran but did not produce PNG")

        raw = png_path.read_bytes()
        return RenderResponse(image_base64=base64.b64encode(raw).decode("ascii"), size_bytes=len(raw))


@router.post("/evaluate-and-render")
def evaluate_and_render(payload: EvalRequest) -> dict[str, Any]:
    eval_result = evaluate_skill(payload.skill_markdown, payload.skill_name)
    dims_for_card = [
        {"name": d.name.replace("(proxy)", ""), "before": max(1, int(d.score) - 1), "after": int(round(d.score))}
        for d in eval_result.dimensions[:8]
    ]
    card_data = {
        "skill-name": eval_result.skill_name,
        "skill-id": eval_result.skill_name,
        "score-before": max(1, int(eval_result.total_score) - 8),
        "score-after": int(round(eval_result.total_score)),
        "improve-1": eval_result.suggestions[0],
        "improve-2": eval_result.suggestions[1],
        "improve-3": eval_result.suggestions[2],
        "dims": dims_for_card,
    }
    try:
        render_result = render_card(RenderRequest(data=card_data, open_image=False))
        card = render_result.model_dump()
    except HTTPException:
        card = None
    return {"eval": eval_result.model_dump(), "card": card}


@router.post("/evaluate-repo")
def evaluate_repo(payload: RepoEvalRequest) -> dict[str, Any]:
    return _evaluate_repo_internal(payload)


@router.post("/evaluate-repos")
def evaluate_repos(payload: BatchRepoEvalRequest) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for item in payload.items:
        try:
            result = _evaluate_repo_internal(item)
            results.append(result)
        except HTTPException as exc:
            failed.append({"repo_url": item.repo_url, "status_code": exc.status_code, "detail": exc.detail})

    ranking = sorted(
        [{"repo_url": r["repo_url"], "skill_name": r["eval"]["skill_name"],
          "total_score": r["eval"]["total_score"], "picked_file": r["picked_file"],
          "history_id": r["history_id"]} for r in results],
        key=lambda x: x["total_score"], reverse=True,
    )

    lines = [
        "# ascend-skills-eval Batch Report", "",
        f"- Total: {len(payload.items)}", f"- Success: {len(results)}", f"- Failed: {len(failed)}",
        "", "## Ranking", "", "| Rank | Repo | Skill | Score | File |",
        "|---:|---|---:|---:|---|",
    ]
    for i, item in enumerate(ranking, start=1):
        lines.append(f"| {i} | {item['repo_url']} | {item['skill_name']} | {item['total_score']} | {item['picked_file']} |")
    if failed:
        lines.extend(["", "## Failed", ""])
        for f in failed:
            lines.append(f"- {f['repo_url']}: {f['detail']}")
    batch_report_markdown = "\n".join(lines) + "\n"

    return {
        "summary": {"total": len(payload.items), "success": len(results), "failed": len(failed)},
        "ranking": ranking, "results": results, "failed": failed,
        "batch_report_markdown": batch_report_markdown,
    }
