"""Scan all SKILL.md, deduplicate, evaluate quality, classify, generate tiers.json."""
import json
import os
import re
from pathlib import Path

ROOT = Path(r"F:\7-workspace\model-agent")
SCAN_ROOTS = [ROOT / "skills", ROOT / "models", ROOT / "small_model_adapt"]

# ---- Scoring config ----
WEIGHTS = {
    "frontmatter": 8,
    "workflow_clarity": 15,
    "boundary_coverage": 10,
    "checkpoint_design": 7,
    "instruction_specificity": 15,
    "resource_integration": 5,
    "overall_architecture": 15,
    "ascend_relevance": 10,
    "verifiability": 15,
}

# ---- Category keywords for classification ----
CATEGORY_KEYWORDS = {
    "adaptation": [
        "adapt", "migration", "migrate", "migrat", "cuda", "compat",
        "convert", "transform", "porting", "移植", "迁移", "适配",
        "model-migration", "model_adapter", "vendor-detector",
        "fsdp2", "megatron-change", "megatron-impact", "megatron-migration",
        "msverl", "regression-triage", "vllm-ascend-model",
        "adapter-check", "hardware-check",
    ],
    "common": [
        "ssh", "tunnel", "debug", "coverage", "test-gen", "unittest",
        "pytest", "refactor", "skill-auditor", "code-comprehension",
        "repo-reader", "auto-bug-fixer", "auto-develop",
        "long-task", "connect", "deploy", "generate-unit-test",
        "analyse-coverage", "python-refactoring", "unittest-writer",
    ],
    "deployment": [
        "deploy", "serve", "server", "install", "docker", "npu-smi",
        "driver", "atc", "om-model", "model-verifier", "model_verifier",
        "inference-repos", "vllm-ascend-performance",
        "部署", "安装", "服务化", "上线",
    ],
    "documentation": [
        "doc", "document", "readme", "docs-gen", "swanlab",
        "ut-develop", "regbase", "triton-operator-doc",
        "npu-arch",
        "文档", "readme", "教程",
    ],
    "optimization": [
        "optim", "tune", "tuning", "profile", "profiling", "perf",
        "precision", "latency", "throughput", "speedup", "加速",
        "优化", "调优", "性能", "ai4s-basic", "ai4s-main",
        "ai4s-perf", "ai4s-precision", "ai4s-profiling",
        "ascendc", "ascend-affinity", "ascend-optimization", "ascend-profiling",
        "ascend-history-to-skill", "ascendc-api", "ascendc-code-review",
        "ascendc-direct", "ascendc-operator-code-gen", "ascendc-operator-design",
        "ascendc-operator-performance", "ascendc-registry",
        "ascendc-runtime-debug", "ascendc-task-focus", "ascendc-tiling",
        "catlass-operator", "model-infer", "npu-adapter-reviewer",
        "ops-profiling", "perf-analyzer", "pypto", "tilelang",
        "triton-operator", "tune-", "vector-triton",
        "verl-async", "vllm-ascend_faq", "simple-vector",
        "deepfri-tf-npu",
    ],
    "pta": [
        "pta", "pre-trained", "pretrained", "adapt-agent",
        "quantify-agent", "optimizer-agent", "verify-agent",
        "ascend-adaptation", "一键", "自动化",
    ],
    "quantization": [
        "quantiz", "quantis", "量化", "int8", "int4", "w4a",
        "awesome-llm", "msmodelslim",
    ],
    "search": [
        "search", "检索", "查找", "搜索", "docs-search",
    ],
    "verification": [
        "verify", "valid", "test", "check", "assert", "eval",
        "precision-debug", "precision-eval", "precision-compare",
        "mssanitizer", "st-design", "whitebox", "env-check",
        "compile-debug", "operator-code-review", "operator-dev",
        "operator-doc-gen", "operator-project-init", "operator-testcase",
        "profiling-anomaly", "tf-community", "mmlab",
        "boltzgen", "deepfri", "detectron2", "goedel-prover",
        "hccl-test", "issue_autoreply", "issue_solver",
        "mmcv", "mmdet", "oligoformer", "ops-precision",
        "ops-simulator", "pypto-api", "pypto-golden", "pypto-op-design",
        "pypto-op-develop", "tilelang-op-design", "tilelang-review",
        "cann-operator-env", "catlass-operator-code", "catlass-operator-design",
        "model-infer-parallel-impl", "model-training",
        "triton-operator-code-gen", "triton-operator-env",
        "验证", "测试", "检查", "校验",
    ],
    "other": [
        "archived", "diffsbdd", "generative-recommendation",
        "megatron-commit-tracker", "run-mindspeed",
    ],
}


def find_skills():
    """Scan all roots, deduplicate by name, prefer ascend/ paths."""
    skills = {}
    for scan_root in SCAN_ROOTS:
        if not scan_root.is_dir():
            continue
        for md_file in scan_root.rglob("SKILL.md"):
            content = md_file.read_text(encoding="utf-8", errors="ignore")
            frontmatter = parse_frontmatter(content)
            name = frontmatter.get("name", md_file.parent.name)
            desc = frontmatter.get("description", "")
            rel_path = str(md_file.relative_to(ROOT))

            existing = skills.get(name)
            if existing is None:
                skills[name] = {
                    "name": name, "path": rel_path, "description": desc,
                    "content": content, "frontmatter": frontmatter,
                    "dir": str(md_file.parent.relative_to(ROOT)),
                    "file_size": len(content), "line_count": content.count("\n"),
                }
            else:
                # Prefer ascend/ paths or shorter paths
                if "skills\\ascend\\" in rel_path and "skills\\ascend\\" not in existing["path"]:
                    skills[name] = {
                        "name": name, "path": rel_path, "description": desc,
                        "content": content, "frontmatter": frontmatter,
                        "dir": str(md_file.parent.relative_to(ROOT)),
                        "file_size": len(content), "line_count": content.count("\n"),
                    }
    return skills


def parse_frontmatter(text: str) -> dict:
    """Parse YAML-like frontmatter between --- markers."""
    result = {}
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return result
    fm_text = match.group(1)
    for line in fm_text.split("\n"):
        line = line.strip()
        m = re.match(r"^(\w[\w-]*)\s*:\s*(.+)$", line)
        if m:
            result[m.group(1)] = m.group(2).strip()
    return result


def score_skill(skill: dict) -> dict:
    """Score a skill on 9 dimensions on a 0-100 scale."""
    text = skill["content"]
    fm = skill["frontmatter"]
    total_lines = skill["line_count"]

    # Reusable counters
    numbered = len(re.findall(r"^\d+\.\s+", text, re.MULTILINE))
    headings = len(re.findall(r"^##\s+", text, re.MULTILINE))
    sections = len(re.findall(r"^#{1,3}\s+", text, re.MULTILINE))
    code_blocks = len(re.findall(r"^```", text, re.MULTILINE)) // 2
    commands = len(re.findall(
        r"\b(npu-smi|python3?|pip|git|node|npm|uvicorn|bash|curl|wget|docker|kubectl|vllm|torch)\b",
        text, re.IGNORECASE,
    ))
    error_terms = len(re.findall(
        r"(异常|错误|error|fallback|失败|如果.*则|回滚|recover|retry|exception|failed|troubleshoot)",
        text, re.IGNORECASE,
    ))
    checkpoint_terms = len(re.findall(
        r"(确认|checkpoint|暂停|approval|用户确认|验证点|检查点|verify)",
        text, re.IGNORECASE,
    ))
    ascend_terms = len(re.findall(
        r"(ascend|npu|torch_npu|cann|npu-smi|昇腾|Ascend|NPU|CANN|Atlas|mindsp)",
        text,
    ))
    verify_terms = len(re.findall(
        r"(test|验证|benchmark|eval|验证结果|预期输出|expected|assert|compare|accuracy|精度)",
        text, re.IGNORECASE,
    ))

    # Script files in skill directory
    script_files = 0
    skill_dir = ROOT / skill["dir"]
    if skill_dir.is_dir():
        script_files = len(list(skill_dir.rglob("*.py"))) + len(list(skill_dir.rglob("*.sh")))
    resource_refs = len(re.findall(
        r"(scripts?/|templates?/|references?/|evals?\.json|results\.tsv)",
        text, re.IGNORECASE,
    ))

    # ---- Scoring (each dimension 0-10, weighted to 100 total) ----

    # D1: Frontmatter quality (weight 8)
    d1 = 40  # base for having any frontmatter
    if fm:
        d1 += 20
        if fm.get("name"):
            d1 += 15
        if fm.get("description"):
            d1 += 15
        if fm.get("keywords"):
            d1 += 10
    d1 = min(100, d1)

    # D2: Workflow clarity (weight 15)
    d2 = 20 + min(30, numbered * 3) + min(25, headings * 3) + min(25, sections * 2)
    d2 = min(100, max(10, d2))

    # D3: Boundary condition coverage (weight 10)
    d3 = 15 + min(60, error_terms * 8) + min(25, total_lines / 8)
    d3 = min(100, max(5, d3))

    # D4: Checkpoint design (weight 7)
    d4 = 15 + min(60, checkpoint_terms * 10) + min(25, total_lines / 10)
    d4 = min(100, max(5, d4))

    # D5: Instruction specificity (weight 15)
    d5 = 20 + min(35, code_blocks * 4) + min(30, commands * 5) + min(15, numbered * 2)
    d5 = min(100, max(10, d5))

    # D6: Resource integration (weight 5)
    d6 = 15 + min(45, script_files * 10) + min(30, resource_refs * 8) + min(10, total_lines / 20)
    d6 = min(100, max(5, d6))

    # D7: Overall architecture (weight 15)
    d7 = 20 + min(30, sections * 3) + min(30, total_lines / 5) + min(20, headings * 3)
    d7 = min(100, max(10, d7))

    # D8: Ascend relevance (weight 10)
    d8 = 20 + min(60, ascend_terms * 5) + min(20, total_lines / 15)
    d8 = min(100, max(5, d8))

    # D9: Verifiability (weight 15)
    d9 = 15 + min(35, verify_terms * 5) + min(30, script_files * 5) + min(20, commands * 3)
    d9 = min(100, max(5, d9))

    dim_0_10 = {
        "frontmatter": round(d1 / 10, 1),
        "workflow_clarity": round(d2 / 10, 1),
        "boundary_coverage": round(d3 / 10, 1),
        "checkpoint_design": round(d4 / 10, 1),
        "instruction_specificity": round(d5 / 10, 1),
        "resource_integration": round(d6 / 10, 1),
        "overall_architecture": round(d7 / 10, 1),
        "ascend_relevance": round(d8 / 10, 1),
        "verifiability": round(d9 / 10, 1),
    }

    # Weighted total (0-100)
    dims_100 = {"frontmatter": d1, "workflow_clarity": d2, "boundary_coverage": d3,
                 "checkpoint_design": d4, "instruction_specificity": d5,
                 "resource_integration": d6, "overall_architecture": d7,
                 "ascend_relevance": d8, "verifiability": d9}
    weighted_sum = sum(dims_100[k] * WEIGHTS[k] for k in WEIGHTS)
    total = round(weighted_sum / sum(WEIGHTS.values()), 1)

    return {"dimensions": dim_0_10, "total": total, "dimensions_100": dims_100}


# Keywords that are too broad for content matching — only count in skill name
NAME_ONLY_KEYWORDS = {
    "test", "check", "assert", "valid", "verify", "eval",
    "验证", "测试", "检查", "校验",
    "optim", "tune", "perf", "加速", "优化",
}

# Directory path fragments that strongly indicate a category
DIR_CATEGORY_MAP = {
    "deployment": ["deployment", "deploy"],
    "adaptation": ["adaptation", "adapt"],
    "optimization": ["optimization", "optimize", "tuning"],
    "verification": ["verification", "verify"],
    "documentation": ["documentation", "document", "docs"],
    "common": ["common"],
    "pta": ["pta"],
    "quantization": ["quantization", "quantize"],
    "search": ["search"],
}


def classify_skill(skill: dict) -> str:
    """Classify skill into one of 11 categories based on path + content analysis."""
    text = (skill.get("description", "") + " " + skill["content"]).lower()
    name = skill["name"].lower()
    path = skill.get("path", "").lower()

    scores = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in name:
                score += 8
            # Content match: skip overly broad keywords to reduce false positives
            if kw not in NAME_ONLY_KEYWORDS and kw in text:
                score += 2
        scores[cat] = score

    # Directory-based boosting: the directory structure is ground truth
    for cat, dir_keywords in DIR_CATEGORY_MAP.items():
        for dk in dir_keywords:
            if dk in path:
                scores[cat] += 20
                break  # one boost per category

    # NPU-named skills without a stronger signal default to deployment
    if "npu" in name and scores.get("deployment", 0) < 20:
        scores["deployment"] = scores.get("deployment", 0) + 12

    best = max(scores, key=scores.get)
    if scores[best] < 5:
        return "other"
    return best


def main():
    print("Scanning skills...")
    skills = find_skills()
    print(f"Found {len(skills)} unique skills (deduplicated)")

    # Evaluate each
    results = []
    for name, skill in skills.items():
        eval_result = score_skill(skill)
        category = classify_skill(skill)
        tier = 2 if eval_result["total"] >= 85 else 3
        results.append({
            "name": name,
            "path": skill["path"],
            "description": skill["description"][:200],
            "category": category,
            "tier": tier,
            "total_score": eval_result["total"],
            "dimensions": eval_result["dimensions"],
            "line_count": skill["line_count"],
        })

    # Sort by score desc
    results.sort(key=lambda x: x["total_score"], reverse=True)

    # Stats
    tier2 = [r for r in results if r["tier"] == 2]
    tier3 = [r for r in results if r["tier"] == 3]
    print(f"\nTier 2 (>=85): {len(tier2)}")
    print(f"Tier 3 (<85): {len(tier3)}")

    for cat in CATEGORY_KEYWORDS:
        cat_skills = [r for r in results if r["category"] == cat]
        t2 = sum(1 for r in cat_skills if r["tier"] == 2)
        t3 = sum(1 for r in cat_skills if r["tier"] == 3)
        print(f"  {cat}: {len(cat_skills)} skills (T2={t2}, T3={t3})")

    # Top 20 Tier 2
    print("\n=== Top 20 Tier 2 ===")
    for r in tier2[:20]:
        print(f"  [{r['total_score']:.1f}] {r['name']} ({r['category']})")

    # Bottom 20 Tier 3
    print("\n=== Bottom 20 Tier 3 ===")
    for r in tier3[:20]:
        print(f"  [{r['total_score']:.1f}] {r['name']} ({r['category']})")

    # Generate tiers.json structure
    tiers_json = {"tier1": {}, "tier2": {}, "tier3": {}}
    for r in results:
        entry = {"description": r["description"], "category": r["category"], "score": r["total_score"]}
        tier_key = f"tier{r['tier']}"
        tiers_json[tier_key][r["name"]] = entry

    out_path = ROOT / "skills" / "tiers.json"
    out_path.write_text(json.dumps(tiers_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWritten: {out_path}")

    # Also write detailed report
    report_path = ROOT / "scripts" / "skill_evaluation_report.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
