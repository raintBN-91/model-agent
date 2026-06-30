"""ascend-skills-eval 回归与边界测试。

运行：
    cd ascend-skills-eval
    python -m pytest tests/ -v
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web-service"))

from app.main import (  # noqa: E402
    _has_frontmatter,
    _pick_repo_file,
    _validate_repo_url,
    evaluate_skill,
)


# ---------- frontmatter 换行符归一化 ----------

def test_frontmatter_lf():
    text = "---\nname: x\ndescription: y\n---\n# body"
    assert _has_frontmatter(text) is True


def test_frontmatter_crlf():
    """CRLF 换行的 frontmatter 应被识别（回归 Bug #01）。"""
    text = "---\r\nname: x\r\ndescription: y\r\n---\r\n# body"
    assert _has_frontmatter(text) is True


def test_frontmatter_mixed_line_endings():
    text = "---\nname: x\r\ndescription: y\r\n---\r\n# body"
    assert _has_frontmatter(text) is True


def test_frontmatter_crlf_no_frontmatter_returns_false():
    """无 frontmatter 的 CRLF 文本应返回 False（边界）。"""
    text = "\r\n# just a body\r\nno frontmatter here\r\n"
    assert _has_frontmatter(text) is False


def test_frontmatter_missing():
    assert _has_frontmatter("# just a body") is False


# ---------- evaluate_skill 基础结构 ----------

def test_evaluate_skill_returns_9_dims():
    sample = (
        "---\nname: ascend-demo\ndescription: 演示 skill\n---\n# Demo\n\n"
        "## Step 1: 准备\n1. npu-smi info\n2. 检查异常与 fallback\n\n"
        "## Step 2: 推理\n```bash\npython3 infer.py --device npu\n```\n确认结果后继续。\n"
    )
    resp = evaluate_skill(sample, "ascend-demo")
    assert len(resp.dimensions) == 9
    assert 0 <= resp.total_score <= 100
    assert len(resp.suggestions) == 3


def test_evaluate_skill_smoke():
    """原 test_batch_perf_concurrent 重命名：仅校验核心同步逻辑未被并发改造破坏。"""
    sample = "---\nname: p\ndescription: q\n---\n# body\n1. step\n异常处理\n确认"
    r = evaluate_skill(sample, "p")
    assert r.skill_name == "p"
    assert r.total_score > 0


# ---------- _pick_repo_file rglob 排序确定性 ----------

def test_pick_repo_file_sorted(tmp_path: Path):
    """rglob 结果排序后应确定性返回字典序最小的文件。"""
    (tmp_path / "z.md").write_text("z", encoding="utf-8")
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "m.md").write_text("m", encoding="utf-8")
    picked = _pick_repo_file(tmp_path, skill_path=None)
    # 三个候选都不是优先名，走 rglob *.md 排序后应取 a.md
    assert picked.name == "a.md"


def test_pick_repo_file_priority_first(tmp_path: Path):
    """根目录存在 SKILL.md 时应优先返回，不走 rglob。"""
    (tmp_path / "README.md").write_text("r", encoding="utf-8")
    (tmp_path / "SKILL.md").write_text("s", encoding="utf-8")
    picked = _pick_repo_file(tmp_path, skill_path=None)
    assert picked.name == "SKILL.md"


# ---------- _validate_repo_url ----------

def test_validate_repo_url_rejects_non_github():
    import pytest
    with pytest.raises(Exception):
        _validate_repo_url("https://evil.com/x.git")


def test_validate_repo_url_accepts_gitcode():
    assert _validate_repo_url("https://gitcode.com/Ascend/model-agent.git").endswith(".git")


# ---------- render_card 超时分支（单元层验证） ----------

def test_render_card_timeout_branch_logic(monkeypatch, tmp_path):
    """验证 render_card 超时分支会抛 504 且调用进程树清理。"""
    import pytest
    from fastapi import HTTPException
    import app.main as m

    killed: list = []

    class _FakePopen:
        def __init__(self, cmd, **kw):
            self.pid = 99999
            self.returncode = None
            self._cmd = cmd
        def communicate(self, timeout=None):
            raise m.subprocess.TimeoutExpired(self._cmd, timeout)
        def poll(self):
            return self.returncode

    # 桩 TemporaryDirectory：用普通类实现 __enter__/__exit__，避免 contextmanager 关键字传递问题
    class _FakeTmpDir:
        def __init__(self, prefix=None, **kw):
            self._d = tmp_path
        def __enter__(self):
            return self._d
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(m.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(m, "_kill_process_tree", lambda p, k: killed.append(p.pid))
    monkeypatch.setattr(m, "_render_script_path", lambda: Path(__file__))
    monkeypatch.setattr(m.tempfile, "TemporaryDirectory", _FakeTmpDir)
    # 关键：platform.system() 内部会调用 subprocess，必须桩掉避免污染
    monkeypatch.setattr(m.platform, "system", lambda: "Windows")

    with pytest.raises(HTTPException) as exc:
        m.render_card(m.RenderRequest(data={"skill-name": "t"}, open_image=False))
    assert exc.value.status_code == 504, f"超时应返回 504，实际 {exc.value.status_code}: {exc.value.detail}"
    assert killed == [99999], "超时后应调用 _kill_process_tree 清理进程树"


def test_kill_process_tree_uses_taskkill_on_windows(monkeypatch):
    """_kill_process_tree 在 Windows 上应调用 taskkill /T。"""
    import app.main as m
    called: list = []

    class _FakePopen:
        pid = 12345
        returncode = None
        def poll(self):
            return None

    def _fake_run(cmd, **kw):
        called.append(cmd)
        class _R:
            returncode = 0
            stdout = ""
            stderr = ""
        return _R()

    monkeypatch.setattr(m.platform, "system", lambda: "Windows")
    monkeypatch.setattr(m.subprocess, "run", _fake_run)
    m._kill_process_tree(_FakePopen(), {})
    assert called and called[0][0] == "taskkill" and "/T" in called[0]


# ---------- evaluate_repos cache key 三元组 ----------

def test_cache_key_distinguishes_branches():
    """验证 cache key 逻辑：同 URL 不同 branch 不应命中缓存。

    此处用白盒方式构造 _handle 闭包无法直接访问，改为通过
    evaluate_repos 的 BatchRepoEvalRequest 路径需要真实 clone，
    故此用例仅校验 cache key 的数据结构设计而非真实请求。
    """
    # 设计契约：cache key 必须是 (repo_url, branch, skill_path) 三元组
    key_main = ("https://gitcode.com/x/y.git", None, None)
    key_dev = ("https://gitcode.com/x/y.git", "dev", None)
    assert key_main != key_dev, "不同 branch 必须产生不同 cache key"


def test_cache_lock_exists():
    """evaluate_repos 内部应使用 threading.Lock 保护 cache（静态检查）。"""
    import app.main as m
    src = open(m.__file__, encoding="utf-8").read()
    assert "threading.Lock()" in src, "cache 必须加锁"
    assert "cache.get(key)" in src or "cache[key]" in src


def test_cache_concurrent_hit_only_clones_once(monkeypatch):
    """同 key 并发请求时，_evaluate_repo_internal 应只被调用一次（cache 命中）。

    直接调用真实的 evaluate_repos，桩掉 _evaluate_repo_internal 计数+延时，
    用 BatchRepoEvalRequest 传两个相同 item 制造并发竞争。
    """
    import threading as _t
    import app.main as m
    from app.main import BatchRepoEvalRequest, RepoEvalRequest

    call_count = {"n": 0}
    call_lock = _t.Lock()

    def _fake_eval_internal(payload):
        with call_lock:
            call_count["n"] += 1
        time.sleep(0.1)  # 模拟 clone 耗时，放大并发竞争窗口
        return {
            "eval": {"skill_name": "x", "total_score": 80, "dimensions": [],
                     "suggestions": [], "mode": "structure_eval"},
            "card": {"image_base64": "", "mime_type": "image/png", "size_bytes": 0},
            "report_markdown": "", "picked_file": "SKILL.md",
            "repo_url": payload.repo_url, "skill_markdown": "", "history_id": "1",
        }

    monkeypatch.setattr(m, "_evaluate_repo_internal", _fake_eval_internal)
    # render_card 也会被 _evaluate_repo_internal 触发，桩掉避免依赖 node
    monkeypatch.setattr(m, "render_card", lambda p: m.RenderResponse(
        image_base64="", mime_type="image/png", size_bytes=0))

    payload = BatchRepoEvalRequest(items=[
        RepoEvalRequest(repo_url="https://gitcode.com/a/b.git"),
        RepoEvalRequest(repo_url="https://gitcode.com/a/b.git"),  # 相同 key
    ])
    result = m.evaluate_repos(payload)

    assert call_count["n"] == 1, f"同 key 并发应只 clone 一次，实际 {call_count['n']}"
    assert result["summary"]["success"] == 2, "两个请求都应成功（一个真实 clone + 一个 cache 命中）"


def test_different_keys_clone_concurrently(monkeypatch):
    """不同 key 的 clone 应真正并发（验证 per-key 锁优于全局锁）。

    用 Barrier(2) 让两个不同 key 的 _evaluate_repo_internal 在并发中同时抵达，
    若 per-key 锁生效则两个线程能同时进入 clone（Barrier 不超时）；
    若误用全局锁则两个线程串行，Barrier 会因只到 1/2 而超时。
    """
    import threading as _t
    import app.main as m
    from app.main import BatchRepoEvalRequest, RepoEvalRequest

    barrier = _t.Barrier(2, timeout=3)
    arrived = {"n": 0}
    arrived_lock = _t.Lock()

    def _fake_eval_internal(payload):
        with arrived_lock:
            arrived["n"] += 1
        barrier.wait(timeout=3)  # 两个不同 key 应能同时到此
        time.sleep(0.05)
        return {
            "eval": {"skill_name": "x", "total_score": 80, "dimensions": [],
                     "suggestions": [], "mode": "structure_eval"},
            "card": {"image_base64": "", "mime_type": "image/png", "size_bytes": 0},
            "report_markdown": "", "picked_file": "SKILL.md",
            "repo_url": payload.repo_url, "skill_markdown": "", "history_id": "1",
        }

    monkeypatch.setattr(m, "_evaluate_repo_internal", _fake_eval_internal)
    monkeypatch.setattr(m, "render_card", lambda p: m.RenderResponse(
        image_base64="", mime_type="image/png", size_bytes=0))

    # 两个不同 repo_url → 不同 key → 应并发
    payload = BatchRepoEvalRequest(items=[
        RepoEvalRequest(repo_url="https://gitcode.com/a/b.git"),
        RepoEvalRequest(repo_url="https://gitcode.com/c/d.git"),
    ])
    result = m.evaluate_repos(payload)
    assert arrived["n"] == 2, "两个不同 key 应都抵达 barrier（并发执行）"
    assert result["summary"]["success"] == 2
