"""doc-agent 工具模块。

在本地 doc-agent 仓库中执行「一键生成小白部署文档」相关任务。

依赖：DOC_AGENT_ROOT 指向 doc-agent 仓库根目录，且在
``tools/model_readme_generator`` 下已创建 ``.venv`` 并安装依赖。
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from langchain_core.tools import tool

_MAX_OUT = 48_000
_SUBPROCESS_TIMEOUT: int = 600   # generate.py / venv 安装超时
_ONE_CLICK_TIMEOUT: int = 1800  # one-click 脚本超时
_QWEN35_STEM = "Qwen3.5-27B"

DOC_SUBCOMMANDS = {
    "vllm",
    "vllm-ascend",
    "sglang",
    "verl",
    "qwen3-8b",
    "qwen3_8b",
    "qwen38b",
    "qwen3.5-27b",
    "qwen35",
    "qwen3.5",
    "generate",
    "local",
    "discover",
    "one-click",
    "one_click",
}


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _venv_python(tool_dir: Path) -> Path:
    """返回虚拟环境中的 Python 可执行文件路径（跨平台）。"""
    if _is_windows():
        return tool_dir / ".venv" / "Scripts" / "python.exe"
    return tool_dir / ".venv" / "bin" / "python"


def _venv_pip(tool_dir: Path) -> Path:
    """返回虚拟环境中的 pip 可执行文件路径（跨平台）。"""
    if _is_windows():
        return tool_dir / ".venv" / "Scripts" / "pip.exe"
    return tool_dir / ".venv" / "bin" / "pip"


def check_doc_agent_environment(root: Path) -> list[str]:
    """检查 doc-agent 环境是否就绪，返回问题列表（空表示 OK）。"""
    issues = []
    if not root.is_dir():
        issues.append(f"❌ doc-agent 仓库目录不存在: {root}")
        issues.append("请克隆 doc-agent 仓库或设置 DOC_AGENT_ROOT 环境变量")
        return issues

    td = root / "tools" / "model_readme_generator"
    if not td.is_dir():
        issues.append(f"❌ 工具目录不存在: {td}")
        return issues

    req_file = td / "requirements.txt"
    if not req_file.is_file():
        issues.append(f"❌ 依赖文件不存在: {req_file}")
        return issues

    venv_dir = td / ".venv"
    if not venv_dir.is_dir():
        try:
            _setup_venv(td)
        except Exception as e:
            issues.append(f"❌ 自动创建虚拟环境失败: {e}")
            if _is_windows():
                issues.append(
                    "请手动执行：cd tools/model_readme_generator && python -m venv .venv && .venv\\Scripts\\pip install -r requirements.txt"
                )
            else:
                issues.append(
                    "请手动执行：cd tools/model_readme_generator && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
                )

    return issues


def _tool_dir_and_py(root: Path) -> tuple[Path, Path] | None:
    td = root / "tools" / "model_readme_generator"
    py = _venv_python(td)
    if not td.is_dir():
        return None
    if not py.is_file():
        _setup_venv(td)
        if not py.is_file():
            return None
    return td, py


def _setup_venv(tool_dir: Path) -> None:
    venv_dir = tool_dir / ".venv"
    if venv_dir.is_dir():
        return
    print(f"创建虚拟环境: {venv_dir}")
    python_cmd = sys.executable if sys.executable else ("python" if _is_windows() else "python3")
    subprocess.run(
        [python_cmd, "-m", "venv", str(venv_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    venv_pip = _venv_pip(tool_dir)
    print(f"安装依赖: {tool_dir / 'requirements.txt'}")
    subprocess.run(
        [str(venv_pip), "install", "-r", str(tool_dir / "requirements.txt")],
        check=True,
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
    )
    print("虚拟环境配置完成")


def resolve_doc_agent_root() -> Path:
    raw = (os.environ.get("DOC_AGENT_ROOT") or "").strip()
    if raw:
        root = Path(raw).expanduser().resolve()
        os.environ["DOC_AGENT_ROOT"] = str(root)
        return root

    default_path = (Path("..") / "doc-agent").resolve()
    if default_path.is_dir():
        os.environ["DOC_AGENT_ROOT"] = str(default_path)
        return default_path

    for search_path in [
        Path.home() / "Documents" / "doc-agent",
        Path.home() / "doc-agent",
        Path("/opt/doc-agent"),
    ]:
        if search_path.is_dir():
            os.environ["DOC_AGENT_ROOT"] = str(search_path)
            return search_path

    return default_path


def _truncate_out(out: str) -> str:
    if len(out) > _MAX_OUT:
        return out[:_MAX_OUT] + "\n\n…（输出已截断）"
    return out


def _format_doc_agent_cli_output(text: str, repo_root: Path) -> str:
    """把 generate/一键脚本里的 ``Wrote /绝对路径`` 改成相对 doc-agent 根的说明。"""
    repo_root = repo_root.resolve()
    lines: list[str] = []
    saw_wrote = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Wrote "):
            saw_wrote = True
            path_str = s[6:].strip()
            try:
                p = Path(path_str).expanduser().resolve()
                try:
                    rel = p.relative_to(repo_root)
                    lines.append(f"✅ 已生成：{rel.as_posix()}")
                except ValueError:
                    lines.append(f"✅ 已生成：{path_str}")
            except Exception:
                lines.append(f"✅ 已生成：{path_str}")
        else:
            lines.append(line)
    out = "\n".join(lines)
    if saw_wrote and ("qwen3-5-27b" in out.lower() or "Qwen3.5-27B" in out):
        out += (
            "\n\n提示：目录名 `qwen3-5-27b` 来自 vLLM Ascend 官方教程 URL 的 slug，对应模型 **Qwen3.5-27B**；"
            "文件名里的 `A2` 表示 Atlas 800 A2，均为 manifest 约定，并非生成错误。"
        )
    return out


def manifest_has_qwen35_27b_entry(manifest: Path) -> bool:
    """判断 manifest 中是否已有 Qwen3.5-27B 相关条目。"""
    if not manifest.is_file():
        return False
    text = manifest.read_text(encoding="utf-8", errors="ignore")
    if _QWEN35_STEM not in text and "qwen3.5-27b" not in text.lower():
        return False
    try:
        import yaml

        data = yaml.safe_load(text)
    except Exception:
        return True
    for e in data.get("entries") or []:
        if not isinstance(e, dict):
            continue
        blob = " ".join(str(e.get(k, "")) for k in e.keys())
        low = blob.lower()
        if "qwen3.5" in low and "27b" in low:
            return True
    return "qwen3.5" in text.lower() and "27b" in text.lower()


def _exec_generate_local(root: Path, manifest_filename: str) -> str:
    pair = _tool_dir_and_py(root)
    if not pair:
        if _is_windows():
            return (
                "未找到 doc-agent 下 tools/model_readme_generator/.venv ，且自动创建失败。\n"
                "请在该目录手动执行：python -m venv .venv && .venv\\Scripts\\pip install -r requirements.txt"
            )
        return (
            "未找到 doc-agent 下 tools/model_readme_generator/.venv ，且自动创建失败。\n"
            "请在该目录手动执行：python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
        )
    td, venv_py = pair
    name = (os.environ.get("DOC_AGENT_MANIFEST") or "").strip() or manifest_filename
    manifest = (td / name).resolve()
    if not manifest.is_file():
        return (
            f"未找到 manifest：{manifest}\n"
            "请先至少执行一次联网发现，或把 manifest 放到上述路径。"
        )
    gen = td / "generate.py"
    if not gen.is_file():
        return f"未找到 generate.py：{gen}"
    cmd = [
        str(venv_py),
        "generate.py",
        "--manifest",
        str(manifest),
        "--repo-root",
        str(root),
    ]
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(td),
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        return f"generate.py 执行超时（>{_SUBPROCESS_TIMEOUT // 60} 分钟）。"
    except OSError as e:
        return f"无法执行 generate.py：{e}"
    out = (completed.stdout or "") + (completed.stderr or "")
    out = _format_doc_agent_cli_output(out, root)
    out = _truncate_out(out)
    return out + f"\n\n[exit code: {completed.returncode}]"


def _exec_one_click(root: Path, extra_args: str) -> str:
    script = root / "tools" / "model_readme_generator" / "one_click_discover_generate.sh"
    if not script.is_file():
        return (
            f"未找到一键脚本：{script}\n"
            "请设置环境变量 DOC_AGENT_ROOT 为 doc-agent 仓库根目录。"
        )
    pair = _tool_dir_and_py(root)
    if not pair:
        if _is_windows():
            return (
                "未找到 doc-agent 下 tools/model_readme_generator/.venv ，且自动创建失败。\n"
                "请在该目录手动执行：python -m venv .venv && .venv\\Scripts\\pip install -r requirements.txt"
            )
        return (
            "未找到 doc-agent 下 tools/model_readme_generator/.venv ，且自动创建失败。\n"
            "请在该目录手动执行：python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
        )
    cmd: list[str] = ["bash", str(script)]
    rest = (extra_args or "").strip()
    if rest:
        cmd.extend(shlex.split(rest))
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(script.parent),
            capture_output=True,
            text=True,
            timeout=_ONE_CLICK_TIMEOUT,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        return f"执行超时（>{_ONE_CLICK_TIMEOUT // 60} 分钟）。全量抓取可能需更久或网络不稳定。"
    except OSError as e:
        return f"无法执行脚本：{e}"
    out = (completed.stdout or "") + (completed.stderr or "")
    out = _format_doc_agent_cli_output(out, root)
    if len(out) > _MAX_OUT:
        out = (
            out[:_MAX_OUT]
            + f"\n\n…（输出已截断，共约 {len(completed.stdout or '') + len(completed.stderr or '')} 字符）"
        )
    return out + f"\n\n[exit code: {completed.returncode}]"


@tool
def doc_agent_one_click_generate(extra_args: str = "") -> str:
    """在本地 doc-agent 仓库中执行「一键」脚本：联网抓取 vLLM Ascend 官方 Model Tutorials 索引
    → 生成 ``models/vllm-ascend/**`` 下「小白」部署 Markdown。

    使用前须：已克隆 doc-agent；在 ``tools/model_readme_generator`` 下建好虚拟环境并安装依赖。

    参数 ``extra_args`` 为附加命令行参数（空格分隔），例如：``--locale zh-cn`` 或 ``--only Qwen3.5-27B``；
    留空表示全量抓取（耗时长、需稳定网络）。
    """
    root = resolve_doc_agent_root()
    issues = check_doc_agent_environment(root)
    if issues:
        return "环境检查结果：\n" + "\n".join(issues)
    return _exec_one_click(root, extra_args)


@tool
def doc_agent_generate_local(manifest_filename: str = "manifest.discovered.yaml") -> str:
    """仅执行 ``generate.py``：用**已有** manifest 生成 ``models/vllm-ascend/**`` 下 Markdown，**不联网、不爬取**。

    参数 ``manifest_filename`` 为相对 ``tools/model_readme_generator/`` 的文件名，
    默认 ``manifest.discovered.yaml``。也可用环境变量 ``DOC_AGENT_MANIFEST`` 覆盖。
    """
    root = resolve_doc_agent_root()
    issues = check_doc_agent_environment(root)
    if issues:
        return "环境检查结果：\n" + "\n".join(issues)
    return _exec_generate_local(root, manifest_filename)


@tool
def doc_agent_example_qwen35_flow() -> str:
    """「example：Qwen3.5-27B」专用：先查本地 manifest 是否已有 Qwen3.5-27B；
    有则只跑本地 generate，无则联网 discover（--only）再生成。
    """
    root = resolve_doc_agent_root()
    issues = check_doc_agent_environment(root)
    if issues:
        return "环境检查结果：\n" + "\n".join(issues)
    pair = _tool_dir_and_py(root)
    if not pair:
        if _is_windows():
            return (
                "😅 咦，doc-agent 的虚拟环境还没就绪～\n"
                "请在 `tools/model_readme_generator` 里先：`python -m venv .venv && .venv\\Scripts\\pip install -r requirements.txt`，"
                "装好了再来找我玩！"
            )
        return (
            "😅 咦，doc-agent 的虚拟环境还没就绪～\n"
            "请在 `tools/model_readme_generator` 里先：`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`，"
            "装好了再来找我玩！"
        )
    td, _ = pair
    name = (os.environ.get("DOC_AGENT_MANIFEST") or "").strip() or "manifest.discovered.yaml"
    manifest = (td / name).resolve()

    if manifest_has_qwen35_27b_entry(manifest):
        head = (
            "✨ 太好啦！在本地 manifest 里**逮到 Qwen3.5-27B** 啦～索引都在自家硬盘上，咱就不爬网啦，"
            "直接帮你「嗖嗖」本地生成，环保又省心！\n"
            "────────\n\n"
        )
        return head + _exec_generate_local(root, "manifest.discovered.yaml")

    head = (
        "🔍 本地 manifest 里**还没蹲到 Qwen3.5-27B** 这位小伙伴呢～没关系！"
        "我这就帮你去官网**抓索引 + 生成**一条龙，可能要一小会儿，泡杯茶等我一下下～\n"
        "────────\n\n"
    )
    return head + _exec_one_click(root, f"--only {_QWEN35_STEM}")


def _detect_git_remote_url(root: Path) -> str:
    """尝试从本地 .git 目录检测远程仓库 URL。"""
    try:
        git_dir = root / ".git"
        if not git_dir.is_dir():
            config_file = root / ".git"
        else:
            config_file = git_dir / "config"

        if not config_file.is_file():
            return ""

        content = config_file.read_text(encoding="utf-8", errors="ignore")
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("url = "):
                url = line[6:].strip()
                if "gitcode.com" in url or "gitee.com" in url:
                    if url.endswith(".git"):
                        url = url[:-4]
                    return url
                if "github.com" in url:
                    if url.endswith(".git"):
                        url = url[:-4]
                    return url
    except Exception:
        pass
    return ""


def _get_doc_agent_url(root: Path, file_path: Path) -> str:
    """将本地路径转换为 GitCode 可访问的链接。"""
    repo_root = os.environ.get("DOC_AGENT_GITCODE_URL", "").strip()
    if not repo_root:
        repo_root = _detect_git_remote_url(root)
    if not repo_root:
        return file_path.as_posix()
    try:
        rel_path = file_path.relative_to(root)
        return f"{repo_root}/blob/main/{rel_path.as_posix()}"
    except ValueError:
        return file_path.as_posix()


def _list_adapter_docs(root: Path, adapter_name: str) -> str:
    """列出指定适配框架的文档目录和文件列表。"""
    suffix = "" if adapter_name in ("sglang", "verl") else "-ascend"
    adapter_dir = root / "models" / f"{adapter_name}{suffix}"
    if not adapter_dir.is_dir():
        return f"未找到 {adapter_name}{suffix} 适配文档目录：{adapter_dir}"

    lines = [f"📁 {adapter_name.upper()} 适配文档目录："]

    repo_url = os.environ.get("DOC_AGENT_GITCODE_URL", "").strip()
    if not repo_url:
        repo_url = _detect_git_remote_url(root)

    for model_dir in sorted(adapter_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        lines.append(f"\n[{model_dir.name}/]")
        for md_file in sorted(model_dir.glob("*.md")):
            rel_path = md_file.relative_to(root)
            if repo_url:
                url = _get_doc_agent_url(root, md_file)
                lines.append(f"  - [{rel_path.name}]({url})")
            else:
                lines.append(f"  - {rel_path.as_posix()}")
    return "\n".join(lines)


@tool
def doc_agent_vllm_adapter_docs() -> str:
    """展示 vllm-ascend 适配文档列表和地址。"""
    root = resolve_doc_agent_root()
    issues = check_doc_agent_environment(root)
    if issues:
        return "环境检查结果：\n" + "\n".join(issues)
    return _list_adapter_docs(root, "vllm")


@tool
def doc_agent_sglang_adapter_docs() -> str:
    """展示 sglang 适配文档列表和地址。"""
    root = resolve_doc_agent_root()
    issues = check_doc_agent_environment(root)
    if issues:
        return "环境检查结果：\n" + "\n".join(issues)
    return _list_adapter_docs(root, "sglang")


@tool
def doc_agent_verl_adapter_docs() -> str:
    """展示 verl 适配文档列表和地址。"""
    root = resolve_doc_agent_root()
    issues = check_doc_agent_environment(root)
    if issues:
        return "环境检查结果：\n" + "\n".join(issues)
    return _list_adapter_docs(root, "verl")


@tool
def doc_agent_example_qwen3_8b() -> str:
    """「example：Qwen3-8B」专用：查找 Qwen3-8B 适配 vllm-ascend 的文档并返回文档内容。"""
    root = resolve_doc_agent_root()
    issues = check_doc_agent_environment(root)
    if issues:
        return "环境检查结果：\n" + "\n".join(issues)

    vllm_dir = root / "models" / "vllm-ascend"
    if not vllm_dir.is_dir():
        return f"未找到 vllm-ascend 文档目录：{vllm_dir}"

    patterns = [
        ("qwen3-8b", "README.md"),
        ("qwen3-8b-a2", "README.md"),
        ("qwen3-8b-a3", "README.md"),
        ("qwen3-8b-w4a8", "Qwen3-8B-W4A8-A2-Vllm-Ascend.md"),
    ]

    repo_url = os.environ.get("DOC_AGENT_GITCODE_URL", "").strip()
    if not repo_url:
        repo_url = _detect_git_remote_url(root)

    for dir_name, file_name in patterns:
        doc_path = vllm_dir / dir_name / file_name
        if doc_path.is_file():
            content = doc_path.read_text(encoding="utf-8", errors="ignore")
            url = _get_doc_agent_url(root, doc_path) if repo_url else doc_path.as_posix()
            link_text = f"[{doc_path.as_posix()}]({url})" if repo_url else doc_path.as_posix()
            return f"📄 文档路径：{link_text}\n---\n\n{content}"

    return "未找到 Qwen3-8B 相关文档，请先运行生成任务。"


def get_tools() -> list:
    return [
        doc_agent_vllm_adapter_docs,
        doc_agent_sglang_adapter_docs,
        doc_agent_verl_adapter_docs,
        doc_agent_example_qwen3_8b,
        doc_agent_one_click_generate,
        doc_agent_generate_local,
        doc_agent_example_qwen35_flow,
    ]
