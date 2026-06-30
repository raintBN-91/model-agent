from __future__ import annotations
"""Layer D: MCP 调用优化。

记录每次 MCP 调用的响应时间和成功率，智能选择最快/最可靠的服务器。
"""


from typing import Any

from engine.experience.store import ExperienceStore


class MCPOptimLayer:
    """MCP 调用优化层。"""

    def __init__(self, store: ExperienceStore):
        self._store = store

    def record_call(
        self, server: str, tool: str, duration: float, success: bool
    ) -> None:
        """记录一次 MCP 调用。"""
        self._store.append_layer_array("mcp_optim", {
            "server": server,
            "tool": tool,
            "duration": round(duration, 3),
            "success": success,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        })

    def suggest_server(self, tool: str) -> str | None:
        """根据历史性能建议最佳服务器。"""
        data = self._store.load_layer_data("mcp_optim")
        entries = data.get("entries", [])
        if not entries:
            return None

        # 按工具筛选
        relevant = [e for e in entries if e.get("tool") == tool and e.get("success")]
        if not relevant:
            return None

        # 按服务器分组计算平均耗时
        server_stats: dict[str, list[float]] = {}
        for e in relevant:
            server = e.get("server", "")
            server_stats.setdefault(server, []).append(e.get("duration", 0))

        best_server = min(
            server_stats,
            key=lambda s: sum(server_stats[s]) / len(server_stats[s]),
        )
        return best_server

    def get_server_stats(self, tool: str | None = None) -> dict:
        """获取服务器统计信息。"""
        data = self._store.load_layer_data("mcp_optim")
        entries = data.get("entries", [])
        if tool:
            entries = [e for e in entries if e.get("tool") == tool]

        stats: dict[str, dict] = {}
        for e in entries:
            server = e.get("server", "?")
            if server not in stats:
                stats[server] = {"calls": 0, "success": 0, "total_duration": 0.0}
            stats[server]["calls"] += 1
            if e.get("success"):
                stats[server]["success"] += 1
            stats[server]["total_duration"] += e.get("duration", 0)

        for s in stats.values():
            s["avg_duration"] = round(
                s["total_duration"] / s["calls"], 3
            ) if s["calls"] else 0
            s["success_rate"] = round(
                s["success"] / s["calls"], 3
            ) if s["calls"] else 0
        return stats
