"""华为云 LTS 异步日志上报器 — 使用 LTS Python Producer SDK。"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Huawei LTS SDK 懒加载（避免模块导入失败阻塞启动）
_HAS_LTS_SDK = False
Producer = None
lts_store = None
Config = None
_SDK_INIT_ERROR: str | None = None

try:
    _SDK_ROOT = str(Path(__file__).resolve().parent.parent / "huaweicloud-lts-python-sdk")
    if _SDK_ROOT not in sys.path:
        sys.path.insert(0, _SDK_ROOT)
    from producer.core.producer import Producer
    from producer.model import lts_store
    from producer.model.config import Config
    _HAS_LTS_SDK = True
except ImportError as e:
    _SDK_INIT_ERROR = str(e)
    # SDK 未安装时正常降级，不影响主业务


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LTSLogger:
    """华为云 LTS 异步日志上报器。

    使用 LTS Python Producer SDK 直接上报。
    认证信息全部从环境变量读取，失败时降级（不影响主业务）。
    Producer 采用懒加载，在首次调用 log() 时才初始化，避免模块导入阶段阻塞。
    """

    def __init__(self) -> None:
        self._ak = os.getenv("A_A", "")
        self._sk = os.getenv("A_S", "")
        self._project_id = os.getenv("A_P", "")
        self._group_id = os.getenv("A_G", "")
        self._stream_id = os.getenv("A_ST", "")
        self._region = "cn-north-4"
        # 北京四接入端点（Producer SDK 使用 8102 端口）
        self._endpoint = "https://lts-access.cn-north-4.myhuaweicloud.com:8102"

        self._enabled = all(
            [
                self._ak,
                self._sk,
                self._project_id,
                self._group_id,
                self._stream_id,
            ]
        )
        self._producer: Any = None
        self._init_error: str | None = None

    def _build_producer(self) -> Any:
        global _HAS_LTS_SDK, Producer, Config
        if not _HAS_LTS_SDK:
            raise ImportError(f"Huawei LTS SDK 不可用: {_SDK_INIT_ERROR}")
        config = Config()
        config.endpoint = self._endpoint
        config.access_key = self._ak
        config.access_secret = self._sk
        config.region_id = self._region
        config.project_id = self._project_id
        # 调低阈值，让日志更快上报
        config.batch_count_threshold = 50
        config.linger_ms = 1000
        config.retries = 3

        producer = Producer.init_producer(config)
        producer.start_producer()
        return producer

    def _ensure_producer(self) -> bool:
        """懒加载 Producer，返回是否可用。"""
        if self._producer is not None:
            return True
        if not self._enabled:
            return False
        try:
            self._producer = self._build_producer()
            return True
        except Exception as e:
            self._enabled = False
            self._init_error = str(e)
            print(f"[LTS] Producer 初始化失败，降级为禁用：{e}", file=sys.stderr)
            return False

    def log(self, event: str, **kwargs: Any) -> None:
        """记录单条日志，通过 Producer SDK 异步批量上报。"""
        entry = {
            "timestamp": _now_iso(),
            "event": event,
            **kwargs,
        }

        if self._ensure_producer():
            log_str = json.dumps(entry, ensure_ascii=False, default=str)
            log = lts_store.generate_log([log_str], {"source": "mofix"})
            try:
                self._producer.send_log(self._group_id, self._stream_id, log)
            except Exception as e:
                # 发送异常时降级打印
                print(f"[LTS] send_log 失败：{e}", file=sys.stderr)
                print(f"[LTS] {entry}", file=sys.stderr)
        else:
            # LTS 未启用时降级打印到控制台
            print(f"[LTS] {entry}", file=sys.stderr)

    async def shutdown(self) -> None:
        """关闭前等待后台线程将剩余日志发送完成。"""
        if self._producer:
            # Producer 内部有缓冲和后台线程，给 3 秒时间 flush
            import asyncio
            await asyncio.sleep(3)


# 全局单例
lts_logger = LTSLogger()
