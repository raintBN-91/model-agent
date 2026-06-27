#!/usr/bin/env python3
"""
courts_forecasting_optimizer — jiuwen Agent Skill 最终版
Python ≥ 3.11, venv: /root/timesfm_env/timesfm/bin/python
特性: 自行探索 / 自适应采样 / 8卡并行 / 用户随时停止 / 异常暂停
"""

from __future__ import annotations

import os
import re
import json
import time
import shutil
import signal
import argparse
import subprocess
import threading
import itertools
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np


# ==================== jiuwen Context 接口 ====================

class Context:
    """
    jiuwen Agent 上下文接口（经验推测版）
    实际接入时根据jiuwen SDK调整方法名
    """
    
    def __init__(self):
        self._state: Dict[str, Any] = {}
        self._stop_flag = threading.Event()
        self._message_history: List[Dict] = []
        self._last_stream_id: Optional[str] = None
    
    # ---- 状态管理 ----
    def set_state(self, key: str, value: Any):
        self._state[key] = value
    
    def get_state(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)
    
    def clear_state(self):
        self._state.clear()
    
    # ---- 用户交互 ----
    def tell(self, message: str, msg_type: str = "text"):
        """向用户发送消息"""
        print(f"[AGENT→USER] {msg_type}: {message[:300]}...")
        self._message_history.append({"role": "agent", "content": message, "type": msg_type})
    
    def ask(self, question: str, options: Optional[List[str]] = None, timeout: int = 300) -> str:
        """向用户提问，等待回答"""
        print(f"[AGENT→USER] ASK: {question}")
        if options:
            print(f"  选项: {options}")
        return ""
    
    def stream(self, message: str, replace_last: bool = False):
        """流式输出进度"""
        prefix = "[STREAM]" if not replace_last else "[STREAM+REPLACE]"
        print(f"{prefix} {message[:200]}...")
    
    def show_table(self, headers: List[str], rows: List[List], title: Optional[str] = None):
        """展示表格"""
        if title:
            print(f"[TABLE] {title}")
        print(f"  headers: {headers}")
        for r in rows[:5]:
            print(f"  {r}")
    
    # ---- 停止检测 ----
    def check_stop(self) -> bool:
        """检测用户是否发送停止指令"""
        return self._stop_flag.is_set()
    
    def request_stop(self):
        """外部调用设置停止标志"""
        self._stop_flag.set()
    
    def wait_for_stop(self, timeout: float = 1.0) -> bool:
        return self._stop_flag.wait(timeout=timeout)


# ==================== 配置常量 ====================

PYTHON_ENV = "/root/timesfm_env/timesfm/bin/python"
SCRIPT_PATH = "load_courts_forecasting_ascend.py"
COMPARE_SCRIPT = "precision_compare.py"


# ==================== 数据模型 ====================

@dataclass
class ParamSpace:
    """参数空间 — 支持自然语言解析 + 自行探索"""
    
    ridge: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0])
    precision: List[str] = field(default_factory=lambda: ["fp32"])
    predict_step: List[int] = field(default_factory=lambda: [96])
    predict_length: List[int] = field(default_factory=lambda: [96])
    xreg_mode: List[str] = field(default_factory=lambda: ["xreg + timesfm"])
    
    # 自行探索标记
    _self_explore: bool = False
    _no_weather: bool = False
    _user_direction: str = ""
    
    @classmethod
    def from_natural_language(cls, text: str) -> "ParamSpace":
        """从自然语言解析参数空间"""
        space = cls()
        text_lower = text.lower()
        space._user_direction = text
        
        # ===== 检测自行探索模式 =====
        space._self_explore = any(w in text for w in [
            "自行", "你发挥", "你决定", "auto", "随便", "你调", "你探索"
        ])
        
        # ===== 检测无气象 =====
        space._no_weather = any(w in text_lower for w in [
            "无气象", "没有天气", "without weather", "no weather", "无天气"
        ])
        
        # ===== ridge解析 =====
        # 范围格式: "0-10", "0.001到1", "0.01~1.0"
        ridge_range = re.search(
            r'ridge.*?(\d+\.?\d*)\s*(?:到|~|-|\s)\s*(\d+\.?\d*)', 
            text_lower
        )
        if ridge_range:
            low, high = float(ridge_range.group(1)), float(ridge_range.group(2))
            # 自行探索: 更智能的采样
            if space._self_explore:
                space.ridge = cls._smart_ridge_sampling(low, high)
            else:
                space.ridge = np.round(
                    np.logspace(np.log10(max(low, 1e-6)), np.log10(high), 5), 6
                ).tolist()
        elif "ridge" in text_lower:
            # 尝试提取枚举值
            after_ridge = text_lower.split("ridge")[1].split("，")[0].split(",")[0]
            vals = re.findall(r'\d+\.?\d*', after_ridge)
            if vals:
                space.ridge = [float(v) for v in vals]
        
        # ===== precision =====
        if "fp16" in text_lower and "fp32" in text_lower:
            space.precision = ["fp16", "fp32"]
        elif "fp16" in text_lower:
            space.precision = ["fp16"]
        elif "fp32" in text_lower:
            space.precision = ["fp32"]
        # 自行探索且未指定: 默认fp32，不额外探索
        
        # ===== predict_step / predict_length =====
        step_match = re.search(r'(?:step|步长|预测长度).*?(\d+)', text_lower)
        if step_match:
            step = int(step_match.group(1))
            space.predict_step = [step]
            space.predict_length = [step]
        if "固定" in text and "96" in text:
            space.predict_step = [96]
            space.predict_length = [96]
        if "192" in text:
            space.predict_step = [96, 192]
            space.predict_length = [96, 192]
        
        # ===== xreg_mode =====
        if "只用timesfm" in text or "不用xreg" in text or "纯timesfm" in text:
            space.xreg_mode = ["timesfm"]
        elif "xreg" in text_lower and "timesfm" in text_lower:
            space.xreg_mode = ["xreg + timesfm"]
        # 自行探索 + 无气象: 自动对比两种模式！
        elif space._self_explore and space._no_weather:
            space.xreg_mode = ["timesfm", "xreg + timesfm"]
            print(f"[探索模式] 无气象场景，自动对比 xreg_mode: {space.xreg_mode}")
        
        return space
    
    @staticmethod
    def _smart_ridge_sampling(low: float, high: float) -> List[float]:
        """智能ridge采样: 对数均匀 + 端点加密"""
        # 基础5点对数均匀
        base = np.logspace(np.log10(max(low, 1e-6)), np.log10(high), 5)
        # 端点附近加密（最优常在边界）
        extra_low = base[0] * 2 if len(base) > 1 else base[0]
        extra_high = base[-1] / 2 if len(base) > 1 else base[-1]
        combined = sorted(set(list(base) + [extra_low, extra_high]))
        return np.round(combined, 6).tolist()
    
    def to_grid(self) -> List[Dict[str, Any]]:
        """全量网格"""
        keys, values = [], []
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if isinstance(v, list) and len(v) > 0:
                keys.append(k)
                values.append(v)
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    def get_exploration_plan(self) -> str:
        """生成探索计划说明"""
        lines = [
            f"**参数空间**:",
            f"  - ridge: {self.ridge}",
            f"  - precision: {self.precision}",
            f"  - predict_step: {self.predict_step}",
            f"  - xreg_mode: {self.xreg_mode}",
        ]
        if self._no_weather:
            lines.append(f"  - weather_enabled: false (无气象)")
        return "\n".join(lines)


@dataclass
class Experiment:
    """单个实验"""
    trial_id: str
    config_dir: Path
    round_num: int = 1
    device_id: int = -1
    status: str = "pending"
    result_csv: Optional[Path] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    log_file: Path = field(default_factory=Path)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_info: Optional[str] = None
    
    @property
    def duration_sec(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def mape(self) -> float:
        return self.metrics.get("MAPE", float('inf'))


# ==================== 核心类 ====================

class ConfigAutoGenerator:
    """自动拷贝模板 + 修改配置"""
    
    def __init__(self, base_config_dir: Path, output_base_dir: Path, param_space: ParamSpace):
        self.base_config_dir = Path(base_config_dir)
        self.output_base_dir = Path(output_base_dir)
        self.param_space = param_space
        self.base_business = self._load_json(self.base_config_dir / "business_config.json")
        self.base_model = self._load_json(self.base_config_dir / "model_config.json")
        
    def _load_json(self, path: Path) -> Dict:
        with open(path) as f:
            return json.load(f)
    
    def _make_trial_id(self, params: Dict, round_num: int) -> str:
        """生成实验ID"""
        parts = [f"R{round_num}"]
        for k, v in sorted(params.items()):
            short = {
                "ridge": "r", "precision": "p", "predict_step": "s",
                "predict_length": "pl", "xreg_mode": "x"
            }.get(k, k[:3])
            val = str(v).replace(" ", "").replace("+", "-")[:10]
            # xreg_mode特殊处理
            if k == "xreg_mode" and "timesfm" in str(v) and "xreg" not in str(v):
                val = "notime"  # 纯timesfm缩写
            parts.append(f"{short}{val}")
        return "_".join(parts)
    
    def generate(self, params: Dict, round_num: int, exp_base_dir: Path) -> Experiment:
        """生成实验配置"""
        trial_id = self._make_trial_id(params, round_num)
        config_dir = exp_base_dir / f"exp_{trial_id}"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # 拷贝并修改
        business = dict(self.base_business)
        model = dict(self.base_model)
        
        # business修改
        for k in ["predict_step", "predict_length"]:
            if k in params:
                business[k] = params[k]
        
        # 无气象强制关闭
        if getattr(self.param_space, '_no_weather', False):
            business["weather_enabled"] = False
        
        step = business.get("predict_step", 96)
        pred_hours = step // 4  # 15min频率
        output_name = f"all_usrs_forecasting_{pred_hours}h_result_{trial_id}_npu.csv"
        business["output_file_path"] = str(self.output_base_dir / output_name)
        
        # model修改
        for k in ["ridge", "precision", "xreg_mode"]:
            if k in params:
                model[k] = params[k]
        
        # 写入
        with open(config_dir / "business_config.json", "w") as f:
            json.dump(business, f, indent=2)
        with open(config_dir / "model_config.json", "w") as f:
            json.dump(model, f, indent=2)
        
        return Experiment(
            trial_id=trial_id,
            config_dir=config_dir,
            round_num=round_num,
            log_file=config_dir / "run.log"
        )


class NPUScheduler:
    """8卡NPU调度"""
    
    def __init__(self, n_devices: int = 8):
        self.n_devices = n_devices
        self.lock = threading.Lock()
        self.device_status = {i: "idle" for i in range(n_devices)}
        self.faulty_devices: Set[int] = set()
        self.running_exps: Dict[int, str] = {}
        
    def acquire(self, trial_id: str, timeout: float = 300) -> Optional[int]:
        """获取空闲卡"""
        start = time.time()
        while time.time() - start < timeout:
            with self.lock:
                for dev_id in range(self.n_devices):
                    if dev_id in self.faulty_devices:
                        continue
                    if self.device_status[dev_id] == "idle":
                        self.device_status[dev_id] = "running"
                        self.running_exps[dev_id] = trial_id
                        return dev_id
                
                healthy = self.n_devices - len(self.faulty_devices)
                if healthy <= 0:
                    return None
            
            time.sleep(2)
        return None
    
    def release(self, device_id: int, is_faulty: bool = False):
        """释放卡"""
        with self.lock:
            if is_faulty:
                self.faulty_devices.add(device_id)
                print(f"[NPU] 设备 {device_id} 标记故障，已隔离")
            self.device_status[device_id] = "idle"
            self.running_exps.pop(device_id, None)
    
    def get_status(self) -> Tuple[str, List[str]]:
        """获取状态"""
        with self.lock:
            bars = []
            for i in range(self.n_devices):
                if i in self.faulty_devices:
                    bars.append("❌")
                elif self.device_status[i] == "idle":
                    bars.append("🟢")
                else:
                    bars.append("🔴")
            info = [f"NPU:{k}->{v}" for k, v in self.running_exps.items()]
            return "".join(bars), info
    
    def all_faulty(self) -> bool:
        return len(self.faulty_devices) >= self.n_devices
    
    def get_healthy_count(self) -> int:
        return self.n_devices - len(self.faulty_devices)


class ExperimentRunner:
    """实验执行器"""
    
    def __init__(self, scheduler: NPUScheduler, stop_event: threading.Event):
        self.scheduler = scheduler
        self.stop_event = stop_event
        self.python_env = PYTHON_ENV
        
    def run(self, exp: Experiment) -> Experiment:
        """运行单个实验"""
        # 等待NPU或停止
        while not self.stop_event.is_set():
            device_id = self.scheduler.acquire(exp.trial_id, timeout=5)
            if device_id is not None:
                break
            if self.scheduler.all_faulty():
                exp.status = "paused"
                exp.error_info = "全部NPU故障"
                return exp
            time.sleep(1)
        else:
            exp.status = "stopped"
            return exp
        
        exp.device_id = device_id
        exp.start_time = datetime.now()
        exp.status = "running"
        
        # 更新device_id
        model_cfg_path = exp.config_dir / "model_config.json"
        with open(model_cfg_path) as f:
            model_cfg = json.load(f)
        model_cfg["device_id"] = device_id
        with open(model_cfg_path, "w") as f:
            json.dump(model_cfg, f, indent=2)
        
        # 启动
        cmd = (
            f"nohup {self.python_env} {SCRIPT_PATH} "
            f"--config-dir {exp.config_dir} "
            f"> {exp.log_file} 2>&1 &"
        )
        subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 轮询等待
        expected_output = Path(self._get_output_path(exp))
        check_interval = 15
        
        while not self.stop_event.is_set():
            time.sleep(check_interval)
            
            if self._check_complete(expected_output):
                exp.status = "done"
                exp.result_csv = expected_output
                exp.end_time = datetime.now()
                self.scheduler.release(device_id)
                return exp
            
            fail_reason = self._check_failed(exp)
            if fail_reason:
                exp.status = "failed"
                exp.error_info = fail_reason
                exp.end_time = datetime.now()
                is_faulty = any(x in fail_reason.lower() for x in ["acl", "npu", "rtError"])
                self.scheduler.release(device_id, is_faulty=is_faulty)
                return exp
        
        # 用户停止
        exp.status = "stopped"
        self._kill_nohup(exp)
        self.scheduler.release(device_id)
        return exp
    
    def _get_output_path(self, exp: Experiment) -> str:
        with open(exp.config_dir / "business_config.json") as f:
            return json.load(f)["output_file_path"]
    
    def _check_complete(self, path: Path) -> bool:
        if not path.exists() or path.stat().st_size < 100:
            return False
        size1 = path.stat().st_size
        time.sleep(3)
        if not path.exists():
            return False
        return size1 == path.stat().st_size
    
    def _check_failed(self, exp: Experiment) -> Optional[str]:
        if not exp.log_file.exists():
            return None
        
        log = exp.log_file.read_text()
        
        fail_patterns = {
            "RuntimeError": "运行错误",
            "OutOfMemory": "显存溢出",
            "Segmentation fault": "段错误",
            "aclError": "NPU驱动错误",
            "rtError": "NPU运行时错误",
            "CUDA out of memory": "显存不足",
            "Killed": "进程被杀死",
        }
        
        for pattern, reason in fail_patterns.items():
            if pattern in log:
                return reason
        
        # 检查僵死
        if not exp.result_csv or not exp.result_csv.exists():
            log_mtime = datetime.fromtimestamp(exp.log_file.stat().st_mtime)
            if (datetime.now() - log_mtime).seconds > 300:
                return "进程僵死"
        
        return None
    
    def _kill_nohup(self, exp: Experiment):
        """停止进程"""
        try:
            # 通过日志目录匹配
            subprocess.run(
                ["pkill", "-f", f"config-dir {exp.config_dir}"],
                capture_output=True
            )
        except Exception:
            pass


class AdaptiveSampler:
    """自适应采样器"""
    
    def __init__(self, param_space: ParamSpace):
        self.space = param_space
        self.history: List[Tuple[Dict, float]] = []
        self.round_results: Dict[int, List[Experiment]] = {}
        
    def generate_round(self, round_num: int, n_samples: int = 8) -> List[Dict]:
        """生成一轮采样"""
        if round_num == 1 or not self.history:
            return self._first_round(n_samples)
        return self._adaptive_round(n_samples)
    
    def _first_round(self, n: int) -> List[Dict]:
        """第一轮：均匀覆盖"""
        # 主要探索ridge，其他固定
        ridge_vals = self.space.ridge
        if len(ridge_vals) > n:
            indices = np.linspace(0, len(ridge_vals)-1, min(n, len(ridge_vals)), dtype=int)
            ridge_vals = [ridge_vals[i] for i in indices]
        
        params_list = []
        for r in ridge_vals:
            p = {"ridge": r}
            # 固定其他参数
            if len(self.space.precision) == 1:
                p["precision"] = self.space.precision[0]
            if len(self.space.predict_step) == 1:
                p["predict_step"] = self.space.predict_step[0]
                p["predict_length"] = self.space.predict_step[0]
            
            # 无气象场景：生成两种xreg_mode对比
            if len(self.space.xreg_mode) > 1:
                for xreg in self.space.xreg_mode:
                    p_copy = dict(p)
                    p_copy["xreg_mode"] = xreg
                    params_list.append(p_copy)
            else:
                if self.space.xreg_mode:
                    p["xreg_mode"] = self.space.xreg_mode[0]
                params_list.append(p)
        
        return params_list[:n]
    
    def _adaptive_round(self, n: int) -> List[Dict]:
        """自适应聚焦"""
        if not self.history:
            return self._first_round(n)
        
        # 按MAPE排序，找最优
        sorted_hist = sorted(self.history, key=lambda x: x[1])
        best_params = sorted_hist[0][0]
        best_ridge = best_params.get("ridge", 0.01)
        best_mape = sorted_hist[0][1]
        
        # 在最优ridge附近加密
        candidates = [
            best_ridge * 0.5,
            best_ridge * 0.7,
            best_ridge * 0.85,
            best_ridge,
            best_ridge * 1.15,
            best_ridge * 1.3,
            best_ridge * 2.0,
        ]
        
        # 限制范围
        all_ridges = self.space.ridge
        min_r, max_r = min(all_ridges), max(all_ridges)
        candidates = [max(min_r, min(max_r, c)) for c in candidates]
        candidates = sorted(list(set(np.round(candidates, 6))))
        
        # 取前n个，保持其他最优参数
        params_list = []
        for r in candidates[:n]:
            p = {"ridge": r}
            for k in ["precision", "predict_step", "predict_length", "xreg_mode"]:
                if k in best_params:
                    p[k] = best_params[k]
            params_list.append(p)
        
        return params_list
    
    def update_history(self, params: Dict, mape: float):
        self.history.append((params, mape))


class AutoTunerAgent:
    """自动调参Agent"""
    
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.stop_event = threading.Event()
        self.scheduler = NPUScheduler(n_devices=8)
        self.runner = ExperimentRunner(self.scheduler, self.stop_event)
        self.experiments: List[Experiment] = []
        self.current_round = 0
        
    def run(self,
            user_intent: str,
            base_config_dir: str = "config",
            true_csv: str = "data/usr/all_usrs_high_quality_interpolate.csv",
            predict_start: str = "2023-12-01 00:00:00",
            predict_end: str = "2023-12-01 23:45:00",
            sampling: str = "adaptive",
            max_rounds: int = 3,
            top_n: int = 3,
            area_top_n: int = 5) -> Dict:
        
        """主入口"""
        
        # 解析参数空间
        param_space = ParamSpace.from_natural_language(user_intent)
        
        # 确定采样策略
        if "grid" in user_intent or "全量" in user_intent:
            sampling = "grid"
        elif "random" in user_intent or "随机" in user_intent:
            sampling = "random"
        else:
            sampling = "adaptive"
        
        # 生成实验计划
        if sampling == "grid":
            all_trials = param_space.to_grid()
            trial_batches = [all_trials]
            total_planned = len(all_trials)
        elif sampling == "random":
            all_trials = param_space.to_random(n=20)
            trial_batches = [all_trials]
            total_planned = len(all_trials)
        else:  # adaptive
            sampler = AdaptiveSampler(param_space)
            trial_batches = None  # 动态生成
            total_planned = "动态"
        
        # 发送计划
        plan_msg = self._build_plan_message(param_space, sampling, total_planned, max_rounds)
        self.ctx.tell(plan_msg)
        
        # 启动停止监听
        stop_thread = threading.Thread(target=self._listen_stop)
        stop_thread.daemon = True
        stop_thread.start()
        
        # 执行
        exp_base_dir = Path(f"experiments/auto_{datetime.now():%Y%m%d_%H%M%S}")
        config_gen = ConfigAutoGenerator(base_config_dir, "infer_result", param_space)
        
        if sampling == "adaptive":
            # 逐轮执行
            for round_num in range(1, max_rounds + 1):
                if self.stop_event.is_set():
                    break
                
                self.current_round = round_num
                trials = sampler.generate_round(round_num, n_samples=self.scheduler.get_healthy_count())
                
                if not trials:
                    break
                
                self.ctx.tell(f"\n📍 **Round {round_num}/{max_rounds}** | 生成 {len(trials)} 组实验")
                
                # 执行本轮
                round_exps = self._execute_round(trials, round_num, config_gen, exp_base_dir)
                
                # 评测本轮
                self._evaluate_round(round_exps, true_csv, predict_start, predict_end, exp_base_dir)
                
                # 更新sampler历史
                for exp in round_exps:
                    if exp.status == "done" and exp.metrics:
                        sampler.update_history(
                            {"ridge": exp.metrics.get("ridge", 0.01)},  # 简化
                            exp.mape
                        )
                
                # 汇报本轮
                self._report_round(round_exps, round_num)
                
                # 检查全部故障
                if self.scheduler.all_faulty():
                    self.ctx.tell("❌ **全部NPU故障，暂停执行！** 请检查硬件后重试。")
                    break
                
                # 检查是否收敛（最优MAPE连续两轮变化<1%）
                if round_num >= 2 and self._check_convergence(round_num):
                    self.ctx.tell("✅ **已收敛，提前结束！**")
                    break
        
        else:
            # grid/random: 一次性执行
            exps = self._execute_round(trial_batches[0], 1, config_gen, exp_base_dir)
            self._evaluate_round(exps, true_csv, predict_start, predict_end, exp_base_dir)
        
        # 最终报告
        return self._build_final_report(top_n, area_top_n, exp_base_dir, param_space)
    
    def _build_plan_message(self, param_space: ParamSpace, sampling: str, total: Any, max_rounds: int) -> str:
        """构建计划消息"""
        lines = [
            f"🎯 **优化任务启动**",
            f"",
            f"**采样策略**: {sampling}",
            f"",
        ]
        
        if param_space._self_explore:
            lines.extend([
                f"**🤖 自行探索模式**",
                f"您的方向: {param_space._user_direction[:100]}...",
                f"",
            ])
        
        lines.extend([
            param_space.get_exploration_plan(),
            f"",
            f"**实验计划**: {total} 组（8卡并行）",
        ])
        
        if sampling == "adaptive":
            lines.append(f"**轮次**: 最多 {max_rounds} 轮，收敛提前结束")
        
        lines.extend([
            f"",
            f"**停止方式**: 说\"停\"立即结束",
            f"",
            f"开始执行...",
        ])
        
        return "\n".join(lines)
    
    def _listen_stop(self):
        """监听停止"""
        while not self.stop_event.is_set():
            if self.ctx.check_stop():
                self.stop_event.set()
                self.ctx.tell("🛑 **收到停止指令，正在优雅终止...**")
                break
            time.sleep(1)
    
    def _execute_round(self, trials: List[Dict], round_num: int,
                       config_gen: ConfigAutoGenerator, exp_base_dir: Path) -> List[Experiment]:
        """执行一轮"""
        exps = []
        for params in trials:
            exp = config_gen.generate(params, round_num, exp_base_dir)
            exps.append(exp)
            self.experiments.append(exp)
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(self.runner.run, exp): exp for exp in exps}
            
            completed = 0
            for future in as_completed(futures):
                if self.stop_event.is_set():
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    break
                
                exp = futures[future]
                try:
                    future.result()
                except Exception as e:
                    exp.status = "failed"
                    exp.error_info = str(e)
                
                completed += 1
                self._stream_progress(exp, completed, len(exps))
        
        return exps
    
    def _evaluate_round(self, exps: List[Experiment], true_csv: str,
                        predict_start: str, predict_end: str, exp_base_dir: Path):
        """评测一轮"""
        done_exps = [e for e in exps if e.status == "done" and e.result_csv]
        if not done_exps:
            return
        
        eval_dir = exp_base_dir / "eval_results"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        pred_args = [f"{e.trial_id}:{e.result_csv}" for e in done_exps]
        cmd = [
            PYTHON_ENV, COMPARE_SCRIPT,
            "--true", true_csv,
            "--pred"
        ] + pred_args + [
            "--start", predict_start,
            "--end", predict_end,
            "--output", str(eval_dir),
            "--date-col", "datetime"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            summary_csv = eval_dir / "summary.csv"
            if summary_csv.exists():
                df = pd.read_csv(summary_csv)
                for _, row in df.iterrows():
                    tid = row["name"]
                    for exp in done_exps:
                        if exp.trial_id == tid:
                            exp.metrics = {
                                "MAPE": row.get("overall_MAPE", np.nan),
                                "RMSE": row.get("overall_RMSE", np.nan),
                                "MAE": row.get("overall_MAE", np.nan),
                                "SMAPE": row.get("overall_SMAPE", np.nan),
                            }
                            # 解析参数到metrics
                            if "r" in tid:
                                try:
                                    r_val = float(re.search(r'r([\d.]+)', tid).group(1))
                                    exp.metrics["ridge"] = r_val
                                except:
                                    pass
                            if "notime" in tid:
                                exp.metrics["xreg_mode"] = "timesfm"
                            elif "xreg" in tid or "x" in tid:
                                exp.metrics["xreg_mode"] = "xreg + timesfm"
        except Exception as e:
            print(f"[ERROR] 评测失败: {e}")
    
    def _stream_progress(self, exp: Experiment, completed: int, total: int):
        """流式进度"""
        status_emoji = {"done": "✅", "failed": "❌", "stopped": "🛑"}.get(exp.status, "❓")
        
        npu_bars, running_info = self.scheduler.get_status()
        
        msg = (
            f"\n🔄 Round {self.current_round} | {completed}/{total} 完成\n"
            f"  {status_emoji} {exp.trial_id}: {exp.status}"
        )
        if exp.metrics:
            msg += f" | MAPE: {exp.mape:.2f}%"
        if exp.duration_sec > 0:
            msg += f" | 耗时: {exp.duration_sec:.0f}s"
        
        msg += f"\n  NPU: {npu_bars}"
        
        self.ctx.stream(msg)
    
    def _report_round(self, exps: List[Experiment], round_num: int):
        """汇报轮次"""
        done = [e for e in exps if e.status == "done" and e.metrics]
        if not done:
            return
        
        done.sort(key=lambda x: x.mape)
        best = done[0]
        
        # 区分xreg_mode对比
        xreg_timesfm = [e for e in done if e.metrics.get("xreg_mode") == "xreg + timesfm"]
        pure_timesfm = [e for e in done if e.metrics.get("xreg_mode") == "timesfm"]
        
        lines = [f"\n📊 **Round {round_num} 完成** | 成功: {len(done)}/{len(exps)}"]
        
        if xreg_timesfm and pure_timesfm:
            avg_xreg = np.mean([e.mape for e in xreg_timesfm])
            avg_pure = np.mean([e.mape for e in pure_timesfm])
            better = "xreg + timesfm" if avg_xreg < avg_pure else "纯timesfm"
            lines.append(f"  xreg+timesfm平均MAPE: {avg_xreg:.2f}% | 纯timesfm: {avg_pure:.2f}% | 当前更优: {better}")
        
        lines.append(f"  最优: {best.trial_id} | MAPE: {best.mape:.2f}% | ridge: {best.metrics.get('ridge', 'N/A')}")
        
        self.ctx.tell("\n".join(lines))
    
    def _check_convergence(self, current_round: int) -> bool:
        """检查是否收敛"""
        # 简化：检查最近两轮最优MAPE变化
        round_exps = defaultdict(list)
        for e in self.experiments:
            if e.status == "done" and e.metrics:
                round_exps[e.round_num].append(e)
        
        if current_round - 1 not in round_exps or current_round not in round_exps:
            return False
        
        prev_best = min(e.mape for e in round_exps[current_round - 1])
        curr_best = min(e.mape for e in round_exps[current_round])
        
        return abs(prev_best - curr_best) / max(prev_best, 0.001) < 0.01
    
    def _build_final_report(self, top_n: int, area_top_n: int, 
                           exp_base_dir: Path, param_space: ParamSpace) -> Dict:
        """构建最终报告"""
        done_exps = [e for e in self.experiments if e.status == "done" and e.metrics]
        
        if not done_exps:
            return {
                "type": "error",
                "report_markdown": "❌ 无成功完成的实验",
                "best_config": None
            }
        
        done_exps.sort(key=lambda x: x.mape)
        best_n = done_exps[:top_n]
        worst_n = done_exps[-top_n:][::-1]
        
        lines = [
            f"# 🎯 台区负荷预测优化报告",
            f"",
            f"**执行摘要**: 共启动 {len(self.experiments)} 组实验，成功完成 {len(done_exps)} 组",
            f"**终止原因**: {'用户停止' if self.stop_event.is_set() else '全部完成/收敛'}",
            f"**无气象模式**: {'是' if param_space._no_weather else '否'}",
            f"",
            f"---",
            f"",
            f"## 🏆 全局最优 TOP-{top_n}",
            f"",
        ]
        
        for i, exp in enumerate(best_n, 1):
            medal = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else f"{i}."
            xreg_label = exp.metrics.get("xreg_mode", "")
            xreg_tag = f" [{xreg_label}]" if xreg_label else ""
            
            lines.extend([
                f"### {medal} {exp.trial_id}{xreg_tag}",
                f"",
                f"| 指标 | 数值 |",
                f"|------|------|",
                f"| MAPE | **{exp.mape:.4f}%** |",
                f"| RMSE | {exp.metrics.get('RMSE', 'N/A'):.4f} |",
                f"| MAE | {exp.metrics.get('MAE', 'N/A'):.4f} |",
                f"| ridge | {exp.metrics.get('ridge', 'N/A')} |",
                f"| 耗时 | {exp.duration_sec:.0f}s |",
                f"| 配置 | `{exp.config_dir}` |",
                f"| 结果 | `{exp.result_csv}` |",
                f"",
            ])
            
            # 台区级明细
            area_csv = exp_base_dir / "eval_results" / f"{exp.trial_id}_area_metrics.csv"
            if not area_csv.exists():
                # 尝试其他路径
                area_csv = Path("eval_results") / f"{exp.trial_id}_area_metrics.csv"
            
            if area_csv.exists():
                try:
                    df_area = pd.read_csv(area_csv).sort_values("MAPE")
                    best_areas = df_area.head(area_top_n)
                    worst_areas = df_area.tail(area_top_n).iloc[::-1]
                    
                    lines.extend([
                        f"<details>",
                        f"<summary>📍 台区级明细（点击展开）</summary>",
                        f"",
                        f"**最优{area_top_n}台区**:",
                        f"",
                        f"| 排名 | 台区 | MAPE | 点数 |",
                        f"|------|------|------|------|",
                    ])
                    for idx, row in best_areas.iterrows():
                        lines.append(f"| 🥇 | {row['area']} | {row['MAPE']:.2f}% | {row.get('points', 'N/A')} |")
                    
                    lines.extend([
                        f"",
                        f"**最差{area_top_n}台区**:",
                        f"",
                        f"| 排名 | 台区 | MAPE | 点数 |",
                        f"|------|------|------|------|",
                    ])
                    for idx, row in worst_areas.iterrows():
                        lines.append(f"| 💥 | {row['area']} | {row['MAPE']:.2f}% | {row.get('points', 'N/A')} |")
                    
                    lines.extend([f"", f"</details>", f""])
                except Exception as e:
                    lines.append(f"*台区级明细读取失败: {e}*")
        
        # 最差
        lines.extend([
            f"---",
            f"",
            f"## 💥 全局最差 TOP-{top_n}（避坑参考）",
            f"",
        ])
        for i, exp in enumerate(worst_n, 1):
            xreg_tag = f" [{exp.metrics.get('xreg_mode', '')}]" if exp.metrics.get('xreg_mode') else ""
            lines.append(f"{i}. **{exp.trial_id}**{xreg_tag} | MAPE: {exp.mape:.2f}% | ridge: {exp.metrics.get('ridge', 'N/A')}")
        
        # 建议
        lines.extend([
            f"",
            f"---",
            f"",
            f"## 💡 智能建议",
            f"",
            self._generate_suggestions(best_n[0] if best_n else None, done_exps, param_space),
            f"",
            f"---",
            f"",
            f"**实验目录**: `{exp_base_dir}`",
            f"**评测结果**: `{exp_base_dir}/eval_results/`",
        ])
        
        return {
            "type": "success",
            "report_markdown": "\n".join(lines),
            "best_config": {
                "trial_id": best_n[0].trial_id,
                "params": {
                    "ridge": best_n[0].metrics.get("ridge"),
                    "precision": best_n[0].metrics.get("precision", "fp32"),
                    "predict_step": best_n[0].metrics.get("predict_step", 96),
                    "xreg_mode": best_n[0].metrics.get("xreg_mode", "xreg + timesfm"),
                },
                "mape": best_n[0].mape,
                "config_dir": str(best_n[0].config_dir),
                "result_csv": str(best_n[0].result_csv) if best_n[0].result_csv else None
            },
            "all_results": [
                {
                    "trial_id": e.trial_id,
                    "status": e.status,
                    "mape": e.mape,
                    "duration_sec": e.duration_sec,
                    "error": e.error_info,
                    "xreg_mode": e.metrics.get("xreg_mode"),
                }
                for e in self.experiments
            ],
            "experiment_dir": str(exp_base_dir)
        }
    
    def _generate_suggestions(self, best_exp: Optional[Experiment], 
                              all_exps: List[Experiment],
                              param_space: ParamSpace) -> str:
        """生成建议"""
        if not best_exp:
            return "无足够数据"
        
        suggestions = [
            f"### 生产推荐配置",
            f"",
            f"```json",
            f"{{",
            f'  "ridge": {best_exp.metrics.get("ridge", 0.1)},',
            f'  "precision": "{best_exp.metrics.get("precision", "fp32")}",',
            f'  "predict_step": {best_exp.metrics.get("predict_step", 96)},',
            f'  "xreg_mode": "{best_exp.metrics.get("xreg_mode", "xreg + timesfm")}"',
            f"}}",
            f"```",
            f"",
            f"### 关键发现",
        ]
        
        # xreg_mode对比分析
        xreg_exps = [e for e in all_exps if e.metrics.get("xreg_mode") == "xreg + timesfm"]
        pure_exps = [e for e in all_exps if e.metrics.get("xreg_mode") == "timesfm"]
        
        if xreg_exps and pure_exps:
            avg_xreg = np.mean([e.mape for e in xreg_exps])
            avg_pure = np.mean([e.mape for e in pure_exps])
            better = "xreg + timesfm" if avg_xreg < avg_pure else "纯timesfm"
            suggestions.append(f"- 无气象场景下，**{better}** 更优")
            suggestions.append(f"  - xreg+timesfm平均MAPE: {avg_xreg:.2f}%")
            suggestions.append(f"  - 纯timesfm平均MAPE: {avg_pure:.2f}%")
        
        # ridge影响
        ridge_mapes = [(e.metrics.get("ridge", 0), e.mape) 
                       for e in all_exps if "ridge" in e.metrics]
        if ridge_mapes:
            ridge_mapes.sort()
            suggestions.append(f"- 最优ridge区间: 约在 {best_exp.metrics.get('ridge', '?')} 附近")
        
        suggestions.extend([
            f"",
            f"### 后续优化",
            f"- 在最优ridge ±20% 范围进一步细搜",
            f"- 对最差台区单独分析数据质量",
        ])
        
        if param_space._no_weather:
            suggestions.append(f"- 建议尝试启用weather_enabled，对比气象输入效果")
        
        return "\n".join(suggestions)


# ==================== jiuwen Skill入口 ====================

class CourtsForecastingSkill:
    """jiuwen Skill注册类"""
    
    name = "courts_forecasting_optimizer"
    display_name = "台区负荷预测自动调参优化"
    
    def handle(self, ctx: Context, **kwargs) -> Dict:
        """
        jiuwen调用入口
        """
        # 提取用户意图
        user_intent = kwargs.get("user_intent", kwargs.get("message", ""))
        if not user_intent:
            ctx.tell("请描述您想优化的参数范围，例如：\n- 'ridge在0.001到1之间，无气象，你自行探索'\n- '无气象，ridge你调，其他你发挥，adaptive跑'")
            return {"type": "waiting_input"}
        
        # 创建Agent执行
        agent = AutoTunerAgent(ctx)
        
        result = agent.run(
            user_intent=user_intent,
            base_config_dir=kwargs.get("base_config_dir", "config"),
            true_csv=kwargs.get("true_csv", "data/usr/all_usrs_high_quality_interpolate.csv"),
            predict_start=kwargs.get("predict_start", "2023-12-01 00:00:00"),
            predict_end=kwargs.get("predict_end", "2023-12-01 23:45:00"),
            sampling=kwargs.get("sampling", "adaptive"),
            max_rounds=kwargs.get("max_rounds", 3),
            top_n=kwargs.get("top_n", 3),
            area_top_n=kwargs.get("area_top_n", 5)
        )
        
        # 发送最终报告
        if "report_markdown" in result:
            ctx.tell(result["report_markdown"], msg_type="markdown")
        
        return result


# ==================== 本地测试入口 ====================

def test_local():
    """本地命令行测试"""
    class MockContext(Context):
        def check_stop(self):
            # 模拟：运行30秒后自动停止，或按Ctrl+C
            return False
    
    ctx = MockContext()
    skill = CourtsForecastingSkill()
    
    import sys
    intent = sys.argv[1] if len(sys.argv) > 1 else "无气象，ridge 0-10你自行探索，step 96，adaptive"
    
    print(f"=" * 60)
    print(f"测试意图: {intent}")
    print(f"=" * 60)
    
    result = skill.handle(ctx, user_intent=intent)
    
    print("\n" + "=" * 60)
    print("RESULT KEYS:", result.keys())
    if result.get("best_config"):
        print("BEST CONFIG:", json.dumps(result["best_config"], indent=2, default=str))


if __name__ == "__main__":
    test_local()