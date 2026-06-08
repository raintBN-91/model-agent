#!/usr/bin/env python3
"""Classify known Ascend adaptation error signatures from a log file."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class Signature:
    name: str
    pattern: str
    root_cause: str
    checks: str
    fix: str


SIGNATURES = [
    Signature(
        name="hccl-init-failed",
        pattern=r"HcclCommInitRootInfo|error code\s*is\s*7",
        root_cause="HCCL init failed because of port/config/device mapping.",
        checks="master_port, visible devices, hccn configuration",
        fix="Change master_port and verify hccn config mount and device mapping.",
    ),
    Signature(
        name="cpu-backend-fallback",
        pattern=r"No backend type associated with device type cpu",
        root_cause="Model/device setup fell back to CPU in distributed flow.",
        checks="rank-device binding and cfg.device parsing",
        fix="Bind NPU device explicitly per rank before model setup.",
    ),
    Signature(
        name="hccl-double-unsupported",
        pattern=r"Unsupported data type at::kDouble|float64",
        root_cause="float64 used on HCCL collective path.",
        checks="collective tensor dtype in logger/reducer",
        fix="Switch collective statistics tensors to float32.",
    ),
    Signature(
        name="autocast-device-unsupported",
        pattern=r"unsupported autocast device_type",
        root_cause="autocast device type is not supported by runtime.",
        checks="autocast context creation path",
        fix="Map to supported autocast device type or runtime API.",
    ),
    Signature(
        name="atc-include-missing",
        pattern=r"cstdint.*file not found",
        root_cause="ATC include/toolchain env not loaded.",
        checks="ASCEND_TOOLKIT_HOME and include path",
        fix="Load CANN environment script before ATC conversion.",
    ),
    Signature(
        name="oom",
        pattern=r"Out of memory|OOM",
        root_cause="Memory pressure exceeds device capacity.",
        checks="batch size, input size, amp usage",
        fix="Lower batch/input, enable AMP, use grad accumulation.",
    ),
]


def scan_log(text: str):
    results = []
    lines = text.splitlines()
    for sig in SIGNATURES:
        regex = re.compile(sig.pattern, re.IGNORECASE)
        hits = [line.strip() for line in lines if regex.search(line)]
        if hits:
            results.append((sig, hits))
    return results


def to_markdown(results) -> str:
    if not results:
        return "# Triage Report\n\nNo known signatures matched. Mark this case as `unknown` and perform manual root-cause analysis.\n"

    rows = [
        "| Signature | Matches | Root Cause | Priority Checks | Fix | Example |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for sig, hits in results:
        example = hits[0].replace("|", "\\|")[:140]
        rows.append(
            f"| {sig.name} | {len(hits)} | {sig.root_cause} | {sig.checks} | {sig.fix} | `{example}` |"
        )

    body = "\n".join(rows)
    return f"# Triage Report\n\n{body}\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify known Ascend adaptation error signatures")
    parser.add_argument("--log", required=True, help="Path to log file")
    parser.add_argument("--out", required=True, help="Output markdown report path")
    args = parser.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.out)

    if not log_path.exists():
        raise SystemExit(f"log file not found: {log_path}")

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    results = scan_log(text)
    report = to_markdown(results)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print(f"write {out_path}")
    print(f"matched signatures: {len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
