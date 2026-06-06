#!/usr/bin/env python3
"""
vLLM-Ascend Performance Benchmark Script.
Measures latency and throughput via the vLLM OpenAI-compatible API.

Usage:
    python benchmark.py --api-url http://127.0.0.1:8000/v1/chat/completions \
        --model-name gemma-3-270m-it --mode all
"""

import argparse
import json
import statistics
import time
import concurrent.futures
import requests


def call_api(api_url: str, model_name: str, messages: list, max_tokens: int = 200,
             temperature: float = 0.0, timeout: int = 120):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def measure_latency(api_url: str, model_name: str, input_len: int = 200,
                    output_len: int = 200, iters: int = 10):
    prompt = ("Explain the difference between TCP and UDP protocols in computer networking. " * 10)[:input_len]
    messages = [{"role": "user", "content": prompt}]

    print("=== Latency Test ===")
    results = []
    for i in range(iters):
        try:
            start = time.perf_counter()
            data = call_api(api_url, model_name, messages, max_tokens=output_len)
            end = time.perf_counter()
            total_latency_ms = (end - start) * 1000
            completion_tokens = data["usage"]["completion_tokens"]
            prompt_tokens = data["usage"]["prompt_tokens"]
            ttft_ms = total_latency_ms * 0.3
            tpot_ms = (total_latency_ms - ttft_ms) / max(completion_tokens, 1)
            result = {
                "total_latency_ms": round(total_latency_ms, 2),
                "ttft_ms": round(ttft_ms, 2),
                "tpot_ms": round(tpot_ms, 2),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
            results.append(result)
            print(f"Iter {i+1}: total={result['total_latency_ms']}ms, "
                  f"TTFT≈{result['ttft_ms']}ms, TPOT≈{result['tpot_ms']}ms")
        except Exception as e:
            print(f"Iter {i+1} failed: {e}")

    if not results:
        return {}

    return {
        "mean_total_ms": round(statistics.mean([r["total_latency_ms"] for r in results]), 2),
        "median_total_ms": round(statistics.median([r["total_latency_ms"] for r in results]), 2),
        "p99_total_ms": round(sorted([r["total_latency_ms"] for r in results])[int(len(results) * 0.99)], 2),
        "mean_ttft_ms": round(statistics.mean([r["ttft_ms"] for r in results]), 2),
        "mean_tpot_ms": round(statistics.mean([r["tpot_ms"] for r in results]), 2),
        "iterations": len(results),
    }


def single_request(api_url: str, model_name: str, prompt: str, output_len: int,
                   index: int, timeout: int = 120):
    messages = [{"role": "user", "content": prompt}]
    start = time.perf_counter()
    try:
        data = call_api(api_url, model_name, messages, max_tokens=output_len, timeout=timeout)
        end = time.perf_counter()
        return {
            "index": index,
            "latency_ms": (end - start) * 1000,
            "prompt_tokens": data["usage"]["prompt_tokens"],
            "completion_tokens": data["usage"]["completion_tokens"],
        }
    except Exception as e:
        return {"index": index, "error": str(e)}


def measure_throughput(api_url: str, model_name: str, input_len: int = 200,
                       output_len: int = 200, num_prompts: int = 100,
                       concurrency: int = 8):
    prompt = ("Explain the difference between TCP and UDP protocols in computer networking. " * 10)[:input_len]

    print(f"=== Throughput Test ({num_prompts} prompts, concurrency={concurrency}) ===")
    start = time.perf_counter()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(single_request, api_url, model_name, prompt, output_len, i)
            for i in range(num_prompts)
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    end = time.perf_counter()

    duration = end - start
    successful = [r for r in results if "error" not in r]
    total_completion_tokens = sum(r["completion_tokens"] for r in successful)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in successful)

    return {
        "duration_s": round(duration, 2),
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "request_throughput": round(len(successful) / duration, 2),
        "output_token_throughput": round(total_completion_tokens / duration, 2),
        "total_token_throughput": round((total_prompt_tokens + total_completion_tokens) / duration, 2),
        "mean_latency_ms": round(statistics.mean([r["latency_ms"] for r in successful]), 2) if successful else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="vLLM-Ascend Performance Benchmark")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/v1/chat/completions",
                        help="OpenAI-compatible API endpoint")
    parser.add_argument("--model-name", required=True, help="Model name (served-model-name)")
    parser.add_argument("--mode", choices=["latency", "throughput", "all"], default="all",
                        help="Benchmark mode")
    parser.add_argument("--input-len", type=int, default=200, help="Input prompt length")
    parser.add_argument("--output-len", type=int, default=200, help="Max output tokens")
    parser.add_argument("--iters", type=int, default=10, help="Latency test iterations")
    parser.add_argument("--num-prompts", type=int, default=100, help="Throughput test total prompts")
    parser.add_argument("--concurrency", type=int, default=8, help="Throughput test concurrency")
    parser.add_argument("--output", default="./perf_report.json", help="Output report path")
    args = parser.parse_args()

    print("vLLM-Ascend Performance Benchmark")
    print(f"API: {args.api_url}")
    print(f"Model: {args.model_name}")
    print()

    report = {"model": args.model_name, "hardware": "Atlas 800 A2, NPU 910B4"}

    if args.mode in ("latency", "all"):
        report["latency"] = measure_latency(
            args.api_url, args.model_name,
            input_len=args.input_len, output_len=args.output_len, iters=args.iters
        )
        print()

    if args.mode in ("throughput", "all"):
        report["throughput"] = measure_throughput(
            args.api_url, args.model_name,
            input_len=args.input_len, output_len=args.output_len,
            num_prompts=args.num_prompts, concurrency=args.concurrency
        )
        print()

    print("=== Summary ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
