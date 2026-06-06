#!/usr/bin/env python3
"""
vLLM-Ascend Lightweight Consistency Check Script.
Evaluates output consistency and basic reasoning via the vLLM OpenAI API.

Usage:
    python check.py --api-url http://127.0.0.1:8000/v1/chat/completions \
        --model-name gemma-3-270m-it --output ./accuracy_report.json
"""

import argparse
import json
import requests


def call_api(api_url: str, model_name: str, messages: list, max_tokens: int = 64,
             temperature: float = 0.0, timeout: int = 60):
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


def evaluate_consistency(api_url: str, model_name: str):
    print("=== Consistency Test ===")
    prompt = "What is the capital of France? Answer in one word."
    results = []
    for i in range(3):
        try:
            data = call_api(api_url, model_name, [{"role": "user", "content": prompt}], max_tokens=16)
            content = data["choices"][0]["message"]["content"].strip()
            results.append(content)
            print(f"  Run {i+1}: {content}")
        except Exception as e:
            print(f"  Run {i+1} failed: {e}")
            results.append(None)

    consistent = len(set([r for r in results if r is not None])) == 1 and None not in results
    print(f"  Consistent: {consistent}")
    return {"prompt": prompt, "outputs": results, "consistent": consistent}


def evaluate_reasoning(api_url: str, model_name: str):
    print("=== Basic Reasoning Test ===")
    tests = [
        {"name": "Arithmetic", "prompt": "What is 15 + 27?", "max_tokens": 16},
        {"name": "Fact", "prompt": "Which planet is known as the Red Planet?", "max_tokens": 16},
        {"name": "Logic", "prompt": "If all cats have tails, and Tom is a cat, does Tom have a tail? Answer yes or no.", "max_tokens": 16},
    ]
    results = []
    for t in tests:
        try:
            data = call_api(api_url, model_name, [{"role": "user", "content": t["prompt"]}], max_tokens=t["max_tokens"])
            content = data["choices"][0]["message"]["content"].strip()
            results.append({"name": t["name"], "prompt": t["prompt"], "output": content})
            print(f"  {t['name']}: {content}")
        except Exception as e:
            print(f"  {t['name']}: failed - {e}")
            results.append({"name": t["name"], "prompt": t["prompt"], "output": None, "error": str(e)})
    return results


def evaluate_chinese(api_url: str, model_name: str):
    print("=== Chinese Test ===")
    prompt = "用一句中文说明 TCP 和 UDP 的核心区别。"
    try:
        data = call_api(api_url, model_name, [{"role": "user", "content": prompt}], max_tokens=64)
        content = data["choices"][0]["message"]["content"].strip()
        print(f"  Prompt: {prompt}")
        print(f"  Output: {content}")
        return {"prompt": prompt, "output": content}
    except Exception as e:
        print(f"  Failed: {e}")
        return {"prompt": prompt, "output": None, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="vLLM-Ascend Lightweight Consistency Check")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/v1/chat/completions",
                        help="OpenAI-compatible API endpoint")
    parser.add_argument("--model-name", required=True, help="Model name (served-model-name)")
    parser.add_argument("--output", default="./accuracy_report.json", help="Output report path")
    args = parser.parse_args()

    print("vLLM-Ascend Consistency Check")
    print(f"API: {args.api_url}")
    print(f"Model: {args.model_name}")
    print()

    report = {
        "model": args.model_name,
        "hardware": "Atlas 800 A2, NPU 910B4",
        "consistency": evaluate_consistency(args.api_url, args.model_name),
        "reasoning": evaluate_reasoning(args.api_url, args.model_name),
        "chinese": evaluate_chinese(args.api_url, args.model_name),
    }

    print()
    print("=== Summary ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
