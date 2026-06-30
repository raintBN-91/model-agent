import urllib.request, json
import sys

def verify():
    req = urllib.request.Request(
        "http://127.0.0.1:8000/v1/completions",
        data=json.dumps({"model":"./","prompt":"What is 2+2?","max_tokens":5,"temperature":0.1}).encode(),
        headers={"Content-Type":"application/json"}
    )
    r = json.loads(urllib.request.urlopen(req, timeout=30).read())
    # 修复：将 assert 替换为显式检查，避免 -O 模式下被跳过
    result = r["choices"][0]["text"].strip()
    if result != "4":
        print(f"❌ 精度验证失败: 期望 '4'，实际 '{result}'")
        sys.exit(1)
    print("✅ 精度通过")

if __name__ == "__main__": verify()