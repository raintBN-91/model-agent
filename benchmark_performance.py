import urllib.request, json, time
BASE="http://127.0.0.1:8000"

def test(n=10):
    lat=[]
    for i in range(n):
        t0=time.time()
        req = urllib.request.Request(
            f"{BASE}/v1/completions",
            data=json.dumps({"model":"./","prompt":"Hello","max_tokens":20,"temperature":0.1}).encode(),
            headers={"Content-Type":"application/json"}
        )
        # 修复：添加 with 和 resp.read()，确保完整接收响应体并正确计时
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
        lat.append(time.time()-t0)
    print(f"串行: avg={sum(lat)/len(lat):.2f}s, throughput={n/sum(lat):.2f} req/s")

if __name__=="__main__": test(10)