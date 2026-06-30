import urllib.request, json, sys
def infer(p, max=20):
    req=urllib.request.Request("http://127.0.0.1:8000/v1/completions",data=json.dumps({"model":"./","prompt":p,"max_tokens":max,"temperature":0.1}).encode(),headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req,timeout=30).read())["choices"][0]["text"]
if __name__=="__main__": print(infer(sys.argv[1] if len(sys.argv)>1 else "Hello"))