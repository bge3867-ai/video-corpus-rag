"""解析 /ask 返回结果并打印 (配合 curl 使用):
    curl -s -X POST http://localhost:8899/ask \
      -H 'Content-Type: application/json' \
      -d '{"question": "有没有森林着火的画面?", "top_k": 4}' | python tools/asktest.py
"""
import json
import sys

d = json.load(sys.stdin)
print("question:", d.get("question"))
print("answer:", (d.get("answer") or "")[:400])
for c in d.get("clips", []):
    print(f"- {c['video']} [{c['start']}-{c['end']}s] dist={c['distance']}")
    print(f"  摘要: {c.get('summary') or '(无)'}")
    for label, key in (("对象", "objects"), ("动作", "actions"),
                       ("场景", "scene"), ("画面文字", "ocr"), ("语音", "asr")):
        if c.get(key):
            print(f"  {label}: {c[key]}")
