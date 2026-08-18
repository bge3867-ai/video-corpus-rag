#!/bin/bash
# ============================================================
# 下载模型到 models/ 目录 (带重试 + 完整性校验 + 双源回退)
#   Qwen3-VL-Embedding-8B  (~17GB, 片段 embedding)
#   Qwen3-VL-8B-Instruct   (~17GB, 片段问答)
# 首选 ModelScope(国内稳定), 失败回退 hf-mirror; 可重复运行断点续传
# ============================================================
set -e
cd "$(dirname "$0")"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate videosrag 2>/dev/null || true
fi
pip install -q modelscope
python - <<'EOF'
import json
import os
import time

local = os.path.abspath("models")


def verify(d):
    """按 index.json 校验所有分片是否齐全, 返回缺失文件名列表。"""
    idx = os.path.join(d, "model.safetensors.index.json")
    if not os.path.exists(idx):
        return None  # 单文件模型, 交给加载阶段验证
    files = set(json.load(open(idx))["weight_map"].values())
    return [f for f in files if not os.path.exists(os.path.join(d, f))]


def dl_ms(repo, out):
    from modelscope import snapshot_download
    return snapshot_download(repo, local_dir=out)


def dl_hf(repo, out):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, local_dir=out, max_workers=8)


for repo in ["Qwen/Qwen3-VL-Embedding-8B", "Qwen/Qwen3-VL-8B-Instruct"]:
    name = repo.split("/")[-1]
    out = os.path.join(local, name)
    ok = False
    for src, fn in (("modelscope", dl_ms), ("hf-mirror", dl_hf)):
        for attempt in range(3):
            try:
                fn(repo, out)
                missing = verify(out)
                if not missing:
                    print(f"OK: {name} (源={src}, 第{attempt+1}次)")
                    ok = True
                    break
                print(f"  分片缺失, 续传重试: {missing}")
            except Exception as e:  # noqa: BLE001
                print(f"  下载出错 ({src}, {type(e).__name__}): {str(e)[:120]}")
                time.sleep(10)
        if ok:
            break
    if not ok:
        raise SystemExit(f"下载失败: {repo}")
print("== 全部模型就绪且校验通过 ==")
EOF
