"""公共配置加载。"""
import logging
import os
import sys

# 必须在 import huggingface_hub 之前设置: 服务器直连 huggingface.co 不稳定
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def load_cfg():
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("videos", "clips", "chroma_db", "models", "logs"):
        cfg["paths"][key] = os.path.join(ROOT, cfg["paths"][key])
        os.makedirs(cfg["paths"][key], exist_ok=True)
    return cfg


def model_local_path(cfg, model_id: str) -> str:
    """返回本地模型目录, 若未下载则尝试从 HF 拉取。"""
    local = os.path.join(cfg["paths"]["models"], model_id.split("/")[-1])
    if os.path.isdir(local) and any(
        f.endswith(".safetensors") or f.endswith(".bin")
        for f in os.listdir(local)
    ):
        return local
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, local_dir=local)