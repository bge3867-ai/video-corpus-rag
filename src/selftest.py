"""embedding 模型冒烟测试: 验证视频输入通路 + 检索语义是否合理。

用法:
    python src/selftest.py videos/debrisFlow_demo.mp4 videos/horse.mp4 "泥石流"

输出: 查询文本与两个视频的余弦相似度, 语义相关的视频应明显更高。
在正式建索引前先跑一遍, 确认 embedding 通路正常。
"""
import sys

from common import load_cfg, model_local_path
from embedder import Embedder


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    va, vb, query = sys.argv[1], sys.argv[2], sys.argv[3]
    cfg = load_cfg()
    model_dir = model_local_path(cfg, cfg["models"]["embed_model_id"])
    emb = Embedder(model_dir, cfg["models"]["embed_device"])
    q = emb.encode_query(query)
    a = emb.encode_clip(va)
    b = emb.encode_clip(vb)
    print(f"query = {query!r}")
    print(f"  sim({va}) = {float(q @ a):.4f}")
    print(f"  sim({vb}) = {float(q @ b):.4f}")
    print(f"  向量维度 = {len(q)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())