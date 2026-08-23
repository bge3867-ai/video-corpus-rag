"""把已入库片段的中文摘要打向量, 写入文本检索 collection (混合检索用)。

用法: python src/embed_summaries.py [--limit 0]
幂等: 直接 upsert, 按 clip id 覆盖。
"""
import argparse
import logging

import chromadb

from common import load_cfg, model_local_path
from embedder import DOC_PROMPT, Embedder

log = logging.getLogger("videosrag.embed_summaries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条(调试用)")
    ap.add_argument("--device", default=None, help="embedding 模型所在 GPU, 默认取配置的 embed_device")
    args = ap.parse_args()

    cfg = load_cfg()
    idx = cfg["index"]
    tname = idx.get("text_collection", "videosrag_text")
    client = chromadb.PersistentClient(path=cfg["paths"]["chroma_db"])
    col = client.get_collection(idx["collection"])
    text_col = client.get_or_create_collection(tname, metadata={"hnsw:space": "cosine"})

    emb = Embedder(
        model_local_path(cfg, cfg["models"]["embed_model_id"]),
        args.device or cfg["models"]["embed_device"],
    )

    got = col.get(include=["metadatas"])
    ids, metas = got["ids"], got["metadatas"]
    if args.limit > 0:
        ids, metas = ids[: args.limit], metas[: args.limit]

    n = 0
    for cid, meta in zip(ids, metas):
        s = (meta.get("summary") or "").strip()
        if not s:
            log.warning("无摘要, 跳过 %s", meta.get("clip_path"))
            continue
        v = emb.encode_text(s, DOC_PROMPT)
        text_col.upsert(ids=[cid], embeddings=[v.tolist()], metadatas=[meta])
        n += 1
    print(f"完成: {n} 条摘要向量写入 {tname}, 共 {text_col.count()} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
