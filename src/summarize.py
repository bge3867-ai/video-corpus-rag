"""为已入库的片段批量补充 VLM 中文摘要 (metadata.summary)。

用法:
    python src/summarize.py [--device cuda:2] [--limit 0]

默认用 cuda:2 跑, 避免与正在服务的问答模型 (cuda:1) 抢显存。
幂等: 已有摘要的片段跳过; --force 重新生成。
"""
import argparse
import logging
import os

import chromadb

from common import load_cfg, model_local_path
from summarizer import Summarizer

log = logging.getLogger("videosrag.summarize")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="摘要模型所在 GPU, 默认取配置的 vlm_device")
    ap.add_argument("--force", action="store_true", help="已有摘要的片段也重新生成")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个片段(调试用)")
    args = ap.parse_args()

    cfg = load_cfg()
    col = chromadb.PersistentClient(path=cfg["paths"]["chroma_db"]).get_collection(
        cfg["index"]["collection"]
    )
    got = col.get(include=["metadatas"])
    ids, metas = got["ids"], got["metadatas"]
    if args.limit > 0:
        ids, metas = ids[: args.limit], metas[: args.limit]

    device = args.device or cfg["models"]["vlm_device"]
    summ = Summarizer(
        model_local_path(cfg, cfg["models"]["vlm_model_id"]),
        device,
        nframes=cfg["server"]["vlm_nframes"],
        max_pixels=cfg["server"]["vlm_max_pixels"],
    )

    n_done = 0
    for i, (cid, meta) in enumerate(zip(ids, metas), 1):
        if meta.get("summary") and not args.force:
            log.info("[%d/%d] 已有摘要, 跳过 %s", i, len(ids), meta["clip_path"])
            continue
        clip_abs = os.path.join(cfg["paths"]["clips"], meta["clip_path"])
        try:
            s = summ.summarize(clip_abs)
        except Exception as e:  # noqa: BLE001 单条失败不影响整体
            log.warning("[%d/%d] 摘要失败 %s: %s", i, len(ids), meta["clip_path"], e)
            continue
        col.update(ids=[cid], metadatas=[{**meta, "summary": s}])
        n_done += 1
        log.info("[%d/%d] %s -> %s", i, len(ids), meta["clip_path"], s)

    print(f"完成: 本次生成 {n_done} 条摘要, 共处理 {len(ids)} 个片段")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
