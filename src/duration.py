"""为已入库片段补充原视频总时长 (metadata.video_duration, 单位: 秒)。

从 metadata.video_path 用 ffprobe 探测, 结果按视频缓存, 很快, 不需要 GPU。
用法: python src/duration.py
"""
import logging
import os
import subprocess

import chromadb

from common import load_cfg

log = logging.getLogger("videosrag.duration")


def probe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    try:
        return round(float(out), 2)
    except ValueError:
        return 0.0


def main():
    cfg = load_cfg()
    col = chromadb.PersistentClient(path=cfg["paths"]["chroma_db"]).get_collection(
        cfg["index"]["collection"]
    )
    got = col.get(include=["metadatas"])
    cache = {}
    n = 0
    for cid, meta in zip(got["ids"], got["metadatas"]):
        vp = meta.get("video_path") or ""
        if not vp or not os.path.exists(vp):
            log.warning("源视频不存在, 跳过 %s", meta.get("clip_path"))
            continue
        if vp not in cache:
            cache[vp] = probe_duration(vp)
        col.update(ids=[cid], metadatas=[{**meta, "video_duration": cache[vp]}])
        n += 1
    print(f"完成: 更新 {n} 条片段的原视频时长, 涉及 {len(cache)} 个视频")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
