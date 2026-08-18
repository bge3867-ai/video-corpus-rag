"""索引管道: 扫描视频库 -> 切片 -> 抽帧 -> embedding -> 写入 Chroma。

用法: python src/index.py [--force]
- 默认幂等: 已经入库的视频(按视频文件 md5 判断)自动跳过
- --force 重新入库所有视频

process_video() 为单视频入库函数, 同时被本脚本和检索服务(上传自动入库)复用。
入库时除了视频向量 (videosrag collection), 还把 VLM 摘要的文本向量写入
text collection (videosrag_text), 供混合检索使用。
"""
import argparse
import hashlib
import logging
import os
import subprocess

import chromadb

from common import load_cfg, model_local_path
from embedder import DOC_PROMPT, Embedder, extract_frames
from enrich import caption_text
from summarizer import Summarizer

log = logging.getLogger("videosrag.index")


def md5_of_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probed(path: str):
    """返回 (时长秒, 是否有音频)。"""
    dur = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    audio = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    return float(dur), bool(audio)


def make_segments(dur: float, seg_len: float, overlap: float, short_limit: float):
    """把视频切成 (start, end) 片段列表。短视频整个算一个片段。"""
    if dur <= 0:
        return []
    if dur <= short_limit:
        return [(0.0, round(dur, 2))]
    segs, t = [], 0.0
    while t < dur:
        end = min(t + seg_len, dur)
        segs.append((round(t, 2), round(end, 2)))
        if end >= dur:
            break
        t = max(end - overlap, 0.0)
    return segs


def cut_clip(src_path: str, start: float, end: float, out_path: str, has_audio: bool, crf: int):
    """用 ffmpeg 精确切出 start~end 片段并重编码为 mp4。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.2f}", "-i", str(src_path),
        "-t", f"{end - start:.2f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    ]
    if has_audio:
        cmd += ["-c:a", "aac"]
    else:
        cmd += ["-an"]
    cmd += [str(out_path)]
    subprocess.run(cmd, check=True)


def list_videos(videos_dir: str, exts):
    out = []
    for name in sorted(os.listdir(videos_dir)):
        p = os.path.join(videos_dir, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in exts:
            out.append(p)
    return out


def process_video(vpath, cfg, col, text_col, embedder,
                  summarizer=None, summarize_fn=None, enrich_fn=None,
                  asr_fn=None, force=False):
    """对单个视频切片并入库; 返回 (新增片段数, 是否因已入库而跳过)。

    enrich_fn: 可选, (clip_path) -> {summary, objects, actions, scene, ocr},
      结构化描述 (旧 summarize_fn 只返回一句话摘要, 兼容保留);
    asr_fn: 可选, (clip_path) -> str, 片段语音转写;
    两者都不传时内部惰性加载 Summarizer。
    """
    name = os.path.basename(vpath)
    idx = cfg["index"]
    vmd5 = md5_of_file(vpath)
    try:
        done = col.get(where={"video_md5": vmd5}, include=[])
        n_existing = len(done["ids"])
    except Exception:  # noqa: BLE001 查询失败按未入库处理
        n_existing = 0
    if n_existing > 0 and not force:
        log.info("跳过(已入库) %s", name)
        return 0, True

    dur, audio = probed(vpath)
    segs = make_segments(dur, idx["seg_len"], idx["overlap"], idx["short_limit"])
    log.info("处理 %s (%.1fs, %d 个片段)", name, dur, len(segs))

    stem = os.path.splitext(name)[0]
    cd = cfg["paths"]["clips"]
    batch = {"ids": [], "embeddings": [], "metadatas": []}
    tbatch = {"ids": [], "embeddings": [], "metadatas": []}
    for si, (s, e) in enumerate(segs):
        rel = os.path.join(stem, f"{stem}_{si:03d}_{int(s)}-{int(e)}.mp4")
        clip_path = os.path.join(cd, rel)
        if not os.path.exists(clip_path):
            cut_clip(vpath, s, e, clip_path, audio, idx["clip_crf"])
        frames = extract_frames(clip_path, idx["frames_per_clip"])
        if not frames:
            log.warning("  片段 %s 抽帧为空, 跳过", rel)
            continue
        vec = embedder.encode_clip(clip_path, frames)
        # 文本侧增强: VLM 结构化描述 (摘要/对象/动作/场景/画面文字) + ASR 语音
        summary = objects = actions = scene = ocr = asr = ""
        enriched = asr_done = False
        if enrich_fn is not None:
            try:
                cap = enrich_fn(clip_path)
                summary = cap.get("summary", "")
                objects = cap.get("objects", "")
                actions = cap.get("actions", "")
                scene = cap.get("scene", "")
                ocr = cap.get("ocr", "")
                enriched = True
            except Exception as exc:  # noqa: BLE001 描述失败不阻断入库
                log.warning("  片段 %s 结构化描述生成失败: %s", rel, exc)
        elif summarize_fn is not None:
            try:
                summary = summarize_fn(clip_path)
            except Exception as exc:  # noqa: BLE001
                log.warning("  片段 %s 摘要生成失败: %s", rel, exc)
        else:
            try:
                if summarizer is None:
                    summarizer = Summarizer(
                        model_local_path(cfg, cfg["models"]["vlm_model_id"]),
                        cfg["models"]["vlm_device"],
                        nframes=cfg["server"]["vlm_nframes"],
                        max_pixels=cfg["server"]["vlm_max_pixels"],
                    )
                summary = summarizer.summarize(clip_path)
            except Exception as exc:  # noqa: BLE001
                log.warning("  片段 %s 摘要生成失败: %s", rel, exc)
        if asr_fn is not None:
            try:
                asr = asr_fn(clip_path) or ""
                asr_done = True
            except Exception as exc:  # noqa: BLE001
                log.warning("  片段 %s ASR 失败: %s", rel, exc)

        cid = hashlib.md5(f"{vmd5}:{s:.2f}:{e:.2f}".encode()).hexdigest()[:24]
        meta = {
            "video": stem,
            "video_path": vpath,
            "video_md5": vmd5,
            "video_duration": round(dur, 2),  # 原视频总秒数
            "start": s,
            "end": e,
            "clip_path": rel,
            "clip_url": "/clips/" + rel,
            "summary": summary,
            "objects": objects,
            "actions": actions,
            "scene": scene,
            "ocr": ocr,
            "asr": asr,
            "enriched": enriched,
            "asr_done": asr_done,
        }
        batch["ids"].append(cid)
        batch["embeddings"].append(vec.tolist())
        batch["metadatas"].append(meta)
        # 文本向量: 摘要 / 结构化描述 / 语音 各一条, id 带类型后缀
        cap_txt = caption_text(meta)
        for suffix, text, ttype in (
            ("_sum", summary, "summary"),
            ("_cap", cap_txt, "caption"),
            ("_asr", asr, "asr"),
        ):
            if not text:
                continue
            tvec = embedder.encode_text(text, DOC_PROMPT)
            tbatch["ids"].append(cid + suffix)
            tbatch["embeddings"].append(tvec.tolist())
            tbatch["metadatas"].append(
                {**meta, "clip_id": cid, "text_type": ttype, "text": text}
            )

    if batch["ids"]:
        col.upsert(ids=batch["ids"], embeddings=batch["embeddings"],
                   metadatas=batch["metadatas"])
    if tbatch["ids"]:
        text_col.upsert(ids=tbatch["ids"], embeddings=tbatch["embeddings"],
                        metadatas=tbatch["metadatas"])
    return len(batch["ids"]), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="重新入库所有视频")
    args = ap.parse_args()

    cfg = load_cfg()
    vd, db = cfg["paths"]["videos"], cfg["paths"]["chroma_db"]
    idx = cfg["index"]

    client = chromadb.PersistentClient(path=db)
    col = client.get_or_create_collection(
        idx["collection"], metadata={"hnsw:space": "cosine"}
    )
    text_col = client.get_or_create_collection(
        idx.get("text_collection", "videosrag_text"),
        metadata={"hnsw:space": "cosine"},
    )
    embedder = Embedder(
        model_local_path(cfg, cfg["models"]["embed_model_id"]),
        cfg["models"]["embed_device"],
    )

    videos = list_videos(vd, idx["video_exts"])
    if not videos:
        log.error("视频库为空: %s 没有视频文件", vd)
        return 1
    log.info("发现 %d 个视频", len(videos))

    n_new = 0
    for i, vpath in enumerate(videos, 1):
        n, skipped = process_video(vpath, cfg, col, text_col, embedder, force=args.force)
        n_new += n
    log.info("完成: 新增 %d 个片段入库, collection 共 %d 个向量, 文本 collection 共 %d 个",
             n_new, col.count(), text_col.count())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
