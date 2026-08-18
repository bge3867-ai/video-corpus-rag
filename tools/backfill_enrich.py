"""一次性回填: 给已入库的片段补文本侧增强 (结构化描述 + ASR)。

同时把旧格式文本向量 (id=cid, 无 text_type) 迁移为 {cid}_sum 新格式。

用法 (建议先停应用服务, 避免 Chroma 多进程并发写):
    python tools/backfill_enrich.py [--limit N] [--asr-only] [--skip-asr] [--force] [--device cuda:0]

- 默认幂等: 已回填 (metadata.enriched=1 且 asr_done=1) 的片段跳过, --force 重做
- --asr-only: 只补语音转写 (跳过已有 asr 文本的片段), 不重跑 VLM
- VLM 走 vLLM 服务 (config.vllm), ASR 走 faster-whisper (config.enrich.asr_device)
- embedding 在本脚本进程内加载, 用 --device 指定卡 (默认 cuda:0)
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import chromadb

from common import load_cfg, model_local_path
from embedder import DOC_PROMPT, Embedder
from enrich import (
    ASRTranscriber,
    CAPTION_PROMPT,
    caption_text,
    frame_parts,
    parse_caption,
)


def migrate_text_col(text_col):
    """旧格式文本向量 (id=cid 无 text_type) -> {cid}_sum。"""
    rows = text_col.get(include=["metadatas"])
    old_ids = [tid for tid, tm in zip(rows["ids"], rows["metadatas"])
               if not (tm or {}).get("text_type")]
    if not old_ids:
        return 0
    old = text_col.get(ids=old_ids, include=["embeddings", "metadatas"])
    text_col.delete(ids=old_ids)
    for tid, emb, tm in zip(old["ids"], old["embeddings"], old["metadatas"]):
        tm = dict(tm or {})
        tm["clip_id"] = tid
        tm["text_type"] = "summary"
        tm["text"] = tm.get("summary", "")
        text_col.upsert(ids=[tid + "_sum"], embeddings=[emb], metadatas=[tm])
    return len(old_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个片段 (调试)")
    ap.add_argument("--asr-only", action="store_true", help="只补语音转写, 不重跑 VLM")
    ap.add_argument("--skip-asr", action="store_true", help="跳过语音转写")
    ap.add_argument("--force", action="store_true", help="重做已回填的片段")
    ap.add_argument("--device", default="cuda:0", help="本进程 embedding 模型所在卡")
    args = ap.parse_args()

    cfg = load_cfg()
    idx = cfg["index"]
    ecfg = cfg.get("enrich") or {}
    vcfg = cfg.get("vllm") or {}

    client = chromadb.PersistentClient(path=cfg["paths"]["chroma_db"])
    col = client.get_or_create_collection(
        idx["collection"], metadata={"hnsw:space": "cosine"}
    )
    text_col = client.get_or_create_collection(
        idx["text_collection"], metadata={"hnsw:space": "cosine"}
    )

    n_mig = migrate_text_col(text_col)
    print(f"文本库迁移: 旧格式 {n_mig} 条 -> {{cid}}_sum")

    embedder = Embedder(
        model_local_path(cfg, cfg["models"]["embed_model_id"]), args.device
    )

    vclient = None
    if not args.asr_only:
        from openai import OpenAI

        vclient = OpenAI(base_url=vcfg["base_url"], api_key="EMPTY", timeout=600)

    asr = None
    if not args.skip_asr and ecfg.get("asr_enabled", True):
        asr = ASRTranscriber(
            ecfg.get("asr_model", "large-v3-turbo"), ecfg.get("asr_device", "cuda:2")
        )

    rows = col.get(include=["metadatas"])
    items = list(zip(rows["ids"], rows["metadatas"]))
    if args.limit:
        items = items[: args.limit]
    print(f"共 {len(items)} 个片段待检查 (--limit={args.limit or '全部'})")

    n = 0
    n_skip = 0
    for cid, meta in items:
        meta = dict(meta)
        if args.asr_only:
            if meta.get("asr") and not args.force:
                n_skip += 1
                continue
        elif meta.get("enriched") and meta.get("asr_done") and not args.force:
            n_skip += 1
            continue
        clip_path = os.path.join(cfg["paths"]["clips"], meta["clip_path"])
        if not os.path.exists(clip_path):
            print(f"[{cid}] 切片文件缺失, 跳过: {clip_path}")
            continue
        t0 = time.time()

        # 1) VLM 结构化描述 (摘要/对象/动作/场景/画面文字)
        if vclient is not None:
            try:
                content = frame_parts(
                    cfg, os.path.abspath(clip_path), cfg["server"]["vlm_nframes"]
                )
                content.append({"type": "text", "text": CAPTION_PROMPT})
                resp = vclient.chat.completions.create(
                    model=vcfg.get("model_name", cfg["models"]["vlm_model_id"]),
                    messages=[{"role": "user", "content": content}],
                    max_tokens=160, temperature=0.0,
                    extra_body={"mm_processor_kwargs": {"max_pixels": cfg["server"]["vlm_max_pixels"]}},
                )
                raw = (resp.choices[0].message.content or "").strip()
                cap = parse_caption(raw)
                meta["summary"] = cap["summary"] or meta.get("summary", "")
                meta["objects"] = cap["objects"]
                meta["actions"] = cap["actions"]
                meta["scene"] = cap["scene"]
                meta["ocr"] = cap["ocr"]
                meta["enriched"] = True
            except Exception as exc:  # noqa: BLE001
                print(f"[{cid}] VLM 描述失败: {exc}")

        # 2) ASR 语音转写 (None=模型不可用; ""=无语音/静音)
        # asr-only 模式只看 asr 是否为空 (历史轮次可能已错置 asr_done 标记)
        need_asr = (not meta.get("asr")) if args.asr_only else not meta.get("asr_done", False)
        if asr is not None and need_asr:
            try:
                r = asr.transcribe(clip_path)
                if r is not None:
                    meta["asr"] = r
                    meta["asr_done"] = True
            except Exception as exc:  # noqa: BLE001
                print(f"[{cid}] ASR 失败: {exc}")
        elif args.skip_asr:
            meta["asr_done"] = True

        # 3) 写回主库 metadata + 文本向量
        col.update(ids=[cid], metadatas=[meta])
        t_ups = {"ids": [], "embeddings": [], "metadatas": []}
        if args.asr_only:
            entries = [("_asr", meta.get("asr", ""), "asr")]
        else:
            entries = [
                ("_sum", meta.get("summary", ""), "summary"),
                ("_cap", caption_text(meta), "caption"),
                ("_asr", meta.get("asr", ""), "asr"),
            ]
        for suffix, text, ttype in entries:
            if not text:
                continue
            vec = embedder.encode_text(text, DOC_PROMPT)
            t_ups["ids"].append(cid + suffix)
            t_ups["embeddings"].append(vec.tolist())
            t_ups["metadatas"].append(
                {**meta, "clip_id": cid, "text_type": ttype, "text": text}
            )
        if t_ups["ids"]:
            text_col.upsert(**t_ups)

        n += 1
        dt = time.time() - t0
        print(f"[{n}/{len(items)}] {meta['video']} {meta['start']}-{meta['end']}s "
              f"({dt:.1f}s) 摘要: {meta['summary'][:18]!r} "
              f"动作: {meta['actions'][:14]!r} "
              f"语音: {meta.get('asr', '')[:18]!r}")

    print(f"完成: 处理 {n} 个, 跳过 {n_skip} 个; "
          f"主库 {col.count()} 条, 文本库 {text_col.count()} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
