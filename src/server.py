"""检索问答服务: 提问 -> 全库混合检索 top-k 视频片段 -> VLM 作答 -> 返回答案+切片。

- 混合检索: 视频向量 + 摘要文本向量双路召回, RRF 融合 (config: server.hybrid)
- 上传入库: POST /upload 上传视频 -> 后台队列自动切片/embedding/摘要入库
- VLM 双模式: config.vllm.enabled=true 时走 vLLM OpenAI 接口 (快, 支持并发);
  否则进程内 transformers 直接生成 (兜底)。
用法: python src/server.py
页面: http://<主机>:8899/   接口: POST /ask {"question": "...", "top_k": 4}
"""
import base64
import io
import itertools
import logging
import os
import queue
import re
import tempfile
import threading

import chromadb
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from common import load_cfg, model_local_path
from embedder import Embedder
from enrich import ASRTranscriber, CAPTION_PROMPT, frame_parts, parse_caption
from index import cut_clip, probed, process_video

log = logging.getLogger("videosrag.server")

cfg = load_cfg()
app = FastAPI(title="视频库语义检索问答", version="1.2")

client = chromadb.PersistentClient(path=cfg["paths"]["chroma_db"])
col = client.get_or_create_collection(
    cfg["index"]["collection"], metadata={"hnsw:space": "cosine"}
)
text_col = client.get_or_create_collection(
    cfg["index"].get("text_collection", "videosrag_text"),
    metadata={"hnsw:space": "cosine"},
)
embedder = Embedder(
    model_local_path(cfg, cfg["models"]["embed_model_id"]),
    cfg["models"]["embed_device"],
)

# ---- 问答 VLM: vLLM OpenAI 接口优先, transformers 本地兜底 ----
VLLM_CFG = cfg.get("vllm") or {}
VLLM_ENABLED = bool(VLLM_CFG.get("enabled")) and bool(VLLM_CFG.get("base_url"))
vllm_client = None
processor = None
vlm = None
process_vision_info = None

if VLLM_ENABLED:
    from openai import OpenAI  # noqa: E402

    vllm_client = OpenAI(base_url=VLLM_CFG["base_url"], api_key="EMPTY", timeout=600)
    log.info("问答走 vLLM 接口 %s (model=%s)", VLLM_CFG["base_url"],
             VLLM_CFG.get("model_name"))
else:
    log.info("加载本地 VLM %s (device=%s) ...", cfg["models"]["vlm_model_id"],
             cfg["models"]["vlm_device"])
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: E402
    from qwen_vl_utils import process_vision_info as _pvi  # noqa: E402

    process_vision_info = _pvi
    vlm_path = model_local_path(cfg, cfg["models"]["vlm_model_id"])
    processor = AutoProcessor.from_pretrained(vlm_path)
    vlm = Qwen3VLForConditionalGeneration.from_pretrained(
        vlm_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=cfg["models"]["vlm_device"],
    )
    vlm.eval()
    log.info("本地 VLM 加载完成")


class AskReq(BaseModel):
    question: str
    top_k: int = cfg["server"]["default_top_k"]
    with_answer: bool = True
    history: list = []  # 多轮对话历史: [{"question": "...", "answer": "..."}, ...]
    last_clips: list = []  # 上一轮返回片段的 clip_path 列表 (追问时排最前)


def _mm_kwargs():
    # vLLM 的 Qwen3-VL 视频路径 token 膨胀严重, 我们统一走图片模态 (自己抽帧)
    return {"max_pixels": cfg["server"]["vlm_max_pixels"]}


def _vllm_frames_per_clip(top_k: int) -> int:
    """按 vLLM 上下文长度自适应每片段帧数 (245760 像素缩图实测约 200 token/帧)。"""
    max_len = (cfg.get("vllm") or {}).get("max_model_len", 8192)
    budget = max(2, (max_len - 1000) // (top_k * 200))
    return min(budget, cfg["server"]["vlm_nframes"])


def _generate(messages, max_new_tokens: int) -> str:
    """vLLM 或本地 transformers 生成文本。"""
    if vllm_client is not None:
        resp = vllm_client.chat.completions.create(
            model=VLLM_CFG.get("model_name", cfg["models"]["vlm_model_id"]),
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.0,
            extra_body={"mm_processor_kwargs": _mm_kwargs()},
        )
        return (resp.choices[0].message.content or "").strip()
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos = process_vision_info(messages)
    inputs = processor(
        text=[text], images=images, videos=videos,
        padding=True, return_tensors="pt",
    ).to(cfg["models"]["vlm_device"])
    with torch.no_grad():
        out = vlm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0][inputs.input_ids.shape[1]:]
    return processor.decode(gen, skip_special_tokens=True).strip()


def enrich_clip_with_vlm(clip_path: str):
    """为片段生成结构化描述 (摘要/对象/动作/场景/画面文字), 一次 VLM 调用。

    返回 {"summary", "objects", "actions", "scene", "ocr"} 字典。
    """
    abs_path = os.path.abspath(clip_path)
    if vllm_client is not None:
        content = frame_parts(cfg, abs_path, cfg["server"]["vlm_nframes"])
        content.append({"type": "text", "text": CAPTION_PROMPT})
        text = _generate([{"role": "user", "content": content}], 160)
    else:
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{abs_path}",
                    "nframes": cfg["server"]["vlm_nframes"],
                    "max_pixels": cfg["server"]["vlm_max_pixels"],
                },
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }]
        text = _generate(messages, 160)
    return parse_caption(text)


_asr = None


def transcribe_clip(clip_path: str) -> str:
    """片段语音转文字 (faster-whisper, 懒加载); 关闭或无语音返回空串。"""
    global _asr
    if not (cfg.get("enrich") or {}).get("asr_enabled", True):
        return ""
    if _asr is None:
        ecfg = cfg.get("enrich") or {}
        _asr = ASRTranscriber(
            ecfg.get("asr_model", "large-v3-turbo"),
            ecfg.get("asr_device", "cuda:2"),
        )
    return _asr.transcribe(clip_path)


def hybrid_search(qvec, top_k):
    """视频向量 + 文本向量 (摘要/结构化描述/语音) 多路召回, RRF 融合。

    文本路每片段可能有多种类型向量 (summary/caption/asr), 先按 clip 分组
    取最优相似度 (乘类型权重), 再以片段为单位参与 RRF 投票。

    返回 [(clip_id, metadata, rrf_score, distance)] 按分数降序。
    distance 优先取视频路余弦距离, 便于前端展示。
    """
    srv = cfg["server"]
    cand = max(srv.get("hybrid_candidates", 12), top_k)
    n_v = min(cand, col.count())
    vres = col.query(query_embeddings=[qvec.tolist()], n_results=n_v,
                     include=["metadatas", "distances"])
    vids = vres["ids"][0]
    vdists = {cid: d for cid, d in zip(vids, vres["distances"][0])}
    vmetas = {cid: (m or {}) for cid, m in zip(vids, vres["metadatas"][0])}

    rrf = {}
    for rank, cid in enumerate(vids):
        rrf[cid] = rrf.get(cid, 0.0) + srv.get("hybrid_video_w", 1.0) / (60 + rank)

    if srv.get("hybrid", True) and text_col.count() > 0:
        weights = srv.get("text_weights", {"caption": 1.0, "summary": 0.8, "asr": 0.8})
        # 每片段平均 3 条文本向量, 多取候选再按 clip 去重
        n_t = min(cand * 3, text_col.count())
        tres = text_col.query(query_embeddings=[qvec.tolist()], n_results=n_t,
                              include=["metadatas", "distances"])
        best = {}  # clip_id -> (加权相似度, metadata, distance)
        for tid, tm, td in zip(tres["ids"][0], tres["metadatas"][0], tres["distances"][0]):
            tm = tm or {}
            cid = tm.get("clip_id") or tid.rsplit("_", 1)[0]
            w = weights.get(tm.get("text_type", "summary"), 0.8)
            sim = (1.0 - td) * w
            if cid not in best or sim > best[cid][0]:
                best[cid] = (sim, tm, td)
        ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:cand]
        for rank, (cid, (_, tm, td)) in enumerate(ranked):
            rrf[cid] = rrf.get(cid, 0.0) + srv.get("hybrid_text_w", 0.8) / (60 + rank)
            vmetas.setdefault(cid, tm)
            vdists.setdefault(cid, td)

    order = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [(cid, vmetas[cid], score, vdists.get(cid)) for cid, score in order]


REWRITE_PROMPT = (
    "把下面的视频检索问题改写成 2 个语义等价的简短检索短语 (每个不超过15字), "
    "每行一个, 直接输出短语本身, 不要序号和任何解释。\n问题: {q}"
)


def rewrite_query(q: str):
    """用 VLM 把问题扩展成多个同义检索短语 (原始问题排第一)。"""
    if not vllm_client or not cfg["server"].get("query_rewrite", True):
        return [q]
    try:
        text = _generate(
            [{"role": "user", "content": REWRITE_PROMPT.format(q=q)}], 64
        )
        out = [q]
        for line in text.splitlines():
            p = line.strip().lstrip("-*#0123456789.、 ").strip()
            if p and p != q and p not in out:
                out.append(p)
        return out[:3]
    except Exception as exc:  # noqa: BLE001 改写失败退回原问题
        log.warning("查询改写失败, 使用原问题: %s", exc)
        return [q]


def search_with_rewrites(q: str, n: int):
    """多短语分路检索, 跨短语累加 RRF 分数。

    返回 [(cid, meta, rrf_score, distance)] 按分数降序, 最多 n 个。
    """
    srv = cfg["server"]
    phrases = rewrite_query(q)
    acc, metas, dists = {}, {}, {}
    for pi, p in enumerate(phrases):
        w = 1.0 if pi == 0 else srv.get("rewrite_w", 0.6)
        qvec = embedder.encode_query(p)
        for cid, m, score, d in hybrid_search(qvec, max(n, 4)):
            acc[cid] = acc.get(cid, 0.0) + score * w
            metas[cid] = m
            dists[cid] = d
    order = sorted(acc.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return [(cid, metas[cid], score, dists.get(cid)) for cid, score in order]


def merge_clips(hits):
    """同视频的重叠片段合并: 跨段重切成一个连续片段, 返回合并后的 hits。

    片段按分数降序, 每个新片段与已保留片段的重叠比例超过
    server.merge_overlap 时, 与其中分数更高者合并为并集区间。
    """
    thr = cfg["server"].get("merge_overlap", 0.5)
    if not thr:
        return hits
    kept = []  # [(cid, meta, rrf, dist)]
    for cid, m, score, d in hits:
        if not m.get("clip_path"):
            continue
        merged = None
        for i, (k0, m0, s0, d0) in enumerate(kept):
            if m0.get("video") != m.get("video"):
                continue
            s0, e0 = m0["start"], m0["end"]
            s1, e1 = m["start"], m["end"]
            ov = min(e0, e1) - max(s0, s1)
            # 重叠占较短片段的比例超过阈值才合并
            if ov > 0 and ov / min(e0 - s0, e1 - s1) >= thr:
                merged = i
                break
        if merged is None:
            kept.append([cid, m, score, d])
            continue
        # 与 kept[merged] 合并成并集区间, 取两段中分数更高的元数据
        k0, m0, s0, d0 = kept[merged]
        if s0 < score:
            m0, s0, d0 = m, score, d if d is not None else d0
        nu, ne = min(m0["start"], m["start"]), max(m0["end"], m["end"])
        m0 = dict(m0)
        m0["start"], m0["end"] = nu, ne
        m0["merged"] = True
        kept[merged] = [k0, m0, s0, min(d0, d) if d is not None else d0]
    # 合并后的片段跨段重切真实视频文件 (缓存)
    out = []
    for cid, m, score, d in kept:
        try:
            src = m.get("video_path")
            if src and os.path.isfile(src) and m.get("merged"):
                has_audio, _ = probed(src)
                stem = os.path.splitext(os.path.basename(src))[0]
                rel = os.path.join("_merged", f"{stem}_{int(m['start'])}-{int(m['end'])}.mp4")
                abs_out = os.path.join(cfg["paths"]["clips"], rel)
                if not os.path.exists(abs_out):
                    os.makedirs(os.path.dirname(abs_out), exist_ok=True)
                    cut_clip(src, m["start"], m["end"], abs_out, has_audio,
                             cfg["index"]["clip_crf"])
                m["clip_path"] = rel
                m["clip_url"] = "/clips/" + rel
        except Exception as exc:  # noqa: BLE001 合并重切失败退回原片段
            log.warning("合并重切失败, 使用原片段: %s", exc)
        out.append((cid, m, score, d))
    return out


# ---- Reranker 候选精排 (Qwen3-Reranker, 懒加载) ----
_reranker = None
_reranker_lock = threading.Lock()


def _get_reranker():
    global _reranker
    rcfg = cfg.get("reranker") or {}
    if not rcfg.get("enabled") or not rcfg.get("model_id"):
        return None
    with _reranker_lock:
        if _reranker is None:
            from transformers import (  # noqa: E402 懒加载
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            path = model_local_path(cfg, rcfg["model_id"])
            log.info("加载 reranker %s (device=%s) ...", path, rcfg.get("device"))
            tok = AutoTokenizer.from_pretrained(path)
            mdl = AutoModelForSequenceClassification.from_pretrained(
                path,
                device_map=rcfg.get("device", "cuda:3"),
                torch_dtype=torch.bfloat16,
            ).eval()
            _reranker = (tok, mdl)
            log.info("reranker 加载完成")
    return _reranker


def rerank_clips(q: str, clips):
    """用 reranker 对候选片段打分重排, 返回重排后的列表。"""
    r = _get_reranker()
    if r is None:
        return clips
    tok, mdl = r

    def doc_text(m):
        parts = [m.get("summary", ""), m.get("objects", ""), m.get("actions", ""),
                 m.get("scene", ""), m.get("ocr", ""), m.get("asr", "")]
        return " ".join(p for p in parts if p)[:300] or m.get("video", "")

    scores = []
    for _, m, _, _ in clips:
        msgs = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": (
                    "<instruct>Given a query A and a passage B, determine whether "
                    "the passage contains an answer to the query by providing a "
                    "prediction of either 'Yes' or 'No'.\n"
                    f"<query>{q}</query>\n<passage>{doc_text(m)}</passage>"
                ),
            }],
        }]
        inputs = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False, return_tensors="pt"
        ).to(mdl.device)
        with torch.no_grad():
            logits = mdl(inputs).logits[0][-1]
        scores.append(float(torch.sigmoid(logits).cpu()))
    order = sorted(range(len(clips)), key=lambda i: scores[i], reverse=True)
    return [clips[i] for i in order]


def build_vlm_messages(question: str, clip_paths, history=None, clip_texts=None):
    """把命中的视频片段组织成 VLM 输入 (vLLM 抽帧图片 / transformers 视频 两种格式)。

    history: 多轮对话历史; clip_texts: 上一轮返回片段的文本描述, 供追问引用。
    """
    content = []
    if cfg["server"].get("conversation", True) and (history or clip_texts):
        ctx = ["(这是多轮对话, 请结合上下文回答当前问题)"]
        for h in (history or [])[-3:]:
            q, a = (h.get("question", "") if isinstance(h, dict) else ""), \
                   (h.get("answer", "") if isinstance(h, dict) else "")
            if q:
                ctx.append(f"此前问题: {q}")
            if a:
                ctx.append(f"此前回答: {a[:200]}")
        if clip_texts:
            ctx.append("上一轮返回的片段信息(用户提到'第N个片段'时按此顺序对应):")
            for i, t in enumerate(clip_texts, 1):
                ctx.append(f"  片段{i}: {t[:150]}")
        content.append({"type": "text", "text": "\n".join(ctx)})
    if vllm_client is not None:
        n = _vllm_frames_per_clip(max(len(clip_paths), 1))
        for i, p in enumerate(clip_paths, 1):
            content.append({"type": "text", "text": f"片段{i} 的视频帧如下:"})
            content.extend(frame_parts(cfg, p, n))
    else:
        for p in clip_paths:
            content.append({
                "type": "video",
                "video": f"file://{p}",
                "nframes": cfg["server"]["vlm_nframes"],
                "max_pixels": cfg["server"]["vlm_max_pixels"],
            })
    content.append({
        "type": "text",
        "text": (
            f"以下是从视频库中检索到的 {len(clip_paths)} 个与问题相关的视频片段。"
            f"请结合片段内容回答用户问题。\n"
            f"要求:\n"
            f"1. 只依据给定片段的内容作答, 不要编造;\n"
            f"2. 引用某个片段时用 [片段N] 标注; 片段不足时说明未在库中找到;\n"
            f"3. 用中文回答。\n\n用户问题: {question}"
        ),
    })
    return [{"role": "user", "content": content}]


@app.post("/ask")
def ask(req: AskReq):
    q = req.question.strip()
    if not q:
        return {"error": "问题不能为空"}
    top_k = max(1, min(req.top_k or cfg["server"]["default_top_k"], cfg["server"]["max_top_k"]))

    # 1) 检索 (查询改写 + 混合检索 + 相邻片段合并)
    hits = search_with_rewrites(q, top_k * 2)

    # 追问型问题 (引用上一轮片段): 上一轮片段排最前, 保证 [片段N] 编号一致
    prev = []
    if req.last_clips:
        try:
            rows = col.get(where={"clip_path": {"$in": list(req.last_clips)}},
                           include=["metadatas"])
            bypath = {m["clip_path"]: (cid, m)
                      for cid, m in zip(rows["ids"], rows["metadatas"])}
            for p in req.last_clips:
                if p in bypath:
                    cid, m = bypath[p]
                    prev.append((cid, m, 1.0, None))
        except Exception:  # noqa: BLE001
            prev = []
    is_followup = bool(prev) and re.search(
        r"第[一二三四五六七八九十\d]+个|片段\s*\d+|刚才|其中|那个|这个|它", q)
    if is_followup:
        seen = {c[0] for c in prev}
        hits = prev + merge_clips([h for h in hits if h[0] not in seen])
    else:
        hits = merge_clips(hits)
    # 候选精排: 对前 N 个候选用 reranker 打分重排
    if (cfg.get("reranker") or {}).get("enabled"):
        n_cand = min(len(hits), (cfg.get("reranker") or {}).get("candidates", 20))
        hits = rerank_clips(q, hits[:n_cand]) + hits[n_cand:]
    hits = hits[:top_k]

    clips = []
    for cid, m, score, dist in hits:
        if not m.get("clip_path"):
            continue
        abs_path = os.path.join(cfg["paths"]["clips"], m["clip_path"])
        clips.append({
            "id": cid,
            "video": m["video"],
            "video_duration": m.get("video_duration", 0),
            "start": m["start"],
            "end": m["end"],
            "distance": round(dist, 4) if dist is not None else None,
            "rrf_score": round(score, 6),
            "clip_url": m["clip_url"],
            "clip_path": m["clip_path"],
            "summary": m.get("summary", ""),
            "objects": m.get("objects", ""),
            "actions": m.get("actions", ""),
            "scene": m.get("scene", ""),
            "ocr": m.get("ocr", ""),
            "asr": m.get("asr", ""),
            "merged": bool(m.get("merged")),
            "_abs_path": abs_path,
        })

    answer = ""
    if req.with_answer and clips:
        clip_texts = [
            " | ".join(x for x in (c.get("summary"), c.get("objects"),
                                   c.get("actions"), c.get("asr")) if x)
            for c in clips
        ]
        msgs = build_vlm_messages(q, [c["_abs_path"] for c in clips],
                                  history=req.history, clip_texts=clip_texts)
        answer = _generate(msgs, cfg["server"]["max_new_tokens"])

    for c in clips:
        c.pop("_abs_path", None)
    return {"question": q, "answer": answer, "clips": clips}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """浏览器录制的语音 -> 文字 (语音提问用, 复用 faster-whisper)。"""
    suffix = os.path.splitext(file.filename or "audio.webm")[1] or ".webm"
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
        return {"text": transcribe_clip(tmp)}
    finally:
        os.unlink(tmp)


# ---- 上传自动入库 (后台单队列, 避免 GPU 并发争抢) ----
_jobs = {}
_jobs_lock = threading.Lock()
_job_ids = itertools.count(1)
_queue = queue.Queue()


def _index_worker():
    while True:
        job_id, vpath = _queue.get()
        try:
            with _jobs_lock:
                _jobs[job_id]["status"] = "切片+向量化+摘要中..."
            n, skipped = process_video(
                vpath, cfg, col, text_col, embedder,
                enrich_fn=enrich_clip_with_vlm,
                asr_fn=transcribe_clip,
            )
            with _jobs_lock:
                _jobs[job_id]["status"] = "已入库" if not skipped else "已存在, 跳过"
                _jobs[job_id]["clips"] = n
        except Exception as exc:  # noqa: BLE001
            log.exception("上传视频入库失败: %s", vpath)
            with _jobs_lock:
                _jobs[job_id]["status"] = "失败"
                _jobs[job_id]["error"] = str(exc)
        finally:
            _queue.task_done()


threading.Thread(target=_index_worker, daemon=True).start()


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    name = os.path.basename(file.filename or "video.mp4")
    if not any(name.lower().endswith(e) for e in cfg["index"]["video_exts"]):
        return {"error": f"不支持的文件类型: {name} (支持 {cfg['index']['video_exts']})"}
    dest = os.path.join(cfg["paths"]["videos"], name)
    with open(dest, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)

    job_id = next(_job_ids)
    with _jobs_lock:
        _jobs[job_id] = {"id": job_id, "status": "排队中", "video": name}
        # 只保留最近 100 个任务记录
        if len(_jobs) > 100:
            for k in sorted(_jobs)[: len(_jobs) - 100]:
                _jobs.pop(k, None)
    _queue.put((job_id, dest))
    log.info("收到上传 %s (job %d), 已入队", name, job_id)
    return {"job_id": job_id, "status": "排队中", "video": name}


@app.get("/jobs/{job_id}")
def job_status(job_id: int):
    with _jobs_lock:
        j = _jobs.get(job_id)
    if j is None:
        return {"error": "任务不存在"}
    return j


@app.get("/health")
def health():
    return {
        "status": "ok",
        "clips_in_index": col.count(),
        "text_index": text_col.count(),
        "text_types": {
            t: len(text_col.get(where={"text_type": t}, include=[])["ids"])
            for t in ("summary", "caption", "asr")
        },
        "hybrid": cfg["server"].get("hybrid", True),
        "vllm": VLLM_ENABLED,
    }


@app.get("/")
def index_page():
    return FileResponse(os.path.join(os.path.dirname(__file__), "webui.html"))


app.mount("/clips", StaticFiles(directory=cfg["paths"]["clips"]), name="clips")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=cfg["server"]["host"], port=cfg["server"]["port"])
