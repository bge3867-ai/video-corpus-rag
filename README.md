# Video Corpus RAG: A Multimodal Video Retrieval-Augmented Generation Framework for Long Video Understanding

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4%2B-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**Video Corpus RAG** is a fully open-source, multimodal Video RAG framework that lets you ask natural-language questions against an *entire video corpus* and receive answers grounded in the actual video clips — retrieved, cut, and cited.

Ask *"What happened before the person opened the door?"* across thousands of hours of video, and the system finds the semantically relevant moments, extracts the exact few-second clips, and answers with evidence annotations.

Built entirely from open-source components (an open alternative to TwelveLabs + Chroma stacks):

- **Video RAG** — semantic retrieval over a whole video library, not a single video
- **Multimodal Retrieval** — video embeddings + text embeddings fused with Reciprocal Rank Fusion (RRF)
- **VLM Reasoning** — Qwen3-VL-8B-Instruct answers directly over the retrieved clips
- **ASR/OCR Fusion** — speech transcripts and on-screen text are indexed and retrievable
- **Action-level Retrieval** — structured captions (objects / actions / scene) make verb queries like *"a man running"* hit precisely
- **vLLM Acceleration** — 3–5 s per query on a single GPU, with concurrent inference

All models are Apache-2.0 licensed.

## ✨ Features

- **Long video semantic retrieval** — 6-second clips (with overlap) embedded by Qwen3-VL-Embedding-8B, indexed in Chroma with cosine similarity
- **Action-level temporal retrieval** — every clip is automatically captioned into objects / actions / scene, so behavior-oriented queries (*"walking"*, *"cooking"*, *"running away"*) match directly
- **Multimodal evidence aggregation** — video vectors + summary/caption/ASR text vectors, fused per-clip with weighted RRF
- **ASR and OCR integration** — faster-whisper transcripts ("who said what") and on-screen text (signs, banners, license plates) are searchable
- **Vision-language model reasoning** — answers are generated only from the retrieved clips, with `[Clip N]` provenance annotations
- **Efficient inference with vLLM** — OpenAI-compatible inference service with graceful fallback to in-process transformers
- **Query rewriting & multi-turn conversation** — queries are expanded into synonymous phrases; follow-up questions carry conversation history
- **Precision reranking** — optional Qwen3-Reranker-8B re-scoring of candidates
- **Adaptive scene slicing** — optional PySceneDetect shot-boundary segmentation instead of fixed windows
- **Batteries-included web UI** — upload, chat, timeline hit visualization, in-browser voice input, clip playback/download

## 🏗️ Architecture

```
Video Input ──► Video Processing ──► Multimodal Indexing ──► Retrieval ──► VLM Reasoning ──► Answer Generation
   │                  │                      │                   │               │                   │
  videos/         6s sliding        Qwen3-VL-Embedding-8B    video path    Qwen3-VL-8B-Instruct    answer with
 (mp4/avi/mov)    window clips        + Chroma (cosine)     + text path    (vLLM service)        [Clip N] citations
                  (1.5s overlap)          │                   RRF fusion         │                   │
                       │                  │                       │               │              clip mp4s
                  VLM caption        text vectors             per-clip       optional 32B       + start/end times
                  (objects/actions/  (summary/caption/asr)    grouping       bnb-4bit mode      + caption/OCR/ASR
                  scene/OCR)              │                     top-k             │
                  ASR (whisper) ──────────┘                     merge           top-k clips
                                                                  │             sampled frames
                                                           query rewrite        as images
                                                           (synonym phrases)
```

**Offline indexing pipeline** (once per video): slice → sample frames → embed video clip → generate structured caption (one VLM call) → transcribe audio → store 1 video vector + up to 3 text vectors per clip in Chroma.

**Online query pipeline**: question → query rewrite (synonym expansion) → hybrid retrieval (video + text, RRF) → adjacent-clip merging → optional reranking → VLM reasoning over top-k clips → cited answer + playable clip files.

GPU layout used in our testbed (adjust in `config.yaml`): `cuda0` = embedding, `cuda1` = vLLM Q&A, `cuda2` = ASR, `cuda3` = reranker.

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/bge3867-ai/video-corpus-rag.git
cd video-corpus-rag
bash setup.sh            # creates conda env `videosrag` + installs deps (incl. CUDA PyTorch)
```

Requirements: Linux, NVIDIA GPU (≥ 24 GB VRAM recommended), CUDA ≥ 12.4, conda.

### 2. Environment setup

```bash
# vLLM inference service (optional, strongly recommended — 3-5 s answers)
conda create -n vllm python=3.11 -y && conda activate vllm
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128   # match your CUDA
pip install -r requirements-vllm.txt

# Download models (~35 GB, ModelScope first with HF fallback, resumable)
bash download_models.sh
```

### 3. Demo

```bash
# drop your videos into videos/
bash run_index.sh        # ① index the corpus (idempotent, incremental)
bash run_vllm.sh         # ② start the vLLM inference service (optional)
bash run_server.sh       # ③ start the app server
```

Open **http://localhost:8899** in your browser, or query via HTTP:

```bash
curl -X POST http://localhost:8899/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What happened before the person opened the door?", "top_k": 4}'
```

That's it — from clone to first answer in about 5 minutes (plus model download time). No vLLM? Set `vllm.enabled: false` in `config.yaml` and the server falls back to in-process transformers (slower, zero extra setup).

## 📖 Example

**Question:**
> "What happened before the person opened the door?"

**Retrieved Evidence:**
```json
{
  "clips": [
    {
      "id": "clip_a1b2c3",
      "video": "office_tour_02",
      "start": 12.0, "end": 18.0,
      "clip_url": "/clips/office_tour_02/office_tour_02_000_12-18.mp4",
      "summary": "A man walks down the corridor carrying a stack of documents",
      "objects": "man, documents, door", "actions": "walking, carrying",
      "scene": "office corridor",
      "ocr": "Conference Room B",
      "asr": "I'll grab the reports and meet you inside."
    }
  ]
}
```

**Answer:**
> Before opening the door, the man walked down the office corridor carrying a stack of documents and said he would grab the reports and meet the other person inside. [Clip 1]

## 📁 Project Structure

```
video-corpus-rag/
├── src/                     # core library
│   ├── server.py            # FastAPI app: /ask, /upload, /transcribe, /health
│   ├── index.py             # slicing (fixed/adaptive), embedding, indexing pipeline
│   ├── embedder.py          # Qwen3-VL-Embedding-8B wrapper (last-token pooling + L2)
│   ├── enrich.py            # structured captioning, frame sampling, ASR transcriber
│   ├── summarizer.py        # local fallback VLM summarizer
│   ├── common.py            # config loading, path handling
│   └── webui.html           # chat web UI (timeline, voice input, upload)
├── tools/                   # maintenance scripts
│   ├── backfill_enrich.py   # backfill text-side enrichment for existing clips
│   ├── cleanup_video.py     # delete a video + its clips + vectors
│   └── asktest.py           # CLI quick query
├── examples/                # usage walkthroughs
├── docs/overview.md         # technical overview: why & how Video RAG works
├── config.yaml              # all tunables (indexing, retrieval, enrichment, vLLM)
├── run_index.sh / run_server.sh / run_vllm.sh (+ run_vllm32.sh)   # launchers
├── Dockerfile / docker-compose.yml / Dockerfile.vllm              # container packaging
├── setup.sh / download_models.sh / requirements*.txt              # env & models
└── CONTRIBUTING.md
```

## ⚙️ Configuration

Key options in `config.yaml` (full file is commented):

| Option | Default | Description |
|---|---|---|
| `index.seg_len / overlap` | `6.0 / 1.5` | clip length and overlap (seconds) |
| `index.adaptive` | `false` | shot-boundary adaptive slicing (PySceneDetect) |
| `index.frames_per_clip` | `8` | frames sampled per clip for embedding |
| `server.default_top_k / max_top_k` | `4 / 8` | number of returned clips |
| `server.query_rewrite / merge_overlap / conversation` | `true / 0.5 / true` | query expansion, clip merging, multi-turn chat |
| `server.hybrid` | `true` | hybrid video+text retrieval switch |
| `server.text_weights` | `caption 1.0 / summary 0.8 / asr 0.8` | per-type text weights |
| `enrich.asr_model / asr_device` | `large-v3-turbo / cuda:2` | ASR model and GPU |
| `reranker.enabled` | `false` | enable Qwen3-Reranker-8B precision reranking |
| `vllm.enabled / max_model_len` | `true / 8192` | vLLM switch and context length (must match `run_vllm.sh`) |

## 🧠 Retrieval Pipeline

1. **Slicing** — videos are cut into 6 s windows with 1.5 s overlap (or shot-aligned segments in adaptive mode)
2. **Video vectors** — 8 frames per clip → Qwen3-VL-Embedding-8B (chat template, last-token pooling, L2) → 4096-dim vectors
3. **Text vectors** — up to 3 texts per clip (summary / structured caption with OCR / ASR transcript) embedded by the same model into a separate Chroma collection
4. **Hybrid retrieval** — video and text paths recall candidates; text hits are grouped per clip (best similarity × type weight); RRF fuses the two rankings
5. **Query rewriting** — the question is expanded into several synonymous phrases, each retrieved and accumulated in the same RRF pool
6. **Merging & reranking** — heavily overlapping clips from the same video are merged and re-cut; optional Qwen3-Reranker-8B re-scores the candidates
7. **Answering** — top-k clips are sampled into frames and sent to Qwen3-VL-8B-Instruct (vLLM), which must answer only from the clips and cite `[Clip N]`

## 📡 HTTP API

| Endpoint | Method | Description |
|---|---|---|
| `/ask` | POST | query the corpus, returns answer + retrieved clips with times and metadata |
| `/upload` | POST | upload a video; auto slice/embed/enrich; returns `job_id` |
| `/jobs/{id}` | GET | indexing progress |
| `/transcribe` | POST | audio file → text (powers the web UI's voice input) |
| `/health` | GET | service status and vector counts per type |
| `/clips/...` | GET | playable clip files |

## 🐳 Docker

```bash
docker compose up -d server        # app service
docker compose up -d vllm          # vLLM inference service (build from Dockerfile.vllm)
```

## 🛠️ Operations

```bash
bash stop_server.sh / stop_vllm.sh     # stop services
bash run_index.sh --force              # rebuild the whole index
python tools/cleanup_video.py <name>   # delete a video + clips + vectors
python tools/backfill_enrich.py        # backfill text enrichment (idempotent, --asr-only)
tail -f logs/server.log logs/vllm.log  # logs
```

## 📊 Performance

Tested on 4 × RTX 3090 (24 GB): 37 videos / 136 six-second clips, one query answered in **3–5 s** with all three text-enrichment paths (caption / OCR / ASR) online.

## ❓ FAQ

- **Slow first start** — the embedding model takes 1–2 min to load; watch `logs/server.log`
- **400 "decoder prompt too long"** — lower `top_k`, or raise both `run_vllm.sh --max-model-len` and `config.yaml vllm.max_model_len` together
- **vLLM won't start** — see `logs/vllm.log`; lower `--gpu-memory-utilization` if VRAM is short. If you hit cache-block errors, set `longest_edge` in the model's `preprocessor_config.json` from `16777216` to `245760`
- **No second GPU for ASR** — set `enrich.asr_device` to a free card or `enrich.asr_enabled: false`
- **Upgrade to 32B** — `bash run_vllm32.sh` serves a bitsandbytes 4-bit Qwen3-VL-32B on port 8901; point `config.yaml vllm.base_url` at it

## 🗺️ Roadmap

- [ ] **Video Agent Memory** — persistent, queryable memory for embodied/video agents
- [ ] **Long Video Understanding** — hierarchical indexing: shot → scene → video summarization
- [ ] **Multimodal Knowledge Graph** — entity/relation extraction across the corpus
- [ ] **VLA Memory** — vision-language-action memory module for robotics pipelines

## 📚 Citation

A technical report is in preparation. If you use this project in your research, please cite the repository for now:

```bibtex
@software{video_corpus_rag,
  title        = {Video Corpus RAG: A Multimodal Video Retrieval-Augmented Generation
                  Framework for Long Video Understanding},
  author       = {bge3867-ai},
  year         = {2026},
  url          = {https://github.com/bge3867-ai/video-corpus-rag},
  note         = {Apache-2.0 licensed open-source software}
}
```

## 🤝 Contributing

Issues, feature requests and pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## 🙏 Acknowledgments

- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) — vision-language models (embedding + instruct)
- [Qwen3-Reranker](https://huggingface.co/Qwen/Qwen3-Reranker-8B) — precision reranking
- [Chroma](https://github.com/chroma-core/chroma) — vector database
- [vLLM](https://github.com/vllm-project/vllm) — high-performance inference engine
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — speech transcription
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — scene-boundary detection

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE). The models it depends on (Qwen3-VL family) are also Apache-2.0.
