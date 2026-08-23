# Contributing to Video Corpus RAG

Thanks for your interest in contributing! This project aims to be the go-to open-source multimodal Video RAG framework, and every contribution — code, docs, bug reports, or ideas — helps.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Environment Setup](#environment-setup)
- [Reporting Issues](#reporting-issues)
- [Contributing Code](#contributing-code)
- [Development Workflow](#development-workflow)
- [Style Guide](#style-guide)
- [Testing](#testing)

## Code of Conduct

Be respectful and constructive. Assume good faith, keep discussions technical, and help newcomers. Harassment of any kind is not tolerated.

## Environment Setup

Requirements: Linux, NVIDIA GPU (≥ 24 GB VRAM recommended), CUDA ≥ 12.4, conda.

```bash
git clone https://github.com/bge3867-ai/video-corpus-rag.git
cd video-corpus-rag

# 1. App environment (slicing / embedding / retrieval / server)
bash setup.sh

# 2. vLLM inference environment (optional but recommended)
conda create -n vllm python=3.11 -y && conda activate vllm
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128  # match your CUDA
pip install -r requirements-vllm.txt

# 3. Models (~35 GB)
bash download_models.sh
```

For a minimal development loop you can also install just the app deps (`requirements.txt`) and set `vllm.enabled: false` in `config.yaml` to use in-process transformers instead of the vLLM service.

## Reporting Issues

Found a bug or want a feature? Open an [issue](https://github.com/bge3867-ai/video-corpus-rag/issues).

A good bug report includes:

1. **Environment** — GPU model & count, CUDA version, `pip list | grep -E "torch|vllm|chromadb|faster-whisper"` output
2. **What you did** — exact command or HTTP request
3. **What you expected vs. what happened** — including the error message
4. **Logs** — relevant tail of `logs/server.log` and `logs/vllm.log`

A good feature request explains the use case first ("I want to query X and get Y") before proposing an implementation.

## Contributing Code

1. **Discuss first for anything large** — open an issue describing the change before writing code, so we can align on the design.
2. **Fork & branch** — work on a feature branch off `main`:
   ```bash
   git checkout -b feat/my-feature
   ```
3. **Keep the core stable** — retrieval/indexing behavior changes need evidence (before/after query examples or benchmark numbers).
4. **One concern per PR** — small, focused PRs are reviewed much faster.
5. **Write a clear PR description** — what problem it solves, how it works, and how you tested it.
6. **Add docs** — new config options go in `config.yaml` (with a comment) and, if user-facing, in the README.
7. **Open the PR** — reference the related issue, keep the branch up to date with `main`, and respond to review comments.

### Areas that welcome contributions

- **Evaluation** — a benchmark harness comparing retrieval quality (recall@k, mAP) across configurations
- **More VLMs** — adapters for other open vision-language models (InternVL, LLaVA, MiniCPM-V…)
- **More embedding models** — pluggable video embedding backends beyond Qwen3-VL-Embedding
- **Long-video hierarchy** — shot → scene → video summarization (see Roadmap)
- **Language coverage** — better prompts/ASR behavior for languages beyond Chinese/English
- **Packaging** — single-command installs, GPU-less dev mode, CI

## Development Workflow

```bash
# index a small test video
mkdir -p videos && cp /path/to/sample.mp4 videos/
bash run_index.sh

# run the server in the foreground to see logs
conda activate videosrag
python src/server.py

# quick query
curl -X POST http://localhost:8899/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "what is happening in this video?", "top_k": 3}'

# maintenance scripts
python tools/backfill_enrich.py --limit 5   # test enrichment on 5 clips
python tools/cleanup_video.py sample        # remove test data
```

## Style Guide

- Python 3.10+, no new runtime dependencies without a good reason — the two-environment split (`requirements.txt` vs `requirements-vllm.txt`) exists to keep the app env light; respect it
- Log with the module logger (`log = logging.getLogger("videosrag.<module>")`), never `print` in library code
- Add a brief docstring to new modules/classes; comments in Chinese are the current convention, English is also fine — but keep it consistent within a file
- Large files (`server.py`, `index.py`) are fine; do not split modules without coordination

## Testing

There is no CI yet (GPU-dependent). Before submitting a PR, please run at least:

```bash
# embedding smoke test: two videos + one query, verify semantic similarity ordering
python src/selftest.py videos/a.mp4 videos/b.mp4 "your test query"

# end-to-end query smoke test (server must be running)
curl -s -X POST http://localhost:8899/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "your test question", "top_k": 3}' | python tools/asktest.py
```

and include the output in your PR description.

## Questions?

Open a [discussion issue](https://github.com/bge3867-ai/video-corpus-rag/issues) — no question is too basic.
