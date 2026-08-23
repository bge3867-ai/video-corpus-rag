# Video Corpus RAG: Technical Overview

This document explains what Video RAG is, why it exists, how this project implements it, and how the retrieval and multimodal fusion actually work. It is written for a first-time visitor who wants to understand the project's value in ~10 minutes.

## 1. Why Video RAG?

Text RAG (retrieval-augmented generation) solved a real problem: LLMs cannot know documents they were not trained on, so we index documents, retrieve the relevant ones at query time, and let the model answer over that evidence. Video has the same problem — only worse:

- **Video is opaque to text models.** A video file has no textual representation. You cannot "grep" a video for "a man in a red jacket running across the street".
- **Video is enormous.** An hour of video is gigabytes of data. Feeding raw video into a context window is impossible; even sparse frame sampling quickly exhausts token budgets.
- **Video meaning is temporal.** The same scene means different things at second 10 and second 70. Retrieval must return *moments*, not files.
- **Video is multimodal.** Speech, on-screen text, objects, actions and scene context all carry meaning simultaneously. None alone tells the whole story.

Traditional video management gives you filenames and metadata (date, title). Traditional video search gives you single-video keyword matching. **Video RAG** gives you semantic, corpus-wide, evidence-grounded question answering over raw video content — the video equivalent of what text RAG did for documents.

## 2. Video RAG vs. Ordinary RAG

| Dimension | Text RAG | Video RAG (this project) |
|---|---|---|
| Indexing unit | text chunk | short video clip (6 s) |
| Embedding model | text encoder | vision-language embedding model (Qwen3-VL-Embedding-8B) |
| Query | text | natural language (optionally rewritten into multiple phrases) |
| Retrieved evidence | text passages | **actual video clips** (playable mp4 + timestamps) |
| Generation | LLM reads passages | VLM *sees* the retrieved clips and answers |
| Extra signals | — | ASR transcripts, on-screen text (OCR), structured captions |
| Output | text with citations | text with `[Clip N]` citations **plus the cut clips themselves** |

The critical difference: the evidence returned to the user is not a re-typed description of the video — it is the video. The system cuts the exact few-second mp4 files out of the source videos, so users can *watch* the evidence.

## 3. System Architecture

The system has two phases: an offline indexing pipeline and an online query pipeline, connected by a shared Chroma vector store.

### 3.1 Offline indexing (once per video)

```
video file
    │  ① slicing: 6 s sliding windows, 1.5 s overlap
    │     (or shot-boundary segmentation in adaptive mode)
    ▼
clips (mp4 files)
    │
    ├──► ② video embedding: 8 sampled frames → Qwen3-VL-Embedding-8B → 4096-dim vector
    │         → Chroma collection "videosrag" (cosine)
    │
    ├──► ③ structured captioning: one VLM call returns
    │         summary / objects / actions / scene / on-screen text (OCR)
    │         → text vector "{clip_id}_sum" and "{clip_id}_cap"
    │
    └──► ④ ASR: faster-whisper large-v3-turbo transcribes the audio track
              → text vector "{clip_id}_asr"
              (silent clips / no audio track are skipped)

      All text vectors live in Chroma collection "videosrag_text" (cosine)
```

Each clip therefore has **1 video vector + up to 3 text vectors**, with metadata carrying clip id, source video, timestamps, caption fields, and transcript.

### 3.2 Online query pipeline

```
question
    │  ① query rewriting: VLM expands the question into several synonymous
    │     phrases (e.g. "fire in the forest" → "forest burning", "wildfire scene")
    │
    ├──► ② hybrid retrieval (per phrase):
    │        video path: query vector vs "videosrag"
    │        text path:  query vector vs "videosrag_text"
    │        (text hits grouped per clip, best similarity × type weight)
    │
    │  ③ RRF fusion: all phrases' rankings merged into one score pool
    ▼
candidate clips (top-N)
    │  ④ merging: clips from the same video with ≥ 50% overlap are merged and re-cut
    │  ⑤ reranking (optional): Qwen3-Reranker-8B re-scores the candidates
    ▼
top-k clips
    │  ⑥ VLM reasoning: frames sampled per clip → Qwen3-VL-8B-Instruct (vLLM)
    │     answers strictly from the clips, citing [Clip N]
    ▼
answer + playable clip files + timestamps + caption/OCR/ASR metadata
```

Multi-turn conversations pass previous Q&A history to the reasoning step, so follow-up questions ("what about the second one?") are resolved against the previous clips.

## 4. Retrieval Pipeline in Detail

### 4.1 Slicing

Videos are cut into 6-second windows with 1.5 s overlap; videos ≤ 6 s become a single clip. Optionally (`index.adaptive: true`), PySceneDetect finds shot boundaries first and clips are aligned to real scene transitions (between `seg_min` 4 s and `seg_max` 12 s). Scene-aligned clips make temporal hits crisper — a retrieved clip starts and ends where the action actually starts and ends.

### 4.2 Video embedding

Qwen3-VL-Embedding-8B embeds each clip by sampling 8 frames, passing them through the official chat template (with the model's system prompt), and pooling via the **last-token** of the final hidden state, followed by L2 normalization — exactly reproducing the model's original sentence-transformers config (1_Pooling → 2_Normalize). The result is a 4096-dim unit vector in the same space as the text embeddings, which is what makes cross-modal cosine similarity meaningful.

### 4.3 Text-side enrichment (the "action-level" trick)

A raw video embedding is good at *visual appearance* similarity but weak at *behavioral* queries ("who is walking?", "someone is cooking"). The fix is to give every clip a rich textual description and let text search cover what visual search misses:

- **Structured caption** — one VLM call per clip returns a fixed 5-line format: summary, objects, actions, scene, on-screen text. Splitting objects/actions/scene into separate fields means the text vectors literally contain the verbs, making action queries hit directly.
- **OCR** — the same call reads on-screen text (signs, banners, license plates, subtitles burned into the frame).
- **ASR** — faster-whisper transcribes speech; silence and audio-less clips are skipped (returns empty, no vector created).

### 4.4 Query rewriting

Raw queries often use words that don't appear in captions ("config blaze" vs the caption "fire"). The query is first expanded by the VLM into several synonymous retrieval phrases. Each phrase runs the full hybrid retrieval independently, and all their hit lists accumulate into a single RRF pool — so a hit found by *any* phrasing contributes to the final ranking. Phrases whose retrieval results overlap with the original query's results are weighted higher.

### 4.5 Hybrid retrieval + RRF

Two recall paths run in parallel:

1. **Video path** — query embedding vs clip embeddings (visual similarity).
2. **Text path** — query embedding vs text embeddings. Because one clip can have up to 3 text vectors, hits are grouped per clip and each clip keeps its *best* text similarity, multiplied by the per-type weight (`caption 1.0 / summary 0.8 / asr 0.8`).

The two paths produce independent rankings, fused with Reciprocal Rank Fusion:

```
rrf(d) = Σ_r 1 / (k + rank_r(d))     # k = 60, r ranges over both paths
```

RRF is a rank-based fusion: it requires no score calibration between the visual and textual spaces, which is exactly why it is robust when the two embeddings behave differently.

### 4.6 Merging and reranking

Sliding-window slicing means the same event usually appears in several overlapping clips. Adjacent clips from the same video with ≥ 50% temporal overlap are merged and re-cut into one clip, so the user gets one clean segment instead of three near-duplicates.

With `reranker.enabled: true`, the top ~20 candidates are then re-scored by Qwen3-Reranker-8B — a cross-encoder that reads the query *against* each candidate clip's text representation and produces a calibrated relevance score. The final top-k is selected from this precision ranking.

### 4.7 Reasoning

The top-k clips are sampled into frames (local scaling keeps each frame ≤ 245,760 pixels to bound token usage; frames-per-clip adapts to the context budget), and sent as images to Qwen3-VL-8B-Instruct running in a separate vLLM service (OpenAI-compatible endpoint). The prompt constrains the model to answer only from the clip content and to cite `[Clip N]`. The response returns the answer, the citation-annotated clips with timestamps, and the playable clip URLs.

## 5. Multimodal Fusion Methods

The project fuses modalities at three distinct points, each with a different technique chosen for its reliability:

| Stage | Fusion | Method | Why |
|---|---|---|---|
| Indexing | vision + text generation | VLM structured captioning (one call) | converts visual/audio content into searchable text without extra models |
| Indexing | audio → text | faster-whisper ASR | speech is indexed as text vectors in the shared embedding space |
| Retrieval | video path + text path | per-clip grouping + weighted RRF | rank-based fusion needs no cross-modal score calibration |
| Retrieval | query expansion | multi-phrase RRF accumulation | multiple phrasings of one question all vote into one ranking |
| Reranking | cross-modal relevance | cross-encoder (Qwen3-Reranker) | calibrated precision scoring of query-vs-clip pairs |
| Reasoning | vision + language | VLM answer generation over clip frames | the final answer is grounded in what the clips actually show |

The design principle throughout: **index richly, fuse late**. Cheap upstream enrichment (captions, transcripts) is normalized into the same embedding space as the query, so the core retrieval stays a simple, robust vector + RRF pipeline, while the expensive multimodal reasoning happens only over the handful of retrieved clips.

## 6. Performance Notes

Measured on 4 × RTX 3090 (24 GB): a corpus of 37 videos / 136 six-second clips, single query answered in **3–5 s** (vLLM inference, ~1 s model time). Memory layout: embedding model ~17 GB, vLLM 8B ~21 GB, ASR ~2 GB, reranker ~15 GB — the four cards are assigned accordingly (`config.yaml`).

## 7. Where This Goes Next

See the [Roadmap](../README.md#-roadmap) in the README. The three directions that build directly on this foundation: hierarchical long-video indexing (shot → scene → video), a multimodal knowledge graph extracted from the corpus, and video-agent memory for embodied AI.
