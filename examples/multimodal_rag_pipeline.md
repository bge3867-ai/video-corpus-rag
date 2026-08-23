# Example: The Full Multimodal RAG Pipeline

This example follows one video end-to-end through the pipeline and shows how each modality (vision, action captioning, OCR, ASR) contributes to retrieval and answering. It is the concrete companion to [docs/overview.md](../docs/overview.md).

## The input

A 30-second clip from a kitchen video, with a person talking while cooking, and a recipe on a board in the background.

## Stage 1 — Video Processing (indexing)

```
kitchen.mp4 (30 s)
    │  5 sliding windows of 6 s (1.5 s overlap)
    ▼
clips: kitchen_000_0-6, kitchen_000_1.5-7.5, ..., kitchen_000_24-30
```

## Stage 2 — Multimodal Indexing

For **each** clip, three things are computed in parallel:

### 2a. Video embedding (vision)

8 frames sampled → Qwen3-VL-Embedding-8B → 4096-dim L2-normalized vector → Chroma collection `videosrag`.

This vector captures *what the clip looks like* — a clip of someone chopping vegetables is now close, in cosine space, to the text query "vegetables being chopped".

### 2b. Structured caption (one VLM call)

The VLM returns a fixed 5-field description:

```
摘要: 一位厨师在厨房切西红柿并把菜倒入锅中
对象: 厨师, 西红柿, 炒锅, 砧板
动作: 切菜, 倒菜, 翻炒
场景: 厨房
画面文字: 3 tablespoons olive oil
```

- The summary and the object/action/scene fields become text vectors `{clip_id}_sum` and `{clip_id}_cap`.
- The **actions field is the action-level retrieval key**: the verb *切菜/翻炒* is literally in the indexed text, so "who is stir-frying?" hits this clip even though the video embedding alone might not.

### 2c. ASR (faster-whisper)

The audio track is transcribed: *"先把西红柿切好，然后开大火翻炒两分钟"* → text vector `{clip_id}_asr`.

Now the clip has **1 video vector + 3 text vectors**, and its metadata carries all fields:

```json
{
  "id": "kitchen_000_1.5-7.5",
  "summary": "一位厨师在厨房切西红柿并把菜倒入锅中",
  "objects": "厨师, 西红柿, 炒锅, 砧板",
  "actions": "切菜, 倒菜, 翻炒",
  "scene": "厨房",
  "ocr": "3 tablespoons olive oil",
  "asr": "先把西红柿切好，然后开大火翻炒两分钟"
}
```

## Stage 3 — Retrieval (one question, three paths)

Question: **"What does the chef say about the stir-fry?"**

1. **Query rewrite** — expanded into retrieval phrases: "厨师说翻炒", "stir-fry instructions", "炒菜步骤说明".
2. **Video path** — query vector vs clip vectors: the kitchen clips score high visually (chef, pan).
3. **Text path** — query vector vs text vectors: the `_asr` vector of clip 1.5–7.5 hits ("翻炒"), the `_cap` vector of clip 0–6 hits ("翻炒" in actions).
4. **Per-clip grouping** — each clip keeps its best text similarity × type weight.
5. **RRF fusion** — both paths' rankings merge; the clip where the speech actually occurs rises to the top because *both* its visual and ASR signals agree.

## Stage 4 — Reasoning (VLM over retrieved clips)

Top-k clips are sampled into frames and sent to Qwen3-VL-8B-Instruct (vLLM):

```
Question: What does the chef say about the stir-fry?
Clips: [Clip 1: kitchen 1.5–7.5s] [Clip 2: kitchen 0–6s] [Clip 3: kitchen 6–12s]
Answer only from the clips and cite [Clip N].
```

Answer:

> "The chef says to cut the tomatoes first, then stir-fry over high heat for two minutes. [Clip 1]"

The response also returns the clip files themselves, so the user can listen to the exact utterance.

## What each modality buys you

| Query | Winning modality | Why |
|---|---|---|
| "a chef stir-frying" | video embedding + actions caption | visual + verb match |
| "what does the chef say?" | ASR vector | speech indexed as text |
| "3 tablespoons olive oil" | OCR (inside caption) | on-screen text indexed |
| "kitchen scene at the start" | temporal metadata | start/end times + scene field |

Remove any one modality and a whole class of queries goes dark — that is why the pipeline enriches all of them at indexing time rather than choosing one at query time.

## Try it yourself

```bash
cp your_kitchen_video.mp4 videos/
bash run_index.sh
bash run_server.sh

curl -X POST http://localhost:8899/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What does the chef say about the stir-fry?", "top_k": 3}'
```
