# Example: Long Video Retrieval

This walkthrough shows how Video Corpus RAG handles a *long* video — dozens of minutes or hours — and how to make temporal queries precise with scene-aware slicing and clip merging.

## Scenario

You have a 45-minute surveillance video `campus_west_2024.mp4`. You want to know: *"When does the delivery van arrive at the loading dock?"* — and you want the exact moment, not a summary.

## Step 1 — Enable adaptive scene slicing (recommended for long videos)

In `config.yaml`:

```yaml
index:
  adaptive: true      # align clips to shot/scene boundaries instead of fixed 6 s windows
  seg_min: 4.0        # shortest clip (seconds)
  seg_max: 12.0       # longest clip (seconds)
```

Then rebuild the index:

```bash
bash run_index.sh --force
```

Why this matters: fixed 6-second windows cut events in the middle. Scene-aligned clips start and end at real shot transitions, so a retrieved clip *is* the event you asked about — the van arriving — not a fragment of it plus the previous event.

## Step 2 — Query with a temporal intent

```bash
curl -X POST http://localhost:8899/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "When does the delivery van arrive at the loading dock?", "top_k": 5}'
```

The query is automatically rewritten into multiple retrieval phrases (e.g. "van pulling into loading dock", "delivery truck arrival") — each retrieved and fused, so the hit is found even if the captions phrase it differently.

## Step 3 — Read the timeline answer

```json
{
  "answer": "The delivery van arrives at the loading dock at 23:14 into the video: a white van reverses toward the dock, the rear doors open, and a worker begins unloading boxes. [Clip 2]",
  "clips": [
    {
      "video": "campus_west_2024",
      "start": 1394.0,
      "end": 1406.0,
      "clip_url": "/clips/campus_west_2024/campus_west_2024_023_1394-1406.mp4",
      "summary": "A white delivery van reverses to the loading dock and unloads boxes",
      "objects": "van, dock, boxes, worker",
      "actions": "reversing, unloading",
      "scene": "loading dock",
      "ocr": "COURIER EXPRESS",
      "asr": ""
    }
  ]
}
```

The `start`/`end` fields give the exact window inside the 45-minute source; `clip_url` is the cut mp4 of just that moment.

## Step 4 — Merged results, not duplicates

Sliding windows (or scene detection) can produce several overlapping hits of the same event. The server merges clips from the same video whose overlap exceeds `server.merge_overlap` (default 0.5) and re-cuts them — so your top-k contains *distinct events*, e.g. van arrival, worker unloading, van departure, rather than three copies of the arrival.

## Step 5 — Optional: reranking for precision

With many candidate hits, enable the reranker in `config.yaml`:

```yaml
reranker:
  enabled: true
  model_id: Qwen/Qwen3-Reranker-8B
  device: "cuda:3"
  candidates: 20
```

The top 20 candidates from hybrid retrieval are re-scored by a cross-encoder that reads the query against each clip's caption/transcript — a much more precise relevance judgment than vector similarity alone. On our testbed this visibly cleaned up the top-5 for long-video temporal queries.

## Scaling up

- **Hours of video** — indexing time scales with clip count (roughly 10 clips/minute of video). Run indexing overnight; queries stay 3–5 s regardless of corpus size.
- **Many videos** — the retrieval is corpus-wide by design; all videos' clips live in one shared Chroma collection.
- **Follow-ups** — pass `history` + `last_clips` to `/ask` to ask "what happened after that?" conversationally; the model reasons with the previous clips as context.
