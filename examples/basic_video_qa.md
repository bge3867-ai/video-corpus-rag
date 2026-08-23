# Example: Basic Video QA

The simplest possible flow: index a few videos, ask a question, watch the evidence.

## Prerequisites

Follow the [Quick Start](../README.md#-quick-start) first — `setup.sh` run, models downloaded, and (optionally) the vLLM service started.

## Step 1 — Add videos

Put any videos into `videos/`:

```bash
cp ~/Downloads/my_vacation.mp4 videos/
cp ~/Downloads/my_pets.mp4 videos/
```

Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`.

## Step 2 — Index

```bash
bash run_index.sh
```

Each video is sliced into 6-second clips, embedded, captioned and transcribed. Watch progress in `logs/index.log`. Indexing is incremental — adding a new video later only processes the new file.

## Step 3 — Ask

```bash
bash run_server.sh   # if not already running

curl -X POST http://localhost:8899/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "Is there a dog playing with a ball?", "top_k": 3}'
```

Response:

```json
{
  "question": "Is there a dog playing with a ball?",
  "answer": "Yes — in the backyard, a golden retriever repeatedly chases and fetches a red ball thrown by a person off-camera. [Clip 1]",
  "clips": [
    {
      "id": "a1b2c3",
      "video": "my_pets",
      "start": 18.0,
      "end": 24.0,
      "clip_url": "/clips/my_pets/my_pets_000_18-24.mp4",
      "distance": 0.132,
      "rrf_score": 0.031,
      "summary": "A golden retriever fetches a red ball in the backyard",
      "objects": "dog, ball, grass",
      "actions": "running, fetching",
      "scene": "backyard",
      "ocr": "",
      "asr": "Go get it! Good boy!"
    }
  ]
}
```

## Step 4 — Watch the evidence

Open the `clip_url` in your browser (`http://localhost:8899/clips/my_pets/my_pets_000_18-24.mp4`) — it is the actual 6-second cut from the source video, not a generated preview. The timestamps tell you exactly where in the original video the moment occurs.

## Step 5 — Use the web UI

Open `http://localhost:8899` instead of curl: same engine, plus chat history, timeline hit visualization, voice input via the microphone button, and video upload via drag-and-drop.

## Tips

- `top_k` controls how many clips are returned (1–8); lower values answer faster
- Questions about *actions* work best ("who is running?", "when does the car turn?") because every clip is captioned with objects/actions/scene
- Add `"history"` and `"last_clips"` from the previous response to ask follow-up questions ("what about the second clip?") with full conversation context
