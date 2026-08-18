"""视频片段 embedding 封装。

使用 Qwen3-VL-Embedding-8B, 用 transformers 原生加载, 精确复刻官方
sentence-transformers 配置的 embedding 流程:
  1. chat template (自动注入 system prompt "Represent the user's input.")
     + add_generation_prompt=True (与模型保存时的 processing_kwargs 一致)
  2. 前向取 last_hidden_state
  3. lasttoken 池化 (取序列最后一个 token, 见 1_Pooling/config.json)
  4. L2 归一化 (见 2_Normalize 模块)
"""
import logging

import av
import torch
import torch.nn.functional as F

log = logging.getLogger("videosrag.embedder")

DOC_PROMPT = "Instruct: 概括该视频片段的内容, 用于文本检索。"   # 入库侧
QUERY_PROMPT = "Instruct: 检索与问题最相关的视频片段。"         # 查询侧
NFRAMES = 8  # 每个视频片段采样帧数


def extract_frames(video_path: str, n_frames: int = 8):
    """从视频均匀抽样 n_frames 帧, 返回 PIL 图片列表。"""
    import PIL.Image

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    n = max(0, container.duration or 0)
    if n <= 0:
        return []
    indices = sorted({int((i + 0.5) * n / n_frames) for i in range(n_frames)})
    frames = []
    for idx in indices:
        container.seek(idx)
        for frame in container.decode(video=0):
            frames.append(frame.to_image().convert("RGB"))
            break
    container.close()
    return frames


class Embedder:
    def __init__(self, model_dir: str, device: str = "cuda:0"):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        log.info("加载 embedding 模型 %s (device=%s) ...", model_dir, device)
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            device_map=device,
        ).eval()

        from qwen_vl_utils import process_vision_info

        self._process_vision_info = process_vision_info
        log.info("embedding 模型加载完成")

    def _embed_messages(self, messages):
        """chat template -> forward -> lasttoken 池化 -> L2 归一化"""
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[-1]                      # (B, seq, D)
        last_idx = inputs["attention_mask"].sum(dim=1) - 1  # 每行最后一个非 pad token
        vec = hidden[0, last_idx[0], :].float()             # lasttoken 池化
        return F.normalize(vec, dim=-1)                     # 2_Normalize: L2

    def encode_text(self, text: str, prompt: str = QUERY_PROMPT):
        """纯文本 embedding, 可用于查询侧或摘要侧(传 DOC_PROMPT)。"""
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt + "\n" + text}],
        }]
        return self._embed_messages(messages).cpu().numpy()

    def encode_query(self, text: str):
        return self.encode_text(text, QUERY_PROMPT)

    def encode_clip(self, video_path: str, frames=None):
        import os

        # qwen_vl_utils 只处理 `file://` + 绝对路径; 相对路径会被静默读成 0 帧
        abs_path = os.path.abspath(video_path)
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{abs_path}",
                    "nframes": NFRAMES,
                },
                {"type": "text", "text": DOC_PROMPT},
            ],
        }]
        return self._embed_messages(messages).cpu().numpy()