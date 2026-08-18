"""VLM 摘要模块: 用 Qwen3-VL-8B-Instruct 为视频片段生成一句话中文语义摘要。"""
import logging
import os

import torch

log = logging.getLogger("videosrag.summarizer")

SUMMARY_PROMPT = (
    "请用一句中文概括这个视频片段的主要内容, 包含关键对象、动作和场景, "
    "不超过40个字, 直接输出摘要本身, 不要加引号、序号或任何前缀。"
)


class Summarizer:
    def __init__(self, model_dir: str, device: str = "cuda:1",
                 nframes: int = 8, max_pixels: int = 245760):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info

        log.info("加载摘要模型 %s (device=%s) ...", model_dir, device)
        self.device = device
        self.nframes = nframes
        self.max_pixels = max_pixels
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=device,
        ).eval()
        self._process_vision_info = process_vision_info
        log.info("摘要模型加载完成")

    def summarize(self, video_path: str) -> str:
        """为单个视频片段生成中文摘要。"""
        # qwen_vl_utils 只认 `file://` + 绝对路径
        abs_path = os.path.abspath(video_path)
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{abs_path}",
                    "nframes": self.nframes,
                    "max_pixels": self.max_pixels,
                },
                {"type": "text", "text": SUMMARY_PROMPT},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos,
            padding=True, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=64, do_sample=False,
            )
        gen = out[0][inputs.input_ids.shape[1]:]
        return self.processor.decode(gen, skip_special_tokens=True).strip()
