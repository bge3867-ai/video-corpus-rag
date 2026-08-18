"""片段文本侧增强: 结构化描述 (对象/动作/场景/画面文字) + ASR 语音转写。

纯逻辑模块, 被 server.py (增量入库) 与 tools/backfill_enrich.py (回填) 共用。
结构化描述走 VLM 一次调用输出固定五行格式, OCR 并入其中 (画面文字行),
ASR 走 faster-whisper。
"""
import base64
import io
import logging
import os
import subprocess

from embedder import extract_frames

log = logging.getLogger("videosrag.enrich")

CAPTION_PROMPT = (
    "请分析这个视频片段, 严格按以下格式输出, 每行一个字段, 不要输出其他内容:\n"
    "摘要: <一句话中文概括, 不超过40字>\n"
    "对象: <画面中的主要对象, 用顿号分隔, 无则写无>\n"
    "动作: <画面中的动作/事件, 用顿号分隔, 无则写无>\n"
    "场景: <地点/环境短语, 无则写无>\n"
    "画面文字: <画面中出现的所有文字内容, 逐条用顿号分隔, 无则写无>"
)


def parse_caption(text: str):
    """解析 VLM 结构化输出 -> {summary, objects, actions, scene, ocr}。

    容错: 任一行缺失不报错; 完全解析失败时整段文本当摘要。
    """
    out = {"summary": "", "objects": "", "actions": "", "scene": "", "ocr": ""}
    if not text:
        return out
    keys = {"摘要": "summary", "对象": "objects", "动作": "actions",
            "场景": "scene", "画面文字": "ocr"}
    found = False
    for line in text.splitlines():
        line = line.strip().lstrip("-*#0123456789.、 ").strip()
        for k, v in keys.items():
            if line.startswith(k):
                val = line[len(k):].lstrip(":： ").strip()
                if val and val != "无":
                    out[v] = val
                found = True
                break
    if not found:
        out["summary"] = text.strip()
    return out


def caption_text(meta):
    """把结构化字段拼成用于文本检索的字符串 (对象+动作+场景+画面文字)。"""
    parts = []
    for k, label in (("objects", "对象"), ("actions", "动作"),
                     ("scene", "场景"), ("ocr", "画面文字")):
        v = (meta or {}).get(k)
        if v:
            parts.append(f"{label}: {v}")
    return "; ".join(parts)


def frame_parts(cfg, abs_path: str, n_frames: int):
    """把视频抽成 n 帧并本地缩放, 返回 vLLM 图片内容块列表 (base64 data URL)。

    缩放上限取 config 的 vlm_max_pixels (默认约 640x384), 保证服务端无论
    是否再缩放, token 数都稳定可控。
    """
    max_px = cfg["server"]["vlm_max_pixels"]
    parts = []
    for pil in extract_frames(abs_path, n_frames):
        w, h = pil.size
        if w * h > max_px:
            f = (max_px / (w * h)) ** 0.5
            nw, nh = max(int(w * f / 32) * 32, 32), max(int(h * f / 32) * 32, 32)
            pil = pil.resize((nw, nh))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        parts.append({"type": "image_url",
                      "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return parts


def clip_has_audio(clip_path: str) -> bool:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(clip_path)],
        capture_output=True, text=True,
    )
    return bool(r.stdout.strip())


class ASRTranscriber:
    """faster-whisper 封装: 懒加载, 无音轨/静音片段返回空串, 加载失败返回 None。"""

    def __init__(self, model: str = "large-v3-turbo", device: str = "cuda:2"):
        self.model_name = model
        # faster-whisper 用 device_index 选卡, 不支持 "cuda:N" 写法
        self.device = "cpu" if str(device) == "cpu" else "cuda"
        self.device_index = (
            int(str(device).split(":")[-1]) if ":" in str(device) else 0
        )
        self._model = None
        self._load_failed = False

    def _load(self):
        from faster_whisper import WhisperModel  # 懒加载, 未安装不影响其他功能

        log.info("加载 faster-whisper %s (device=%s, index=%d) ...",
                 self.model_name, self.device, self.device_index)
        self._model = WhisperModel(
            self.model_name, device=self.device, device_index=self.device_index,
            compute_type="float16",
        )
        log.info("ASR 模型加载完成")
        return self._model

    def transcribe(self, clip_path: str):
        """转写片段语音。

        返回 None=模型不可用(加载失败, 调用方跳过即可);
        ""=片段无音轨或静音; 否则为转写文本。
        """
        if not clip_has_audio(clip_path):
            return ""
        if self._model is None and not self._load_failed:
            try:
                self._load()
            except Exception as exc:  # noqa: BLE001 环境缺依赖时静默跳过
                self._load_failed = True
                log.warning("ASR 模型加载失败, 跳过转写: %s", exc)
        if self._model is None:
            return None
        try:
            segments, _ = self._model.transcribe(
                str(clip_path), beam_size=1, vad_filter=True,
            )
            texts = []
            for seg in segments:
                if seg.no_speech_prob < 0.6:
                    texts.append(seg.text.strip())
            return " ".join(t for t in texts if t).strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("片段 %s ASR 失败: %s", os.path.basename(clip_path), exc)
            return ""
