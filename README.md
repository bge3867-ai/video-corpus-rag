# 🎬 Video Corpus RAG — 视频库级语义检索问答

用自然语言向**整个视频库**提问, 系统跨全部视频检索语义最相关的片段, **切出真实视频切片**返回, 并基于切片内容作答。

完全基于开源组件构建 (替代 TwelveLabs + Chroma 的闭源方案):

- **视频切片** (6s) → **Qwen3-VL-Embedding-8B** 打向量 → **Chroma** 全库统一索引
- 入库时对每个片段自动做**文本侧三重增强**:
  - 📝 一句话中文**语义摘要**
  - 🎬 **结构化描述** (对象/动作/场景) —— 行为类提问「男人在走」「动物奔跑」直接命中
  - 🔤 **画面文字 OCR** —— 车牌/路牌/横幅可检索
  - 🔊 **语音转写 ASR** (faster-whisper) —— 「谁说了什么」可检索
- **提问** → 跨全库**混合检索** (视频向量 + 文本向量 RRF 融合) → 切出 top-k 片段 → **Qwen3-VL-8B-Instruct** 基于片段作答 (标注来源)
- 问答推理走 **vLLM 推理服务** (OpenAI 兼容接口, 独立进程, 秒级响应、支持并发), 可一键回退进程内 transformers

所有模型均 Apache-2.0 开源。Web 界面自带: 提问 / 上传视频自动入库 / 片段播放与下载 / 摘要、结构化描述、语音文本展示。

## 架构

```
                 ┌──────────────────────── 入库管道 (离线) ────────────────────────┐
   videos/ ─────►│  切片(6s) → 抽帧 → Qwen3-VL-Embedding-8B → Chroma 视频向量    │
                 │       └─► VLM 摘要+结构化描述(含OCR) → 文本向量  ─┐             │
                 │       └─► faster-whisper ASR → 文本向量        ───┴─► videosrag │
                 └─────────────────────────────────────────────────────────────────┘
                                            │
   提问 ──► Qwen3-VL-Embedding-8B 查询向量 ─┴─► 视频路 + 文本路 RRF 混合检索
                                            │
                    命中 top-k 片段 ◄────────┘
                          │
                          ▼
              Qwen3-VL-8B-Instruct (vLLM 服务, 8900)
                          │
        ┌─────────────────┴──────────────────┐
        ▼                                    ▼
   自然语言答案 (标注 [片段N])         切片 mp4 + 起止秒数 + 文本增强字段
```

GPU 分配 (按 config.yaml 调整): `cuda0`=embedding, `cuda1`=vLLM 问答, `cuda2`=ASR。

## 功能特性

- **语料库级检索**: 一次提问扫描全部视频的所有片段, 不是单视频内检索
- **混合检索**: 视频向量 + 摘要/结构化描述/语音 三类文本向量, RRF 融合, 各路权重可调
- **行为级检索**: 结构化描述把对象/动作/场景拆开入库, 「奔跑」「炒菜」这类动词查询精准命中
- **网页上传即入库**: 上传视频 → 自动切片/向量化/三重增强, 全程无需命令行
- **vLLM 加速**: 问答与摘要走独立 vLLM 推理服务, 单次问答约 3~5 秒, 支持 8 路并发
- **显存自适应**: 图片本地缩放控制 token 数, 按 top_k 自动调整每片段帧数, 8×3090 实测稳定

## 安装

### 1. 应用环境 (切片/向量化/检索服务)

```bash
bash setup.sh            # 创建 conda 环境 videosrag + 安装依赖 (含 CUDA 版 PyTorch)
```

要求: Linux + NVIDIA GPU (建议 ≥24GB 显存), CUDA ≥ 12.4, conda。

### 2. vLLM 推理服务环境 (可选, 但强烈推荐)

```bash
conda create -n vllm python=3.11 -y && conda activate vllm
# torch 与 vllm 版本需匹配本机 CUDA (下例为 CUDA 12.8; 其他版本见 vLLM 官方文档)
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-vllm.txt
```

> ⚠️ **vLLM 部署注意事项** (实测踩坑, 按需处理):
> 1. vllm 0.11.2 + 驱动 570 (CUDA 12.8) 组合可用; 若报 CUDA 版本不符, 请对齐 torch/vllm 的 cuXXX 版本
> 2. 若启动时报 "No available memory for cache blocks" / KV 缓存异常: 把
>    `models/Qwen3-VL-8B-Instruct/preprocessor_config.json` 的 `longest_edge`
>    从 `16777216` 改为 `245760`
> 3. 若视频输入 token 膨胀: 本项目问答路径已改为**本地抽帧送图片** (见 `src/enrich.py: frame_parts`),
>    不走 vLLM 视频模态, 无需额外处理
> 4. 若 `qwen_vl_utils` 报 `video_fps` 相关错误, 见 `src/common.py` 说明

### 3. 下载模型 (~35GB)

```bash
bash download_models.sh   # ModelScope 优先, 失败自动回退 hf-mirror, 断点续传
```

## 快速开始

```bash
# 1. 把你的视频放进 videos/ (mp4/avi/mov/mkv/webm/flv)
# 2. 建索引 (幂等, 新视频自动增量)
bash run_index.sh
# 3. 启动 vLLM 服务 (可选; 不启动则走进程内 transformers 兜底)
bash run_vllm.sh
# 4. 启动应用服务
bash run_server.sh
```

浏览器打开 **http://localhost:8899** 即可提问、上传视频。

不想用 vLLM 时, 把 `config.yaml` 里 `vllm.enabled` 改为 `false` 重启服务即可 (慢但零额外部署)。

## HTTP API

```bash
# 提问 (跨全库检索 + 基于命中片段作答)
curl -X POST http://localhost:8899/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "有没有森林着火的画面?", "top_k": 4}'

# 上传视频自动入库
curl -F "file=@myvideo.mp4" http://localhost:8899/upload   # 返回 job_id
curl http://localhost:8899/jobs/1                          # 查询入库进度

# 服务状态
curl http://localhost:8899/health
```

`/ask` 返回:

```json
{
  "question": "有没有森林着火的画面?",
  "answer": "片段1 展示了森林燃烧的场景 [片段1] ...",
  "clips": [
    {
      "id": "abc123", "video": "forestfire_3",
      "start": 0.0, "end": 6.0, "video_duration": 42.5,
      "clip_url": "/clips/forestfire_3/forestfire_3_000_0-6.mp4",
      "distance": 0.123, "rrf_score": 0.030,
      "summary": "森林大火燃烧, 浓烟滚滚",
      "objects": "火焰、树木", "actions": "燃烧、蔓延", "scene": "森林",
      "ocr": "iStock, by Getty Images", "asr": ""
    }
  ]
}
```

`clip_url` 即切好的视频片段, 可直接播放/下载 (物理文件在 `clips/`)。

## 配置 (config.yaml)

| 参数 | 默认 | 说明 |
|---|---|---|
| index.seg_len / overlap | 6.0 / 1.5 | 切片长度与重叠(秒) |
| index.frames_per_clip | 8 | 每片段 embedding 抽帧数 |
| server.default_top_k / max_top_k | 4 / 8 | 返回片段数范围 |
| server.vlm_max_pixels | 245760 | 送 VLM 每帧最大像素 (控 token/显存) |
| server.hybrid | true | 混合检索开关 |
| server.text_weights | caption 1.0 / summary 0.8 / asr 0.8 | 文本路各类型权重 |
| enrich.asr_model / asr_device | large-v3-turbo / cuda:2 | ASR 模型与卡位 |
| vllm.enabled / max_model_len | true / 8192 | vLLM 开关与上下文长度 (须与 run_vllm.sh 一致) |

## 检索原理

1. **切片**: 视频按 6s 窗口 (1.5s 重叠) 切成片段 mp4, 短视频 (≤6s) 整体成段
2. **视频向量**: 每片段抽 8 帧, Qwen3-VL-Embedding-8B 按官方 chat-template + lasttoken 池化 + L2 归一化 → 4096 维
3. **文本向量**: 每片段最多 3 条文本 (摘要/结构化描述/语音) 用同模型文本侧打向量, 独立 collection
4. **混合检索**: 视频路 + 文本路分别召回, 文本路按片段分组取最优 (乘类型权重), RRF 融合排序
5. **作答**: top-k 片段抽帧送 Qwen3-VL-8B-Instruct, 要求只依据片段内容作答并标注 [片段N]

## 常用运维

```bash
bash stop_server.sh / stop_vllm.sh      # 停服务
bash run_index.sh --force               # 重建全部索引
python tools/cleanup_video.py 文件名     # 删除视频+切片+向量
python tools/backfill_enrich.py          # 给已入库片段回填文本增强 (幂等, 支持 --asr-only)
tail -f logs/server.log logs/vllm.log   # 看日志
```

## 硬件实测

4 × RTX 3090 (24GB): 37 个视频 / 136 个 6s 片段, 单次问答 **3~5 秒**, 三路文本增强全部在线。

## 常见问题

- **服务启动慢**: 首次加载 embedding 模型约 1-2 分钟, `tail -f logs/server.log`
- **/ask 报 token 超限 (400)**: 调低 top_k, 或同步调大 `run_vllm.sh --max-model-len` 与 `config.yaml vllm.max_model_len`
- **vLLM 起不来**: 看 `logs/vllm.log`; 显存不足时调低 `--gpu-memory-utilization`
- **没有 GPU 2 号卡**: 把 `enrich.asr_device` 改到空闲卡, 或 `enrich.asr_enabled: false`
- **换更大模型**: 换成 `Qwen/Qwen3-VL-32B-Instruct` (~65GB, 需多卡 tensor parallel), 改 config 的 `vlm_model_id` 与 run_vllm.sh 参数

## 许可

本项目代码使用 [Apache License 2.0](LICENSE)。所依赖的模型 (Qwen3-VL 系列) 亦为 Apache-2.0。

## 致谢

- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) — 视觉语言模型 (Embedding / Instruct)
- [Chroma](https://github.com/chroma-core/chroma) — 向量数据库
- [vLLM](https://github.com/vllm-project/vllm) — 高性能推理引擎
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 语音转写
