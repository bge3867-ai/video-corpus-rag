#!/bin/bash
# 启动 vLLM 推理服务 (OpenAI 兼容接口, 端口 8900, 独占指定 GPU 卡)
# 需要独立的 conda 环境 vllm, 见 README "安装" 一节
cd "$(dirname "$0")"
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate vllm
fi
mkdir -p logs
if [ -f logs/vllm.pid ] && kill -0 "$(cat logs/vllm.pid)" 2>/dev/null; then
  echo "vLLM 已在运行 (pid $(cat logs/vllm.pid)), 先跑 stop_vllm.sh"
  exit 1
fi
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup vllm serve \
  models/Qwen3-VL-8B-Instruct \
  --served-model-name Qwen3-VL-8B-Instruct \
  --host 0.0.0.0 --port 8900 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 1024 \
  --limit-mm-per-prompt '{"video": 10, "image": 40}' \
  --mm-processor-kwargs '{"max_pixels": 245760}' \
  > logs/vllm.log 2>&1 &
echo $! > logs/vllm.pid
echo "vLLM 启动中 pid=$(cat logs/vllm.pid), 约 1-2 分钟"
echo "查看状态: tail -f logs/vllm.log  或  curl http://localhost:8900/v1/models"
