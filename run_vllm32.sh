#!/bin/bash
# 启动 vLLM 推理服务: Qwen3-VL-32B 4bit 量化版 (端口 8901, GPU 卡 1)
# 使用前先跑 stop_vllm32.sh 停掉旧实例; 8B 版用 run_vllm.sh (8900)
cd "$(dirname "$0")"
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate vllm
fi
mkdir -p logs
if [ -f logs/vllm32.pid ] && kill -0 "$(cat logs/vllm32.pid)" 2>/dev/null; then
  echo "vLLM-32B 已在运行 (pid $(cat logs/vllm32.pid)), 先跑 stop_vllm32.sh"
  exit 1
fi
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup vllm serve \
  models/Qwen3-VL-32B-bnb4 \
  --served-model-name Qwen3-VL-32B-Instruct \
  --host 0.0.0.0 --port 8901 \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --gpu-memory-utilization 0.95 \
  --max-model-len 6144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 1024 \
  --limit-mm-per-prompt '{"video": 10, "image": 40}' \
  --mm-processor-kwargs '{"max_pixels": 245760}' \
  > logs/vllm32.log 2>&1 &
echo $! > logs/vllm32.pid
echo "vLLM-32B 启动中 pid=$(cat logs/vllm32.pid), 约 2-4 分钟"
echo "查看状态: tail -f logs/vllm32.log  或  curl http://localhost:8901/v1/models"
