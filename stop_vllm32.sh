#!/bin/bash
# 停止 vLLM-32B 推理服务 (端口 8901)
cd "$(dirname "$0")"
if [ -f logs/vllm32.pid ]; then
  kill "$(cat logs/vllm32.pid)" 2>/dev/null
  rm -f logs/vllm32.pid
  echo "vLLM-32B 已停止 (pid 文件方式)"
else
  echo "未找到 pid 文件"
fi
pkill -f "[8]901" 2>/dev/null
sleep 2
pgrep -f "[8]901" >/dev/null && echo "仍有残留进程" || echo "vLLM-32B 进程已清空"
