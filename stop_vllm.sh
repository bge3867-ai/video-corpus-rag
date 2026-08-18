#!/bin/bash
# 停止 vLLM 推理服务
cd "$(dirname "$0")"
if [ -f logs/vllm.pid ]; then
  kill "$(cat logs/vllm.pid)" 2>/dev/null
  rm -f logs/vllm.pid
  echo "vLLM 已停止 (pid 文件方式)"
else
  echo "未找到 pid 文件"
fi
# 兜底: 按进程名清掉残留
pkill -f "[v]llm serve" 2>/dev/null
sleep 2
pgrep -f "[v]llm serve" >/dev/null && echo "仍有残留进程" || echo "vLLM 进程已清空"
