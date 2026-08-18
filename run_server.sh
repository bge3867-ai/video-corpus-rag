#!/bin/bash
# 启动检索问答服务 (后台运行, 日志 logs/server.log)
cd "$(dirname "$0")"
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate videosrag
fi
mkdir -p logs
if [ -f logs/server.pid ] && kill -0 "$(cat logs/server.pid)" 2>/dev/null; then
  echo "服务已在运行 (pid $(cat logs/server.pid)), 先跑 stop_server.sh"
  exit 1
fi
nohup python src/server.py > logs/server.log 2>&1 &
echo $! > logs/server.pid
echo "服务启动中 pid=$(cat logs/server.pid)"
echo "模型加载约 1-2 分钟, 用以下命令查看状态:"
echo "  curl http://localhost:8899/health"
echo "  tail -f logs/server.log"
