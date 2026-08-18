#!/bin/bash
# 停止检索问答服务
cd "$(dirname "$0")"
if [ -f logs/server.pid ]; then
  kill "$(cat logs/server.pid)" 2>/dev/null && echo "服务已停止 (pid $(cat logs/server.pid))"
  rm -f logs/server.pid
else
  echo "没有找到运行中的服务"
fi
