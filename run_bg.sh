#!/bin/bash
# 通用后台启动器: usage: run_bg.sh <日志文件> <命令...>
# 用 setsid + stdin 重定向彻底脱离 SSH 会话, 避免连接被后台进程拖住
cd "$(dirname "$0")" || exit 1
LOG="$1"; shift
mkdir -p logs
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate videosrag
fi
if command -v setsid >/dev/null 2>&1; then
  nohup setsid "$@" > "$LOG" 2>&1 < /dev/null &
else
  nohup "$@" > "$LOG" 2>&1 < /dev/null &
fi
echo "已后台启动: $*  (log=$LOG, pid=$!)"
