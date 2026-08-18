#!/bin/bash
# 建索引: 切片 -> 抽帧 -> embedding -> 三重文本增强 -> 写入 Chroma
# 幂等: 已入库的视频(按 md5 判断)自动跳过; --force 全部重建
cd "$(dirname "$0")"
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate videosrag
fi
python src/index.py "$@"
echo "== 索引完成, 入库片段数见上方日志 =="
