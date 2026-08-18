#!/bin/bash
# ============================================================
# 一键初始化: conda 环境 videosrag + Python 依赖 (首次运行, 约 5-10 分钟)
# ============================================================
set -e
cd "$(dirname "$0")"

# 自动探测 conda 安装位置
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE=$(conda info --base)
elif [ -d "$HOME/miniconda3" ]; then
  CONDA_BASE="$HOME/miniconda3"
elif [ -d "$HOME/anaconda3" ]; then
  CONDA_BASE="$HOME/anaconda3"
else
  echo "未找到 conda, 请先安装 Miniconda: https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | grep -qw videosrag; then
  echo "== 创建 conda 环境 videosrag (python 3.11) =="
  conda create -y -n videosrag python=3.11
fi
conda activate videosrag

echo "== 安装 PyTorch (默认 CUDA 12.8; 老驱动失败时回退 CUDA 12.6) =="
pip install -U pip wheel
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 \
  || pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126

echo "== 安装其余依赖 =="
pip install -r requirements.txt

echo "== 验证 =="
python -c "import torch; print('torch', torch.__version__, '| cuda ok:', torch.cuda.is_available(), '| gpu num:', torch.cuda.device_count())"
echo "== 环境安装完成, 下一步: bash download_models.sh =="
