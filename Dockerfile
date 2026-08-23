# 应用服务镜像: 切片/向量化/混合检索/三重文本增强/Web UI
# GPU 推理依赖 torch cu128 wheel (自带 CUDA 运行库, 无需系统 CUDA toolkit)
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_DISABLE_XET=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 \
       --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY tools ./tools
COPY config.yaml .

EXPOSE 8899
CMD ["python", "src/server.py"]
