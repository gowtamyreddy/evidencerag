# EvidenceRAG on Hugging Face Spaces (Docker SDK).
# The corpus is NOT baked into the image: at first start, start_space.sh
# downloads the papers from arXiv and builds the index, then serves the app.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt ./
# CPU-only torch first, so sentence-transformers reuses it (see requirements note).
RUN pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY app ./app
COPY src ./src
COPY scripts ./scripts
COPY data ./data

USER user
EXPOSE 7860
CMD ["bash", "scripts/start_space.sh"]
