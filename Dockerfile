FROM python:3.10-slim

LABEL maintainer="Yusei406"
LABEL description="Calvano et al. (2020) Q-learning Replication Environment"

WORKDIR /app

# システム依存関係
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python依存関係
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ソースコード
COPY . .
RUN pip install -e .

# デフォルトコマンド
CMD ["./run_all.sh"]

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s \
  CMD python -c "import src.environment; print('OK')" || exit 1
