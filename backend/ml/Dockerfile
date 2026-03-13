FROM python:3.11-slim

WORKDIR /app

# LightGBM runtime dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/ml/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
