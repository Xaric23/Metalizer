FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_VIRTUALENVS_CREATE=false \
    PORT=8000 \
    METALIZER_MEDIA_ROOT=/data \
    METALIZER_INPUT_DIR=/data/inputs \
    METALIZER_OUTPUT_DIR=/data/outputs \
    METALIZER_ASSETS_DIR=/app/assets \
    METALIZER_JOBS_DB_URL=sqlite:////data/jobs.sqlite3

WORKDIR /app

# System dependencies for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg build-essential libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY app ./app
COPY assets ./assets
COPY README.md ./README.md

# Ensure runtime directories exist (and can be mounted as volumes)
RUN mkdir -p /data/inputs /data/outputs && \
    groupadd -r metalizer && useradd -r -g metalizer metalizer && \
    chown -R metalizer:metalizer /app /data

USER metalizer

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
