FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-kor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

COPY mobility_service ./mobility_service
COPY .env.example ./.env.example

EXPOSE 8002

CMD ["sh", "-c", "exec uv run uvicorn mobility_service.app:app --host 0.0.0.0 --port ${PORT:-8002}"]
