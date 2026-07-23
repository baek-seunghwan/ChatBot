FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

COPY mobility_service ./mobility_service
COPY pool-feature-diagram.html ./pool-feature-diagram.html
COPY .env.example ./.env.example

EXPOSE 8002

CMD ["sh", "-c", "exec uv run uvicorn mobility_service.app:app --host 0.0.0.0 --port ${PORT:-8002}"]
