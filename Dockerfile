# 📝 로그인형 Leon's ChatBot 배포용 Dockerfile
#
# 빌드:  docker build -t chatbot .
# 실행:  docker run -p 8001:8001 --env-file .env chatbot
# (또는 docker compose up --build)
FROM python:3.12-slim

# uv 설치 (공식 이미지에서 바이너리만 복사 — 가장 빠른 방법)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV LANGSMITH_TRACING=false \
    LANGCHAIN_TRACING_V2=false \
    RAG_EMBEDDINGS_LOCAL_ONLY=1 \
    RAG_EMBEDDING_CACHE_DIR=/app/artifacts/hf_cache

# 📝 의존성 레이어 분리: 코드만 바뀌면 의존성 설치를 다시 하지 않는다 (빌드 캐시 활용)
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project || uv sync --no-dev

# 앱 코드와 문서, 벡터DB 인덱스 복사
COPY chatbot ./chatbot
COPY scripts ./scripts
COPY artifacts/chroma_db ./artifacts/chroma_db

# 📝 임베딩 모델을 이미지 빌드 시점에 미리 다운로드 (컨테이너 첫 요청 지연 방지)
RUN uv run python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_folder='/app/artifacts/hf_cache')"

EXPOSE 8001

# API 키는 실행 시 환경변수로 주입 (--env-file .env)
CMD ["uv", "run", "uvicorn", "chatbot.local_chat.app:app", "--host", "0.0.0.0", "--port", "8001"]
