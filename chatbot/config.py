# 📝 RAG 설정 파일: API KEY 로드, chunk_size, top_k 같은 설정값 관리
# 📝 ingest.py / rag_chain.py / main.py / eval.py가 전부 이 값을 공유한다.
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 📝 .env 파일 로드 (없어도 동작)
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

# 로컬 과제/데모 실행에서는 LangSmith 추적을 기본으로 끈다.
# 공식 현재 환경 변수인 LANGSMITH_TRACING만 사용한다.
os.environ.setdefault("LANGSMITH_TRACING", "false")

LANGSMITH_TRACING_ENABLED = os.getenv("LANGSMITH_TRACING", "false").lower() in (
    "1",
    "true",
    "yes",
)
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "leon-rag-chatbot")


def langsmith_status() -> dict[str, object]:
    """API 키 값은 노출하지 않고 LangSmith 설정 상태만 반환한다."""
    return {
        "enabled": LANGSMITH_TRACING_ENABLED,
        "project": LANGSMITH_PROJECT,
        "api_key_configured": bool(os.getenv("LANGSMITH_API_KEY")),
        "workspace_configured": bool(os.getenv("LANGSMITH_WORKSPACE_ID")),
        "endpoint": os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
    }

# 📝 RAG에 넣을 문서 폴더 (txt / md / pdf 지원)
RAG_DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", REPO_ROOT / "chatbot" / "rag_docs"))

# 📝 벡터DB(ChromaDB) 저장 위치
CHROMA_DIR = Path(os.getenv("RAG_CHROMA_DIR", REPO_ROOT / "artifacts" / "chroma_db"))
FAISS_DIR = Path(os.getenv("RAG_FAISS_DIR", REPO_ROOT / "artifacts" / "faiss"))
BM25_DIR = Path(os.getenv("RAG_BM25_DIR", REPO_ROOT / "artifacts" / "bm25"))
COLLECTION_NAME = "rag_docs"

# 📝 청크 설정
#    CHUNK_SIZE    = 한 조각 크기(글자 수)
#    CHUNK_OVERLAP = 앞뒤 문맥 겹치는 크기
#    TOP_K         = 검색 결과 몇 개 가져올지
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "300"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("RAG_TOP_K", "3"))

# 검색 백엔드: 하나만 쓰거나 쉼표로 여러 개를 지정해 하이브리드 검색한다.
RETRIEVAL_BACKENDS = tuple(
    item.strip().lower()
    for item in os.getenv("RAG_RETRIEVAL_BACKENDS", "chroma,faiss,bm25").split(",")
    if item.strip()
)
HYBRID_CANDIDATE_MULTIPLIER = int(
    os.getenv("RAG_HYBRID_CANDIDATE_MULTIPLIER", "3")
)
HYBRID_RRF_K = int(os.getenv("RAG_HYBRID_RRF_K", "60"))
RETRIEVAL_WEIGHTS = {
    "chroma": float(os.getenv("RAG_CHROMA_WEIGHT", "0.25")),
    "faiss": float(os.getenv("RAG_FAISS_WEIGHT", "0.25")),
    "bm25": float(os.getenv("RAG_BM25_WEIGHT", "0.50")),
}

# 📝 이 통합 관련도 점수보다 낮은 청크는 근거로 쓰지 않는다.
#    관련 문서가 하나도 없으면 "문서에서 확인할 수 없습니다"로 답한다.
MIN_RELEVANCE_SCORE = float(os.getenv("RAG_MIN_RELEVANCE_SCORE", "0.35"))

# 📝 임베딩 모델: 한국어를 포함한 다국어 지원, 크기가 작아 로컬 실행에 적합
EMBEDDING_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
_embedding_cache_dir = os.getenv("RAG_EMBEDDING_CACHE_DIR")
EMBEDDING_CACHE_DIR = Path(_embedding_cache_dir) if _embedding_cache_dir else None
if EMBEDDING_CACHE_DIR is not None and not EMBEDDING_CACHE_DIR.is_absolute():
    EMBEDDING_CACHE_DIR = REPO_ROOT / EMBEDDING_CACHE_DIR
EMBEDDING_LOCAL_FILES_ONLY = os.getenv("RAG_EMBEDDINGS_LOCAL_ONLY", "1").lower() in (
    "1",
    "true",
    "yes",
)
if EMBEDDING_LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 📝 답변 프롬프트 버전 (rag_chain.PROMPT_VERSIONS 참고. v2가 근거 규칙이 더 강함)
PROMPT_VERSION = os.getenv("RAG_PROMPT_VERSION", "v2")

# 📝 검색 후 재정렬(reranker) 사용 여부 ("1"이면 켜짐)
USE_RERANKER = os.getenv("RAG_USE_RERANKER", "0") == "1"
