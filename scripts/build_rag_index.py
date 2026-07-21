# 📝 벡터DB 인덱스 (재)생성 CLI
#
# 실행:
#   uv run python -m scripts.build_rag_index                          # 기본 설정
#   uv run python -m scripts.build_rag_index --chunk-size 700 --overlap 100
#   uv run python -m scripts.build_rag_index --chunk-size 400 --collection rag_docs_cs400_ov50
from __future__ import annotations

import argparse
from pathlib import Path

from chatbot.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    RAG_DOCS_DIR,
    RETRIEVAL_BACKENDS,
)
from chatbot.ingest import build


def main() -> None:
    parser = argparse.ArgumentParser(
        description="문서 → 청크 → 임베딩 → Chroma/FAISS/BM25 인덱스 저장"
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument(
        "--backends",
        default=",".join(RETRIEVAL_BACKENDS),
        help="쉼표로 구분: chroma,faiss,bm25 (예: faiss,bm25)",
    )
    parser.add_argument("--docs-dir", type=Path, default=RAG_DOCS_DIR)
    args = parser.parse_args()
    build(
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        collection_name=args.collection,
        backends=args.backends,
        docs_dir=args.docs_dir,
    )


if __name__ == "__main__":
    main()
