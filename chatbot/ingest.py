# 📝 ingest: 문서 읽기 → chunk 분할 → embedding → ChromaDB 저장
# 📝 실행: uv run python -m chatbot.ingest
from __future__ import annotations

from pathlib import Path

from .config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_CACHE_DIR,
    EMBEDDING_LOCAL_FILES_ONLY,
    EMBEDDING_MODEL,
    RAG_DOCS_DIR,
)


def read_document(path: Path) -> str:
    """txt / md / pdf 파일에서 텍스트를 읽는다."""
    if path.suffix.lower() in (".txt", ".md"):
        return path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".pdf":
        # 📝 PDF는 pypdf가 설치된 경우에만 지원 (uv add pypdf)
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"지원하지 않는 형식: {path.name}")


def load_documents(docs_dir: Path) -> list[tuple[str, str]]:
    """문서 폴더에서 (파일이름, 내용) 목록을 읽는다."""
    documents = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() in (".txt", ".md", ".pdf"):
            documents.append((path.name, read_document(path)))
    if not documents:
        raise FileNotFoundError(f"{docs_dir}에 txt/md/pdf 문서가 없습니다.")
    return documents


def split_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """문서를 청크(작은 덩어리)로 나눈다.

    1) 빈 줄 기준으로 문단을 먼저 나누고
    2) 문단이 chunk_size보다 길면 chunk_overlap만큼 겹치게 잘라낸다.
    문서를 통째로 임베딩하면 여러 주제가 섞여 검색이 잘 안 되기 때문이다.
    """
    chunks: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)
            continue
        # 📝 긴 문단은 chunk_size 간격으로, 앞부분을 chunk_overlap만큼 겹쳐서 자른다.
        start = 0
        while start < len(paragraph):
            piece = paragraph[start : start + chunk_size].strip()
            if piece:
                chunks.append(piece)
            start += chunk_size - chunk_overlap
    return chunks


def build(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    collection_name: str | None = None,
    quiet: bool = False,
) -> int:
    """문서 전체를 임베딩해서 ChromaDB에 저장하고, 저장한 청크 수를 돌려준다.

    chunk_size / chunk_overlap / collection_name을 넘기면 기본 설정 대신 그 값을 쓴다.
    (실험 스크립트가 여러 설정의 인덱스를 나란히 만들 때 사용)
    """
    # 📝 무거운 라이브러리는 함수 안에서 import (앱 시작 속도를 위해)
    import chromadb
    from sentence_transformers import SentenceTransformer

    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP
    collection_name = collection_name or COLLECTION_NAME
    log = (lambda *a: None) if quiet else print

    log(f"[1/4] 문서 로딩: {RAG_DOCS_DIR}")
    documents = load_documents(RAG_DOCS_DIR)

    log(f"[2/4] 청크 분리 (CHUNK_SIZE={chunk_size}, CHUNK_OVERLAP={chunk_overlap})")
    ids, texts, metadatas = [], [], []
    for filename, text in documents:
        for i, chunk in enumerate(split_into_chunks(text, chunk_size, chunk_overlap)):
            ids.append(f"{filename}-{i}")
            texts.append(chunk)
            metadatas.append({"source": filename, "chunk_index": i})
    log(f"      문서 {len(documents)}개 → 청크 {len(texts)}개")

    log(f"[3/4] 임베딩 생성: {EMBEDDING_MODEL}")
    model_kwargs = {"local_files_only": EMBEDDING_LOCAL_FILES_ONLY}
    if EMBEDDING_CACHE_DIR is not None:
        model_kwargs["cache_folder"] = str(EMBEDDING_CACHE_DIR)
    try:
        model = SentenceTransformer(EMBEDDING_MODEL, **model_kwargs)
    except Exception as exc:
        if EMBEDDING_LOCAL_FILES_ONLY:
            raise RuntimeError(
                "임베딩 모델을 로컬 캐시에서 찾지 못했습니다. "
                f"모델: {EMBEDDING_MODEL}, 캐시: {EMBEDDING_CACHE_DIR}. "
                "처음 1회는 네트워크가 되는 환경에서 "
                "`RAG_EMBEDDINGS_LOCAL_ONLY=0 uv run python -m chatbot.ingest`를 "
                "실행해 캐시를 만든 뒤 다시 실행하세요."
            ) from exc
        raise
    embeddings = model.encode(texts, show_progress_bar=not quiet).tolist()

    log(f"[4/4] ChromaDB 저장: {CHROMA_DIR} (컬렉션: {collection_name})")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # 📝 다시 실행해도 깨끗하게 새로 만들도록 기존 컬렉션은 지운다.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        collection_name, metadata={"hnsw:space": "cosine"}
    )
    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    log(f"완료: 청크 {collection.count()}개 저장됨")
    return collection.count()


if __name__ == "__main__":
    build()
