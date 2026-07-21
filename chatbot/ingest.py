"""문서 로딩 → 청킹 → 임베딩 → Chroma/FAISS/BM25 인덱스 생성."""

from __future__ import annotations

from pathlib import Path

from .chunking import DocumentChunker
from .config import (
    BM25_DIR,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    FAISS_DIR,
    RAG_DOCS_DIR,
    RETRIEVAL_BACKENDS,
)
from .embeddings import Embedder, SentenceTransformerEmbedder
from .indexes import BM25Index, ChromaIndex, FaissIndex, parse_backends
from .rag_types import SourceDocument

SUPPORTED_SUFFIXES = (".txt", ".md", ".pdf")


def read_document(path: Path) -> str:
    """UTF-8 텍스트/Markdown 또는 PDF에서 텍스트를 읽는다."""
    if path.suffix.lower() in (".txt", ".md"):
        return path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"지원하지 않는 형식: {path.name}")


def load_documents(docs_dir: Path) -> list[tuple[str, str]]:
    """기존 API 호환용: 문서 폴더를 (파일명, 본문) 목록으로 읽는다."""
    if not docs_dir.exists():
        raise FileNotFoundError(f"문서 폴더가 없습니다: {docs_dir}")
    documents = [
        (path.name, read_document(path))
        for path in sorted(docs_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not documents:
        raise FileNotFoundError(f"{docs_dir}에 txt/md/pdf 문서가 없습니다.")
    return documents


def split_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """기존 API 호환용: 경계 인식 청커로 본문 문자열만 반환한다."""
    chunker = DocumentChunker(chunk_size, chunk_overlap)
    return [
        chunk.text
        for chunk in chunker.split(SourceDocument(source="document", text=text))
    ]


def build(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    collection_name: str | None = None,
    quiet: bool = False,
    *,
    backends: str | tuple[str, ...] | list[str] | None = None,
    docs_dir: Path | None = None,
    embedder: Embedder | None = None,
) -> int:
    """선택한 모든 검색 인덱스를 같은 청크/임베딩으로 생성한다.

    반환값은 기존 호출부와 호환되도록 저장한 청크 수이다.
    """
    chunk_size = chunk_size if chunk_size is not None else CHUNK_SIZE
    chunk_overlap = (
        chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP
    )
    collection_name = collection_name or COLLECTION_NAME
    docs_dir = docs_dir or RAG_DOCS_DIR
    backend_names = parse_backends(backends or RETRIEVAL_BACKENDS)
    log = (lambda *args: None) if quiet else print

    log(f"[1/4] 문서 로딩: {docs_dir}")
    source_documents = [
        SourceDocument(source=filename, text=text)
        for filename, text in load_documents(docs_dir)
    ]

    log(
        f"[2/4] 청크 분리 (CHUNK_SIZE={chunk_size}, "
        f"CHUNK_OVERLAP={chunk_overlap})"
    )
    chunks = DocumentChunker(chunk_size, chunk_overlap).split_documents(
        source_documents
    )
    if not chunks:
        raise ValueError("문서에서 인덱싱할 텍스트를 찾지 못했습니다.")
    log(f"      문서 {len(source_documents)}개 → 청크 {len(chunks)}개")

    vector_backends = set(backend_names) & {"chroma", "faiss"}
    embeddings = None
    active_embedder = embedder
    if vector_backends:
        active_embedder = active_embedder or SentenceTransformerEmbedder(
            show_progress=not quiet
        )
        log(f"[3/4] 임베딩 생성: 청크 {len(chunks)}개")
        embeddings = active_embedder.embed_documents([chunk.text for chunk in chunks])
    else:
        log("[3/4] 임베딩 생략: BM25만 선택됨")

    log(f"[4/4] 인덱스 저장: {', '.join(backend_names)}")
    counts: dict[str, int] = {}
    for backend in backend_names:
        if backend == "chroma":
            assert active_embedder is not None and embeddings is not None
            counts[backend] = ChromaIndex(
                CHROMA_DIR, collection_name, active_embedder
            ).build(chunks, embeddings)
        elif backend == "faiss":
            assert active_embedder is not None and embeddings is not None
            counts[backend] = FaissIndex(
                FAISS_DIR, collection_name, active_embedder
            ).build(chunks, embeddings)
        elif backend == "bm25":
            counts[backend] = BM25Index(BM25_DIR, collection_name).build(chunks)
        log(f"      {backend}: {counts[backend]}개")

    if len(set(counts.values())) != 1:
        raise RuntimeError(f"인덱스별 저장 개수가 일치하지 않습니다: {counts}")
    log(f"완료: 청크 {len(chunks)}개 × 인덱스 {len(counts)}개")
    return len(chunks)


if __name__ == "__main__":
    build()
