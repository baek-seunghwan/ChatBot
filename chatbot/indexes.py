"""Chroma, FAISS, BM25 인덱스와 하이브리드 검색기."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

from .embeddings import Embedder, normalize_vectors
from .rag_types import DocumentChunk, SearchHit

SUPPORTED_BACKENDS = ("chroma", "faiss", "bm25")


def parse_backends(value: str | Iterable[str]) -> tuple[str, ...]:
    """쉼표 문자열이나 이름 목록을 검증하고 중복 없이 반환한다."""
    names = value.split(",") if isinstance(value, str) else list(value)
    result = tuple(dict.fromkeys(name.strip().lower() for name in names if name.strip()))
    if not result:
        raise ValueError("검색 백엔드를 하나 이상 지정해야 합니다.")
    unknown = sorted(set(result) - set(SUPPORTED_BACKENDS))
    if unknown:
        raise ValueError(
            f"지원하지 않는 검색 백엔드: {', '.join(unknown)} "
            f"(지원: {', '.join(SUPPORTED_BACKENDS)})"
        )
    return result


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _collection_dir(root: Path, collection_name: str) -> Path:
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", collection_name).strip("._")
    if not safe_name:
        raise ValueError("collection_name에 유효한 문자가 없습니다.")
    return root / safe_name


class SearchIndex(Protocol):
    name: str

    def search(self, query: str, top_k: int) -> list[SearchHit]: ...


class ChromaIndex:
    name = "chroma"

    def __init__(self, path: Path, collection_name: str, embedder: Embedder) -> None:
        self.path = path
        self.collection_name = collection_name
        self.embedder = embedder
        self._collection = None

    def build(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("청크 수와 임베딩 수가 다릅니다.")
        import chromadb

        self.path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.path))
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass
        collection = client.create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        collection.add(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=normalize_vectors(embeddings).tolist(),
            metadatas=[
                {
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "metadata_json": json.dumps(chunk.metadata, ensure_ascii=False),
                }
                for chunk in chunks
            ],
        )
        self._collection = collection
        return collection.count()

    def _load(self):
        if self._collection is not None:
            return self._collection
        import chromadb

        client = chromadb.PersistentClient(path=str(self.path))
        try:
            self._collection = client.get_collection(self.collection_name)
        except Exception as exc:
            raise RuntimeError(
                f"Chroma 인덱스({self.collection_name})가 없습니다. "
                "먼저 인덱스 생성 명령을 실행하세요."
            ) from exc
        return self._collection

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        if top_k <= 0:
            return []
        collection = self._load()
        count = collection.count()
        if count == 0:
            return []
        vector = normalize_vectors(self.embedder.embed_query(query)).tolist()
        result = collection.query(
            query_embeddings=vector,
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[SearchHit] = []
        for chunk_id, text, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            extra = json.loads(metadata.get("metadata_json", "{}"))
            chunk = DocumentChunk(
                id=chunk_id,
                text=text,
                source=metadata["source"],
                chunk_index=int(metadata["chunk_index"]),
                start_char=int(metadata.get("start_char", 0)),
                end_char=int(metadata.get("end_char", len(text))),
                metadata=extra,
            )
            hits.append(SearchHit(chunk=chunk, score=score, backend_scores={self.name: score}))
        return hits


class FaissIndex:
    name = "faiss"

    def __init__(self, root: Path, collection_name: str, embedder: Embedder) -> None:
        self.directory = _collection_dir(root, collection_name)
        self.embedder = embedder
        self._index = None
        self._chunks: list[DocumentChunk] | None = None

    @property
    def index_path(self) -> Path:
        return self.directory / "index.faiss"

    @property
    def manifest_path(self) -> Path:
        return self.directory / "chunks.json"

    @staticmethod
    def _faiss():
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "FAISS를 사용할 수 없습니다. `uv sync`로 faiss-cpu를 설치하세요."
            ) from exc
        return faiss

    def build(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("청크 수와 임베딩 수가 다릅니다.")
        if not chunks:
            raise ValueError("FAISS에 저장할 청크가 없습니다.")
        faiss = self._faiss()
        vectors = normalize_vectors(embeddings)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.index_path.with_suffix(".faiss.tmp")
        faiss.write_index(index, str(temporary))
        temporary.replace(self.index_path)
        _write_json(
            self.manifest_path,
            {
                "version": 1,
                "dimension": int(vectors.shape[1]),
                "chunks": [chunk.to_dict() for chunk in chunks],
            },
        )
        self._index = index
        self._chunks = chunks
        return index.ntotal

    def _load(self):
        if self._index is not None and self._chunks is not None:
            return self._index, self._chunks
        if not self.index_path.exists() or not self.manifest_path.exists():
            raise RuntimeError(
                f"FAISS 인덱스({self.directory.name})가 없습니다. "
                "먼저 인덱스 생성 명령을 실행하세요."
            )
        faiss = self._faiss()
        index = faiss.read_index(str(self.index_path))
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        chunks = [DocumentChunk.from_dict(item) for item in payload["chunks"]]
        if index.ntotal != len(chunks):
            raise RuntimeError("FAISS 인덱스와 청크 메타데이터 개수가 일치하지 않습니다.")
        self._index, self._chunks = index, chunks
        return index, chunks

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        if top_k <= 0:
            return []
        index, chunks = self._load()
        vector = normalize_vectors(self.embedder.embed_query(query))
        scores, positions = index.search(vector, min(top_k, index.ntotal))
        hits: list[SearchHit] = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue
            normalized_score = max(0.0, min(1.0, float(score)))
            hits.append(
                SearchHit(
                    chunk=chunks[int(position)],
                    score=normalized_score,
                    backend_scores={self.name: normalized_score},
                )
            )
        return hits


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_+#.-]+|[가-힣]+")


def bm25_tokenize(text: str) -> list[str]:
    """영문 단어와 한글 어절/부분어를 함께 사용하는 가벼운 토크나이저."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for token in _TOKEN_RE.findall(normalized):
        tokens.append(token)
        # 조사 변화는 흡수하되 흔한 2글자 조각의 오탐은 줄이기 위해 3-gram을 쓴다.
        if re.fullmatch(r"[가-힣]+", token) and len(token) >= 3:
            tokens.extend(f"ko:{token[i:i + 3]}" for i in range(len(token) - 2))
    return tokens


class BM25Index:
    name = "bm25"

    def __init__(
        self,
        root: Path,
        collection_name: str,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.directory = _collection_dir(root, collection_name)
        self.path = self.directory / "index.json"
        self.k1 = k1
        self.b = b
        self._chunks: list[DocumentChunk] | None = None
        self._term_frequencies: list[Counter[str]] | None = None
        self._document_frequencies: Counter[str] | None = None
        self._lengths: list[int] | None = None

    def build(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            raise ValueError("BM25에 저장할 청크가 없습니다.")
        token_lists = [bm25_tokenize(chunk.text) for chunk in chunks]
        term_frequencies = [Counter(tokens) for tokens in token_lists]
        document_frequencies: Counter[str] = Counter()
        for frequencies in term_frequencies:
            document_frequencies.update(frequencies.keys())
        lengths = [len(tokens) for tokens in token_lists]
        _write_json(
            self.path,
            {
                "version": 1,
                "k1": self.k1,
                "b": self.b,
                "chunks": [chunk.to_dict() for chunk in chunks],
                "term_frequencies": [dict(value) for value in term_frequencies],
                "document_frequencies": dict(document_frequencies),
                "lengths": lengths,
            },
        )
        self._chunks = chunks
        self._term_frequencies = term_frequencies
        self._document_frequencies = document_frequencies
        self._lengths = lengths
        return len(chunks)

    def _load(self):
        if all(
            value is not None
            for value in (
                self._chunks,
                self._term_frequencies,
                self._document_frequencies,
                self._lengths,
            )
        ):
            return
        if not self.path.exists():
            raise RuntimeError(
                f"BM25 인덱스({self.directory.name})가 없습니다. "
                "먼저 인덱스 생성 명령을 실행하세요."
            )
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.k1 = float(payload.get("k1", self.k1))
        self.b = float(payload.get("b", self.b))
        self._chunks = [DocumentChunk.from_dict(item) for item in payload["chunks"]]
        self._term_frequencies = [Counter(item) for item in payload["term_frequencies"]]
        self._document_frequencies = Counter(payload["document_frequencies"])
        self._lengths = [int(value) for value in payload["lengths"]]

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        if top_k <= 0:
            return []
        self._load()
        assert self._chunks is not None
        assert self._term_frequencies is not None
        assert self._document_frequencies is not None
        assert self._lengths is not None
        query_terms = bm25_tokenize(query)
        if not query_terms or not self._chunks:
            return []
        document_count = len(self._chunks)
        average_length = sum(self._lengths) / document_count or 1.0
        scores = np.zeros(document_count, dtype=np.float64)
        for term in query_terms:
            df = self._document_frequencies.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
            for index, frequencies in enumerate(self._term_frequencies):
                tf = frequencies.get(term, 0)
                if tf == 0:
                    continue
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * self._lengths[index] / average_length
                )
                scores[index] += idf * (tf * (self.k1 + 1.0)) / denominator
        ranked = np.argsort(-scores)[: min(top_k, document_count)]
        if not len(ranked) or float(scores[ranked[0]]) <= 0:
            return []
        return [
            SearchHit(
                chunk=self._chunks[int(index)],
                score=float(scores[index] / (scores[index] + 1.0)),
                backend_scores={
                    self.name: float(scores[index] / (scores[index] + 1.0))
                },
            )
            for index in ranked
            if scores[index] > 0
        ]


class HybridRetriever:
    """여러 검색기의 순위와 점수를 결합해 중복 없는 결과를 반환한다."""

    def __init__(
        self,
        indexes: Iterable[SearchIndex],
        *,
        weights: dict[str, float] | None = None,
        candidate_multiplier: int = 3,
        rrf_k: int = 60,
    ) -> None:
        self.indexes = list(indexes)
        if not self.indexes:
            raise ValueError("검색 인덱스를 하나 이상 전달해야 합니다.")
        self.weights = weights or {index.name: 1.0 for index in self.indexes}
        if any(self.weights.get(index.name, 0.0) <= 0 for index in self.indexes):
            raise ValueError("각 검색 인덱스 가중치는 0보다 커야 합니다.")
        self.candidate_multiplier = max(1, candidate_multiplier)
        self.rrf_k = max(1, rrf_k)

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("검색 질의는 비어 있을 수 없습니다.")
        if top_k <= 0:
            return []
        fetch_k = top_k * self.candidate_multiplier
        results = {index.name: index.search(query, fetch_k) for index in self.indexes}
        if len(self.indexes) == 1:
            return next(iter(results.values()))[:top_k]

        chunks: dict[str, DocumentChunk] = {}
        backend_scores: dict[str, dict[str, float]] = {}
        reciprocal_scores: Counter[str] = Counter()
        denominator = sum(
            self.weights[index.name] / (self.rrf_k + 1) for index in self.indexes
        )
        total_weight = sum(self.weights[index.name] for index in self.indexes)
        for backend, hits in results.items():
            weight = self.weights[backend]
            for rank, hit in enumerate(hits, start=1):
                chunk_id = hit.chunk.id
                chunks[chunk_id] = hit.chunk
                backend_scores.setdefault(chunk_id, {})[backend] = hit.score
                reciprocal_scores[chunk_id] += weight / (self.rrf_k + rank)

        fused: list[SearchHit] = []
        for chunk_id, reciprocal_score in reciprocal_scores.items():
            scores = backend_scores[chunk_id]
            rank_component = reciprocal_score / denominator
            value_component = sum(
                self.weights[index.name] * scores.get(index.name, 0.0)
                for index in self.indexes
            ) / total_weight
            fused.append(
                SearchHit(
                    chunk=chunks[chunk_id],
                    score=max(
                        0.0,
                        min(1.0, value_component * (0.9 + 0.1 * rank_component)),
                    ),
                    backend_scores=scores,
                )
            )
        fused.sort(key=lambda hit: (-hit.score, hit.chunk.source, hit.chunk.chunk_index))
        return fused[:top_k]
