"""배치 임베딩 생성과 벡터 정규화."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import (
    EMBEDDING_CACHE_DIR,
    EMBEDDING_LOCAL_FILES_ONLY,
    EMBEDDING_MODEL,
)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """내적 검색이 코사인 유사도가 되도록 행 벡터를 L2 정규화한다."""
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(values / norms, dtype=np.float32)


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Sentence Transformers 모델을 지연 로드하는 임베딩 파이프라인."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        *,
        batch_size: int = 32,
        local_files_only: bool = EMBEDDING_LOCAL_FILES_ONLY,
        cache_dir=EMBEDDING_CACHE_DIR,
        show_progress: bool = False,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.local_files_only = local_files_only
        self.cache_dir = cache_dir
        self.show_progress = show_progress
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        kwargs = {"local_files_only": self.local_files_only}
        if self.cache_dir is not None:
            kwargs["cache_folder"] = str(self.cache_dir)
        try:
            self._model = SentenceTransformer(self.model_name, **kwargs)
        except Exception as exc:
            if self.local_files_only:
                raise RuntimeError(
                    "임베딩 모델을 로컬 캐시에서 찾지 못했습니다. "
                    f"모델: {self.model_name}, 캐시: {self.cache_dir}. "
                    "처음 한 번은 RAG_EMBEDDINGS_LOCAL_ONLY=0으로 인덱스를 생성하세요."
                ) from exc
            raise
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self._load().encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=self.show_progress,
        )
        return normalize_vectors(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("검색 질의는 비어 있을 수 없습니다.")
        return self.embed_documents([text])[0]

