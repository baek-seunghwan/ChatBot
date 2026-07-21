"""RAG 파이프라인 전 단계가 공유하는 데이터 모델."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceDocument:
    """청킹 전 원본 문서."""

    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunk:
    """검색과 인덱싱의 최소 단위."""

    id: str
    text: str
    source: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DocumentChunk":
        return cls(
            id=str(value["id"]),
            text=str(value["text"]),
            source=str(value["source"]),
            chunk_index=int(value["chunk_index"]),
            start_char=int(value.get("start_char", 0)),
            end_char=int(value.get("end_char", len(str(value["text"])))),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class SearchHit:
    """하나 이상의 검색 인덱스에서 검색된 청크."""

    chunk: DocumentChunk
    score: float
    backend_scores: dict[str, float] = field(default_factory=dict)

