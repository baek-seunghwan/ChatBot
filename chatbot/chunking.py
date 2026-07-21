"""문서 구조와 문맥 겹침을 보존하는 청킹 로직."""

from __future__ import annotations

import hashlib
import re

from .rag_types import DocumentChunk, SourceDocument


class DocumentChunker:
    """문단/문장/단어 경계를 우선해 문서를 겹치는 청크로 나눈다."""

    _SEPARATORS = ("\n\n", "\n", ". ", "。", "! ", "? ", " ")

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size는 1 이상이어야 합니다.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap은 0 이상이어야 합니다.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap은 chunk_size보다 작아야 합니다.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _boundary(self, text: str, start: int, hard_end: int) -> int:
        """너무 작은 조각을 만들지 않으면서 가장 자연스러운 끝 경계를 찾는다."""
        if hard_end >= len(text):
            return len(text)
        min_end = start + max(1, self.chunk_size // 2)
        window = text[start:hard_end]
        for separator in self._SEPARATORS:
            pos = window.rfind(separator)
            if pos >= 0:
                end = start + pos + len(separator)
                if end >= min_end:
                    return end
        return hard_end

    @staticmethod
    def _chunk_id(source: str, index: int, text: str) -> str:
        digest = hashlib.sha256(
            f"{source}\0{index}\0{text}".encode("utf-8")
        ).hexdigest()[:24]
        return f"chunk-{digest}"

    def split(self, document: SourceDocument) -> list[DocumentChunk]:
        text = self._normalize(document.text)
        if not text:
            return []

        chunks: list[DocumentChunk] = []
        start = 0
        while start < len(text):
            hard_end = min(len(text), start + self.chunk_size)
            end = self._boundary(text, start, hard_end)

            raw = text[start:end]
            left_trim = len(raw) - len(raw.lstrip())
            right_trimmed = raw.rstrip()
            piece_start = start + left_trim
            piece_end = start + len(right_trimmed)
            piece = text[piece_start:piece_end]
            if piece:
                index = len(chunks)
                chunks.append(
                    DocumentChunk(
                        id=self._chunk_id(document.source, index, piece),
                        text=piece,
                        source=document.source,
                        chunk_index=index,
                        start_char=piece_start,
                        end_char=piece_end,
                        metadata=dict(document.metadata),
                    )
                )

            if end >= len(text):
                break
            next_start = max(0, end - self.chunk_overlap)
            if next_start <= start:
                next_start = end
            start = next_start
        return chunks

    def split_documents(self, documents: list[SourceDocument]) -> list[DocumentChunk]:
        return [chunk for document in documents for chunk in self.split(document)]

