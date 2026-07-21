from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from chatbot.chunking import DocumentChunker
from chatbot.embeddings import normalize_vectors
from chatbot.indexes import BM25Index, ChromaIndex, FaissIndex, HybridRetriever
from chatbot.providers import LLMResult
from chatbot.rag_chain import NO_ANSWER_TEXT, RagChain
from chatbot.rag_types import DocumentChunk, SearchHit, SourceDocument


class KeywordEmbedder:
    """외부 모델 없이 인덱스 파이프라인을 검증하는 결정적 임베더."""

    terms = ("rag", "fastapi", "로그인")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = [float(lowered.count(term)) for term in self.terms]
            if not any(vector):
                vector = [0.0, 0.0, 1.0]
            vectors.append(vector)
        return normalize_vectors(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def sample_chunks() -> list[DocumentChunk]:
    values = [
        ("rag.txt", "RAG는 검색 문서를 프롬프트에 넣어 답변을 생성한다."),
        ("api.txt", "FastAPI는 Python API 서버를 만드는 웹 프레임워크다."),
        ("auth.txt", "로그인은 사용자 인증을 거쳐 세션을 만든다."),
    ]
    return [
        DocumentChunk(
            id=f"chunk-{index}",
            text=text,
            source=source,
            chunk_index=0,
            start_char=0,
            end_char=len(text),
        )
        for index, (source, text) in enumerate(values)
    ]


class ChunkingTests(unittest.TestCase):
    def test_preserves_overlap_and_natural_boundaries(self) -> None:
        document = SourceDocument(
            source="guide.md",
            text=(
                "첫 번째 문단은 RAG의 검색 단계를 설명합니다. "
                "검색 결과는 다음 단계로 전달됩니다.\n\n"
                "두 번째 문단은 프롬프트에 컨텍스트를 붙이는 단계를 설명합니다."
            ),
        )
        chunks = DocumentChunker(chunk_size=55, chunk_overlap=12).split(document)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(0 < len(chunk.text) <= 55 for chunk in chunks))
        self.assertLess(chunks[1].start_char, chunks[0].end_char)
        self.assertEqual([chunk.chunk_index for chunk in chunks], list(range(len(chunks))))
        self.assertEqual(len({chunk.id for chunk in chunks}), len(chunks))

    def test_rejects_invalid_overlap(self) -> None:
        with self.assertRaises(ValueError):
            DocumentChunker(chunk_size=20, chunk_overlap=20)


class PersistentIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.embedder = KeywordEmbedder()
        self.chunks = sample_chunks()
        self.embeddings = self.embedder.embed_documents(
            [chunk.text for chunk in self.chunks]
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_faiss_build_reload_and_search(self) -> None:
        index = FaissIndex(self.root / "faiss", "test", self.embedder)
        self.assertEqual(index.build(self.chunks, self.embeddings), 3)

        reloaded = FaissIndex(self.root / "faiss", "test", self.embedder)
        hits = reloaded.search("FastAPI 서버", top_k=2)
        self.assertEqual(hits[0].chunk.source, "api.txt")
        self.assertGreater(hits[0].score, 0.9)

    def test_chroma_build_reload_and_search(self) -> None:
        index = ChromaIndex(self.root / "chroma", "test_collection", self.embedder)
        self.assertEqual(index.build(self.chunks, self.embeddings), 3)

        reloaded = ChromaIndex(
            self.root / "chroma", "test_collection", self.embedder
        )
        hits = reloaded.search("RAG 검색", top_k=2)
        self.assertEqual(hits[0].chunk.source, "rag.txt")
        self.assertGreater(hits[0].score, 0.9)

    def test_bm25_build_reload_and_korean_search(self) -> None:
        index = BM25Index(self.root / "bm25", "test")
        self.assertEqual(index.build(self.chunks), 3)

        reloaded = BM25Index(self.root / "bm25", "test")
        hits = reloaded.search("사용자 로그인 인증", top_k=2)
        self.assertEqual(hits[0].chunk.source, "auth.txt")
        self.assertGreater(hits[0].score, 0.5)


class StaticIndex:
    def __init__(self, name: str, hits: list[SearchHit]) -> None:
        self.name = name
        self.hits = hits

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        return self.hits[:top_k]


class HybridAndGenerationTests(unittest.TestCase):
    def test_hybrid_retriever_deduplicates_and_fuses(self) -> None:
        chunks = sample_chunks()
        vector_hits = [
            SearchHit(chunks[0], 0.92, {"faiss": 0.92}),
            SearchHit(chunks[1], 0.70, {"faiss": 0.70}),
        ]
        lexical_hits = [
            SearchHit(chunks[0], 1.0, {"bm25": 1.0}),
            SearchHit(chunks[2], 0.60, {"bm25": 0.60}),
        ]
        retriever = HybridRetriever(
            [StaticIndex("faiss", vector_hits), StaticIndex("bm25", lexical_hits)]
        )

        hits = retriever.search("RAG", top_k=3)
        self.assertEqual(hits[0].chunk.id, chunks[0].id)
        self.assertEqual(set(hits[0].backend_scores), {"faiss", "bm25"})
        self.assertEqual(len({hit.chunk.id for hit in hits}), len(hits))

    def test_retrieved_context_is_attached_to_generation_prompt(self) -> None:
        chunks = sample_chunks()

        class FakeRetriever:
            def search(self, query: str, top_k: int) -> list[SearchHit]:
                return [SearchHit(chunks[0], 0.95, {"bm25": 0.95})]

        class FakeRouter:
            def __init__(self) -> None:
                self.prompt = ""

            def generate(self, prompt: str, **kwargs) -> LLMResult:
                self.prompt = prompt
                return LLMResult("검색 문서를 사용한 답변", "fake", "fake-model")

        router = FakeRouter()
        chain = RagChain(
            retrieval_backends="bm25",
            retriever=FakeRetriever(),
            router=router,
            min_score=0.3,
        )
        answer = chain.ask("RAG는 어떻게 답변해?")

        self.assertIn("<retrieved_context>", router.prompt)
        self.assertIn("[rag.txt#chunk-0]", router.prompt)
        self.assertIn(chunks[0].text, router.prompt)
        self.assertEqual(answer.sources, ["rag.txt"])
        self.assertEqual(answer.provider, "fake")

    def test_no_relevant_context_skips_llm(self) -> None:
        class EmptyRetriever:
            def search(self, query: str, top_k: int) -> list[SearchHit]:
                return []

        class FailingRouter:
            def generate(self, *args, **kwargs):
                raise AssertionError("근거가 없을 때 LLM을 호출하면 안 됩니다.")

        chain = RagChain(
            retrieval_backends="bm25",
            retriever=EmptyRetriever(),
            router=FailingRouter(),
        )
        self.assertEqual(chain.ask("없는 질문").answer, NO_ANSWER_TEXT)


if __name__ == "__main__":
    unittest.main()
