# 📝 rag_chain: retriever 생성 → prompt 구성 → LLM 호출 → 답변 생성
# 📝 터미널 테스트: uv run python -m chatbot.rag_chain "RAG가 뭐야?"
from __future__ import annotations

import sys
from dataclasses import dataclass, field

from .config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MIN_RELEVANCE_SCORE,
    TOP_K,
)
from .providers import LLMRouter

# 📝 18번 규칙: LLM이 문서 근거 없이 답하지 못하게 막는 시스템 프롬프트
RAG_SYSTEM_PROMPT = """너는 문서 기반 RAG 챗봇이다.

반드시 지켜야 할 규칙:
1. 제공된 문서 내용만 사용해서 답변한다.
2. 문서에 없는 내용은 추측하지 않는다.
3. 문서에 없으면 "문서에서 확인할 수 없습니다"라고 답한다.
4. 답변 마지막에 참고한 문서명을 표시한다.
5. 답변은 한국어로 한다.
6. 사용자가 초보자라면 쉬운 말로 설명한다.
7. 코드나 개념 설명이 필요하면 단계별로 설명한다.
8. 문서 근거와 답변 내용을 분리해서 보여준다."""

RAG_PROMPT_TEMPLATE = """아래 문서를 참고해서 질문에 답하세요.

문서:
{context}

질문:
{question}"""

# 📝 문서에 답이 없을 때의 고정 응답
NO_ANSWER_TEXT = "문서에서 확인할 수 없습니다."
NO_ANSWER_MESSAGE = "관련 문서를 찾지 못했습니다."
GROUNDED_MESSAGE = "문서 근거 기반으로 답변했습니다."


@dataclass
class RetrievedChunk:
    """검색된 청크 하나: 내용 + 출처 파일 + 유사도 점수(1에 가까울수록 관련 높음)"""

    text: str
    source: str
    score: float


@dataclass
class RagAnswer:
    """RAG 답변 결과 (19번 응답 형식과 1:1 대응)"""

    question: str
    answer: str
    sources: list[str] = field(default_factory=list)   # 참고한 문서명 목록
    confidence: float = 0.0                            # 가장 관련 높은 청크의 유사도
    retrieved_chunks: int = 0                          # 근거로 사용한 청크 수
    message: str = NO_ANSWER_MESSAGE
    chunks: list[RetrievedChunk] = field(default_factory=list)
    provider: str = ""
    model: str = ""

    def to_dict(self) -> dict:
        """API 응답용 JSON 형식으로 변환"""
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "confidence": round(self.confidence, 2),
            "retrieved_chunks": self.retrieved_chunks,
            "message": self.message,
        }


class RagChain:
    """질문을 받아 검색하고 답변을 생성하는 RAG 파이프라인.

    임베딩 모델과 벡터DB는 처음 질문이 들어올 때 한 번만 로드한다(lazy loading).
    """

    def __init__(self, top_k: int = TOP_K, min_score: float = MIN_RELEVANCE_SCORE) -> None:
        self.top_k = top_k
        self.min_score = min_score
        self._embedder = None
        self._collection = None
        self._router = LLMRouter()

    def _ensure_loaded(self) -> None:
        """임베딩 모델과 ChromaDB 컬렉션을 최초 1회만 로드한다."""
        if self._collection is not None:
            return
        import chromadb
        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            self._collection = client.get_collection(COLLECTION_NAME)
        except Exception as exc:
            raise RuntimeError(
                "벡터DB가 없습니다. 먼저 `uv run python -m chatbot.ingest`를 실행하세요."
            ) from exc

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """질문을 임베딩해서 벡터DB에서 비슷한 청크 TOP_K개를 찾는다."""
        self._ensure_loaded()
        query_embedding = self._embedder.encode([question]).tolist()
        result = self._collection.query(
            query_embeddings=query_embedding,
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for text, meta, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            # 📝 코사인 거리(distance)를 유사도 점수로 변환: score = 1 - distance
            chunks.append(
                RetrievedChunk(text=text, source=meta["source"], score=1 - distance)
            )
        return chunks

    def ask(self, question: str) -> RagAnswer:
        """검색 → 관련성 필터 → 프롬프트 구성 → LLM 호출까지 한 번에 수행한다."""
        # 📝 1) 검색 후, 유사도가 기준(min_score) 이상인 청크만 근거로 쓴다.
        relevant = [c for c in self.retrieve(question) if c.score >= self.min_score]

        # 📝 2) 근거 문서가 하나도 없으면 LLM을 호출하지 않고 바로 "확인 불가"로 답한다.
        #       (문서 근거 없이 LLM 지식으로 답하는 환각을 막는 장치)
        if not relevant:
            return RagAnswer(question=question, answer=NO_ANSWER_TEXT)

        # 📝 3) 검색된 청크를 문서 컨텍스트로 넣어 답변을 생성한다.
        context = "\n\n".join(f"[{c.source}]\n{c.text}" for c in relevant)
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        result = self._router.generate(
            prompt, system=RAG_SYSTEM_PROMPT, max_tokens=800, temperature=0.2
        )

        # 📝 4) 출처 문서명은 중복 없이 순서를 유지해서 모은다.
        sources = list(dict.fromkeys(c.source for c in relevant))
        return RagAnswer(
            question=question,
            answer=result.text,
            sources=sources,
            confidence=max(c.score for c in relevant),
            retrieved_chunks=len(relevant),
            message=GROUNDED_MESSAGE,
            chunks=relevant,
            provider=result.provider,
            model=result.model,
        )


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "RAG가 뭐야?"
    chain = RagChain()
    result = chain.ask(question)
    print(f"질문: {question}\n")
    print(f"답변({result.provider}/{result.model}):\n{result.answer}\n")
    print(f"신뢰도: {result.confidence:.2f} / 사용 청크: {result.retrieved_chunks}개")
    print(f"메시지: {result.message}")
    print("참고 문서:")
    for chunk in result.chunks:
        print(f"- [{chunk.source}] (유사도 {chunk.score:.2f}) {chunk.text[:60]}...")
