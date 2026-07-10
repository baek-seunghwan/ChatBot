# 📝 rag_chain: LangGraph 기반 retriever → relevance filter → answer generation
# 📝 터미널 테스트: uv run python -m chatbot.rag_chain "RAG가 뭐야?"
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Literal, TypedDict

from .config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_CACHE_DIR,
    EMBEDDING_LOCAL_FILES_ONLY,
    EMBEDDING_MODEL,
    MIN_RELEVANCE_SCORE,
    PROMPT_VERSION,
    TOP_K,
    USE_RERANKER,
)
from .providers import LLMRouter

from langgraph.graph import END, START, StateGraph

# 📝 프롬프트 버전: 실험(run_rag_experiments)에서 버전별 성능을 비교할 수 있다.
#    기본은 v2. .env의 RAG_PROMPT_VERSION으로 바꿀 수 있다.
PROMPT_VERSIONS: dict[str, str] = {
    # v1: 초기 8규칙 프롬프트
    "v1": """너는 문서 기반 RAG 챗봇이다.

반드시 지켜야 할 규칙:
1. 제공된 문서 내용만 사용해서 답변한다.
2. 문서에 없는 내용은 추측하지 않는다.
3. 문서에 없으면 "문서에서 확인할 수 없습니다"라고 답한다.
4. 답변 마지막에 참고한 문서명을 표시한다.
5. 답변은 한국어로 한다.
6. 사용자가 초보자라면 쉬운 말로 설명한다.
7. 코드나 개념 설명이 필요하면 단계별로 설명한다.
8. 문서 근거와 답변 내용을 분리해서 보여준다.""",
    # v2: 근거 규칙 강화 (질문에 직접 답변, 관련 없는 문서 무시, 부분 근거 처리)
    "v2": """너는 문서 기반 RAG 챗봇이다.

반드시 지켜야 할 규칙:
1. 반드시 제공된 문서 내용만 근거로 답변한다. 너의 사전 지식은 사용하지 않는다.
2. 문서에 없는 내용은 추측하지 않는다.
3. 근거가 부족하면 "제공된 문서에서 확인할 수 없습니다"라고 답한다.
4. 질문과 관련 없는 문서가 섞여 있으면 그 문서는 무시하고 답변에 사용하지 않는다.
5. 답변의 첫 문장은 질문에 직접적으로 답한다. 서론을 붙이지 않는다.
6. 문서 일부만 관련 있으면, 관련 있는 부분만 답하고 나머지는 "문서에서 확인할 수 없습니다"라고 밝힌다.
7. 답변은 한국어로, 초보자도 이해할 수 있게 쉬운 말로 설명한다.
8. 답변 마지막 줄에 "(참고: 문서명)" 형식으로 실제로 사용한 문서만 표시한다.""",
}

RAG_SYSTEM_PROMPT = PROMPT_VERSIONS[PROMPT_VERSION]

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


class RagGraphState(TypedDict, total=False):
    """LangGraph 노드들이 공유하는 RAG 상태."""

    question: str
    retrieved_chunks: list[RetrievedChunk]
    relevant_chunks: list[RetrievedChunk]
    route: Literal["generate", "no_answer"]
    answer: RagAnswer


class RagChain:
    """질문을 받아 검색하고 답변을 생성하는 LangGraph 기반 RAG 파이프라인.

    임베딩 모델과 벡터DB는 처음 질문이 들어올 때 한 번만 로드한다(lazy loading).
    """

    def __init__(
        self,
        top_k: int = TOP_K,
        min_score: float = MIN_RELEVANCE_SCORE,
        collection_name: str = COLLECTION_NAME,
        prompt_version: str = PROMPT_VERSION,
        use_reranker: bool = USE_RERANKER,
    ) -> None:
        self.top_k = top_k
        self.min_score = min_score
        self.collection_name = collection_name
        self.prompt_version = prompt_version
        self.system_prompt = PROMPT_VERSIONS[prompt_version]
        self.use_reranker = use_reranker
        self._embedder = None
        self._collection = None
        self._router = LLMRouter()
        self._graph = self._build_graph()

    def _build_graph(self):
        """RAG 실행 순서를 LangGraph StateGraph로 구성한다."""
        graph = StateGraph(RagGraphState)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("filter_context", self._filter_context_node)
        graph.add_node("generate", self._generate_node)
        graph.add_node("no_answer", self._no_answer_node)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "filter_context")
        graph.add_conditional_edges(
            "filter_context",
            self._route_after_filter,
            {
                "generate": "generate",
                "no_answer": "no_answer",
            },
        )
        graph.add_edge("generate", END)
        graph.add_edge("no_answer", END)
        return graph.compile()

    def _ensure_loaded(self) -> None:
        """임베딩 모델과 ChromaDB 컬렉션을 최초 1회만 로드한다."""
        if self._collection is not None:
            return
        import chromadb
        from sentence_transformers import SentenceTransformer

        model_kwargs = {"local_files_only": EMBEDDING_LOCAL_FILES_ONLY}
        if EMBEDDING_CACHE_DIR is not None:
            model_kwargs["cache_folder"] = str(EMBEDDING_CACHE_DIR)
        try:
            self._embedder = SentenceTransformer(EMBEDDING_MODEL, **model_kwargs)
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
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            self._collection = client.get_collection(self.collection_name)
        except Exception as exc:
            raise RuntimeError(
                f"벡터DB 컬렉션({self.collection_name})이 없습니다. "
                "먼저 `uv run python -m chatbot.ingest`를 실행하세요."
            ) from exc

    @staticmethod
    def _lexical_overlap(question: str, text: str) -> float:
        """질문과 청크의 단어 겹침 비율 (0~1). reranker에 사용."""
        q_tokens = set(question.split())
        t_tokens = set(text.split())
        if not q_tokens:
            return 0.0
        return len(q_tokens & t_tokens) / len(q_tokens)

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """질문을 임베딩해서 벡터DB에서 비슷한 청크 TOP_K개를 찾는다.

        reranker가 켜져 있으면 TOP_K의 3배를 가져온 뒤,
        임베딩 유사도 + 단어 겹침 점수로 다시 정렬해서 TOP_K개만 남긴다.
        (임베딩만으로는 놓치는 키워드 일치를 보완하는 장치)
        """
        self._ensure_loaded()
        fetch_k = self.top_k * 3 if self.use_reranker else self.top_k
        query_embedding = self._embedder.encode([question]).tolist()
        result = self._collection.query(
            query_embeddings=query_embedding,
            n_results=fetch_k,
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
        if self.use_reranker:
            chunks.sort(
                key=lambda c: 0.7 * c.score + 0.3 * self._lexical_overlap(question, c.text),
                reverse=True,
            )
            chunks = chunks[: self.top_k]
        return chunks

    def _retrieve_node(self, state: RagGraphState) -> RagGraphState:
        """질문과 관련된 청크 후보를 검색한다."""
        return {"retrieved_chunks": self.retrieve(state["question"])}

    def _filter_context_node(self, state: RagGraphState) -> RagGraphState:
        """유사도 기준을 넘는 청크만 답변 근거로 남긴다."""
        relevant = [
            chunk
            for chunk in state.get("retrieved_chunks", [])
            if chunk.score >= self.min_score
        ]
        return {
            "relevant_chunks": relevant,
            "route": "generate" if relevant else "no_answer",
        }

    @staticmethod
    def _route_after_filter(state: RagGraphState) -> Literal["generate", "no_answer"]:
        """근거 청크 유무에 따라 생성 또는 확인 불가 응답으로 분기한다."""
        return state.get("route", "no_answer")

    def _no_answer_node(self, state: RagGraphState) -> RagGraphState:
        """근거 문서가 없으면 LLM을 호출하지 않고 고정 응답을 반환한다."""
        return {
            "answer": RagAnswer(
                question=state["question"],
                answer=NO_ANSWER_TEXT,
            )
        }

    def _generate_node(self, state: RagGraphState) -> RagGraphState:
        """검색된 근거 청크를 프롬프트에 넣어 LLM 답변을 생성한다."""
        relevant = state.get("relevant_chunks", [])
        context = "\n\n".join(f"[{c.source}]\n{c.text}" for c in relevant)
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=state["question"],
        )
        result = self._router.generate(
            prompt,
            system=self.system_prompt,
            max_tokens=800,
            temperature=0.2,
        )

        sources = list(dict.fromkeys(chunk.source for chunk in relevant))
        return {
            "answer": RagAnswer(
                question=state["question"],
                answer=result.text,
                sources=sources,
                confidence=max(chunk.score for chunk in relevant),
                retrieved_chunks=len(relevant),
                message=GROUNDED_MESSAGE,
                chunks=relevant,
                provider=result.provider,
                model=result.model,
            )
        }

    def ask(self, question: str) -> RagAnswer:
        """LangGraph 실행: 검색 → 필터 → 조건 분기 → 답변 생성."""
        state = self._graph.invoke(
            {"question": question},
            config={
                "run_name": "rag-chat",
                "tags": ["rag", "chat", self.prompt_version],
                "metadata": {
                    "top_k": self.top_k,
                    "min_score": self.min_score,
                    "collection": self.collection_name,
                    "prompt_version": self.prompt_version,
                    "reranker": self.use_reranker,
                },
            },
        )
        return state["answer"]


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
