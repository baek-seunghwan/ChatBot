# 📝 rag_langchain: LangChain 컴포넌트 기반 RAG
#
# 기존 rag_chain.py는 chromadb/sentence-transformers를 직접 호출하는 "직접 구현 RAG"이고,
# 이 파일은 같은 흐름을 LangChain 표준 컴포넌트로 다시 구성한 버전이다.
#
#   Document Loader  → TextLoader / PyPDFLoader
#   Text Splitter    → RecursiveCharacterTextSplitter
#   Embeddings       → HuggingFaceEmbeddings
#   Vector Store     → Chroma (langchain-chroma)
#   Retriever        → vectorstore.as_retriever()
#   Chain            → LCEL: {context, question} | ChatPromptTemplate | LLM | StrOutputParser
#
# 인덱스 생성: uv run python -m chatbot.rag_langchain --ingest
# 질문 테스트: uv run python -m chatbot.rag_langchain "RAG가 뭐야?"
from __future__ import annotations

import sys
from dataclasses import dataclass, field

from .config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_CACHE_DIR,
    EMBEDDING_LOCAL_FILES_ONLY,
    EMBEDDING_MODEL,
    MIN_RELEVANCE_SCORE,
    RAG_DOCS_DIR,
    TOP_K,
)
from .providers import LLMRouter
from .rag_chain import NO_ANSWER_TEXT, RAG_SYSTEM_PROMPT

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# 📝 직접 구현 RAG(rag_docs)와 분리된 LangChain 전용 컬렉션
LC_COLLECTION_NAME = "rag_docs_langchain"


@dataclass
class LangChainRagAnswer:
    """LangChain RAG 답변 결과 (rag_chain.RagAnswer와 같은 형식)"""

    question: str
    answer: str
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.0
    retrieved_chunks: int = 0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "confidence": round(self.confidence, 2),
            "retrieved_chunks": self.retrieved_chunks,
        }


def _load_documents() -> list[Document]:
    """LangChain Document Loader로 rag_docs 폴더의 문서를 읽는다."""
    from langchain_community.document_loaders import TextLoader

    documents: list[Document] = []
    for path in sorted(RAG_DOCS_DIR.iterdir()):
        if path.suffix.lower() in (".txt", ".md"):
            docs = TextLoader(str(path), encoding="utf-8").load()
        elif path.suffix.lower() == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader

            docs = PyPDFLoader(str(path)).load()
        else:
            continue
        # 📝 답변에 출처를 표시할 수 있도록 파일명을 메타데이터에 넣는다.
        for doc in docs:
            doc.metadata["source"] = path.name
        documents.extend(docs)
    if not documents:
        raise FileNotFoundError(f"{RAG_DOCS_DIR}에 문서가 없습니다.")
    return documents


def _embeddings():
    """HuggingFaceEmbeddings: 기존과 같은 다국어 임베딩 모델을 LangChain 인터페이스로 감싼 것."""
    from sentence_transformers import SentenceTransformer

    kwargs = {"local_files_only": EMBEDDING_LOCAL_FILES_ONLY}
    if EMBEDDING_CACHE_DIR is not None:
        kwargs["cache_folder"] = str(EMBEDDING_CACHE_DIR)

    try:
        model = SentenceTransformer(EMBEDDING_MODEL, **kwargs)
    except Exception as exc:
        if EMBEDDING_LOCAL_FILES_ONLY:
            raise RuntimeError(
                "임베딩 모델을 로컬 캐시에서 찾지 못했습니다. "
                f"모델: {EMBEDDING_MODEL}, 캐시: {EMBEDDING_CACHE_DIR}. "
                "처음 1회는 네트워크가 되는 환경에서 "
                "`RAG_EMBEDDINGS_LOCAL_ONLY=0 uv run python -m chatbot.rag_langchain --ingest`를 "
                "실행해 캐시를 만든 뒤 다시 실행하세요."
            ) from exc
        raise

    class SentenceTransformerEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return model.encode(texts).tolist()

        def embed_query(self, text: str) -> list[float]:
            return model.encode([text])[0].tolist()

    return SentenceTransformerEmbeddings()


def build_vectorstore() -> int:
    """문서 로딩 → 분할 → 임베딩 → Chroma 저장. 저장한 청크 수를 돌려준다."""
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print(f"[1/3] 문서 로딩: {RAG_DOCS_DIR}")
    documents = _load_documents()

    print(f"[2/3] 청크 분리 (CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"      문서 {len(documents)}개 → 청크 {len(chunks)}개")

    print(f"[3/3] Chroma 저장: {CHROMA_DIR} (컬렉션: {LC_COLLECTION_NAME})")
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(LC_COLLECTION_NAME)
    except Exception:
        pass
    Chroma.from_documents(
        documents=chunks,
        embedding=_embeddings(),
        client=client,
        collection_name=LC_COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"완료: 청크 {len(chunks)}개 저장됨")
    return len(chunks)


class LangChainRag:
    """LangChain 컴포넌트로 구성한 RAG 파이프라인.

    LCEL(LangChain Expression Language) 체인:
        {"context": retriever | format_docs, "question": passthrough}
        | ChatPromptTemplate
        | LLM
        | StrOutputParser
    """

    def __init__(self, top_k: int = TOP_K, min_score: float = MIN_RELEVANCE_SCORE) -> None:
        self.top_k = top_k
        self.min_score = min_score
        self._router = LLMRouter()
        self._vectorstore = None
        self._chain = None

    def _ensure_loaded(self) -> None:
        if self._vectorstore is not None:
            return
        import chromadb
        from langchain_chroma import Chroma

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            client.get_collection(LC_COLLECTION_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"벡터DB 컬렉션({LC_COLLECTION_NAME})이 없습니다. "
                "먼저 `uv run python -m chatbot.rag_langchain --ingest`를 실행하세요."
            ) from exc
        self._vectorstore = Chroma(
            client=client,
            collection_name=LC_COLLECTION_NAME,
            embedding_function=_embeddings(),
        )
        self._chain = self._build_chain()

    def _build_chain(self):
        """LCEL 체인 구성. LLM은 기존 LLMRouter(Claude→Gemini fallback)를 Runnable로 감싼다."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RAG_SYSTEM_PROMPT),
                ("human", "아래 문서를 참고해서 질문에 답하세요.\n\n문서:\n{context}\n\n질문:\n{question}"),
            ]
        )

        def call_llm(messages) -> str:
            # ChatPromptValue → (system, user) 텍스트로 풀어서 기존 라우터 호출
            msgs = messages.to_messages()
            system = next((m.content for m in msgs if m.type == "system"), "")
            user = "\n".join(m.content for m in msgs if m.type == "human")
            return self._router.generate(user, system=system, max_tokens=800, temperature=0.2).text

        # 📝 {"context": ..., "question": ...} → 프롬프트 → LLM → 문자열
        return prompt | RunnableLambda(call_llm) | StrOutputParser()

    @staticmethod
    def format_docs(docs: list[Document]) -> str:
        """검색된 청크를 [출처] 본문 형태의 컨텍스트 문자열로 합친다."""
        return "\n\n".join(
            f"[{d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs
        )

    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        """유사도 점수와 함께 TOP_K 청크를 검색하고, 기준 미달 청크는 제외한다."""
        self._ensure_loaded()
        # similarity_search_with_relevance_scores: 1에 가까울수록 관련 높음
        results = self._vectorstore.similarity_search_with_relevance_scores(
            question, k=self.top_k
        )
        return [(doc, score) for doc, score in results if score >= self.min_score]

    def ask(self, question: str) -> LangChainRagAnswer:
        self._ensure_loaded()
        pairs = self.retrieve(question)
        if not pairs:
            return LangChainRagAnswer(question=question, answer=NO_ANSWER_TEXT)
        context = self.format_docs([doc for doc, _ in pairs])
        answer = self._chain.invoke({"context": context, "question": question})
        sources = list(dict.fromkeys(d.metadata.get("source", "?") for d, _ in pairs))
        return LangChainRagAnswer(
            question=question,
            answer=answer,
            sources=sources,
            confidence=max(score for _, score in pairs),
            retrieved_chunks=len(pairs),
        )


if __name__ == "__main__":
    if "--ingest" in sys.argv:
        build_vectorstore()
        sys.exit(0)
    question = sys.argv[1] if len(sys.argv) > 1 else "RAG가 뭐야?"
    rag = LangChainRag()
    result = rag.ask(question)
    print(f"질문: {question}\n")
    print(f"답변:\n{result.answer}\n")
    print(f"신뢰도: {result.confidence:.2f} / 사용 청크: {result.retrieved_chunks}개 / 출처: {result.sources}")
