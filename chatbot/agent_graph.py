# 📝 agent_graph: LangGraph 기반 RAG Agent
#
# rag_chain.py의 최소 그래프(retrieve → filter → generate)와 달리,
# 이 파일은 "스스로 판단하고 재시도하는" 진짜 Agent 구조다.
#
#   analyze_question ─┬─ (잡담/인사) → direct_answer → END
#                     └─ (문서 질문) → retrieve
#   retrieve → grade_documents ─┬─ (근거 충분) → generate
#                               ├─ (근거 부족 + 재시도 남음) → rewrite_question → retrieve  (루프)
#                               └─ (근거 부족 + 재시도 소진) → no_answer → END
#   generate → self_check ─┬─ (근거 기반 OK) → END
#                          └─ (근거 없음 판정 + 재시도 남음) → rewrite_question → retrieve
#
# 터미널 테스트: uv run python -m chatbot.agent_graph "LoRA가 뭐야?"
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Literal, TypedDict

from . import config as _config  # noqa: F401 - loads offline/tracing env before LangGraph imports
from .providers import LLMRouter
from .rag_chain import (
    NO_ANSWER_TEXT,
    RAG_SYSTEM_PROMPT,
    RagChain,
    RetrievedChunk,
)

from langgraph.graph import END, START, StateGraph

# 📝 재검색(질문 재작성) 최대 횟수. 무한 루프 방지 장치.
MAX_REWRITES = 2

ANALYZE_PROMPT = """다음 사용자 입력을 분류하세요.

입력: {question}

분류 기준:
- "chitchat": 인사, 잡담, 감정 표현 등 문서 검색이 필요 없는 입력
- "doc_question": 기술/개념/문서 내용에 대한 질문

반드시 chitchat 또는 doc_question 중 하나의 단어만 출력하세요."""

GRADE_PROMPT = """질문과 문서 청크가 주어집니다. 이 청크가 질문에 답하는 데 유용한지 판단하세요.

질문: {question}

청크:
{chunk}

유용하면 yes, 아니면 no 한 단어만 출력하세요."""

REWRITE_PROMPT = """검색이 실패한 질문을 벡터 검색에 더 잘 걸리도록 다시 작성하세요.

원래 질문: {question}

규칙:
- 핵심 키워드를 명시적으로 포함시킨다 (약어는 풀어 쓴다).
- 한 문장의 검색용 질문만 출력한다. 설명은 붙이지 않는다."""

SELF_CHECK_PROMPT = """답변이 문서 근거에 기반했는지 검증하세요.

문서:
{context}

질문: {question}

답변: {answer}

답변의 핵심 내용이 문서에서 확인 가능하면 grounded,
문서에 없는 내용을 지어냈으면 hallucinated 한 단어만 출력하세요."""

CHITCHAT_SYSTEM = "너는 친절한 한국어 챗봇이다. 짧고 자연스럽게 답한다."


class AgentState(TypedDict, total=False):
    """LangGraph 노드들이 공유하는 Agent 상태."""

    question: str            # 현재 검색에 쓰는 질문 (재작성될 수 있음)
    original_question: str   # 사용자가 처음 입력한 질문
    question_type: Literal["chitchat", "doc_question"]
    chunks: list[RetrievedChunk]          # 검색 결과
    graded_chunks: list[RetrievedChunk]   # LLM 평가를 통과한 근거
    rewrite_count: int
    answer: str
    grounded: bool
    trace: list[str]          # 지나간 노드 기록 (디버깅/데모용)
    sources: list[str]
    confidence: float


@dataclass
class AgentAnswer:
    """Agent 최종 응답."""

    question: str
    answer: str
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.0
    retrieved_chunks: int = 0
    rewrites: int = 0
    grounded: bool = False
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "confidence": round(self.confidence, 2),
            "retrieved_chunks": self.retrieved_chunks,
            "rewrites": self.rewrites,
            "grounded": self.grounded,
            "trace": self.trace,
        }


class RagAgent:
    """질문 분석 → 검색 → 문서 평가 → 재검색 → 답변 → 자기검증을 수행하는 LangGraph Agent."""

    def __init__(self) -> None:
        self._router = LLMRouter()
        self._rag = RagChain()  # 검색기(retriever)는 기존 구현 재사용
        self._graph = self._build_graph()

    # ── 그래프 구성 ──────────────────────────────────────────────
    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("analyze_question", self._analyze_question)
        graph.add_node("direct_answer", self._direct_answer)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade_documents", self._grade_documents)
        graph.add_node("rewrite_question", self._rewrite_question)
        graph.add_node("generate", self._generate)
        graph.add_node("self_check", self._self_check)
        graph.add_node("no_answer", self._no_answer)

        graph.add_edge(START, "analyze_question")
        graph.add_conditional_edges(
            "analyze_question",
            self._route_by_type,
            {"chitchat": "direct_answer", "doc_question": "retrieve"},
        )
        graph.add_edge("retrieve", "grade_documents")
        graph.add_conditional_edges(
            "grade_documents",
            self._route_after_grade,
            {
                "generate": "generate",
                "rewrite": "rewrite_question",
                "no_answer": "no_answer",
            },
        )
        graph.add_edge("rewrite_question", "retrieve")
        graph.add_edge("generate", "self_check")
        graph.add_conditional_edges(
            "self_check",
            self._route_after_check,
            {"done": END, "rewrite": "rewrite_question"},
        )
        graph.add_edge("direct_answer", END)
        graph.add_edge("no_answer", END)
        return graph.compile()

    def _llm(self, prompt: str, system: str = "간결하게 답하세요.", max_tokens: int = 300) -> str:
        return self._router.generate(
            prompt, system=system, max_tokens=max_tokens, temperature=0.0
        ).text.strip()

    # ── 노드 ────────────────────────────────────────────────────
    def _analyze_question(self, state: AgentState) -> AgentState:
        """1단계: 질문 유형 분석. 잡담이면 검색을 건너뛴다."""
        question = state["question"]
        verdict = self._llm(ANALYZE_PROMPT.format(question=question)).lower()
        qtype = "chitchat" if "chitchat" in verdict else "doc_question"
        return {
            "original_question": question,
            "question_type": qtype,
            "rewrite_count": 0,
            "trace": [f"analyze_question:{qtype}"],
        }

    @staticmethod
    def _route_by_type(state: AgentState) -> str:
        return state.get("question_type", "doc_question")

    def _direct_answer(self, state: AgentState) -> AgentState:
        """잡담/인사는 문서 검색 없이 바로 답한다."""
        answer = self._llm(state["question"], system=CHITCHAT_SYSTEM)
        return {
            "answer": answer,
            "grounded": False,
            "trace": state.get("trace", []) + ["direct_answer"],
        }

    def _retrieve(self, state: AgentState) -> AgentState:
        """2단계: 벡터DB에서 관련 청크 검색 (기존 retriever 재사용)."""
        chunks = self._rag.retrieve(state["question"])
        return {
            "chunks": chunks,
            "trace": state.get("trace", []) + [f"retrieve:{len(chunks)}개"],
        }

    def _grade_documents(self, state: AgentState) -> AgentState:
        """3단계: 검색된 청크를 LLM이 하나씩 평가해서 관련 있는 것만 남긴다.

        유사도 점수(숫자)만 믿지 않고, LLM이 의미 기준으로 한 번 더 거른다.
        """
        question = state["original_question"]
        graded = []
        for chunk in state.get("chunks", []):
            verdict = self._llm(
                GRADE_PROMPT.format(question=question, chunk=chunk.text[:800]),
                max_tokens=8,
            ).lower()
            if verdict.startswith("yes"):
                graded.append(chunk)
        return {
            "graded_chunks": graded,
            "trace": state.get("trace", [])
            + [f"grade_documents:{len(graded)}/{len(state.get('chunks', []))} 통과"],
        }

    def _route_after_grade(self, state: AgentState) -> str:
        if state.get("graded_chunks"):
            return "generate"
        if state.get("rewrite_count", 0) < MAX_REWRITES:
            return "rewrite"
        return "no_answer"

    def _rewrite_question(self, state: AgentState) -> AgentState:
        """4단계: 검색 실패 시 질문을 검색 친화적으로 재작성하고 다시 검색한다."""
        rewritten = self._llm(
            REWRITE_PROMPT.format(question=state["original_question"]), max_tokens=100
        )
        count = state.get("rewrite_count", 0) + 1
        return {
            "question": rewritten,
            "rewrite_count": count,
            "trace": state.get("trace", []) + [f"rewrite_question({count}): {rewritten[:50]}"],
        }

    def _generate(self, state: AgentState) -> AgentState:
        """5단계: 평가를 통과한 근거로 답변 생성."""
        graded = state["graded_chunks"]
        context = "\n\n".join(f"[{c.source}]\n{c.text}" for c in graded)
        answer = self._router.generate(
            f"아래 문서를 참고해서 질문에 답하세요.\n\n문서:\n{context}\n\n질문:\n{state['original_question']}",
            system=RAG_SYSTEM_PROMPT,
            max_tokens=800,
            temperature=0.2,
        ).text
        return {
            "answer": answer,
            "sources": list(dict.fromkeys(c.source for c in graded)),
            "confidence": max(c.score for c in graded),
            "trace": state.get("trace", []) + ["generate"],
        }

    def _self_check(self, state: AgentState) -> AgentState:
        """6단계: 자기검증. 답변이 문서 근거에 기반했는지 LLM이 확인한다."""
        graded = state.get("graded_chunks", [])
        context = "\n\n".join(c.text for c in graded)
        verdict = self._llm(
            SELF_CHECK_PROMPT.format(
                context=context[:4000],
                question=state["original_question"],
                answer=state["answer"],
            ),
            max_tokens=8,
        ).lower()
        grounded = "grounded" in verdict
        return {
            "grounded": grounded,
            "trace": state.get("trace", [])
            + [f"self_check:{'grounded' if grounded else 'hallucinated'}"],
        }

    def _route_after_check(self, state: AgentState) -> str:
        if state.get("grounded"):
            return "done"
        if state.get("rewrite_count", 0) < MAX_REWRITES:
            return "rewrite"
        return "done"  # 재시도 소진: 검증 실패 표시를 남긴 채 종료

    def _no_answer(self, state: AgentState) -> AgentState:
        return {
            "answer": NO_ANSWER_TEXT,
            "grounded": False,
            "trace": state.get("trace", []) + ["no_answer"],
        }

    # ── 실행 ────────────────────────────────────────────────────
    def ask(self, question: str) -> AgentAnswer:
        state = self._graph.invoke(
            {"question": question},
            config={
                "run_name": "rag-agent",
                "tags": ["rag", "agent", "self-correction"],
                "metadata": {"max_rewrites": MAX_REWRITES},
            },
        )
        return AgentAnswer(
            question=question,
            answer=state.get("answer", NO_ANSWER_TEXT),
            sources=state.get("sources", []),
            confidence=state.get("confidence", 0.0),
            retrieved_chunks=len(state.get("graded_chunks", [])),
            rewrites=state.get("rewrite_count", 0),
            grounded=state.get("grounded", False),
            trace=state.get("trace", []),
        )


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "LoRA가 뭐야?"
    agent = RagAgent()
    result = agent.ask(question)
    print(f"질문: {question}\n")
    print(f"답변:\n{result.answer}\n")
    print(f"근거 검증: {'통과' if result.grounded else '미통과'} / 재검색 {result.rewrites}회 / 출처: {result.sources}")
    print("실행 경로:")
    for step in result.trace:
        print(f"  → {step}")
