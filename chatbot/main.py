# 📝 main: RAG 챗봇 전용 FastAPI 서버, /chat API 제공
# 📝 실행: uv run uvicorn chatbot.main:app --reload --port 8002
# 📝 테스트: http://127.0.0.1:8002/docs 에서 /chat 호출
#
# 로그인이 필요한 통합 앱은 chatbot/local_chat/app.py(/api/rag/ask)를 사용하고,
# 이 파일은 과제 제출용 최소 RAG API 서버다.
from __future__ import annotations

import asyncio

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .rag_chain import RagChain

app = FastAPI(
    title="문서 기반 RAG 챗봇",
    description="질문과 관련된 문서를 검색하고, 검색된 문서를 근거로 답변을 생성합니다.",
    version="1.0.0",
)

# 📝 첫 질문이 들어올 때 임베딩 모델과 벡터DB를 로드한다(lazy loading).
_chain = RagChain()


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    """19번 응답 형식"""

    question: str
    answer: str
    sources: list[str]
    confidence: float
    retrieved_chunks: int
    message: str


@app.get("/")
def health() -> dict:
    return {"status": "ok", "docs": "/docs", "chat": "POST /chat"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """질문 → 문서 검색 → 근거 기반 답변"""
    result = await asyncio.to_thread(_chain.ask, request.question)
    return ChatResponse(**result.to_dict())
