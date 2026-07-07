# 📝 로그인형 단일 챗봇의 FastAPI 메인 파일
# 📝 실행: uvicorn chatbot.local_chat.app:app --reload --port 8001
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..providers import LLMRouter
from ..rag_chain import RagChain
from . import auth
from .db import get_conn, init_db
from .web import INDEX_HTML

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# 📝 .env에 있는 ANTHROPIC_API_KEY / GEMINI_API_KEY를 불러옴
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

app = FastAPI(
    title="Leon's ChatBot",
    description="로그인 후 단일 채팅 화면에서 하나의 AI 답변만 제공하는 챗봇",
    version="2.0.0",
)

# 📝 서버가 시작될 때 DB 테이블을 준비함
init_db()

# 📝 providers.py의 LLMRouter가 Claude 우선 → Gemini 폴백으로 하나의 답변을 만든다.
_router = LLMRouter()

# 📝 RAG 체인: 첫 질문이 들어올 때 임베딩 모델과 벡터DB를 로드한다(lazy loading).
_rag_chain = RagChain()


# ── 요청/응답 데이터 모양 ──────────────────────────────
class AuthRequest(BaseModel):
    username: str = Field(min_length=2, max_length=30, pattern=r"^[a-zA-Z0-9_가-힣]+$")
    password: str = Field(min_length=6, max_length=100)


class AuthResponse(BaseModel):
    token: str
    username: str


class AskRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ChatAnswer(BaseModel):
    answer: str | None = None
    error: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str | None = None
    error: str | None = None


class RagAskResponse(BaseModel):
    """RAG 응답 형식: 답변 + 출처 + 신뢰도 + 사용 청크 수"""

    question: str
    answer: str | None = None
    error: str | None = None
    sources: list[str] = []
    confidence: float = 0.0
    retrieved_chunks: int = 0
    message: str = ""


# ── 인증 도우미 ──────────────────────────────
def current_user(authorization: str = Header(default="")) -> dict:
    """Authorization: Bearer <token> 헤더에서 로그인 사용자를 찾음."""
    token = authorization.removeprefix("Bearer ").strip()
    user = auth.get_user_by_token(token) if token else None
    if user is None:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    return user


# ── 인증 API ──────────────────────────────
@app.post("/api/auth/signup", response_model=AuthResponse)
def signup(request: AuthRequest) -> AuthResponse:
    try:
        user_id = auth.create_user(request.username, request.password)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    # 📝 가입 직후 바로 로그인 상태로 만들어 줌
    return AuthResponse(token=auth.create_session(user_id), username=request.username)


@app.post("/api/auth/login", response_model=AuthResponse)
def login(request: AuthRequest) -> AuthResponse:
    user_id = auth.verify_user(request.username, request.password)
    if user_id is None:
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다.")
    return AuthResponse(token=auth.create_session(user_id), username=request.username)


@app.post("/api/auth/logout")
def logout(authorization: str = Header(default="")) -> dict:
    auth.delete_session(authorization.removeprefix("Bearer ").strip())
    return {"ok": True}


@app.get("/api/auth/me")
def me(user: dict = Depends(current_user)) -> dict:
    return {"username": user["username"]}


# ── 모델 호출 ──────────────────────────────
_SYSTEM_PROMPT = (
    "당신은 친절한 개인 비서 챗봇입니다. 사용자의 질문에 한국어로 명확하고 "
    "간결하게 답하세요. 모르는 내용은 추측하지 말고 모른다고 답하세요."
)


def _chatbot_call(prompt: str) -> ChatAnswer:
    """LLMRouter를 통해 하나의 챗봇 답변만 생성함."""
    try:
        result = _router.generate(
            prompt, system=_SYSTEM_PROMPT, max_tokens=800, temperature=0.5
        )
        return ChatAnswer(answer=result.text)
    except Exception as exc:
        return ChatAnswer(error=f"{type(exc).__name__}: {str(exc)[:200]}")


def _row_value(row, column: str) -> str | None:
    """기존 DB에만 있거나 새 DB에만 있는 컬럼을 안전하게 읽음."""
    try:
        return row[column]
    except (IndexError, KeyError):
        return None


def _stored_answer(row) -> dict | None:
    """예전 모델별 기록을 통합 챗봇 기록 모양으로 변환함."""
    answer = _row_value(row, "answer")
    error = _row_value(row, "error")
    if answer is not None or error is not None:
        return {
            "answer": answer,
            "error": error,
        }

    answer = _row_value(row, "assistant_answer")
    error = _row_value(row, "assistant_error")
    if answer is not None or error is not None:
        return {
            "answer": answer,
            "error": error,
        }

    for field in ("answer", "error"):
        for name in ("claude", "gemini", "local"):
            value = _row_value(row, f"{name}_{field}")
            if value is None:
                continue
            return {
                "answer": _row_value(row, f"{name}_answer"),
                "error": _row_value(row, f"{name}_error"),
            }
    return None


# ── 챗봇 API ──────────────────────────────
@app.post("/api/chat/ask", response_model=AskResponse)
async def ask(request: AskRequest, user: dict = Depends(current_user)) -> AskResponse:
    answer = await asyncio.to_thread(_chatbot_call, request.message)
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO messages (
                user_id, question, answer, error
            ) VALUES (?, ?, ?, ?)
            """,
            (
                user["id"], request.message,
                answer.answer, answer.error,
            ),
        )
    return AskResponse(question=request.message, **answer.model_dump())


@app.get("/api/chat/history")
def history(user: dict = Depends(current_user), limit: int = 30) -> list[dict]:
    """이전 대화 기록을 오래된 순으로, 프론트가 그대로 그릴 수 있는 모양으로 돌려줌."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user["id"], min(limit, 100)),
        ).fetchall()
    turns = []
    for row in reversed(rows):
        stored = _stored_answer(row) or {"answer": None, "error": None}
        turns.append(
            {
                "question": row["question"],
                **stored,
                "created_at": row["created_at"],
            }
        )
    return turns


@app.delete("/api/chat/history")
def clear_history(user: dict = Depends(current_user)) -> dict:
    with get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE user_id = ?", (user["id"],))
    return {"ok": True}


# ── RAG API ──────────────────────────────
def _rag_call(question: str) -> RagAskResponse:
    """RAG 체인(검색 → 답변 생성)을 호출함."""
    try:
        result = _rag_chain.ask(question)
        return RagAskResponse(**result.to_dict())
    except Exception as exc:
        return RagAskResponse(
            question=question, error=f"{type(exc).__name__}: {str(exc)[:200]}"
        )


@app.post("/api/rag/ask", response_model=RagAskResponse)
async def rag_ask(
    request: AskRequest, user: dict = Depends(current_user)
) -> RagAskResponse:
    """문서 검색 기반(RAG) 답변. 일반 채팅과 같은 messages 테이블에 기록함."""
    response = await asyncio.to_thread(_rag_call, request.message)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (user_id, question, answer, error) VALUES (?, ?, ?, ?)",
            (user["id"], request.message, response.answer, response.error),
        )
    return response


# ── 화면 ──────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML
