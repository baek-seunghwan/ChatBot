# 📝 로그인형 단일 챗봇의 FastAPI 메인 파일
# 📝 실행: uvicorn chatbot.local_chat.app:app --reload --port 8001
from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..model import generate_text, load_checkpoint
from ..providers import LLMRouter
from ..qa_match import best_match
from ..rag_chain import RagChain
from ..config import langsmith_status
from . import auth
from .db import get_conn, init_db
from .web import INDEX_HTML

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = Path(
    os.getenv("CHATBOT_MODEL_PATH", REPO_ROOT / "artifacts" / "chatbot.pt")
)

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

ResponderMode = Literal["ai", "local"]


# ── 요청/응답 데이터 모양 ──────────────────────────────
class AuthRequest(BaseModel):
    username: str = Field(min_length=2, max_length=30, pattern=r"^[a-zA-Z0-9_가-힣]+$")
    password: str = Field(min_length=6, max_length=100)


class AuthResponse(BaseModel):
    token: str
    username: str


class AskRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    responder: ResponderMode = "ai"


class ChatAnswer(BaseModel):
    responder: ResponderMode = "ai"
    answer: str | None = None
    error: str | None = None


class AskResponse(BaseModel):
    question: str
    responder: ResponderMode = "ai"
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
        return ChatAnswer(responder="ai", answer=result.text)
    except Exception as exc:
        return ChatAnswer(responder="ai", error=f"{type(exc).__name__}: {str(exc)[:200]}")


@lru_cache(maxsize=1)
def get_local_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"학습 모델이 없습니다: {MODEL_PATH}. 먼저 `python -m chatbot.train`을 실행하세요."
        )
    return load_checkpoint(MODEL_PATH, device="cpu")


# 📝 학습/추론에서 반드시 같은 형식을 써야 한다 (scripts/build_qa_corpus.py와 동일)
_QA_PROMPT_FORMAT = "질문: {q} 답변:"
_LOCAL_FALLBACK_ANSWER = (
    "아직 배우지 못한 질문이에요. 저는 학습한 범위 안에서만 답할 수 있는 작은 모델이라, "
    "이 질문은 AI 모드로 물어봐 주세요!"
)

# 📝 학습 질문과의 유사도가 이 값보다 낮으면 생성도 하지 않고 솔직하게 모른다고 답함
#    (엉뚱한 입력에 외운 답변을 내뱉는 것을 막는 장치)
_GENERATION_MIN_SIMILARITY = 0.30

_TIME_PATTERN = re.compile(r"몇\s*시|지금\s*시간|시간\s*(알려|좀|뭐)")
_DATE_PATTERN = re.compile(r"며칠|몇\s*일이|무슨\s*요일|오늘\s*날짜|날짜\s*(알려|좀|뭐)")

_WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]


def _dynamic_answer(prompt: str) -> str | None:
    """시간/날짜처럼 실시간 정보가 필요한 질문은 datetime으로 직접 답한다."""
    now = datetime.now()
    if _TIME_PATTERN.search(prompt):
        return f"지금은 {now.hour}시 {now.minute}분이에요."
    if _DATE_PATTERN.search(prompt):
        weekday = _WEEKDAYS[now.weekday()]
        return f"오늘은 {now.year}년 {now.month}월 {now.day}일 {weekday}요일이에요."
    return None


def _local_llm_call(prompt: str) -> ChatAnswer:
    """로컬 학습 모델로 답변을 생성함. (질문 이어쓰기가 아니라 질문→답변 방식)

    0차: 시간/날짜 질문은 datetime으로 실시간 답변
    1차: 학습한 QA 쌍과 직접 매칭 → 비슷한 질문이면 학습된 답변을 그대로 반환
    2차: 유사도가 어느 정도 있으면 "질문: X 답변:" 형식으로 모델이 답변 생성
    3차: 유사도가 너무 낮거나 생성 실패면 솔직한 안내 메시지
    """
    try:
        # 📝 0차: 실시간 정보(시간/날짜)는 코드로 직접 답함
        dynamic = _dynamic_answer(prompt)
        if dynamic is not None:
            return ChatAnswer(responder="local", answer=dynamic)

        # 📝 1차: 아는 질문이면 정확한 답을 바로 돌려줌 (생성 오류 방지)
        matched, score = best_match(prompt)
        if matched is not None:
            return ChatAnswer(responder="local", answer=matched)

        # 📝 배운 것과 전혀 다른 입력("ddd" 등)이면 생성하지 않고 모른다고 답함
        #    작은 모델은 모르는 질문에도 외운 답을 자신 있게 내뱉기 때문
        if score < _GENERATION_MIN_SIMILARITY:
            return ChatAnswer(responder="local", answer=_LOCAL_FALLBACK_ANSWER)

        # 📝 2차: QA 형식으로 모델 생성 (질문을 그대로 이어쓰지 않게 함)
        model, tokenizer, config, metadata = get_local_model_bundle()

        # 📝 옛날 체크포인트(문장 이어쓰기용)는 질문에 답할 수 없으므로 생성하지 않음
        #    → scripts.build_qa_corpus로 말뭉치를 만들고 qa_corpus.txt로 재학습해야 함
        if "qa" not in str(metadata.get("corpus", "")):
            return ChatAnswer(responder="local", answer=_LOCAL_FALLBACK_ANSWER)

        qa_prompt = _QA_PROMPT_FORMAT.format(q=prompt.strip())
        generated = generate_text(
            model,
            tokenizer,
            config,
            qa_prompt,
            max_new_tokens=120,
            temperature=0.5,
            top_k=10,
            device="cpu",
        )
        # 📝 "질문: X 답변:" 뒤에 생성된 부분만 잘라냄
        answer = generated[len(qa_prompt):].strip() if generated.startswith(qa_prompt) else ""

        # 📝 3차: 빈 답/너무 짧은 답/질문 복붙이면 솔직하게 모른다고 답함
        if len(answer) < 5 or answer in prompt:
            answer = _LOCAL_FALLBACK_ANSWER
        return ChatAnswer(responder="local", answer=answer)
    except Exception as exc:
        return ChatAnswer(
            responder="local", error=f"{type(exc).__name__}: {str(exc)[:200]}"
        )


def _answer_for_mode(prompt: str, responder: ResponderMode) -> ChatAnswer:
    if responder == "local":
        return _local_llm_call(prompt)
    return _chatbot_call(prompt)


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
        responder = _row_value(row, "responder") or "ai"
        return {
            "responder": responder if responder in ("ai", "local") else "ai",
            "answer": answer,
            "error": error,
        }

    answer = _row_value(row, "assistant_answer")
    error = _row_value(row, "assistant_error")
    if answer is not None or error is not None:
        return {
            "responder": "ai",
            "answer": answer,
            "error": error,
        }

    for field in ("answer", "error"):
        for name in ("claude", "gemini", "local"):
            value = _row_value(row, f"{name}_{field}")
            if value is None:
                continue
            return {
                "responder": "local" if name == "local" else "ai",
                "answer": _row_value(row, f"{name}_answer"),
                "error": _row_value(row, f"{name}_error"),
            }
    return None


# ── 챗봇 API ──────────────────────────────
@app.post("/api/chat/ask", response_model=AskResponse)
async def ask(request: AskRequest, user: dict = Depends(current_user)) -> AskResponse:
    answer = await asyncio.to_thread(_answer_for_mode, request.message, request.responder)
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO messages (
                user_id, question, responder, answer, error
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                user["id"], request.message, answer.responder,
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
        stored = _stored_answer(row) or {"responder": "ai", "answer": None, "error": None}
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
            """
            INSERT INTO messages (user_id, question, responder, answer, error)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user["id"], request.message, "ai", response.answer, response.error),
        )
    return response


# ── 화면 ──────────────────────────────
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "app": "Leon's ChatBot", "langsmith": langsmith_status()}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML
