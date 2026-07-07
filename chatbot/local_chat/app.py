# 📝 로그인형 통합 챗봇의 FastAPI 메인 파일
# 📝 실행: uvicorn chatbot.local_chat.app:app --reload --port 8001
from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# 📝 5주차 과제에서 학습한 내 모델(chatbot.pt)을 불러오는 함수들
from ..model import generate_text_hybrid, load_checkpoint
from ..providers import LLMRouter
from . import auth
from .db import get_conn, init_db
from .web import INDEX_HTML

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# 📝 학습된 모델 파일 위치 (기존 앱과 동일한 경로)
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
    description="로그인 후 단일 채팅 화면에서 하나의 AI 답변만 제공하는 통합 챗봇",
    version="2.0.0",
)

# 📝 서버가 시작될 때 DB 테이블을 준비함
init_db()

# 📝 기존 providers.py의 LLMRouter를 재사용함 (Claude 우선 → Gemini 폴백 로직 내장)
_router = LLMRouter()


# ── 요청/응답 데이터 모양 ──────────────────────────────
class AuthRequest(BaseModel):
    username: str = Field(min_length=2, max_length=30, pattern=r"^[a-zA-Z0-9_가-힣]+$")
    password: str = Field(min_length=6, max_length=100)


class AuthResponse(BaseModel):
    token: str
    username: str


class AskRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


# 📝 name은 화면에서 항상 assistant로 내려 보내고, 실제 사용 모델은 model에 남김
class NamedAnswer(BaseModel):
    name: str
    answer: str | None = None
    model: str | None = None
    error: str | None = None


class AskResponse(BaseModel):
    question: str
    mode: str
    results: list[NamedAnswer]


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


def _cloud_call(prompt: str) -> NamedAnswer:
    """Claude 우선 호출, 실패하면 LLMRouter가 자동으로 Gemini로 폴백함."""
    try:
        result = _router.generate(
            prompt, system=_SYSTEM_PROMPT, max_tokens=800, temperature=0.5
        )
        provider = "Claude" if result.provider == "anthropic" else "Gemini"
        return NamedAnswer(
            name="assistant", answer=result.text, model=f"{provider} · {result.model}"
        )
    except Exception as exc:
        return NamedAnswer(name="assistant", error=f"{type(exc).__name__}: {str(exc)[:200]}")


# 📝 내 모델은 무거우니 한 번만 불러오고 계속 재사용함
@lru_cache(maxsize=1)
def get_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"학습 모델이 없습니다: {MODEL_PATH}. 먼저 `python -m chatbot.train`을 실행하세요."
        )
    return load_checkpoint(MODEL_PATH, device="cpu")


def _local_call(prompt: str) -> NamedAnswer:
    """내 학습 모델 호출. 외부 API를 못 쓸 때 통합 챗봇의 마지막 폴백으로 사용함."""
    try:
        model, tokenizer, config, metadata = get_model_bundle()
        generated = generate_text_hybrid(
            model,
            tokenizer,
            config,
            prompt,
            max_new_words=15,
            top_k=8,
            next_word_index=metadata.get("next_word_index"),
            device="cpu",
        )
        answer = (
            "외부 AI를 사용할 수 없어 로컬 학습 모델로 문장을 이어 생성했습니다.\n\n"
            f"{generated}"
        )
        return NamedAnswer(
            name="assistant",
            answer=answer,
            model="Local · character-transformer (문장 이어쓰기 모델)",
        )
    except Exception as exc:
        return NamedAnswer(name="assistant", error=f"{type(exc).__name__}: {str(exc)[:200]}")


def _assistant_call(prompt: str) -> NamedAnswer:
    """외부 LLM과 로컬 모델을 하나의 챗봇 응답 경로로 묶음."""
    cloud_result = _cloud_call(prompt)
    if cloud_result.answer:
        return cloud_result

    local_result = _local_call(prompt)
    if local_result.answer:
        return local_result

    return NamedAnswer(
        name="assistant",
        error=(
            "사용 가능한 응답 엔진이 없습니다. "
            f"외부 API: {cloud_result.error or '실패'} / "
            f"로컬 모델: {local_result.error or '실패'}"
        ),
    )


def _legacy_answer(row) -> dict | None:
    """예전 모델별 기록을 통합 챗봇 기록 모양으로 변환함."""
    labels = {"claude": "Claude", "gemini": "Gemini", "local": "Local"}
    for field in ("answer", "error"):
        for name in ("claude", "gemini", "local"):
            value = row[f"{name}_{field}"]
            if value is None:
                continue
            model = row[f"{name}_model"]
            return {
                "name": "assistant",
                "answer": row[f"{name}_answer"],
                "model": f"{labels[name]} · {model}" if model else labels[name],
                "error": row[f"{name}_error"],
            }
    return None


# ── 챗봇 API ──────────────────────────────
@app.post("/api/chat/ask", response_model=AskResponse)
async def ask(request: AskRequest, user: dict = Depends(current_user)) -> AskResponse:
    # 📝 내부 엔진은 여러 개여도 화면과 API에는 하나의 assistant 답변만 내려간다.
    answer = await asyncio.to_thread(_assistant_call, request.message)
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO messages (
                user_id, question, mode,
                assistant_answer, assistant_model, assistant_error
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user["id"], request.message, "chat",
                answer.answer, answer.model, answer.error,
            ),
        )
    return AskResponse(question=request.message, mode="chat", results=[answer])


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
        if row["assistant_answer"] is not None or row["assistant_error"] is not None:
            results = [
                {
                    "name": "assistant",
                    "answer": row["assistant_answer"],
                    "model": row["assistant_model"],
                    "error": row["assistant_error"],
                }
            ]
        else:
            legacy = _legacy_answer(row)
            results = [legacy] if legacy else []
        turns.append(
            {
                "question": row["question"],
                "mode": "chat",
                "results": results,
                "created_at": row["created_at"],
            }
        )
    return turns


@app.delete("/api/chat/history")
def clear_history(user: dict = Depends(current_user)) -> dict:
    with get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE user_id = ?", (user["id"],))
    return {"ok": True}


# ── 화면 ──────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML
