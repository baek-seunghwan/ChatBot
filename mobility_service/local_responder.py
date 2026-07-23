from __future__ import annotations

import os
import re
from datetime import datetime

import httpx

from .knowledge import SERVICE_FACTS
from .my_model import own_model_reply

# 로컬 챗봇: Ollama(http://localhost:11434)로 답한다.
# 시간/날짜 같은 실시간 질문은 모델 없이 코드로 즉답하고,
# Ollama가 꺼져 있으면 실행 방법을 안내한다.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b")
# 첫 호출은 모델을 메모리에 올리느라 오래 걸릴 수 있다.
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))

_SYSTEM_PROMPT = (
    "당신은 MOVB(택시·퀵 합승 서비스)의 로컬 챗봇입니다. "
    "한국어로 짧고 친절하게 답하세요. 아래 서비스 정보를 근거로 퀵/택시/합승 관련 "
    "질문에 구체적으로 답하고, 정보에 없는 내용은 지어내지 마세요. "
    "실제 주문 접수/조회/취소는 이 모드에서 처리할 수 없으니, "
    "접수를 원하면 'AI 채팅' 모드를 쓰라고 안내하세요.\n\n"
    + SERVICE_FACTS
)

_TIME_PATTERN = re.compile(r"몇\s*시|지금\s*시간|시간\s*(알려|좀|뭐)")
_DATE_PATTERN = re.compile(r"며칠|몇\s*일이|무슨\s*요일|오늘\s*날짜|날짜\s*(알려|좀|뭐)")
_WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]


def ollama_status() -> dict[str, object]:
    """Return the Ollama availability seen by the FastAPI server.

    On Render, localhost means the Render container rather than the user's PC,
    so this check prevents the browser from showing a misleading ON state.
    """
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
        response.raise_for_status()
        body = response.json()
        models = body.get("models", []) if isinstance(body, dict) else []
        names = {
            str(model.get("name", ""))
            for model in models
            if isinstance(model, dict)
        }
        model_available = OLLAMA_MODEL in names
        return {
            "available": model_available,
            "model": OLLAMA_MODEL,
            "message": (
                "Ollama와 모델이 준비되어 있습니다."
                if model_available
                else f"Ollama에 {OLLAMA_MODEL} 모델이 없습니다."
            ),
        }
    except (httpx.HTTPError, ValueError):
        return {
            "available": False,
            "model": OLLAMA_MODEL,
            "message": "이 서버에서 Ollama에 연결할 수 없습니다.",
        }


def _dynamic_answer(prompt: str) -> str | None:
    now = datetime.now()
    if _TIME_PATTERN.search(prompt):
        return f"지금은 {now.hour}시 {now.minute}분입니다."
    if _DATE_PATTERN.search(prompt):
        weekday = _WEEKDAYS[now.weekday()]
        return f"오늘은 {now.year}년 {now.month}월 {now.day}일 {weekday}요일입니다."
    return None


def local_model_reply(prompt: str, engine: str = "ollama") -> str:
    """'내 로컬 채팅' 모드 응답. 동기 함수라 호출부에서 asyncio.to_thread로 감싼다.

    engine:
      - "ollama": Ollama(gemma4) 사용. 꺼져 있으면 나만의 모델로 자동 폴백.
      - "own": 나만의 모델(자체 QA 매칭)만 사용 — 외부 서버 불필요.
    """
    text = (prompt or "").strip()
    if not text:
        return "메시지를 입력해주세요."

    dynamic = _dynamic_answer(text)
    if dynamic is not None:
        return dynamic

    if engine == "own":
        return own_model_reply(text)

    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "stream": False,
                # gemma4 같은 thinking 모델이 토큰을 전부 '생각'에 쓰고
                # 빈 답변을 내는 것을 막는다.
                "think": False,
                "options": {"temperature": 0.5, "num_predict": 600},
            },
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()
    except httpx.ConnectError:
        # Ollama가 꺼져 있으면 나만의 모델로 자동 전환한다.
        return "🔌 Ollama가 꺼져 있어 나만의 모델로 답했어요.\n" + own_model_reply(text)
    except httpx.TimeoutException:
        return "로컬 모델 응답이 너무 오래 걸려요. 잠시 후 다시 시도해주세요."
    except (httpx.HTTPError, ValueError) as exc:
        return f"로컬 모델 응답 중 오류가 발생했어요: {type(exc).__name__}"

    answer = ""
    if isinstance(body, dict):
        message = body.get("message")
        if isinstance(message, dict):
            answer = str(message.get("content", "")).strip()
    return answer or "로컬 모델이 빈 응답을 반환했어요. 다시 물어봐 주세요."
