from __future__ import annotations

import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from chatbot.model import generate_text, load_checkpoint
from chatbot.qa_match import best_match

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("CHATBOT_MODEL_PATH", REPO_ROOT / "artifacts" / "chatbot.pt"))

_QA_PROMPT_FORMAT = "질문: {q} 답변:"
_LOCAL_FALLBACK_ANSWER = (
    "아직 배우지 못한 질문이에요. 저는 학습한 범위 안에서만 답할 수 있는 작은 모델이라, "
    "이 질문은 'AI 채팅' 모드로 물어봐 주세요!"
)
_GENERATION_MIN_SIMILARITY = 0.30
_TIME_PATTERN = re.compile(r"몇\s*시|지금\s*시간|시간\s*(알려|좀|뭐)")
_DATE_PATTERN = re.compile(r"며칠|몇\s*일이|무슨\s*요일|오늘\s*날짜|날짜\s*(알려|좀|뭐)")
_WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]


@lru_cache(maxsize=1)
def _get_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"학습 모델이 없습니다: {MODEL_PATH}. 먼저 `python -m chatbot.train`을 실행하세요."
        )
    return load_checkpoint(MODEL_PATH, device="cpu")


def _dynamic_answer(prompt: str) -> str | None:
    now = datetime.now()
    if _TIME_PATTERN.search(prompt):
        return f"지금은 {now.hour}시 {now.minute}분이에요."
    if _DATE_PATTERN.search(prompt):
        weekday = _WEEKDAYS[now.weekday()]
        return f"오늘은 {now.year}년 {now.month}월 {now.day}일 {weekday}요일이에요."
    return None


def local_model_reply(prompt: str) -> str:
    """chatbot/local_chat의 '로컬 LLM이 답변' 모드와 동일한 로직(직접 학습한 소형 모델)을 재사용한다.

    MoveOps 배송 챗봇 위젯의 '내 로컬 채팅' 모드용 — 배송 주문 처리는 하지 않고,
    학습한 QA 매칭/생성으로 캐주얼한 질문에만 답한다.
    """
    try:
        dynamic = _dynamic_answer(prompt)
        if dynamic is not None:
            return dynamic

        matched, score = best_match(prompt)
        if matched is not None:
            return matched

        if score < _GENERATION_MIN_SIMILARITY:
            return _LOCAL_FALLBACK_ANSWER

        model, tokenizer, config, metadata = _get_model_bundle()
        if "qa" not in str(metadata.get("corpus", "")):
            return _LOCAL_FALLBACK_ANSWER

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
        answer = generated[len(qa_prompt):].strip() if generated.startswith(qa_prompt) else ""
        if len(answer) < 5 or answer in prompt:
            answer = _LOCAL_FALLBACK_ANSWER
        return answer
    except FileNotFoundError as exc:
        return f"로컬 모델을 찾지 못했어요: {exc}"
    except Exception as exc:
        return f"로컬 모델 응답 중 오류가 발생했어요: {type(exc).__name__}"
