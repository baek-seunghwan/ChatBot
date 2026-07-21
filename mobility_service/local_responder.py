from __future__ import annotations

import re
from datetime import datetime

_TIME_PATTERN = re.compile(r"몇\s*시|지금\s*시간|시간\s*(알려|좀|뭐)")
_DATE_PATTERN = re.compile(r"며칠|몇\s*일이|무슨\s*요일|오늘\s*날짜|날짜\s*(알려|좀|뭐)")
_WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]


def local_model_reply(prompt: str) -> str:
    """MoveOps 전용 로컬 응답.

    이 모드는 더 이상 외부 챗봇 학습 모델을 사용하지 않는다.
    배송 관련 요청은 agent 모드를 사용하도록 안내하고,
    시간/날짜 질문만 간단히 로컬 처리한다.
    """
    text = (prompt or "").strip()
    now = datetime.now()
    if _TIME_PATTERN.search(text):
        return f"지금은 {now.hour}시 {now.minute}분입니다."
    if _DATE_PATTERN.search(text):
        weekday = _WEEKDAYS[now.weekday()]
        return f"오늘은 {now.year}년 {now.month}월 {now.day}일 {weekday}요일입니다."
    return (
        "MoveOps 로컬 모드는 일반 잡담 모델을 포함하지 않습니다. "
        "배송 요청은 기본 AI 모드로 출발지/도착지/물품 정보를 알려주세요."
    )

