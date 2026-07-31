from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


GREETING_REPLY = "안녕하세요! MOVB 서비스에 대해 물어보세요 🙂"


@dataclass(frozen=True)
class CachedChatResponse:
    key: str
    reply: str
    match_type: Literal["exact", "semantic"]


# 이 문장들은 공백만 정리한 뒤 완전히 같을 때 즉시 반환한다.
# 모델·임베딩·외부 API를 호출하지 않는 가장 빠른 경로다.
_EXACT_RESPONSES = {
    "안녕하세요": ("greeting", GREETING_REPLY),
}

# 짧은 일상 표현은 외부 임베딩 API 대신 의도별 별칭으로 묶는다.
# 업무 문장이 인사 캐시에 잘못 들어가지 않도록 전체 문장이 인사일 때만 매칭한다.
_SEMANTIC_GREETING_PATTERN = re.compile(
    r"^(?:"
    r"안녕(?:하세요|하세용|하세여)?|"
    r"안뇽(?:하세요|하세용|하세여)?|"
    r"하이+|헬로+|ㅎㅇ+|"
    r"방가(?:워)?|반가워(?:요)?|반갑(?:습니다|다)"
    r")$"
)


def _exact_key(message: str) -> str:
    return " ".join(message.strip().split())


def _semantic_key(message: str) -> str:
    return re.sub(
        r"[^가-힣ㄱ-ㅎㅏ-ㅣa-z0-9]",
        "",
        message.lower(),
    )


def cached_chat_response(message: str) -> CachedChatResponse | None:
    """Return a deterministic response before any model or external API call."""

    exact = _EXACT_RESPONSES.get(_exact_key(message))
    if exact is not None:
        key, reply = exact
        return CachedChatResponse(key=key, reply=reply, match_type="exact")

    semantic_key = _semantic_key(message)
    if _SEMANTIC_GREETING_PATTERN.fullmatch(semantic_key):
        return CachedChatResponse(
            key="greeting",
            reply=GREETING_REPLY,
            match_type="semantic",
        )
    return None
