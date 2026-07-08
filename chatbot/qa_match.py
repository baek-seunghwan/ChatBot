# 📝 QA 직접 매칭: 사용자의 질문이 학습한 질문과 비슷하면 학습된 답변을 그대로 돌려준다.
#
# 작은 문자 단위 모델은 생성 품질이 불안정하기 때문에,
# "아는 질문"은 생성하지 않고 정확한 답을 돌려주는 안전장치를 먼저 둔다.
# (1차: 여기서 매칭 → 실패하면 2차: 모델 생성)
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

QA_PAIRS_PATH = Path(__file__).resolve().parent / "qa_pairs.jsonl"


def _normalize(text: str) -> str:
    """비교를 위해 공백/문장부호를 없애고 소문자로 만든다."""
    return re.sub(r"[^가-힣a-z0-9]", "", text.lower())


def _bigrams(text: str) -> set[str]:
    """문자 2글자 조각 집합. 한국어 유사도 비교에 잘 동작한다."""
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


def similarity(a: str, b: str) -> float:
    """두 문장의 유사도 (0~1). 문자 2글자 조각의 자카드 유사도."""
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    ga, gb = _bigrams(na), _bigrams(nb)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


@lru_cache(maxsize=1)
def load_qa_index() -> list[tuple[str, str]]:
    """(질문, 답변) 목록. 변형 질문도 각각 항목으로 펼친다."""
    if not QA_PAIRS_PATH.exists():
        return []
    index = []
    with QA_PAIRS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pair = json.loads(line)
            for q in [pair["question"], *pair.get("variants", [])]:
                index.append((q, pair["answer"]))
    return index


def best_match(question: str, threshold: float = 0.55) -> tuple[str | None, float]:
    """학습 질문 중 가장 비슷한 것의 답변을 돌려준다. 기준 미달이면 (None, 점수)."""
    best_answer, best_score = None, 0.0
    for known_question, answer in load_qa_index():
        score = similarity(question, known_question)
        if score > best_score:
            best_answer, best_score = answer, score
    if best_score >= threshold:
        return best_answer, best_score
    return None, best_score
