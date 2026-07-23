from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

# "나만의 모델": 외부 서버(Ollama) 없이 동작하는 자체 QA 매칭 모델.
# 직접 만든 학습 데이터(chatbot/qa_corpus.txt)의 질문-답변 쌍과
# MOVB 서비스 QA를 문자 2글자(bigram) 자카드 유사도로 매칭해 답한다.
REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = REPO_ROOT / "chatbot" / "qa_corpus.txt"

MATCH_THRESHOLD = 0.55

# MOVB 서비스 전용 QA (코퍼스에 없는 서비스 지식 보강)
MOVB_QA: list[tuple[str, str]] = [
    (
        "MOVB가 뭐야",
        "MOVB는 택시 합승과 묶음 퀵을 더 편리하게 접수하고, 경로와 예상 요금을 확인하는 포트폴리오용 모빌리티 서비스예요.",
    ),
    (
        "퀵 합승이 뭐야",
        "퀵 합승은 서로 다른 사람들의 물건이 같은 방향이면 한 차량에 묶어 요금을 나누는 MOVB의 핵심 기능이에요. "
        "진행 방향 차이가 30도 이내이고, 묶었을 때 총 주행거리가 각자 따로 갈 때의 85% 이하면 매칭됩니다.",
    ),
    (
        "합승 요금은 어떻게 나눠",
        "합승 요금은 각자 혼자 보냈을 때의 견적에 비례해서 나눠요. 멀리 보내는 사람이 조금 더 내되, 모두가 혼자보다 쌉니다.",
    ),
    (
        "택시 합승은 어떻게 해",
        "택시 합승 페이지(/taxi)에서 공통 출발지와 각자의 목적지를 입력하면, 최적 하차 순서와 공정한 요금 분담을 계산해드려요.",
    ),
    (
        "배송 상품 종류 알려줘",
        "퀵(기본), 퀵 이코노미(더 싸지만 느림), 퀵 급송(가장 빠르고 비쌈), 도보 배송(초단거리 저가)이 있어요.",
    ),
    (
        "물품 크기는 어떤 게 있어",
        "XS(서류/초소형), S(소형), M(중형), L(대형) 네 가지예요.",
    ),
    (
        "주문은 어떻게 해",
        "채팅에 출발지, 도착지, 물품, 연락처를 말씀해주시면 견적을 보여드리고, '네'라고 확정하시면 접수돼요. "
        "실제 접수는 'AI 채팅' 모드에서만 가능해요.",
    ),
]

_FALLBACK = (
    "자체 QA에 아직 등록되지 않은 질문이에요. 퀵 종류, 물품 크기, 합승 요금처럼 MOVB 서비스에 관해 물어보거나 "
    "더 자유로운 대화는 'AI 채팅' 모드를 이용해주세요."
)

_GREETING_PATTERN = re.compile(r"^(안녕|안녕하세요|하이|헬로|헬|ㅎㅇ|반가워|ㅇㅇ)+$")
_OLLAMA_PATTERN = re.compile(
    r"^(올라마|ollama|라마)$|(올라마|ollama|라마).*(켜|상태|연결)|(?:켜|연결).*(올라마|ollama|라마)|지금.*켰"
)


def _normalize(text: str) -> str:
    return re.sub(r"[^가-힣a-z0-9]", "", text.lower())


def _bigrams(text: str) -> set[str]:
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


def similarity(a: str, b: str) -> float:
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
    pairs = list(MOVB_QA)
    if CORPUS_PATH.exists():
        pattern = re.compile(r"질문:\s*(.+?)\s*답변:\s*(.+)")
        for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines():
            match = pattern.match(line.strip())
            if match:
                pairs.append((match.group(1), match.group(2)))
    return pairs


def own_model_reply(prompt: str) -> str:
    """학습 QA 중 가장 비슷한 질문의 답을 돌려준다. 기준 미달이면 솔직히 모른다고 답한다."""
    conversation_key = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-z0-9]", "", prompt.lower())
    if _GREETING_PATTERN.fullmatch(conversation_key):
        return "안녕하세요! 지금은 서버 없이 동작하는 MOVB 자체 QA 모드예요. 퀵·택시·합승 서비스에 대해 물어보세요 🙂"
    if _OLLAMA_PATTERN.search(conversation_key):
        return "현재 연결 상태는 채팅창 위의 Ollama 표시가 가장 정확해요. '연결 안 됨'이면 이 자체 QA 모드로 답변합니다."
    best_answer, best_score = None, 0.0
    for question, answer in load_qa_index():
        score = similarity(prompt, question)
        if score > best_score:
            best_answer, best_score = answer, score
    if best_answer is not None and best_score >= MATCH_THRESHOLD:
        return best_answer
    return _FALLBACK
