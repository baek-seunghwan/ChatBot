from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from .knowledge import default_knowledge_base

# "나만의 모델": 외부 서버(Ollama) 없이 동작하는 자체 QA 검색 모델.
# 배포에 포함되는 일상 대화 QA와 MOVB 서비스 QA만 사용한다.
BUNDLED_QA_PATH = Path(__file__).resolve().parent / "local_chat_qa.jsonl"

MATCH_THRESHOLD = 0.48

# MOVB 서비스 전용 QA. 서비스 행동과 핵심 개념은 일반 문서 검색보다 먼저
# 처리해 "주문해줘"에 인증 키 설명이 나오는 식의 오답을 막는다.
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
        "내 로컬 채팅은 대화와 질문 답변만 지원해요. 실제 배송 견적·접수·조회·취소는 "
        "상단의 **AI 채팅**으로 바꾼 뒤 출발지, 도착지, 물품, 연락처를 말씀해주세요.",
    ),
    (
        "배송 주문해줘",
        "내 로컬 채팅에서는 실제 배송을 접수할 수 없어요. 상단에서 **AI 채팅**을 선택하면 "
        "자연어로 정보를 모아 견적을 보여드리고, 확인 후 Sandbox 주문까지 진행할 수 있어요.",
    ),
    (
        "퀵과 도보 배송 차이",
        "퀵 배송은 오토바이·승용차 같은 차량을 이용하는 일반적인 배송이고, 도보 배송은 "
        "가까운 거리를 도보로 전달하는 상품이에요. 실제 가능 여부와 요금은 출발지·도착지로 "
        "견적을 조회해야 확인할 수 있어요.",
    ),
    (
        "묶음배송 합승 택시동승 차이",
        "묶음배송은 한 사람이 여러 목적지로 보내는 배송, 퀵 합승은 서로 다른 사용자의 "
        "비슷한 방향 배송을 묶는 기능, 택시 동승은 함께 출발한 승객들의 하차 순서와 "
        "요금을 나누는 기능이에요.",
    ),
    (
        "Sandbox에서 실제 결제돼",
        "아니요. 현재 연결된 Kakao Mobility Sandbox는 포트폴리오 시연 환경이라 실제 결제나 "
        "실배송이 일어나지 않아요. 화면에서도 테스트용 이름과 전화번호를 사용해야 해요.",
    ),
]

_FALLBACK = (
    "아직 답을 찾지 못했어요. MOVB 서비스에 대해 질문해 주세요 🙂"
)

_GREETING_PATTERN = re.compile(r"^(안녕|안녕하세요|하이|헬로|헬|ㅎㅇ|반가워|ㅇㅇ)+$")
_OLLAMA_PATTERN = re.compile(
    r"^(올라마|ollama|라마)$|(올라마|ollama|라마).*(켜|상태|연결)|(?:켜|연결).*(올라마|ollama|라마)|지금.*켰"
)
_MOVB_TOPIC_PATTERN = re.compile(
    r"movb|모브|퀵|배송|도보|묶음|합승|카풀|동승|주문|견적|요금|물품|sandbox|샌드박스",
    re.IGNORECASE,
)
_ORDER_ACTION_PATTERN = re.compile(
    r"(배송|퀵|도보).*(주문|접수|보내|견적|조회|취소)|"
    r"(주문|접수|견적).*(해줘|할래|하고\s*싶|보여줘)",
    re.IGNORECASE,
)
_NEGATIVE_FEELING_PATTERN = re.compile(
    r"기분.*(안\s*좋|별로|우울)|힘들|지쳤|짜증|스트레스|속상|괴로",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-z0-9]", "", text.lower())


def _ngrams(text: str, size: int) -> set[str]:
    if len(text) < size:
        return {text} if text else set()
    return {text[i : i + size] for i in range(len(text) - size + 1)}


def similarity(a: str, b: str) -> float:
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0

    shorter, longer = sorted((na, nb), key=len)
    if len(shorter) >= 4 and shorter in longer:
        coverage = len(shorter) / len(longer)
        if coverage >= 0.55:
            return 0.78 + 0.2 * coverage

    def dice(size: int) -> float:
        left, right = _ngrams(na, size), _ngrams(nb, size)
        if not left or not right:
            return 0.0
        return 2 * len(left & right) / (len(left) + len(right))

    return 0.7 * dice(2) + 0.3 * dice(3)


def _load_bundled_qa() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if not BUNDLED_QA_PATH.exists():
        return pairs
    for raw_line in BUNDLED_QA_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        item = json.loads(line)
        answer = str(item["answer"]).strip()
        questions = [item["question"], *item.get("variants", [])]
        pairs.extend((str(question).strip(), answer) for question in questions)
    return pairs


@lru_cache(maxsize=1)
def load_qa_index() -> list[tuple[str, str]]:
    pairs = [*MOVB_QA, *_load_bundled_qa()]
    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for question, answer in pairs:
        key = _normalize(question)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append((question, answer))
    return unique


def _best_match(prompt: str) -> tuple[str | None, float]:
    best_answer, best_score = None, 0.0
    for question, answer in load_qa_index():
        score = similarity(prompt, question)
        if score > best_score:
            best_answer, best_score = answer, score
    return best_answer, best_score


def own_model_reply(prompt: str) -> str:
    """외부 모델 없이 로컬 QA와 MOVB 근거 문서만으로 답한다."""
    conversation_key = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-z0-9]", "", prompt.lower())
    if _GREETING_PATTERN.fullmatch(conversation_key):
        return "안녕하세요! MOVB 서비스에 대해 물어보세요 🙂"
    if _OLLAMA_PATTERN.search(conversation_key):
        return (
            "Ollama를 꺼도 괜찮아요. 지금은 외부 서버를 쓰지 않는 Leon의 로컬 QA로 "
            "답하고 있어요. Ollama 연결 상태는 채팅창 위 표시에서 확인할 수 있어요."
        )
    if _ORDER_ACTION_PATTERN.search(prompt):
        return next(answer for question, answer in MOVB_QA if question == "배송 주문해줘")

    best_answer, best_score = _best_match(prompt)
    if best_answer is not None and best_score >= MATCH_THRESHOLD:
        return best_answer

    if _NEGATIVE_FEELING_PATTERN.search(prompt):
        return (
            "많이 힘드셨겠어요. 잠깐 하던 일을 멈추고 천천히 숨을 쉬어 보세요. "
            "혼자 버티기 어렵다면 가까운 사람에게 지금 기분을 이야기해보는 것도 좋아요."
        )
    if _MOVB_TOPIC_PATTERN.search(prompt):
        results = default_knowledge_base().search(prompt, limit=2)
        if results:
            return default_knowledge_base().fallback_answer(results)
    return _FALLBACK
