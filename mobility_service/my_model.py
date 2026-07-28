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
        "MOVB는 여러 사람의 배송을 한 동선으로 묶는 스마트 딜리버리의 경로와 예상 요금을 비교하고 접수하는 포트폴리오용 서비스예요.",
    ),
    (
        "스마트 딜리버리 묶음퀵이 뭐야",
        "스마트 딜리버리는 보내는 사람 1~5명의 물품을 픽업해 받는 사람 2~5명에게 하나의 경유 배송으로 접수하는 기능이에요. "
        "각각 보낼 때의 합계와 묶음 가격, 픽업·배송 순서를 접수 전에 비교할 수 있어요.",
    ),
    (
        "묶음퀵 요금은 어떻게 계산해",
        "각 목적지를 따로 보낼 때의 견적 합계와 추천 순서대로 한 번에 경유하는 스마트 딜리버리 견적을 비교해요. "
        "주소와 차량 조건에 따라 달라지므로 접수 화면의 실제 견적을 확인해야 해요.",
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
        "스마트 딜리버리 묶음퀵 주문은 어떻게 해",
        "홈페이지 스마트 딜리버리 영역에서 보내는 사람과 받는 사람을 + 버튼으로 추가하고, 주소·연락처와 물품 정보를 입력하세요. "
        "추천 경로와 비교 견적을 확인하고 동의하면 하나의 주문으로 접수할 수 있어요.",
    ),
    (
        "Sandbox에서 실제 결제돼",
        "아니요. 현재 연결된 Kakao Mobility Sandbox는 포트폴리오 시연 환경이라 실제 결제나 "
        "실배송이 일어나지 않아요. 화면에서도 테스트용 이름과 전화번호를 사용해야 해요.",
    ),
    (
        "퀵 취소 수수료가 얼마야",
        "기사님이 픽업지로 출발하기 전에는 취소 수수료가 없어요. 픽업지로 출발한 뒤에는 이용 요금의 15%, "
        "상품을 픽업한 뒤에는 100%가 부과될 수 있어요. 픽업지 도착 후 10분 이상 노쇼도 100% 기준이에요. "
        "실제 취소 전에는 현재 주문 화면에 표시되는 상태와 수수료를 확인해주세요.",
    ),
    (
        "기사님이 배정되지 않아",
        "주문량이 많거나 근처에 가능한 기사님이 없으면 배정이 늦어질 수 있어요. 배정 가능 시간을 넘겨 자동 취소되면 "
        "다시 접수해야 해요. MOVB 스마트 딜리버리가 취소·만료·실패한 경우에는 이용 내역의 "
        "'단일 퀵으로 다시 이용하기'를 사용할 수 있어요.",
    ),
    (
        "퀵 취급 불가 품목 알려줘",
        "세 변의 합 140cm 초과 또는 20kg 초과 물품, 현금·수표·유가증권·상품권, 독극물·화약류·인화물질, "
        "마약류·개인 간 의약품·밀수품, 살아 있는 동물이나 동물 사체 등은 취급할 수 없어요. "
        "포장이 배송에 적합하지 않은 물품도 거절될 수 있어요.",
    ),
    (
        "퀵 추가 요금 알려줘",
        "대표 기준은 픽업지 대기 10분당 2,000원, 과적료 5,000원이에요. 엘리베이터가 없는 상·하차는 "
        "층당 10,000원이 각각 발생할 수 있고 차량별 운반료도 달라요. 현장에서 직접 지불하지 말고 "
        "카카오 T 고객센터를 통해 확인·결제해야 해요.",
    ),
    (
        "기사님 위치는 언제 보여",
        "기사님 위치는 픽업 출발~픽업 완료, 배송 출발~배송 완료 상태에서 확인할 수 있어요. "
        "도착이 늦거나 추가 확인이 필요하면 주문 상세의 '기사님 통화'를 이용해주세요. "
        "MOVB Sandbox에서 GPS가 없으면 화면에 모의 위치라고 명확히 표시해요.",
    ),
    (
        "카카오 퀵 고객센터 어디야",
        "카카오 T 앱에서 '내 정보 → 서비스 문의 → 채널 메뉴 → 고객센터 → 상담사 연결하기' 순서로 문의할 수 있어요. "
        "추가 요금·환불처럼 결제와 관련된 내용은 공식 고객센터에서 최종 확인해주세요.",
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
    r"movb|모브|퀵|배송|도보|묶음|주문|견적|요금|물품|sandbox|샌드박스",
    re.IGNORECASE,
)
_ORDER_ACTION_PATTERN = re.compile(
    r"(배송|퀵|도보).*(주문|접수|보내|견적|조회|취소)|"
    r"(주문|접수|견적).*(해줘|할래|하고\s*싶|보여줘)",
    re.IGNORECASE,
)
_SUPPORT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"취급\s*불가|금지\s*품목|보내면\s*안|위험\s*물품", re.IGNORECASE),
        "퀵 취급 불가 품목 알려줘",
    ),
    (
        re.compile(r"취소.*(수수료|환불)|수수료.*취소", re.IGNORECASE),
        "퀵 취소 수수료가 얼마야",
    ),
    (
        re.compile(r"기사.*(배정.*안|안.*배정)|배정\s*실패|자동\s*취소", re.IGNORECASE),
        "기사님이 배정되지 않아",
    ),
    (
        re.compile(r"추가\s*요금|대기료|과적료|상.?하차", re.IGNORECASE),
        "퀵 추가 요금 알려줘",
    ),
    (
        re.compile(r"기사.*(위치|도착|어디)|현\s*위치|기사님\s*통화", re.IGNORECASE),
        "기사님 위치는 언제 보여",
    ),
    (
        re.compile(r"고객\s*센터|상담사|서비스\s*문의", re.IGNORECASE),
        "카카오 퀵 고객센터 어디야",
    ),
]
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
    for pattern, question in _SUPPORT_PATTERNS:
        if pattern.search(prompt):
            return next(answer for item, answer in MOVB_QA if item == question)
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
