from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, TypedDict
from zoneinfo import ZoneInfo

os.environ.setdefault("LANGSMITH_TRACING", "false")

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from .providers import LLMRouter

from .client import KakaoApiError, KakaoMobilityClient
from .config import Settings
from .conversation_store import ConversationStore
from .directions import RoutePlanner
from .geocode import KakaoGeocodeClient
from .knowledge import MobilityKnowledgeBase, SERVICE_FACTS, default_knowledge_base
from .bundle import bundle_quote
from .models import (
    CreateDeliveryRequest,
    DeliveryDraft,
    Fleet,
    OrderType,
    PaymentType,
    ProductSize,
)
from .orders import cancel_order_by_id, get_order_steps, place_order
from .pool_store import PoolStore
from .pooling import build_pool_order, is_compatible, pool_quote
from .rideshare import carpool_plan
from .store import MobilityStore

MAX_HISTORY_TURNS = 6

KOREAN_FIELD_LABELS = {
    "pickup.location.basicAddress": "출발지 주소",
    "pickup.location.latitude": "출발지 좌표(위도) — 주소를 조금 더 구체적으로 알려주세요",
    "pickup.location.longitude": "출발지 좌표(경도) — 주소를 조금 더 구체적으로 알려주세요",
    "pickup.contact.name": "보내는 분 성함",
    "pickup.contact.phone": "보내는 분 연락처",
    "dropoff.location.basicAddress": "도착지 주소",
    "dropoff.location.latitude": "도착지 좌표(위도) — 주소를 조금 더 구체적으로 알려주세요",
    "dropoff.location.longitude": "도착지 좌표(경도) — 주소를 조금 더 구체적으로 알려주세요",
    "dropoff.contact.name": "받는 분 성함",
    "dropoff.contact.phone": "받는 분 연락처",
    "productName": "물품명",
}

CHITCHAT_SYSTEM = (
    "당신은 택시·퀵 합승 서비스 'MOVB(모브)'의 도우미입니다. "
    "친절하고 간결한 한국어로 답하세요. 아래 서비스 정보를 근거로 퀵/택시/합승 관련 "
    "질문에 구체적으로 답하고, 정보에 없는 내용은 지어내지 마세요. "
    "배송 주문을 원하면 출발지/도착지/물품 정보를 알려달라고 안내하세요.\n\n"
    + SERVICE_FACTS
)

KNOWLEDGE_SYSTEM = """당신은 MOVB AI 모빌리티 운영 서비스의 지식 안내자입니다.
반드시 제공된 근거 안에서만 한국어로 답하세요.
- 먼저 사용자의 질문에 직접 답합니다.
- 근거에 없는 실제 가격, 법적 제한, 운영 정책은 추측하지 않습니다.
- 필요한 경우 AI 채팅에서 이어서 할 수 있는 행동을 한 문장으로 안내합니다.
- 답변 끝에 사용한 근거 제목을 [제목] 형식으로 표시합니다.
"""

INTENT_PROMPT = """다음은 배송 주문 챗봇과 사용자의 대화입니다.

현재 단계: {stage}
지금까지 모은 정보: {slots}

최근 대화:
{history}

사용자의 새 메시지: {message}

사용자의 의도를 다음 중 하나의 단어로만 분류하세요:
- provide_info: 배송 정보(주소/물품/시간 등)를 알려주거나 새로 요청하는 경우
- modify: 이미 말한 정보를 수정/변경하는 경우
- confirm: 견적을 보고 주문을 확정/진행하겠다는 의사표시 (예/네/진행해줘/주문해줘)
- cancel: 주문 작성을 그만두거나, 이미 접수된 주문을 취소하려는 경우
- status_query: 주문 상태/배송 현황을 물어보는 경우
- bundle: 여러 도착지에 한꺼번에 보내는 묶음 배송 견적/할인을 물어보는 경우
- carpool: 동승/카풀 시 경유 순서나 요금을 어떻게 나눌지 물어보는 경우
- pool: 합승 배송 — 다른 사람 물건과 같은 차에 묶어서 싸게 보내려는 경우 ("합승으로", "같이 보내서 싸게" 등)
- question: 서비스 자체에 대한 궁금증 — 요금 체계, 배송 상품 차이, 합승/택시 동승 방식 등을 물어보는 경우 (지금 주문하려는 게 아님)
- vehicle_select: 주문에 사용할 차량 종류를 선택하거나 선택지를 보고 싶은 경우
- chitchat: 배송과 무관한 인사/잡담

단어 하나만 출력하세요."""

POOL_EXTRACT_PROMPT = """사용자 메시지에서 합승 배송 요청 정보를 JSON으로 추출하세요.

사용자 메시지: {message}

형식: {{"pickupAddress": "출발지 주소", "dropoffAddress": "도착지 주소",
"productName": "물품명", "productSize": "XS/S/M/L 중 하나(모르면 null)",
"senderName": "보내는 사람(모르면 null)", "senderPhone": "보내는 사람 연락처(모르면 null)",
"receiverName": "받는 사람(모르면 null)", "receiverPhone": "받는 사람 연락처(모르면 null)"}}
모르는 값은 null. JSON 객체 하나만 출력하세요."""

BUNDLE_EXTRACT_PROMPT = """사용자 메시지에서 묶음 배송 정보를 JSON으로 추출하세요.

사용자 메시지: {message}

형식: {{"pickup": "출발지 주소", "dropoffs": ["도착지 주소1", "도착지 주소2"]}}
모르는 값은 null로 두세요. JSON 객체 하나만 출력하세요."""

CARPOOL_EXTRACT_PROMPT = """사용자 메시지에서 동승(카풀) 정보를 JSON으로 추출하세요.

사용자 메시지: {message}

형식: {{"origin": "공통 출발지 주소", "passengers": [{{"name": "이름(모르면 null)", "destination": "목적지 주소"}}]}}
JSON 객체 하나만 출력하세요."""

SLOT_EXTRACT_PROMPT = """사용자의 배송 요청 메시지에서 알 수 있는 정보만 JSON으로 추출하세요.

이미 알고 있는 정보: {known_slots}

사용자 메시지: {message}

가능한 키 (알 수 있는 것만 포함, 모르면 키 자체를 넣지 마세요):
- orderType: QUICK, QUICK_ECONOMY, QUICK_EXPRESS, DOBO 중 하나
- productSize: XS, S, M, L 중 하나
- pickupAddress, pickupName, pickupPhone
- dropoffAddress, dropoffName, dropoffPhone
- productName, declaredValue(숫자), quantity, wishTime
- paymentType: CARD, CASH_ON_PICKUP, CASH_ON_DROPOFF 중 하나
- fleet: MOTORCYCLE, JIMBAJI_MOTORCYCLE, PASSENGER_CAR, DAMAS, LABO, TON 중 하나

JSON 객체 하나만 출력하세요. 설명 문장은 쓰지 마세요."""


class AgentState(TypedDict, total=False):
    session_id: str
    message: str
    turns: list[dict[str, str]]
    slots: dict[str, Any]
    stage: str
    intent: str
    missing_summary: str | None
    quote: dict[str, Any] | None
    quote_hash: str | None
    partner_order_id: str | None
    reply: str
    order: dict[str, Any] | None
    sources: list[dict[str, Any]]
    actions: list[dict[str, str]]
    trace: list[str]


@dataclass
class AgentChatResult:
    session_id: str
    reply: str
    stage: str
    slots: dict[str, Any] = field(default_factory=dict)
    quote: dict[str, Any] | None = None
    order: dict[str, Any] | None = None
    sources: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, str]] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "reply": self.reply,
            "stage": self.stage,
            "slots": self.slots,
            "quote": self.quote,
            "order": self.order,
            "sources": self.sources,
            "actions": self.actions,
            "trace": self.trace,
        }


def _slots_payload(slots: dict[str, Any]) -> dict[str, Any]:
    """flat slots dict를 CreateDeliveryRequest/DeliveryDraft가 받는 중첩 payload로 조립."""
    payload: dict[str, Any] = {
        "orderType": slots.get("orderType", "QUICK"),
        "productSize": slots.get("productSize", "XS"),
        "pickup": {
            "location": {
                "basicAddress": slots.get("pickupAddress"),
                "latitude": slots.get("pickupLat"),
                "longitude": slots.get("pickupLng"),
            },
            "contact": {
                "name": slots.get("pickupName"),
                "phone": slots.get("pickupPhone"),
            },
        },
        "dropoff": {
            "location": {
                "basicAddress": slots.get("dropoffAddress"),
                "latitude": slots.get("dropoffLat"),
                "longitude": slots.get("dropoffLng"),
            },
            "contact": {
                "name": slots.get("dropoffName"),
                "phone": slots.get("dropoffPhone"),
            },
        },
        "productName": slots.get("productName", "배송 물품"),
        "waypoints": [],
        "paymentType": slots.get("paymentType", "CARD"),
    }
    if slots.get("declaredValue") is not None:
        payload["declaredValue"] = slots["declaredValue"]
    if slots.get("quantity") is not None:
        payload["quantity"] = slots["quantity"]
    if slots.get("wishTime") is not None:
        payload["wishTime"] = slots["wishTime"]
    if slots.get("fleet") is not None:
        payload["fleetOption"] = {
            "fleet": slots["fleet"],
            "type": slots.get("fleetDispatchType", "REQUIRED"),
        }
    return payload


def _quote_hash(slots: dict[str, Any]) -> str:
    canonical = json.dumps(slots, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DeliveryAgent:
    """자연어로 카카오 T 배송 요청을 받아 견적을 보여주고, 확정 시 주문까지 만드는 LangGraph 에이전트."""

    def __init__(
        self,
        client: KakaoMobilityClient,
        geocoder: KakaoGeocodeClient,
        store: MobilityStore,
        conversations: ConversationStore,
        router: LLMRouter | None = None,
        pools: PoolStore | None = None,
        knowledge_base: MobilityKnowledgeBase | None = None,
        route_planner: RoutePlanner | None = None,
    ) -> None:
        self._client = client
        self._geocoder = geocoder
        self._store = store
        self._conversations = conversations
        self._router = router or LLMRouter()
        self._pools = pools
        self._knowledge = knowledge_base or default_knowledge_base()
        self._routes = route_planner
        self._graph = self._build_graph()

    async def _llm(
        self, prompt: str, system: str, max_tokens: int = 300, temperature: float = 0.2
    ) -> str:
        result = await asyncio.to_thread(
            self._router.generate,
            prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return result.text.strip()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("load_session", self._load_session)
        graph.add_node("classify_intent", self._classify_intent)
        graph.add_node("extract_slots", self._extract_slots)
        graph.add_node("geocode_addresses", self._geocode_addresses)
        graph.add_node("check_completeness", self._check_completeness)
        graph.add_node("ask_clarification", self._ask_clarification)
        graph.add_node("quote_price", self._quote_price)
        graph.add_node("confirm_and_create_order", self._confirm_and_create_order)
        graph.add_node("cancel_flow", self._cancel_flow)
        graph.add_node("status_query", self._status_query)
        graph.add_node("bundle_flow", self._bundle_flow)
        graph.add_node("carpool_flow", self._carpool_flow)
        graph.add_node("pool_flow", self._pool_flow)
        graph.add_node("knowledge_qa", self._knowledge_qa)
        graph.add_node("vehicle_select", self._vehicle_select)
        graph.add_node("chitchat", self._chitchat)
        graph.add_node("finalize", self._finalize)

        graph.add_edge(START, "load_session")
        graph.add_edge("load_session", "classify_intent")
        graph.add_conditional_edges(
            "classify_intent",
            self._route_by_intent,
            {
                "provide_info": "extract_slots",
                "modify": "extract_slots",
                "confirm": "confirm_and_create_order",
                "cancel": "cancel_flow",
                "status_query": "status_query",
                "bundle": "bundle_flow",
                "carpool": "carpool_flow",
                "pool": "pool_flow",
                "question": "knowledge_qa",
                "vehicle_select": "vehicle_select",
                "chitchat": "chitchat",
            },
        )
        graph.add_edge("extract_slots", "geocode_addresses")
        graph.add_edge("geocode_addresses", "check_completeness")
        graph.add_conditional_edges(
            "check_completeness",
            self._route_completeness,
            {"clarify": "ask_clarification", "quote": "quote_price"},
        )
        graph.add_edge("ask_clarification", "finalize")
        graph.add_edge("quote_price", "finalize")
        graph.add_edge("confirm_and_create_order", "finalize")
        graph.add_edge("cancel_flow", "finalize")
        graph.add_edge("status_query", "finalize")
        graph.add_edge("bundle_flow", "finalize")
        graph.add_edge("carpool_flow", "finalize")
        graph.add_edge("pool_flow", "finalize")
        graph.add_edge("knowledge_qa", "finalize")
        graph.add_edge("vehicle_select", "finalize")
        graph.add_edge("chitchat", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    # ── 노드 ────────────────────────────────────────────────────
    async def _load_session(self, state: AgentState) -> AgentState:
        session = self._conversations.get_or_create(state["session_id"])
        return {
            "slots": session["slots"],
            "turns": session["turns"],
            "stage": session["stage"],
            "quote": session["quote"],
            "quote_hash": session["quote_hash"],
            "partner_order_id": session["partner_order_id"],
            "trace": [f"load_session:{session['stage']}"],
        }

    @staticmethod
    def _history_text(state: AgentState) -> str:
        turns = state.get("turns", [])[-MAX_HISTORY_TURNS:]
        if not turns:
            return "(대화 이력 없음)"
        return "\n".join(f"{t['role']}: {t['content']}" for t in turns)

    @staticmethod
    def _heuristic_intent(message: str, stage: str) -> str | None:
        """명확한 업무 표현은 빠르고 재현 가능하게 분류하고 애매한 문장만 LLM에 맡긴다."""
        text = message.lower().strip()
        compact = re.sub(r"\s+", "", text)

        if stage in {"confirming", "pool_confirming", "pool_consent"} and compact in {
            "네", "예", "응", "ㅇㅇ", "그래", "진행", "진행해줘", "주문해줘", "좋아",
        }:
            return "confirm"
        if re.search(r"(취소|그만|철회|없던\s*일)", text):
            return "cancel"
        if re.search(r"(상태|현황|어디쯤|배송\s*조회|기사.*배정)", text):
            return "status_query"
        if re.fullmatch(r"(안녕|안녕하세요|하이|헬로|반가워)[!?.~ ]*", text):
            return "chitchat"
        if re.search(r"(차량|차종).*(선택|종류|골라|뭐가)", text) and not re.search(
            r"(오토바이|다마스|라보|1\s*톤|일톤|승용차|짐받이)", text
        ):
            return "vehicle_select"
        if re.search(
            r"(오토바이|다마스|라보|1\s*톤|일톤|승용차|짐받이).*(선택|배송|퀵|할래|해줘)",
            text,
        ):
            return "provide_info"

        definition_question = bool(
            re.search(r"(뭐야|무엇|뜻|차이|종류|설명|어떤\s*기능|어떻게\s*동작)", text)
        )
        action_request = bool(
            re.search(r"(보내|접수|등록|주문|견적|계산|나눠|매칭|싸게|진행)", text)
        )
        trip_details = bool(
            re.search(r"(에서|출발).*(으로|까지|목적지|도착)", text)
            or re.search(r"010[- ]?\d{4}", text)
        )

        if re.search(r"(택시\s*합승|카풀|동승)", text):
            return (
                "carpool"
                if trip_details or re.search(r"(계산해|경유\s*순서)", text)
                else "question"
            )
        if "묶음" in text:
            return "question" if definition_question and not action_request else "bundle"
        if re.search(r"(퀵\s*합승|배송\s*합승|합승)", text):
            return (
                "pool"
                if trip_details or re.search(r"(보내|접수|등록|매칭|싸게\s*보내)", text)
                else "question"
            )

        service_topic = bool(
            re.search(
                r"(movb|모브|퀵|도보\s*배송|배송\s*상품|물품\s*크기|sandbox|샌드박스|"
                r"요금|가격|결제|경유지|주문|배송|개인정보|관리자)",
                text,
            )
        )
        question_expression = bool(
            definition_question
            or re.search(r"(알려|궁금|가능해|되나요|돼요|인가요|왜|어떻게|얼마)", text)
            or "?" in text
        )
        if service_topic and question_expression and not re.search(
            r"(에서|부터).*(으로|까지).*(보내|배송)", text
        ):
            return "question"

        if re.search(r"(010[- ]?\d{4}|출발지|도착지|받는\s*사람|보내는\s*사람)", text):
            return "provide_info"
        if re.search(r"(퀵|도보).*(보내|접수|주문|배송)", text):
            return "provide_info"
        if re.search(r"(예약\s*배송).*(시작|접수|할래|하고\s*싶)", text):
            return "provide_info"
        return None

    async def _classify_intent(self, state: AgentState) -> AgentState:
        heuristic = self._heuristic_intent(
            state["message"], state.get("stage", "collecting")
        )
        if heuristic is not None:
            return {
                "intent": heuristic,
                "trace": state.get("trace", [])
                + [f"classify_intent:{heuristic}:heuristic"],
            }

        prompt = INTENT_PROMPT.format(
            stage=state.get("stage", "collecting"),
            slots=json.dumps(state.get("slots", {}), ensure_ascii=False),
            history=self._history_text(state),
            message=state["message"],
        )
        try:
            verdict = (
                await self._llm(
                    prompt, system="분류만 하는 어시스턴트입니다.", max_tokens=10, temperature=0.0
                )
            ).lower()
        except RuntimeError:
            verdict = ""

        intent = None
        for label in (
            "question",
            "provide_info",
            "modify",
            "confirm",
            "status_query",
            "bundle",
            "carpool",
            "pool",
            "cancel",
            "vehicle_select",
            "chitchat",
        ):
            if label in verdict:
                intent = label
                break
        if intent is None:
            intent = (
                "provide_info"
                if state.get("stage") not in {"confirming", "pool_confirming", "pool_consent"}
                else "chitchat"
            )

        return {
            "intent": intent,
            "trace": state.get("trace", []) + [f"classify_intent:{intent}:llm"],
        }

    @staticmethod
    def _route_by_intent(state: AgentState) -> str:
        return state.get("intent", "chitchat")

    @staticmethod
    def _coerce_slot_value(key: str, value: Any) -> Any | None:
        enum_map = {
            "orderType": OrderType,
            "productSize": ProductSize,
            "paymentType": PaymentType,
            "fleet": Fleet,
        }
        if key in enum_map:
            try:
                return enum_map[key](str(value).strip().upper()).value
            except ValueError:
                return None
        if key == "declaredValue":
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @staticmethod
    def _heuristic_slots(message: str) -> dict[str, Any]:
        """버튼 선택과 명확한 한국어 차량명을 LLM 없이도 슬롯으로 반영한다."""
        text = message.lower()
        slots: dict[str, Any] = {}
        for pattern, value in (
            (r"짐받이\s*오토바이", "JIMBAJI_MOTORCYCLE"),
            (r"오토바이|바이크", "MOTORCYCLE"),
            (r"승용차", "PASSENGER_CAR"),
            (r"다마스", "DAMAS"),
            (r"라보", "LABO"),
            (r"1\s*톤|일톤", "TON"),
        ):
            if re.search(pattern, text):
                slots["fleet"] = value
                break
        if "퀵 이코노미" in text or "퀵이코노미" in text:
            slots["orderType"] = "QUICK_ECONOMY"
        elif "퀵 급송" in text or "급송" in text:
            slots["orderType"] = "QUICK_EXPRESS"
        elif "도보" in text:
            slots["orderType"] = "DOBO"
            slots.pop("fleet", None)
        elif "퀵" in text or "배송" in text:
            slots["orderType"] = "QUICK"
        if "예약" in text:
            slots["_reservationRequested"] = True
        iso_time = re.search(
            r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?\b",
            message,
        )
        if iso_time:
            slots["wishTime"] = iso_time.group(0)
        size_match = re.search(r"(?:크기|사이즈)\s*(xs|s|m|l)\b", text, re.I)
        if size_match:
            slots["productSize"] = size_match.group(1).upper()
        return slots

    async def _extract_slots(self, state: AgentState) -> AgentState:
        slots = dict(state.get("slots", {}))
        heuristic_delta = self._heuristic_slots(state["message"])
        prompt = SLOT_EXTRACT_PROMPT.format(
            known_slots=json.dumps(slots, ensure_ascii=False), message=state["message"]
        )
        try:
            raw = await self._llm(
                prompt, system="JSON만 출력하는 정보 추출기입니다.", max_tokens=400, temperature=0.0
            )
        except RuntimeError:
            raw = "{}"

        match = re.search(r"\{.*\}", raw, re.S)
        delta: dict[str, Any] = {}
        if match:
            try:
                delta = json.loads(match.group())
            except json.JSONDecodeError:
                delta = {}

        allowed_keys = {
            "orderType", "productSize", "pickupAddress", "pickupName", "pickupPhone",
            "dropoffAddress", "dropoffName", "dropoffPhone", "productName",
            "declaredValue", "quantity", "wishTime", "paymentType", "fleet",
        }
        applied = []
        for key, value in heuristic_delta.items():
            slots[key] = value
            applied.append(key)
        for key, value in delta.items():
            if key not in allowed_keys or value in (None, ""):
                continue
            coerced = self._coerce_slot_value(key, value)
            if coerced is not None:
                slots[key] = coerced
                applied.append(key)

        self._conversations.save_slots(state["session_id"], slots, state.get("stage", "collecting"))
        return {"slots": slots, "trace": state.get("trace", []) + [f"extract_slots:{applied}"]}

    async def _geocode_addresses(self, state: AgentState) -> AgentState:
        slots = dict(state.get("slots", {}))
        for kind in ("pickup", "dropoff"):
            address_key = f"{kind}Address"
            lat_key, lng_key = f"{kind}Lat", f"{kind}Lng"
            geocoded_key = f"{kind}AddressGeocoded"
            address = slots.get(address_key)
            if not address:
                continue
            if slots.get(geocoded_key) == address and slots.get(lat_key) is not None:
                continue
            location = await self._geocoder.search_address(address)
            if location is None:
                continue
            slots[lat_key] = location.latitude
            slots[lng_key] = location.longitude
            slots[address_key] = location.basic_address
            slots[geocoded_key] = location.basic_address

        self._conversations.save_slots(state["session_id"], slots, state.get("stage", "collecting"))
        return {"slots": slots, "trace": state.get("trace", []) + ["geocode_addresses"]}

    async def _check_completeness(self, state: AgentState) -> AgentState:
        slots = state.get("slots", {})
        reservation_missing = bool(
            slots.get("_reservationRequested") and not slots.get("wishTime")
        )
        payload = _slots_payload(slots)
        try:
            CreateDeliveryRequest(**payload, partnerOrderId="validation-check-0000")
        except ValidationError as exc:
            lines: list[str] = ["- 픽업 예약 시간"] if reservation_missing else []
            seen: set[str] = set()
            for error in exc.errors():
                loc = ".".join(str(part) for part in error["loc"])
                if loc in seen:
                    continue
                seen.add(loc)
                label = KOREAN_FIELD_LABELS.get(loc)
                lines.append(f"- {label}" if label else f"- {error['msg']}")
            return {
                "missing_summary": "\n".join(lines),
                "trace": state.get("trace", []) + [f"check_completeness:invalid({len(lines)})"],
            }
        if reservation_missing:
            return {
                "missing_summary": "- 픽업 예약 시간",
                "trace": state.get("trace", [])
                + ["check_completeness:reservation_time_missing"],
            }
        return {
            "missing_summary": None,
            "trace": state.get("trace", []) + ["check_completeness:ok"],
        }

    @staticmethod
    def _route_completeness(state: AgentState) -> str:
        return "clarify" if state.get("missing_summary") else "quote"

    async def _ask_clarification(self, state: AgentState) -> AgentState:
        summary = state.get("missing_summary") or "필요한 정보가 더 있어요."
        reply = f"주문을 진행하려면 아래 정보가 더 필요해요.\n{summary}"
        self._conversations.save_slots(state["session_id"], state.get("slots", {}), "collecting")
        actions = [
            {"label": "차량 선택", "message": "차량 선택지를 보여줘"},
            {"label": "처음부터", "message": "주문 작성을 취소하고 처음부터 할래"},
        ]
        if state.get("slots", {}).get("_reservationRequested") and not state.get(
            "slots", {}
        ).get("wishTime"):
            tomorrow = datetime.now(ZoneInfo("Asia/Seoul")) + timedelta(days=1)
            tomorrow = tomorrow.replace(hour=15, minute=0, second=0, microsecond=0)
            actions.insert(
                0,
                {
                    "label": "내일 15시",
                    "message": f"예약 시간은 {tomorrow.isoformat()}로 할게",
                },
            )
        return {
            "reply": reply,
            "stage": "collecting",
            "actions": actions,
            "trace": state.get("trace", []) + ["ask_clarification"],
        }

    @staticmethod
    def _format_quote(data: Any) -> str:
        # `price`는 특정 orderType 하나에 대한 단일 확정 요금(totalPrice)을,
        # `estimate`는 여러 orderType/차량 옵션 비교 목록(lists)을 반환한다 (client.py 참고).
        if isinstance(data, dict) and "price" in data:
            price = data.get("price")
            estimate = data.get("estimate")
            route = data.get("route")
            lines = [DeliveryAgent._format_quote(price)]
            estimate_rows = (
                estimate.get("lists") if isinstance(estimate, dict) else None
            )
            if isinstance(estimate_rows, list) and estimate_rows:
                selected = estimate_rows[0]
                seconds = int(selected.get("estimatedTime") or 0)
                fleet_option = selected.get("fleetOption")
                fleet = (
                    fleet_option.get("fleet")
                    if isinstance(fleet_option, dict)
                    else selected.get("fleet")
                )
                if seconds:
                    lines.append(
                        f"- 배송 예상 시간: 약 {max(1, round(seconds / 60))}분"
                    )
                if fleet:
                    lines.append(f"- 카카오 추천 차량: {fleet}")
            if isinstance(route, dict):
                source = (
                    "카카오 실도로"
                    if route.get("actualRoadData")
                    else "보정 거리(길찾기 키 미연결)"
                )
                future = (
                    " · 예약 교통량 반영"
                    if route.get("futureTrafficApplied")
                    else ""
                )
                lines.append(
                    f"- 이동 경로: {route.get('distanceKm', 0)}km · "
                    f"약 {route.get('durationMinutes', 0)}분 ({source}{future})"
                )
            return "\n".join(lines)

        if isinstance(data, dict) and isinstance(data.get("totalPrice"), (int, float)):
            return f"- 예상 요금: {int(data['totalPrice']):,}원"

        rows = data.get("lists") if isinstance(data, dict) else None
        if not isinstance(rows, list) or not rows:
            return (
                "가격 정보를 확인했지만 형식을 파악하지 못했어요. 원본 결과: "
                + json.dumps(data, ensure_ascii=False)[:500]
            )
        lines = []
        for row in sorted(
            (r for r in rows if isinstance(r, dict)),
            key=lambda r: r.get("totalFareAmount") or 0,
        ):
            fare = row.get("totalFareAmount")
            order_type = row.get("orderType", "")
            fleet_option = row.get("fleetOption")
            fleet = (
                fleet_option.get("fleet")
                if isinstance(fleet_option, dict)
                else row.get("fleet", "")
            )
            if isinstance(fare, (int, float)):
                lines.append(f"- {order_type} ({fleet}): {int(fare):,}원")
            else:
                lines.append(f"- {order_type} ({fleet}): 가격 정보 없음")
        return "\n".join(lines) if lines else "가격 옵션을 찾지 못했어요."

    async def _quote_price(self, state: AgentState) -> AgentState:
        slots = state.get("slots", {})
        payload = _slots_payload(slots)
        try:
            draft = DeliveryDraft(**payload)
        except ValidationError:
            return {
                "reply": "죄송해요, 입력하신 정보로 견적을 계산할 수 없었어요. 다시 한 번 말씀해주시겠어요?",
                "stage": "collecting",
                "trace": state.get("trace", []) + ["quote_price:draft_invalid"],
            }
        try:
            price = await self._client.price(draft)
        except KakaoApiError as exc:
            return {
                "reply": f"가격 조회 중 문제가 있었어요: {exc}",
                "stage": "collecting",
                "trace": state.get("trace", []) + ["quote_price:api_error"],
            }

        estimate = None
        try:
            estimate = await self._client.estimate(draft)
        except KakaoApiError:
            pass
        route = (
            await self._routes.route_summary(
                draft.pickup.location,
                draft.dropoff.location,
                waypoints=[item.location for item in draft.waypoints],
                departure_time=draft.wish_time,
            )
            if self._routes
            else None
        )
        data = {"price": price, "estimate": estimate, "route": route}
        quote_hash = _quote_hash(slots)
        self._conversations.save_quote(state["session_id"], data, quote_hash)
        summary = self._format_quote(data)
        reply = f"견적을 확인했어요!\n{summary}\n\n이대로 진행할까요? '네' 또는 '진행해줘'라고 답해주세요."
        return {
            "quote": data,
            "quote_hash": quote_hash,
            "stage": "confirming",
            "reply": reply,
            "actions": [
                {"label": "이대로 접수", "message": "네, 이대로 주문해줘"},
                {"label": "차량 변경", "message": "차량 선택지를 보여줘"},
                {"label": "취소", "message": "주문 작성을 취소할래"},
            ],
            "trace": state.get("trace", []) + ["quote_price:ok"],
        }

    async def _confirm_and_create_order(self, state: AgentState) -> AgentState:
        slots = state.get("slots", {})
        stage = state.get("stage")

        # 합승 분기: 매칭 제안에 대한 확정 / 자동진행 사전 동의
        if stage == "pool_confirming":
            return await self._execute_pool_order(state)
        if stage == "pool_consent":
            slots = dict(slots)
            request_id = slots.get("_poolMyRequestId")
            if self._pools is not None and request_id:
                self._pools.set_auto_consent(int(request_id), True)
            self._conversations.save_slots(state["session_id"], slots, "collecting")
            return {
                "slots": slots,
                "stage": "collecting",
                "reply": "네! 같은 방향 합승이 들어오면 자동으로 묶어서 접수하고, 다음 대화에서 결과를 알려드릴게요.",
                "trace": state.get("trace", []) + ["pool_consent:granted"],
            }

        if state.get("stage") != "confirming":
            return {
                "reply": "아직 확인할 주문 내용이 없어요. 먼저 배송 정보를 알려주시겠어요? (출발지, 도착지, 물품 등)",
                "trace": state.get("trace", []) + ["confirm:no_quote_yet"],
            }

        current_hash = _quote_hash(slots)
        if current_hash != state.get("quote_hash"):
            # 견적을 보여준 이후 정보가 바뀌었으므로, 예전 가격으로 주문하지 않도록 재견적부터 다시 수행
            completeness = await self._check_completeness(state)
            if completeness.get("missing_summary"):
                return await self._ask_clarification({**state, **completeness})
            return await self._quote_price(state)

        payload = _slots_payload(slots)
        partner_order_id = f"agent-{state['session_id'][:10]}-{current_hash[:12]}"
        try:
            request = CreateDeliveryRequest(**payload, partnerOrderId=partner_order_id)
        except ValidationError:
            return {
                "reply": "주문 정보가 유효하지 않아요. 다시 확인해주시겠어요?",
                "stage": "collecting",
                "trace": state.get("trace", []) + ["confirm:invalid_request"],
            }

        try:
            result = await place_order(self._client, self._store, request, partner_order_id)
        except KakaoApiError as exc:
            return {
                "reply": f"주문 접수 중 문제가 발생했어요: {exc}\n잠시 후 다시 시도해주시겠어요?",
                "trace": state.get("trace", []) + ["confirm:api_error"],
            }

        order_id = result.get("partnerOrderId") or partner_order_id
        self._conversations.set_partner_order_id(state["session_id"], order_id)
        reply = f"주문이 접수됐어요! 주문번호: {order_id}\n'상태 확인해줘'라고 물어보시면 진행 상황을 알려드릴게요."
        return {
            "order": result,
            "partner_order_id": order_id,
            "stage": "placed",
            "reply": reply,
            "actions": [
                {"label": "상세 배송 상태", "message": "출발지부터 목적지까지 상세 상태 보여줘"},
                {"label": "주문 취소", "message": "이 주문 취소해줘"},
            ],
            "trace": state.get("trace", []) + ["confirm:order_created"],
        }

    async def _cancel_flow(self, state: AgentState) -> AgentState:
        stage = state.get("stage", "collecting")
        session_id = state["session_id"]
        if stage == "placed" and state.get("partner_order_id"):
            try:
                await cancel_order_by_id(self._client, self._store, state["partner_order_id"])
            except KakaoApiError as exc:
                return {
                    "reply": f"주문 취소 중 문제가 발생했어요: {exc}",
                    "trace": state.get("trace", []) + ["cancel:api_error"],
                }
            self._conversations.reset_draft(session_id)
            return {
                "reply": "주문을 취소했어요.",
                "stage": "collecting",
                "slots": {},
                "trace": state.get("trace", []) + ["cancel:order_canceled"],
            }

        self._conversations.reset_draft(session_id)
        canceled_pools = (
            self._pools.cancel_open_by_session(session_id) if self._pools else 0
        )
        pool_text = f" 대기 중이던 합승 요청 {canceled_pools}건도 취소했어요." if canceled_pools else ""
        return {
            "reply": f"주문 작성을 취소했어요.{pool_text} 처음부터 다시 시작할 수 있어요.",
            "stage": "collecting",
            "slots": {},
            "trace": state.get("trace", []) + ["cancel:draft_reset"],
        }

    def _pool_status_lines(self, session_id: str) -> list[str]:
        """이 세션의 합승 요청 현황을 요약한다 (status_query 브리핑용)."""
        if self._pools is None:
            return []
        lines = []
        for request in self._pools.get_by_session(session_id):
            route = f"{request['pickup']['address']} → {request['dropoff']['address']}"
            if request["status"] == "open":
                lines.append(f"- [합승 대기중] {route}")
            elif request["status"] == "ordered":
                share = f", 분담금 {request['sharePrice']:,}원" if request["sharePrice"] else ""
                lines.append(f"- [합승 매칭 완료!] {route} (주문번호 {request['poolId']}{share})")
        return lines

    async def _status_query(self, state: AgentState) -> AgentState:
        pool_lines = self._pool_status_lines(state["session_id"])
        partner_order_id = state.get("partner_order_id")
        explicit_id = re.search(
            r"\b(?:agent|moveops|pool)-[A-Za-z0-9._-]+\b",
            state["message"],
        )
        if explicit_id:
            partner_order_id = explicit_id.group(0)
        if not partner_order_id:
            if pool_lines:
                return {
                    "reply": "현재 합승 요청 현황이에요.\n" + "\n".join(pool_lines),
                    "trace": state.get("trace", []) + ["status_query:pool_only"],
                }
            return {
                "reply": "아직 이 대화에서 접수된 주문이 없어요. 주문번호가 있다면 알려주세요.",
                "actions": [
                    {"label": "배송 주문 시작", "message": "퀵 배송 주문하고 싶어"},
                    {"label": "합승 현황", "message": "합승 배송 현황 알려줘"},
                ],
                "trace": state.get("trace", []) + ["status_query:no_order"],
            }
        try:
            details = await get_order_steps(
                self._client,
                self._store,
                partner_order_id,
                refresh_order=True,
            )
        except KakaoApiError as exc:
            return {
                "reply": f"상태 확인 중 문제가 발생했어요: {exc}",
                "trace": state.get("trace", []) + ["status_query:api_error"],
            }
        order = details.get("order")
        status = order.get("status") if order else None
        reply = (
            f"주문번호 {partner_order_id}의 현재 상태는 '{status}'예요."
            if status
            else "주문 상태를 찾지 못했어요."
        )
        steps = details.get("steps") or []
        if steps:
            status_labels = {
                "waiting": "대기",
                "started": "진행 중",
                "completed": "완료",
                "WAITING": "대기",
                "STARTED": "진행 중",
                "COMPLETED": "완료",
            }
            kind_labels = {
                "PICKUP": "출발지",
                "DROPOFF": "목적지",
            }
            step_lines = []
            for step in steps:
                kind = step.get("kind", "STEP")
                label = kind_labels.get(
                    kind,
                    kind.replace("WAYPOINT_", "경유지 "),
                )
                step_status = str(step.get("status") or "UNKNOWN")
                friendly = status_labels.get(step_status, step_status)
                eta = step.get("estimatedEndedAt")
                eta_text = f" · 예상 {eta}" if eta else ""
                step_lines.append(f"- {label}: {friendly}{eta_text}")
            reply += "\n\n정차지별 상세 상태\n" + "\n".join(step_lines)
        if pool_lines:
            reply += "\n" + "\n".join(pool_lines)
        return {
            "reply": reply,
            "actions": [
                {"label": "상태 새로고침", "message": "배송 상태 다시 확인해줘"},
                {"label": "배송원 확인", "message": "기사 배정 상태 알려줘"},
            ],
            "trace": state.get("trace", [])
            + [f"status_query:ok(steps={len(steps)})"],
        }

    async def _llm_json(self, prompt: str) -> dict[str, Any]:
        try:
            raw = await self._llm(
                prompt, system="JSON만 출력하는 정보 추출기입니다.", max_tokens=400, temperature=0.0
            )
        except RuntimeError:
            return {}
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    async def _bundle_flow(self, state: AgentState) -> AgentState:
        """묶음 배송: 따로 보낼 때 vs 한 번에 묶어 보낼 때 요금 비교 + 묶음 할인."""
        parsed = await self._llm_json(BUNDLE_EXTRACT_PROMPT.format(message=state["message"]))
        pickup = parsed.get("pickup")
        dropoffs = [d for d in (parsed.get("dropoffs") or []) if isinstance(d, str) and d.strip()]
        if not pickup or len(dropoffs) < 2:
            return {
                "reply": (
                    "묶음 배송 견적을 내려면 출발지 1곳과 도착지 2곳 이상이 필요해요.\n"
                    "예: \"판교역에서 정자역이랑 서현역으로 서류 보낼 건데 묶으면 얼마야?\""
                ),
                "trace": state.get("trace", []) + ["bundle:need_info"],
            }
        try:
            result = await bundle_quote(
                self._client,
                self._geocoder,
                pickup,
                dropoffs,
                route_planner=self._routes,
            )
        except (ValueError, KakaoApiError) as exc:
            return {
                "reply": f"묶음 견적 계산 중 문제가 있었어요: {exc}",
                "trace": state.get("trace", []) + ["bundle:error"],
            }

        individual_lines = "\n".join(
            f"- {item['address']}: {item['price']:,}원" for item in result["individual"]
        )
        route_text = " → ".join(result["route"])
        route_info = result.get("routeInfo") or {}
        road_text = (
            f"실도로 {route_info.get('distanceKm')}km · "
            f"약 {route_info.get('durationMinutes')}분\n"
            if route_info
            else ""
        )
        reply = (
            f"묶음 배송 비교 견적이에요! (출발: {result['pickup']})\n\n"
            f"[각각 따로 보낼 때]\n{individual_lines}\n"
            f"합계: {result['individualTotal']:,}원\n\n"
            f"[한 번에 묶어 보낼 때 — 경유지 추가 혜택 적용]\n"
            f"추천 경유 순서: {route_text}\n"
            f"{road_text}"
            f"묶음 견적: {result['bundledPrice']:,}원\n\n"
        )
        if result["recommendBundle"]:
            reply += f"→ 묶어서 보내면 {result['saving']:,}원 절약돼요! 이대로 접수할까요?"
        else:
            reply += "→ 이번 건은 따로 보내는 게 더 유리해요."
        return {
            "quote": result,
            "reply": reply,
            "trace": state.get("trace", []) + [f"bundle:ok({len(dropoffs)}곳)"],
        }

    async def _carpool_flow(self, state: AgentState) -> AgentState:
        """동승(카풀): 누구를 먼저 내려줄지 경유 순서 + 요금 분배안 계산."""
        parsed = await self._llm_json(CARPOOL_EXTRACT_PROMPT.format(message=state["message"]))
        origin_address = parsed.get("origin")
        raw_passengers = parsed.get("passengers") or []
        passengers_in = [
            p for p in raw_passengers
            if isinstance(p, dict) and isinstance(p.get("destination"), str) and p["destination"].strip()
        ]
        if not origin_address or len(passengers_in) < 2:
            return {
                "reply": (
                    "동승 요금을 나누려면 공통 출발지와 각자의 목적지(2곳 이상)가 필요해요.\n"
                    "예: \"강남역에서 A는 잠실역, B는 건대입구역 가는데 요금 어떻게 나눠?\""
                ),
                "trace": state.get("trace", []) + ["carpool:need_info"],
            }

        origin = await self._geocoder.search_address(origin_address)
        if origin is None:
            return {
                "reply": f"출발지 주소를 찾지 못했어요: {origin_address}",
                "trace": state.get("trace", []) + ["carpool:geocode_fail"],
            }
        passengers = []
        for p in passengers_in:
            location = await self._geocoder.search_address(p["destination"])
            if location is None:
                return {
                    "reply": f"목적지 주소를 찾지 못했어요: {p['destination']}",
                    "trace": state.get("trace", []) + ["carpool:geocode_fail"],
                }
            passengers.append({"name": p.get("name"), "location": location})

        try:
            plan = await carpool_plan(
                origin,
                passengers,
                route_planner=self._routes,
            )
        except ValueError as exc:
            return {
                "reply": str(exc),
                "trace": state.get("trace", []) + ["carpool:error"],
            }

        stop_lines = "\n".join(
            f"{s['dropOrder']}번째 하차 — {s['name']} ({s['address']})\n"
            f"   분담금 {s['share']:,}원 (혼자 타면 {s['soloFare']:,}원 → {s['saving']:,}원 절약)"
            for s in plan["stops"]
        )
        reply = (
            f"동승 플랜을 짜봤어요! (출발: {origin.basic_address}, 총 이동 약 {plan['totalKm']}km)\n\n"
            f"{stop_lines}\n\n"
            f"예상 총 요금: {plan['sharedFare']:,}원 → 다 같이 {plan['groupSaving']:,}원 절약!\n"
            f"※ {plan['note']}"
        )
        return {
            "reply": reply,
            "trace": state.get("trace", []) + [f"carpool:ok({len(passengers)}명)"],
        }

    # ── 퀵 합승 (패키지 카풀) ──────────────────────────────────
    POOL_REQUIRED = {
        "pickupAddress": "출발지 주소",
        "dropoffAddress": "도착지 주소",
        "productName": "물품명",
        "senderName": "보내는 분 성함",
        "senderPhone": "보내는 분 연락처",
        "receiverName": "받는 분 성함",
        "receiverPhone": "받는 분 연락처",
    }

    async def _pool_flow(self, state: AgentState) -> AgentState:
        """합승 요청 등록 → 같은 방향 대기 건과 매칭 → 분담금 제안."""
        if self._pools is None:
            return {
                "reply": "합승 기능이 아직 준비되지 않았어요.",
                "trace": state.get("trace", []) + ["pool:unavailable"],
            }
        slots = dict(state.get("slots", {}))
        draft = dict(slots.get("_poolDraft", {}))
        parsed = await self._llm_json(POOL_EXTRACT_PROMPT.format(message=state["message"]))
        for key in self.POOL_REQUIRED:
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                draft[key] = value.strip()
        if isinstance(parsed.get("productSize"), str):
            try:
                draft["productSize"] = ProductSize(parsed["productSize"].strip().upper()).value
            except ValueError:
                pass

        missing = [label for key, label in self.POOL_REQUIRED.items() if not draft.get(key)]
        slots["_poolDraft"] = draft
        if missing:
            self._conversations.save_slots(state["session_id"], slots, state.get("stage", "collecting"))
            return {
                "slots": slots,
                "reply": "합승 등록에 아래 정보가 더 필요해요.\n" + "\n".join(f"- {m}" for m in missing),
                "trace": state.get("trace", []) + [f"pool:need_info({len(missing)})"],
            }

        pickup_loc = await self._geocoder.search_address(draft["pickupAddress"])
        dropoff_loc = await self._geocoder.search_address(draft["dropoffAddress"])
        if pickup_loc is None or dropoff_loc is None:
            bad = draft["pickupAddress"] if pickup_loc is None else draft["dropoffAddress"]
            return {
                "reply": f"주소를 찾지 못했어요: {bad}. 조금 더 구체적으로 알려주시겠어요?",
                "trace": state.get("trace", []) + ["pool:geocode_fail"],
            }

        pickup = {
            "address": pickup_loc.basic_address, "lat": pickup_loc.latitude,
            "lng": pickup_loc.longitude, "name": draft["senderName"], "phone": draft["senderPhone"],
        }
        dropoff = {
            "address": dropoff_loc.basic_address, "lat": dropoff_loc.latitude,
            "lng": dropoff_loc.longitude, "name": draft["receiverName"], "phone": draft["receiverPhone"],
        }
        product = {
            "productName": draft["productName"],
            "productSize": draft.get("productSize", "XS"),
        }

        try:
            solo_data = await self._client.price(DeliveryDraft(
                orderType=OrderType.QUICK,
                productSize=ProductSize(product["productSize"]),
                pickup={"location": {"basicAddress": pickup["address"], "latitude": pickup["lat"], "longitude": pickup["lng"]}},
                dropoff={"location": {"basicAddress": dropoff["address"], "latitude": dropoff["lat"], "longitude": dropoff["lng"]}},
            ))
            solo_price = solo_data.get("totalPrice") if isinstance(solo_data, dict) else None
        except KakaoApiError:
            solo_price = None

        my_request = self._pools.create_request(
            state["session_id"], pickup, dropoff, product,
            int(solo_price) if solo_price else None,
        )

        candidates = []
        for request in self._pools.list_open(exclude_session=state["session_id"]):
            if await is_compatible(request, my_request, self._routes):
                candidates.append(request)
        if not candidates:
            slots["_poolMyRequestId"] = my_request["id"]
            slots.pop("_poolPending", None)
            self._conversations.save_slots(state["session_id"], slots, "pool_consent")
            solo_text = f"(단독 견적 {int(solo_price):,}원) " if solo_price else ""
            return {
                "slots": slots,
                "stage": "pool_consent",
                "reply": (
                    f"지금은 같은 방향으로 가는 합승 대기 건이 없어요. {solo_text}합승 대기로 등록해뒀어요!\n"
                    "같은 방향 요청이 들어와서 매칭되면 확인 없이 바로 진행해도 될까요? ('네'라고 하시면 자동 진행돼요)"
                ),
                "trace": state.get("trace", []) + ["pool:registered_waiting"],
            }

        other = candidates[0]
        try:
            quote = await pool_quote(
                self._client,
                [other, my_request],
                route_planner=self._routes,
            )
        except (ValueError, KakaoApiError) as exc:
            return {
                "reply": f"합승 견적 계산 중 문제가 있었어요: {exc}",
                "trace": state.get("trace", []) + ["pool:quote_error"],
            }

        slots["_poolPending"] = {
            "requestIds": [other["id"], my_request["id"]],
            "pooledPrice": quote["pooledPrice"],
            "shares": {str(other["id"]): quote["shares"][0], str(my_request["id"]): quote["shares"][1]},
        }
        slots["_poolMyRequestId"] = my_request["id"]
        self._conversations.save_slots(state["session_id"], slots, "pool_confirming")
        reply = (
            f"🚚 같은 방향 합승 상대를 찾았어요!\n"
            f"- 대기 중: {other['pickup']['address']} → {other['dropoff']['address']} ({other['product'].get('productName', '물품')})\n"
            f"- 내 요청: {pickup['address']} → {dropoff['address']} ({product['productName']})\n\n"
            f"[요금 비교]\n"
            f"- 각자 따로: {quote['soloTotal']:,}원 (상대 {quote['soloPrices'][0]:,}원 + 나 {quote['soloPrices'][1]:,}원)\n"
            f"- 합승 1건: {quote['pooledPrice']:,}원\n"
            f"- 내 분담금: {quote['shares'][1]:,}원 (단독 대비 {quote['savings'][1]:,}원 절약)\n"
            f"- 상대 분담금: {quote['shares'][0]:,}원 ({quote['savings'][0]:,}원 절약)\n\n"
            f"합승으로 진행할까요? '네'라고 하면 접수돼요."
        )
        return {
            "slots": slots,
            "stage": "pool_confirming",
            "reply": reply,
            "trace": state.get("trace", []) + [f"pool:matched(req={my_request['id']}↔{other['id']})"],
        }

    async def _execute_pool_order(self, state: AgentState) -> AgentState:
        """합승 확정: 두 요청을 경유지로 묶은 Sandbox 주문 1건을 생성하고 분담금을 기록한다."""
        slots = dict(state.get("slots", {}))
        pending = slots.get("_poolPending") or {}
        request_ids = pending.get("requestIds") or []
        requests = [self._pools.get_request(rid) for rid in request_ids]
        if len(requests) != 2 or any(r is None for r in requests):
            return {"reply": "합승 정보를 찾지 못했어요. 다시 시도해주세요.", "stage": "collecting",
                    "trace": state.get("trace", []) + ["pool_confirm:missing"]}
        other, mine = requests
        if other["status"] != "open":
            slots.pop("_poolPending", None)
            self._conversations.save_slots(state["session_id"], slots, "collecting")
            return {"slots": slots, "stage": "collecting",
                    "reply": "아쉽지만 상대 합승 건이 방금 취소되거나 이미 접수됐어요. 대기 목록을 다시 확인해볼게요 — 합승 요청을 다시 말씀해주세요.",
                    "trace": state.get("trace", []) + ["pool_confirm:other_gone"]}

        id_seed = "-".join(str(rid) for rid in sorted(request_ids))
        pool_id = f"pool-{hashlib.sha256(id_seed.encode()).hexdigest()[:12]}"
        try:
            order_request = build_pool_order(requests, pool_id)
        except ValidationError:
            return {"reply": "합승 주문 정보가 유효하지 않아요 (연락처 누락 등). 다시 확인해주세요.",
                    "trace": state.get("trace", []) + ["pool_confirm:invalid"]}
        try:
            result = await place_order(self._client, self._store, order_request, pool_id)
        except KakaoApiError as exc:
            return {"reply": f"합승 주문 접수 중 문제가 발생했어요: {exc}",
                    "trace": state.get("trace", []) + ["pool_confirm:api_error"]}

        shares = {int(k): v for k, v in (pending.get("shares") or {}).items()}
        self._pools.mark_ordered(request_ids, pool_id, shares)
        self._conversations.set_partner_order_id(state["session_id"], pool_id)
        self._conversations.set_partner_order_id(other["sessionId"], pool_id)
        slots.pop("_poolPending", None)
        slots.pop("_poolDraft", None)
        self._conversations.save_slots(state["session_id"], slots, "placed")
        my_share = shares.get(mine["id"])
        share_text = f"내 분담금은 {my_share:,}원이에요. " if my_share else ""
        return {
            "slots": slots, "order": result, "partner_order_id": pool_id, "stage": "placed",
            "reply": f"합승 주문이 접수됐어요! 주문번호: {pool_id}\n{share_text}상대방도 다음 대화에서 매칭 결과를 확인하게 돼요.",
            "trace": state.get("trace", []) + ["pool_confirm:order_created"],
        }

    async def _knowledge_qa(self, state: AgentState) -> AgentState:
        """MOVB 문서에서 근거를 검색하고, LLM이 없을 때도 추출형 답변을 제공한다."""
        results = self._knowledge.search(state["message"], limit=3)
        sources = [result.to_source() for result in results]
        if not results:
            return {
                "reply": self._knowledge.fallback_answer(results),
                "sources": [],
                "trace": state.get("trace", []) + ["knowledge_qa:no_match"],
            }

        prompt = (
            f"사용자 질문:\n{state['message']}\n\n"
            f"검색된 MOVB 근거:\n{self._knowledge.context(results)}"
        )
        try:
            reply = await self._llm(
                prompt,
                system=KNOWLEDGE_SYSTEM,
                max_tokens=450,
                temperature=0.1,
            )
            cited_titles = [result.title for result in results[:2]]
            if not any(f"[{title}]" in reply for title in cited_titles):
                reply += "\n\n근거 문서: " + ", ".join(
                    f"[{title}]" for title in cited_titles
                )
            generation = "llm"
        except RuntimeError:
            reply = self._knowledge.fallback_answer(results)
            generation = "extractive"

        return {
            "reply": reply,
            "sources": sources,
            "actions": [
                {"label": "배송 주문", "message": "퀵 배송 주문하고 싶어"},
                {"label": "차량 선택", "message": "차량 선택지를 보여줘"},
                {"label": "예약 ETA", "message": "예약 배송을 시작하고 싶어"},
            ],
            "trace": state.get("trace", [])
            + [f"knowledge_qa:{generation}:{results[0].chunk_id}"],
        }

    async def _vehicle_select(self, state: AgentState) -> AgentState:
        slots = state.get("slots", {})
        selected = slots.get("fleet")
        selected_text = f"\n현재 선택: **{selected}**" if selected else ""
        return {
            "reply": (
                "카카오 T 퀵에서 사용할 차량을 골라주세요.\n"
                "- 오토바이: 서류·소형 물품\n"
                "- 다마스: 부피 있는 중소형 짐\n"
                "- 라보: 더 큰 적재 공간이 필요한 짐\n"
                "- 1톤: 대형·중량 화물\n\n"
                "최종 배차 가능 여부와 요금은 카카오 Sandbox 견적에서 확인합니다."
                f"{selected_text}"
            ),
            "actions": [
                {"label": "오토바이", "message": "오토바이로 퀵 배송할래"},
                {"label": "다마스", "message": "다마스로 퀵 배송할래"},
                {"label": "라보", "message": "라보로 퀵 배송할래"},
                {"label": "1톤", "message": "1톤으로 퀵 배송할래"},
            ],
            "trace": state.get("trace", []) + ["vehicle_select"],
        }

    async def _chitchat(self, state: AgentState) -> AgentState:
        try:
            reply = await self._llm(state["message"], system=CHITCHAT_SYSTEM, max_tokens=200)
        except RuntimeError:
            reply = (
                "안녕하세요! MOVB에서는 퀵 견적과 주문, 묶음배송, 퀵 합승, "
                "택시 동승 요금 계산을 도와드려요. "
                "예를 들어 “퀵과 도보 배송은 뭐가 달라?” 또는 "
                "“판교역에서 정자역으로 서류 보내줘”라고 말씀해보세요."
            )
        return {
            "reply": reply,
            "actions": [
                {"label": "배송 주문", "message": "퀵 배송 주문하고 싶어"},
                {"label": "차량 선택", "message": "차량 선택지를 보여줘"},
                {"label": "예약 ETA", "message": "예약 배송을 시작하고 싶어"},
                {"label": "배송 상태", "message": "배송 상태를 확인하고 싶어"},
            ],
            "trace": state.get("trace", []) + ["chitchat"],
        }

    async def _finalize(self, state: AgentState) -> AgentState:
        session_id = state["session_id"]
        self._conversations.append_turn(session_id, "user", state["message"])
        self._conversations.append_turn(session_id, "assistant", state.get("reply", ""))
        return {}

    # ── 실행 ────────────────────────────────────────────────────
    async def achat(self, session_id: str, message: str) -> AgentChatResult:
        state = await self._graph.ainvoke(
            {"session_id": session_id, "message": message},
            config={
                "run_name": "delivery-agent",
                "tags": ["mobility", "agent", "delivery-chat"],
                "metadata": {"session_id": session_id},
            },
        )
        return AgentChatResult(
            session_id=session_id,
            reply=state.get("reply", "죄송해요, 응답을 만들지 못했어요."),
            stage=state.get("stage", "collecting"),
            slots=state.get("slots", {}),
            quote=state.get("quote"),
            order=state.get("order"),
            sources=state.get("sources", []),
            actions=state.get("actions", []),
            trace=state.get("trace", []),
        )


if __name__ == "__main__":
    import uuid

    async def _main() -> None:
        settings = Settings.from_env()
        client = KakaoMobilityClient(settings)
        geocoder = KakaoGeocodeClient(settings)
        store = MobilityStore(settings.database_path)
        conversations = ConversationStore(settings.database_path)
        agent = DeliveryAgent(client, geocoder, store, conversations)
        message = sys.argv[1] if len(sys.argv) > 1 else "판교역에서 정자동으로 서류 하나 퀵으로 보내줘"
        result = await agent.achat(str(uuid.uuid4()), message)
        print(result.reply)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        await client.close()
        await geocoder.close()

    asyncio.run(_main())
