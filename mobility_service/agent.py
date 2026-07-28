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
    "당신은 스마트 딜리버리 서비스 'MOVB(모브)'의 도우미입니다. "
    "친절하고 간결한 한국어로 답하세요. 아래 서비스 정보를 근거로 스마트 딜리버리와 배송 관련 "
    "질문에 구체적으로 답하고, 정보에 없는 내용은 지어내지 마세요. "
    "배송 주문을 원하면 출발지/도착지/물품 정보를 알려달라고 안내하세요.\n\n"
    + SERVICE_FACTS
)

KNOWLEDGE_SYSTEM = """당신은 MOVB AI 모빌리티 운영 서비스의 지식 안내자입니다.
반드시 제공된 근거 안에서만 한국어로 답하세요.
- 먼저 사용자의 질문에 직접 답합니다.
- 근거에 없는 실제 가격, 법적 제한, 운영 정책은 추측하지 않습니다.
- 필요한 경우 AI 채팅에서 이어서 할 수 있는 행동을 한 문장으로 안내합니다.
- 내부 문서 제목이나 출처 표시는 답변 본문에 노출하지 않습니다.
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
- bundle: 여러 사람의 배송을 한 동선으로 묶는 스마트 딜리버리 견적/할인을 물어보는 경우
- question: 서비스 자체에 대한 궁금증 — 요금 체계, 배송 상품 차이, 스마트 딜리버리 방식 등을 물어보는 경우 (지금 주문하려는 게 아님)
- vehicle_select: 주문에 사용할 차량 종류를 선택하거나 선택지를 보고 싶은 경우
- chitchat: 배송과 무관한 인사/잡담

단어 하나만 출력하세요."""

BUNDLE_EXTRACT_PROMPT = """사용자 메시지에서 묶음 배송 정보를 JSON으로 추출하세요.

사용자 메시지: {message}

형식: {{"pickup": "출발지 주소", "dropoffs": ["도착지 주소1", "도착지 주소2"]}}
모르는 값은 null로 두세요. JSON 객체 하나만 출력하세요."""

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
        knowledge_base: MobilityKnowledgeBase | None = None,
        route_planner: RoutePlanner | None = None,
    ) -> None:
        self._client = client
        self._geocoder = geocoder
        self._store = store
        self._conversations = conversations
        self._router = router or LLMRouter()
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

        if stage == "confirming" and compact in {
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
        if "묶음" in text or "스마트 딜리버리" in text:
            if re.search(
                r"(요금|가격).*(어떻게|기준|방식|계산)|"
                r"(어떻게|기준|방식).*(요금|가격|계산)",
                text,
            ):
                return "question"
            return "question" if definition_question and not action_request else "bundle"
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
                if state.get("stage") != "confirming"
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
        return {
            "reply": "주문 작성을 취소했어요. 처음부터 다시 시작할 수 있어요.",
            "stage": "collecting",
            "slots": {},
            "trace": state.get("trace", []) + ["cancel:draft_reset"],
        }

    async def _status_query(self, state: AgentState) -> AgentState:
        partner_order_id = state.get("partner_order_id")
        explicit_id = re.search(
            r"\b(?:agent|moveops|bundle|smart)-[A-Za-z0-9._-]+\b",
            state["message"],
        )
        if explicit_id:
            partner_order_id = explicit_id.group(0)
        if not partner_order_id:
            return {
                "reply": "아직 이 대화에서 접수된 주문이 없어요. 주문번호가 있다면 알려주세요.",
                "actions": [
                    {"label": "배송 주문 시작", "message": "퀵 배송 주문하고 싶어"},
                    {
                        "label": "스마트 딜리버리",
                        "message": "스마트 딜리버리 이용 방법 알려줘",
                        "target": "smartDelivery",
                    },
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
        """스마트 딜리버리: 개별 배송과 하나의 묶음 경로를 비교한다."""
        parsed = await self._llm_json(BUNDLE_EXTRACT_PROMPT.format(message=state["message"]))
        pickup = parsed.get("pickup")
        dropoffs = [d for d in (parsed.get("dropoffs") or []) if isinstance(d, str) and d.strip()]
        if not pickup or len(dropoffs) < 2:
            return {
                "reply": (
                    "스마트 딜리버리는 여러 사람의 배송을 한 동선으로 묶어 "
                    "비교해요. 홈페이지 접수 영역에서 보내는 사람 1~5명과 "
                    "받는 사람 2~5명을 연결해주세요. 서울뿐 아니라 대전 등 "
                    "국내 주소도 입력할 수 있어요."
                ),
                "actions": [
                    {
                        "label": "접수 화면 열기",
                        "message": "스마트 딜리버리 접수 화면 열기",
                        "target": "smartDelivery",
                    }
                ],
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
                "reply": f"스마트 딜리버리 견적 계산 중 문제가 있었어요: {exc}",
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
            f"스마트 딜리버리 비교 견적이에요! (출발: {result['pickup']})\n\n"
            f"[각각 따로 보낼 때]\n{individual_lines}\n"
            f"합계: {result['individualTotal']:,}원\n\n"
            f"[한 번에 묶어 보낼 때 — 경유지 추가 혜택 적용]\n"
            f"추천 경유 순서: {route_text}\n"
            f"{road_text}"
            f"스마트 딜리버리 견적: {result['bundledPrice']:,}원\n\n"
        )
        if result["recommendBundle"]:
            reply += (
                f"→ 스마트 딜리버리로 보내면 {result['saving']:,}원 절약돼요!\n"
                "홈페이지 접수 영역에서 보내는 분과 받는 분 연락처를 확인한 "
                "뒤 진행할 수 있어요."
            )
        else:
            reply += "→ 이번 건은 따로 보내는 게 더 유리해요."
        return {
            "quote": result,
            "reply": reply,
            "actions": [
                {
                    "label": "접수 화면 열기",
                    "message": "스마트 딜리버리 접수 화면 열기",
                    "target": "smartDelivery",
                },
                {
                    "label": "준비물",
                    "message": "스마트 딜리버리 접수에 필요한 정보를 알려줘",
                },
            ],
            "trace": state.get("trace", []) + [f"bundle:ok({len(dropoffs)}곳)"],
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
                "안녕하세요! MOVB에서는 일반 퀵과 스마트 딜리버리의 "
                "견적·주문·상태 확인을 "
                "도와드려요. "
                "예를 들어 “퀵과 도보 배송은 뭐가 달라?” 또는 "
                "“판교역에서 정자역으로 서류 보내줘”라고 말씀해보세요."
            )
        return {
            "reply": reply,
            "actions": [
                {
                    "label": "스마트 딜리버리",
                    "message": "스마트 딜리버리 접수 화면 열기",
                    "target": "smartDelivery",
                },
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
