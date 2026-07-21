from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, TypedDict

os.environ.setdefault("LANGSMITH_TRACING", "false")

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from chatbot.providers import LLMRouter

from .client import KakaoApiError, KakaoMobilityClient
from .config import Settings
from .conversation_store import ConversationStore
from .geocode import KakaoGeocodeClient
from .models import CreateDeliveryRequest, DeliveryDraft, OrderType, PaymentType, ProductSize
from .orders import cancel_order_by_id, get_order_status, place_order
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
    "당신은 카카오 T 퀵·도보 배송 주문을 도와주는 'MoveOps 배송 도우미'입니다. "
    "친절하고 간결한 한국어로 답하세요. 배송과 무관한 잡담/인사에는 짧게 응대하고, "
    "배송 주문을 하고 싶다면 출발지/도착지/물품 정보를 알려달라고 안내하세요."
)

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
- chitchat: 배송과 무관한 인사/잡담

단어 하나만 출력하세요."""

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
    trace: list[str]


@dataclass
class AgentChatResult:
    session_id: str
    reply: str
    stage: str
    slots: dict[str, Any] = field(default_factory=dict)
    quote: dict[str, Any] | None = None
    order: dict[str, Any] | None = None
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "reply": self.reply,
            "stage": self.stage,
            "slots": self.slots,
            "quote": self.quote,
            "order": self.order,
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
    ) -> None:
        self._client = client
        self._geocoder = geocoder
        self._store = store
        self._conversations = conversations
        self._router = router or LLMRouter()
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

    async def _classify_intent(self, state: AgentState) -> AgentState:
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
            "provide_info",
            "modify",
            "confirm",
            "cancel",
            "status_query",
            "chitchat",
        ):
            if label in verdict:
                intent = label
                break
        if intent is None:
            intent = "provide_info" if state.get("stage") != "confirming" else "chitchat"

        return {"intent": intent, "trace": state.get("trace", []) + [f"classify_intent:{intent}"]}

    @staticmethod
    def _route_by_intent(state: AgentState) -> str:
        return state.get("intent", "chitchat")

    @staticmethod
    def _coerce_slot_value(key: str, value: Any) -> Any | None:
        enum_map = {"orderType": OrderType, "productSize": ProductSize, "paymentType": PaymentType}
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

    async def _extract_slots(self, state: AgentState) -> AgentState:
        slots = dict(state.get("slots", {}))
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
            "declaredValue", "quantity", "wishTime", "paymentType",
        }
        applied = []
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
        payload = _slots_payload(slots)
        try:
            CreateDeliveryRequest(**payload, partnerOrderId="validation-check-0000")
        except ValidationError as exc:
            lines: list[str] = []
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
        return {
            "reply": reply,
            "stage": "collecting",
            "trace": state.get("trace", []) + ["ask_clarification"],
        }

    @staticmethod
    def _format_quote(data: Any) -> str:
        # `price`는 특정 orderType 하나에 대한 단일 확정 요금(totalPrice)을,
        # `estimate`는 여러 orderType/차량 옵션 비교 목록(lists)을 반환한다 (client.py 참고).
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
            fleet = row.get("fleet", "")
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
            data = await self._client.price(draft)
        except KakaoApiError as exc:
            return {
                "reply": f"가격 조회 중 문제가 있었어요: {exc}",
                "stage": "collecting",
                "trace": state.get("trace", []) + ["quote_price:api_error"],
            }

        quote_hash = _quote_hash(slots)
        self._conversations.save_quote(state["session_id"], data, quote_hash)
        summary = self._format_quote(data)
        reply = f"견적을 확인했어요!\n{summary}\n\n이대로 진행할까요? '네' 또는 '진행해줘'라고 답해주세요."
        return {
            "quote": data,
            "quote_hash": quote_hash,
            "stage": "confirming",
            "reply": reply,
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
        if not partner_order_id:
            return {
                "reply": "아직 이 대화에서 접수된 주문이 없어요. 주문번호가 있다면 알려주세요.",
                "trace": state.get("trace", []) + ["status_query:no_order"],
            }
        try:
            order = await get_order_status(self._client, self._store, partner_order_id, refresh=True)
        except KakaoApiError as exc:
            return {
                "reply": f"상태 확인 중 문제가 발생했어요: {exc}",
                "trace": state.get("trace", []) + ["status_query:api_error"],
            }
        status = order.get("status") if order else None
        reply = (
            f"주문번호 {partner_order_id}의 현재 상태는 '{status}'예요."
            if status
            else "주문 상태를 찾지 못했어요."
        )
        return {"reply": reply, "trace": state.get("trace", []) + ["status_query:ok"]}

    async def _chitchat(self, state: AgentState) -> AgentState:
        try:
            reply = await self._llm(state["message"], system=CHITCHAT_SYSTEM, max_tokens=200)
        except RuntimeError:
            reply = "지금은 답변을 만들지 못했어요. 잠시 후 다시 시도해주세요."
        return {"reply": reply, "trace": state.get("trace", []) + ["chitchat"]}

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
