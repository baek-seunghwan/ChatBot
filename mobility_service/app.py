from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import Body, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .agent import DeliveryAgent
from .bundle import bundle_quote
from .client import KakaoApiError, KakaoMobilityClient
from .config import Settings
from .conversation_store import ConversationStore
from .geocode import KakaoGeocodeClient
from .local_responder import local_model_reply
from .models import (
    AgentChatRequest,
    ApiEnvelope,
    BundleQuoteRequest,
    CallbackBody,
    CarpoolPlanRequest,
    CreateDeliveryRequest,
    DeliveryDraft,
)
from .rideshare import carpool_plan
from .orders import cancel_order_by_id, get_order_status, place_order
from .pool_store import PoolStore
from .store import MobilityStore
from .web import INDEX_HTML, TAXI_HTML


def create_app(
    *,
    settings: Settings | None = None,
    client: KakaoMobilityClient | None = None,
    store: MobilityStore | None = None,
    geocoder: KakaoGeocodeClient | None = None,
    conversations: ConversationStore | None = None,
    agent: DeliveryAgent | None = None,
    pool_store: PoolStore | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    resolved_store = store or MobilityStore(resolved_settings.database_path)
    resolved_client = client or KakaoMobilityClient(resolved_settings)
    resolved_geocoder = geocoder or KakaoGeocodeClient(resolved_settings)
    resolved_conversations = conversations or ConversationStore(resolved_settings.database_path)
    resolved_pools = pool_store or PoolStore(resolved_settings.database_path)
    resolved_agent = agent or DeliveryAgent(
        resolved_client, resolved_geocoder, resolved_store, resolved_conversations,
        pools=resolved_pools,
    )
    owns_client = client is None
    owns_geocoder = geocoder is None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if owns_client:
            await resolved_client.close()
        if owns_geocoder:
            await resolved_geocoder.close()

    application = FastAPI(
        title="모브 (MOVB)",
        description="Kakao Mobility Sandbox API를 연동한 FastAPI 백엔드 서비스",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.state.kakao_client = resolved_client
    application.state.store = resolved_store
    application.state.geocoder = resolved_geocoder
    application.state.conversations = resolved_conversations
    application.state.agent = resolved_agent

    @application.exception_handler(KakaoApiError)
    async def kakao_api_error_handler(
        _: Request, exc: KakaoApiError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={
                "ok": False,
                "message": str(exc),
                "providerStatus": exc.status_code,
            },
        )

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> str:
        return INDEX_HTML

    @application.get("/taxi", response_class=HTMLResponse, include_in_schema=False)
    async def taxi_page() -> str:
        return TAXI_HTML

    @application.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "moveops",
            "kakaoConfigured": resolved_settings.configured,
            "mapConfigured": resolved_settings.map_configured,
            "sandbox": True,
        }

    @application.get("/api/config", response_model=ApiEnvelope)
    async def public_config() -> ApiEnvelope:
        return ApiEnvelope(
            data={
                "configured": resolved_settings.configured,
                "mapConfigured": resolved_settings.map_configured,
                "geocodingConfigured": resolved_settings.geocoding_configured,
                # JavaScript 키는 등록된 웹 도메인에서 사용하는 공개 식별자다.
                # REST API 키와 Native App 키는 절대 클라이언트에 전달하지 않는다.
                "kakaoJavascriptKey": (
                    resolved_settings.kakao_javascript_key
                    if resolved_settings.map_configured
                    else None
                ),
                "environment": "sandbox",
                "callbackBaseUrl": resolved_settings.callback_base_url or None,
                "database": Path(resolved_settings.database_path).name,
            }
        )

    @application.get("/api/kakao/auth-check", response_model=ApiEnvelope)
    async def auth_check() -> ApiEnvelope:
        return ApiEnvelope(data=await resolved_client.auth_check())

    @application.post("/api/deliveries/estimate", response_model=ApiEnvelope)
    async def estimate(request: DeliveryDraft) -> ApiEnvelope:
        return ApiEnvelope(data=await resolved_client.estimate(request))

    @application.post("/api/deliveries/price", response_model=ApiEnvelope)
    async def price(request: DeliveryDraft) -> ApiEnvelope:
        return ApiEnvelope(data=await resolved_client.price(request))

    @application.post("/api/orders", response_model=ApiEnvelope)
    async def create_order(
        request: CreateDeliveryRequest,
        idempotency_key: str | None = Header(
            default=None, alias="Idempotency-Key", max_length=100
        ),
    ) -> ApiEnvelope:
        partner_order_id = (
            request.partner_order_id
            or idempotency_key
            or f"moveops-{uuid4().hex[:20]}"
        )
        result = await place_order(resolved_client, resolved_store, request, partner_order_id)
        message = result.pop("message", None)
        return ApiEnvelope(data=result, message=message)

    @application.get("/api/orders", response_model=ApiEnvelope)
    async def list_orders(
        limit: int = Query(default=30, ge=1, le=100)
    ) -> ApiEnvelope:
        return ApiEnvelope(data=resolved_store.list_orders(limit))

    @application.get("/api/orders/{partner_order_id}", response_model=ApiEnvelope)
    async def get_order(
        partner_order_id: str,
        refresh: bool = Query(default=False),
    ) -> ApiEnvelope:
        local_order = await get_order_status(
            resolved_client, resolved_store, partner_order_id, refresh
        )
        if local_order is None:
            raise HTTPException(status_code=404, detail="저장된 주문이 없습니다.")
        return ApiEnvelope(data=local_order)

    @application.get(
        "/api/orders/{partner_order_id}/picker", response_model=ApiEnvelope
    )
    async def get_picker(partner_order_id: str) -> ApiEnvelope:
        return ApiEnvelope(data=await resolved_client.get_picker(partner_order_id))

    @application.patch(
        "/api/orders/{partner_order_id}/cancel", response_model=ApiEnvelope
    )
    async def cancel_order(partner_order_id: str) -> ApiEnvelope:
        if resolved_store.get_order(partner_order_id) is None:
            raise HTTPException(status_code=404, detail="저장된 주문이 없습니다.")
        result = await cancel_order_by_id(resolved_client, resolved_store, partner_order_id)
        return ApiEnvelope(data=result)

    @application.put(
        "/api/v1/callback/orders/{partner_order_id}/{event}",
        response_model=ApiEnvelope,
    )
    async def order_callback(
        partner_order_id: str,
        event: str,
        body: CallbackBody | None = Body(default=None),
    ) -> ApiEnvelope:
        callback_body = (
            body.model_dump(mode="json", by_alias=True, exclude_none=True)
            if body
            else {}
        )
        result = resolved_store.record_callback(
            partner_order_id, event, callback_body
        )
        return ApiEnvelope(data=result)

    @application.put(
        "/api/v1/callback/orders/{order_id}/steps/{step_id}",
        response_model=ApiEnvelope,
    )
    async def step_callback(
        order_id: str,
        step_id: str,
        body: dict[str, Any] = Body(default_factory=dict),
    ) -> ApiEnvelope:
        payload = {"stepId": step_id, **body}
        result = resolved_store.record_callback(
            order_id, f"step:{body.get('status', 'updated')}", payload
        )
        return ApiEnvelope(data=result)

    @application.post("/api/agent/chat", response_model=ApiEnvelope)
    async def agent_chat(
        request: AgentChatRequest,
        x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
    ) -> ApiEnvelope:
        session_id = request.session_id or x_session_id or f"sess-{uuid4().hex}"
        if request.mode == "local":
            reply = await asyncio.to_thread(local_model_reply, request.message)
            return ApiEnvelope(
                data={
                    "sessionId": session_id,
                    "reply": reply,
                    "stage": "collecting",
                    "slots": {},
                    "quote": None,
                    "order": None,
                    "trace": ["local_model"],
                }
            )
        result = await resolved_agent.achat(session_id=session_id, message=request.message)
        return ApiEnvelope(data=result.to_dict())

    @application.post("/api/bundle/quote", response_model=ApiEnvelope)
    async def bundle_quote_route(request: BundleQuoteRequest) -> ApiEnvelope:
        try:
            result = await bundle_quote(
                resolved_client,
                resolved_geocoder,
                request.pickup_address,
                request.dropoff_addresses,
                product_size=request.product_size.value,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return ApiEnvelope(data=result)

    @application.post("/api/carpool/plan", response_model=ApiEnvelope)
    async def carpool_plan_route(request: CarpoolPlanRequest) -> ApiEnvelope:
        origin = await resolved_geocoder.search_address(request.origin_address)
        if origin is None:
            raise HTTPException(
                status_code=422, detail=f"출발지 주소를 찾지 못했습니다: {request.origin_address}"
            )
        passengers = []
        for passenger in request.passengers:
            location = await resolved_geocoder.search_address(passenger.address)
            if location is None:
                raise HTTPException(
                    status_code=422, detail=f"목적지 주소를 찾지 못했습니다: {passenger.address}"
                )
            passengers.append({"name": passenger.name, "location": location})
        try:
            plan = carpool_plan(origin, passengers)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return ApiEnvelope(data=plan)

    @application.get("/api/pool/requests", response_model=ApiEnvelope)
    async def list_pool_requests() -> ApiEnvelope:
        """합승 대기 보드: 진행 중인 합승 요청 목록 (개인정보 제외 요약)."""
        board = [
            {
                "id": request["id"],
                "pickupAddress": request["pickup"]["address"],
                "dropoffAddress": request["dropoff"]["address"],
                "productName": request["product"].get("productName", "물품"),
                "soloPrice": request["soloPrice"],
                "createdAt": request["createdAt"],
            }
            for request in resolved_pools.list_open()
        ]
        return ApiEnvelope(data=board)

    @application.delete("/api/pool/requests/{request_id}", response_model=ApiEnvelope)
    async def cancel_pool_request(
        request_id: int,
        x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
    ) -> ApiEnvelope:
        if not x_session_id:
            raise HTTPException(status_code=422, detail="X-Session-Id 헤더가 필요합니다.")
        if not resolved_pools.cancel_request(request_id, x_session_id):
            raise HTTPException(
                status_code=404, detail="취소할 수 있는 합승 요청을 찾지 못했습니다."
            )
        return ApiEnvelope(data={"canceled": request_id})

    return application


app = create_app()
