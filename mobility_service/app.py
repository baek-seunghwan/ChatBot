from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import (
    Body,
    Cookie,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
)
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from .agent import DeliveryAgent
from .bundle import bundle_quote
from .client import KakaoApiError, KakaoMobilityClient
from .config import Settings
from .conversation_store import ConversationStore
from .directions import KakaoDirectionsClient, RoutePlanner
from .geocode import KakaoGeocodeClient
from .knowledge import default_knowledge_base
from .local_responder import local_model_reply, ollama_status
from .models import (
    AgentChatRequest,
    ApiEnvelope,
    BundleQuoteRequest,
    CallbackBody,
    CarpoolBookingRequest,
    CarpoolPlanRequest,
    CreateDeliveryRequest,
    DeliveryDraft,
    LoginRequest,
    RegisterRequest,
    RouteSummaryRequest,
    SandboxStatusChange,
)
from .rideshare import carpool_plan
from .orders import (
    cancel_order_by_id,
    get_order_status,
    get_order_steps,
    place_order,
)
from .pool_store import PoolStore
from .store import MobilityStore
from .user_store import DuplicateEmailError, SESSION_TTL_SECONDS, UserStore
from .web import ADMIN_HTML, FEATURES_HTML, INDEX_HTML, TAXI_HTML


SESSION_COOKIE_NAME = "movb_session"


def create_app(
    *,
    settings: Settings | None = None,
    client: KakaoMobilityClient | None = None,
    store: MobilityStore | None = None,
    geocoder: KakaoGeocodeClient | None = None,
    directions: KakaoDirectionsClient | None = None,
    conversations: ConversationStore | None = None,
    agent: DeliveryAgent | None = None,
    pool_store: PoolStore | None = None,
    user_store: UserStore | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    resolved_store = store or MobilityStore(resolved_settings.database_path)
    resolved_client = client or KakaoMobilityClient(resolved_settings)
    resolved_geocoder = geocoder or KakaoGeocodeClient(resolved_settings)
    resolved_directions = directions or KakaoDirectionsClient(resolved_settings)
    resolved_routes = RoutePlanner(resolved_directions)
    resolved_conversations = conversations or ConversationStore(resolved_settings.database_path)
    resolved_pools = pool_store or PoolStore(resolved_settings.database_path)
    resolved_users = user_store or UserStore(resolved_settings.database_path)
    resolved_knowledge = default_knowledge_base()
    if resolved_settings.admin_configured:
        resolved_users.ensure_admin(
            username=resolved_settings.admin_username,
            password=resolved_settings.admin_password,
        )
    resolved_agent = agent or DeliveryAgent(
        resolved_client, resolved_geocoder, resolved_store, resolved_conversations,
        pools=resolved_pools, knowledge_base=resolved_knowledge,
        route_planner=resolved_routes,
    )
    owns_client = client is None
    owns_geocoder = geocoder is None
    owns_directions = directions is None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if owns_client:
            await resolved_client.close()
        if owns_geocoder:
            await resolved_geocoder.close()
        if owns_directions:
            await resolved_directions.close()

    application = FastAPI(
        title="모브 (MOVB)",
        description=(
            "LangGraph Agent와 근거 기반 Knowledge RAG가 Kakao Mobility Sandbox "
            "업무를 연결하는 AI 모빌리티 운영 서비스"
        ),
        version="1.2.0",
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.state.kakao_client = resolved_client
    application.state.store = resolved_store
    application.state.geocoder = resolved_geocoder
    application.state.directions = resolved_directions
    application.state.routes = resolved_routes
    application.state.conversations = resolved_conversations
    application.state.agent = resolved_agent
    application.state.knowledge = resolved_knowledge
    application.state.users = resolved_users

    def set_session_cookie(response: Response, request: Request, token: str) -> None:
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=token,
            max_age=SESSION_TTL_SECONDS,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="lax",
            path="/",
        )

    def require_current_user(
        session_token: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
    ) -> dict[str, Any]:
        user = (
            resolved_users.get_user_by_session(session_token)
            if session_token
            else None
        )
        if user is None:
            raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
        return user

    def require_admin(
        user: dict[str, Any] = Depends(require_current_user),
    ) -> dict[str, Any]:
        if user.get("role") != "ADMIN":
            raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
        return user

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

    @application.get("/features", response_class=HTMLResponse, include_in_schema=False)
    async def features_page() -> str:
        return FEATURES_HTML

    @application.get(
        "/admin",
        response_class=HTMLResponse,
        response_model=None,
        include_in_schema=False,
    )
    async def admin_page(
        session_token: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
    ) -> Response:
        user = (
            resolved_users.get_user_by_session(session_token)
            if session_token
            else None
        )
        if user is None or user.get("role") != "ADMIN":
            return RedirectResponse(url="/?admin=1", status_code=303)
        return HTMLResponse(ADMIN_HTML)

    @application.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "moveops",
            "kakaoConfigured": resolved_settings.configured,
            "mapConfigured": resolved_settings.map_configured,
            "directionsConfigured": resolved_settings.directions_configured,
            "adminConfigured": resolved_settings.admin_configured,
            "knowledgeChunks": len(resolved_knowledge.chunks),
            "sandbox": True,
        }

    @application.get("/api/config", response_model=ApiEnvelope)
    async def public_config() -> ApiEnvelope:
        return ApiEnvelope(
            data={
                "configured": resolved_settings.configured,
                "mapConfigured": resolved_settings.map_configured,
                "geocodingConfigured": resolved_settings.geocoding_configured,
                "directionsConfigured": resolved_settings.directions_configured,
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

    @application.get("/api/local-chat/status", response_model=ApiEnvelope)
    async def local_chat_status() -> ApiEnvelope:
        return ApiEnvelope(data=await asyncio.to_thread(ollama_status))

    @application.get("/api/knowledge/search", response_model=ApiEnvelope)
    async def knowledge_search(
        q: str = Query(min_length=2, max_length=500),
        limit: int = Query(default=3, ge=1, le=5),
    ) -> ApiEnvelope:
        results = resolved_knowledge.search(q, limit=limit)
        return ApiEnvelope(
            data={
                "query": q,
                "results": [
                    {
                        **result.to_source(),
                        "excerpt": result.content[:500],
                    }
                    for result in results
                ],
            }
        )

    @application.get("/api/kakao/auth-check", response_model=ApiEnvelope)
    async def auth_check() -> ApiEnvelope:
        return ApiEnvelope(data=await resolved_client.auth_check())

    @application.post(
        "/api/auth/register", response_model=ApiEnvelope, status_code=201
    )
    async def register(
        payload: RegisterRequest,
        request: Request,
        response: Response,
    ) -> ApiEnvelope:
        try:
            user = resolved_users.create_user(
                name=payload.name,
                email=payload.email,
                password=payload.password,
            )
        except DuplicateEmailError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        token = resolved_users.create_session(user["id"])
        set_session_cookie(response, request, token)
        return ApiEnvelope(data={"user": user}, message="회원가입이 완료되었습니다.")

    @application.post("/api/auth/login", response_model=ApiEnvelope)
    async def login(
        payload: LoginRequest,
        request: Request,
        response: Response,
    ) -> ApiEnvelope:
        user = resolved_users.authenticate(
            identifier=payload.identifier,
            password=payload.password,
        )
        if user is None:
            raise HTTPException(
                status_code=401,
                detail="아이디·이메일 또는 비밀번호가 올바르지 않습니다.",
            )
        token = resolved_users.create_session(user["id"])
        set_session_cookie(response, request, token)
        return ApiEnvelope(data={"user": user}, message="로그인되었습니다.")

    @application.get("/api/auth/me", response_model=ApiEnvelope)
    async def current_user(
        user: dict[str, Any] = Depends(require_current_user),
    ) -> ApiEnvelope:
        return ApiEnvelope(data={"user": user})

    @application.post("/api/auth/logout", response_model=ApiEnvelope)
    async def logout(
        response: Response,
        session_token: str | None = Cookie(
            default=None, alias=SESSION_COOKIE_NAME
        ),
    ) -> ApiEnvelope:
        if session_token:
            resolved_users.revoke_session(session_token)
        response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")
        return ApiEnvelope(message="로그아웃되었습니다.")

    @application.get("/api/admin/summary", response_model=ApiEnvelope)
    async def admin_summary(
        _: dict[str, Any] = Depends(require_admin),
    ) -> ApiEnvelope:
        return ApiEnvelope(
            data={
                "users": resolved_users.user_counts(),
                "orders": resolved_store.order_counts(),
                "openPoolRequests": len(resolved_pools.list_open()),
            }
        )

    @application.get("/api/admin/users", response_model=ApiEnvelope)
    async def admin_users(
        limit: int = Query(default=100, ge=1, le=500),
        _: dict[str, Any] = Depends(require_admin),
    ) -> ApiEnvelope:
        return ApiEnvelope(data=resolved_users.list_users(limit))

    @application.get("/api/admin/orders", response_model=ApiEnvelope)
    async def admin_orders(
        limit: int = Query(default=100, ge=1, le=500),
        _: dict[str, Any] = Depends(require_admin),
    ) -> ApiEnvelope:
        return ApiEnvelope(data=resolved_store.list_orders(limit))

    @application.post("/api/deliveries/estimate", response_model=ApiEnvelope)
    async def estimate(request: DeliveryDraft) -> ApiEnvelope:
        provider = await resolved_client.estimate(request)
        route = await resolved_routes.route_summary(
            request.pickup.location,
            request.dropoff.location,
            waypoints=[item.location for item in request.waypoints],
            departure_time=request.wish_time,
        )
        data = dict(provider) if isinstance(provider, dict) else {"provider": provider}
        data["routeInfo"] = route
        return ApiEnvelope(data=data)

    @application.post("/api/routes/summary", response_model=ApiEnvelope)
    async def route_summary(request: RouteSummaryRequest) -> ApiEnvelope:
        return ApiEnvelope(
            data=await resolved_routes.route_summary(
                request.origin,
                request.destination,
                waypoints=request.waypoints,
                departure_time=request.departure_time,
            )
        )

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

    @application.get(
        "/api/orders/{partner_order_id}/steps", response_model=ApiEnvelope
    )
    async def order_steps(
        partner_order_id: str,
        refresh: bool = Query(default=True),
    ) -> ApiEnvelope:
        if resolved_store.get_order(partner_order_id) is None:
            raise HTTPException(status_code=404, detail="저장된 주문이 없습니다.")
        return ApiEnvelope(
            data=await get_order_steps(
                resolved_client,
                resolved_store,
                partner_order_id,
                refresh_order=refresh,
            )
        )

    @application.patch(
        "/api/orders/{partner_order_id}/cancel", response_model=ApiEnvelope
    )
    async def cancel_order(partner_order_id: str) -> ApiEnvelope:
        if resolved_store.get_order(partner_order_id) is None:
            raise HTTPException(status_code=404, detail="저장된 주문이 없습니다.")
        result = await cancel_order_by_id(resolved_client, resolved_store, partner_order_id)
        return ApiEnvelope(data=result)

    @application.patch(
        "/api/admin/orders/{partner_order_id}/sandbox-status",
        response_model=ApiEnvelope,
    )
    async def change_sandbox_status(
        partner_order_id: str,
        payload: SandboxStatusChange,
        _: dict[str, Any] = Depends(require_admin),
    ) -> ApiEnvelope:
        if resolved_store.get_order(partner_order_id) is None:
            raise HTTPException(status_code=404, detail="저장된 주문이 없습니다.")
        provider = await resolved_client.change_sandbox_status(
            partner_order_id,
            payload.order_status.value,
            cancel_by=payload.cancel_by,
        )
        local_status = {
            "ABORT": "ABORTED",
            "MATCH_PICKER": "MATCHED",
            "CANCEL": "CANCELED",
            "PICKUP_COMPLETED": "PICKUP_COMPLETED",
            "DROPOFF_COMPLETED": "DROPOFF_COMPLETED",
        }[payload.order_status.value]
        resolved_store.set_status(partner_order_id, local_status)
        return ApiEnvelope(
            data={
                "provider": provider,
                "order": resolved_store.get_order(partner_order_id),
            },
            message=f"Sandbox 주문 상태를 {local_status}(으)로 변경했습니다.",
        )

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
            reply = await asyncio.to_thread(
                local_model_reply, request.message, request.local_engine
            )
            return ApiEnvelope(
                data={
                    "sessionId": session_id,
                    "reply": reply,
                    "stage": "collecting",
                    "slots": {},
                    "quote": None,
                    "order": None,
                    "trace": [f"local_model:{request.local_engine}"],
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
                route_planner=resolved_routes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return ApiEnvelope(data=result)

    async def build_carpool_plan(
        request: CarpoolPlanRequest, *, departure_time: str | None = None
    ) -> dict[str, Any]:
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
            plan = await carpool_plan(
                origin,
                passengers,
                route_planner=resolved_routes,
                departure_time=departure_time,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return plan

    @application.post("/api/carpool/plan", response_model=ApiEnvelope)
    async def carpool_plan_route(request: CarpoolPlanRequest) -> ApiEnvelope:
        return ApiEnvelope(
            data=await build_carpool_plan(
                request, departure_time=request.departure_at
            )
        )

    @application.post("/api/carpool/requests", response_model=ApiEnvelope)
    async def create_carpool_request(
        request: CarpoolBookingRequest,
    ) -> ApiEnvelope:
        plan = await build_carpool_plan(
            request,
            departure_time=(
                request.departure_at if request.departure_mode == "scheduled" else None
            ),
        )
        saved = resolved_store.create_taxi_request(
            request.model_dump(mode="json", by_alias=True), plan
        )
        return ApiEnvelope(
            data=saved,
            message=(
                "택시 합승 접수가 저장되었습니다. "
                "현재는 실제 택시 배차가 아닌 경로·요금 접수 기능입니다."
            ),
        )

    @application.get("/api/carpool/requests/{request_id}", response_model=ApiEnvelope)
    async def get_carpool_request(request_id: str) -> ApiEnvelope:
        saved = resolved_store.get_taxi_request(request_id)
        if saved is None:
            raise HTTPException(status_code=404, detail="택시 합승 접수를 찾지 못했습니다.")
        return ApiEnvelope(data=saved)

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
