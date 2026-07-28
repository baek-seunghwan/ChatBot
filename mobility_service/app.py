from __future__ import annotations

import asyncio
import hashlib
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
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
from .bundle import multi_pickup_bundle_quote, prepare_bundle_order
from .client import KakaoApiError, KakaoMobilityClient
from .config import Settings
from .conversation_store import ConversationStore
from .directions import KakaoDirectionsClient, RoutePlanner
from .geocode import KakaoGeocodeClient
from .knowledge import default_knowledge_base
from .local_responder import local_model_reply, ollama_status
from .matching import build_pooled_order, compatible_for_pooling
from .models import (
    AgentChatRequest,
    ApiEnvelope,
    BundleOrderRequest,
    BundleQuoteRequest,
    CallbackBody,
    CreateDeliveryRequest,
    DeliveryDraft,
    LoginRequest,
    RegisterRequest,
    RouteSummaryRequest,
    SandboxStatusChange,
)
from .orders import (
    cancel_order_by_id,
    get_order_status,
    get_order_steps,
    place_order,
)
from .store import MobilityStore
from .user_store import DuplicateEmailError, SESSION_TTL_SECONDS, UserStore
from .web import ADMIN_HTML, FEATURES_HTML, HISTORY_HTML, INDEX_HTML


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
    user_store: UserStore | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    resolved_store = store or MobilityStore(resolved_settings.database_path)
    resolved_client = client or KakaoMobilityClient(resolved_settings)
    resolved_geocoder = geocoder or KakaoGeocodeClient(resolved_settings)
    resolved_directions = directions or KakaoDirectionsClient(resolved_settings)
    resolved_routes = RoutePlanner(resolved_directions)
    resolved_conversations = conversations or ConversationStore(resolved_settings.database_path)
    resolved_users = user_store or UserStore(resolved_settings.database_path)
    resolved_knowledge = default_knowledge_base()
    if resolved_settings.admin_configured:
        resolved_users.ensure_admin(
            username=resolved_settings.admin_username,
            password=resolved_settings.admin_password,
        )
    resolved_agent = agent or DeliveryAgent(
        resolved_client, resolved_geocoder, resolved_store, resolved_conversations,
        knowledge_base=resolved_knowledge, route_planner=resolved_routes,
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

    @application.get(
        "/bundle",
        response_class=HTMLResponse,
        response_model=None,
        include_in_schema=False,
    )
    async def bundle_page() -> RedirectResponse:
        return RedirectResponse(url="/#smartDelivery", status_code=307)

    @application.get(
        "/smart-delivery/form",
        response_class=HTMLResponse,
        response_model=None,
        include_in_schema=False,
    )
    async def smart_delivery_form() -> RedirectResponse:
        return RedirectResponse(url="/#smartDelivery", status_code=307)

    @application.get("/features", response_class=HTMLResponse, include_in_schema=False)
    async def features_page() -> str:
        return FEATURES_HTML

    @application.get("/history", response_class=HTMLResponse, include_in_schema=False)
    async def history_page() -> str:
        return HISTORY_HTML

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

    def public_match_request(match_request: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in match_request.items()
            if key not in {"request", "clientId"}
        }

    def match_owner_id(
        x_client_id: str,
        session_token: str | None,
    ) -> str:
        user = (
            resolved_users.get_user_by_session(session_token)
            if session_token
            else None
        )
        return (
            f"user:{user['id']}"
            if user is not None
            else f"browser:{x_client_id}"
        )

    @application.post(
        "/api/delivery-matches",
        response_model=ApiEnvelope,
        status_code=202,
    )
    async def create_delivery_match(
        request: CreateDeliveryRequest,
        x_client_id: str = Header(
            alias="X-Client-Id",
            min_length=8,
            max_length=100,
        ),
        session_token: str | None = Cookie(
            default=None,
            alias=SESSION_COOKIE_NAME,
        ),
        idempotency_key: str | None = Header(
            default=None,
            alias="Idempotency-Key",
            max_length=100,
        ),
    ) -> ApiEnvelope:
        if request.order_type.value not in {"QUICK", "QUICK_EXPRESS"}:
            raise HTTPException(
                status_code=422,
                detail="공동배송 자동 매칭은 일반 퀵과 퀵 급송만 지원합니다.",
            )
        request_id = idempotency_key or f"match-request-{uuid4().hex[:20]}"
        expires_at = (
            datetime.now(timezone.utc) + timedelta(minutes=15)
        ).isoformat()
        request_payload = request.model_dump(
            mode="json",
            by_alias=True,
            exclude_none=True,
        )
        owner_id = match_owner_id(x_client_id, session_token)
        saved, created = resolved_store.reserve_match_request(
            request_id,
            owner_id,
            request_payload,
            expires_at,
        )
        if not created and saved["clientId"] != owner_id:
            raise HTTPException(
                status_code=409,
                detail="이미 다른 브라우저가 사용한 공동배송 요청 키입니다.",
            )
        if not created or saved["status"] != "WAITING":
            return ApiEnvelope(
                data={"matching": public_match_request(saved)},
                message="기존 공동배송 매칭 요청을 반환했습니다.",
            )

        candidates = resolved_store.list_pending_match_requests(
            exclude_request_id=request_id,
            exclude_client_id=owner_id,
        )
        for candidate in candidates:
            if not compatible_for_pooling(candidate["request"], request_payload):
                continue
            match_id = f"match-{uuid4().hex[:20]}"
            if not resolved_store.claim_match_requests(
                candidate["requestId"],
                request_id,
                match_id,
            ):
                continue

            partner_order_id = f"smart-pool-{uuid4().hex[:18]}"
            request_ids = (candidate["requestId"], request_id)
            try:
                pooled_order = build_pooled_order(
                    candidate["request"],
                    request_payload,
                    partner_order_id,
                )
                order_result = await place_order(
                    resolved_client,
                    resolved_store,
                    pooled_order,
                    partner_order_id,
                )
            except Exception as exc:
                resolved_store.fail_match_requests(request_ids, str(exc))
                raise
            resolved_store.complete_match_requests(
                request_ids,
                partner_order_id,
            )
            matched = resolved_store.get_match_request(request_id)
            return ApiEnvelope(
                data={
                    "matching": public_match_request(matched or saved),
                    "orderResult": order_result,
                },
                message=(
                    "다른 사용자의 배송과 매칭되어 스마트 딜리버리 한 건으로 "
                    "접수되었습니다."
                ),
            )

        waiting = resolved_store.get_match_request(request_id)
        return ApiEnvelope(
            data={"matching": public_match_request(waiting or saved)},
            message=(
                "공동배송 대기열에 접수했습니다. 15분 안에 출발지·도착 방향과 "
                "배송 조건이 비슷한 다른 사용자 주문을 찾습니다."
            ),
        )

    @application.get("/api/delivery-matches", response_model=ApiEnvelope)
    async def list_delivery_matches(
        x_client_id: str = Header(
            alias="X-Client-Id",
            min_length=8,
            max_length=100,
        ),
        session_token: str | None = Cookie(
            default=None,
            alias=SESSION_COOKIE_NAME,
        ),
        limit: int = Query(default=30, ge=1, le=100),
    ) -> ApiEnvelope:
        return ApiEnvelope(
            data=resolved_store.list_match_requests(
                client_id=match_owner_id(x_client_id, session_token),
                limit=limit,
            )
        )

    @application.patch(
        "/api/delivery-matches/{request_id}/cancel",
        response_model=ApiEnvelope,
    )
    async def cancel_delivery_match(
        request_id: str,
        x_client_id: str = Header(
            alias="X-Client-Id",
            min_length=8,
            max_length=100,
        ),
        session_token: str | None = Cookie(
            default=None,
            alias=SESSION_COOKIE_NAME,
        ),
    ) -> ApiEnvelope:
        canceled = resolved_store.cancel_match_request(
            request_id,
            client_id=match_owner_id(x_client_id, session_token),
        )
        if canceled is None:
            raise HTTPException(
                status_code=409,
                detail="대기 중인 본인의 공동배송 요청만 취소할 수 있습니다.",
            )
        return ApiEnvelope(
            data={"matching": public_match_request(canceled)},
            message="공동배송 매칭 대기를 취소했습니다.",
        )

    @application.post(
        "/api/delivery-matches/{request_id}/single-order",
        response_model=ApiEnvelope,
        status_code=201,
    )
    async def retry_delivery_match_as_single(
        request_id: str,
        x_client_id: str = Header(
            alias="X-Client-Id",
            min_length=8,
            max_length=100,
        ),
        session_token: str | None = Cookie(
            default=None,
            alias=SESSION_COOKIE_NAME,
        ),
    ) -> ApiEnvelope:
        owner_id = match_owner_id(x_client_id, session_token)
        match_request = resolved_store.get_match_request(
            request_id,
            client_id=owner_id,
        )
        if match_request is None:
            raise HTTPException(
                status_code=404,
                detail="본인의 스마트 딜리버리 요청을 찾을 수 없습니다.",
            )

        retryable = match_request["status"] in {"CANCELED", "EXPIRED", "FAILED"}
        if match_request["status"] == "MATCHED":
            pooled_order_id = match_request.get("partnerOrderId")
            pooled_order = (
                resolved_store.get_order(pooled_order_id)
                if pooled_order_id
                else None
            )
            retryable = bool(
                pooled_order
                and pooled_order["status"]
                in {
                    "CANCELED",
                    "ABORTED",
                    "MATCHING_FAILED",
                    "REQUEST_FAILED",
                }
            )
        if not retryable:
            raise HTTPException(
                status_code=409,
                detail=(
                    "매칭이 취소·만료·실패했거나 공동배송 주문이 취소된 뒤에만 "
                    "단일 퀵으로 다시 접수할 수 있습니다."
                ),
            )

        digest = hashlib.sha256(
            f"{owner_id}|{request_id}|single-retry".encode("utf-8")
        ).hexdigest()[:20]
        partner_order_id = f"quick-retry-{digest}"
        single_request = CreateDeliveryRequest.model_validate(
            match_request["request"]
        )
        result = await place_order(
            resolved_client,
            resolved_store,
            single_request,
            partner_order_id,
        )
        return ApiEnvelope(
            data={"orderResult": result},
            message="원래 배송 정보로 단일 퀵을 다시 접수했습니다.",
        )

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
    @application.post("/api/smart-delivery/quote", response_model=ApiEnvelope)
    async def bundle_quote_route(request: BundleQuoteRequest) -> ApiEnvelope:
        try:
            result = await multi_pickup_bundle_quote(
                resolved_client,
                resolved_geocoder,
                request,
                route_planner=resolved_routes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return ApiEnvelope(data=result)

    @application.post(
        "/api/bundle/orders",
        response_model=ApiEnvelope,
        status_code=201,
    )
    @application.post(
        "/api/smart-delivery/orders",
        response_model=ApiEnvelope,
        status_code=201,
    )
    async def create_bundle_order(
        request: BundleOrderRequest,
        idempotency_key: str | None = Header(
            default=None, alias="Idempotency-Key", max_length=100
        ),
    ) -> ApiEnvelope:
        partner_order_id = (
            request.partner_order_id
            or idempotency_key
            or f"smart-{uuid4().hex[:20]}"
        )
        try:
            quote, order_request = await prepare_bundle_order(
                resolved_client,
                resolved_geocoder,
                request,
                partner_order_id,
                route_planner=resolved_routes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        result = await place_order(
            resolved_client,
            resolved_store,
            order_request,
            partner_order_id,
        )
        message = result.pop("message", None)
        return ApiEnvelope(
            data={"quote": quote, "orderResult": result},
            message=message or "스마트 딜리버리 Sandbox 주문이 접수되었습니다.",
        )

    return application


app = create_app()
