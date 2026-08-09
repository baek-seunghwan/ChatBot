from __future__ import annotations

import asyncio
import copy
import hashlib
import secrets
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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from .agent import DeliveryAgent
from .bundle import multi_pickup_bundle_quote, prepare_bundle_order
from .chat_cache import cached_chat_response
from .client import KakaoApiError, KakaoMobilityClient
from .config import Settings
from .conversation_store import ConversationStore
from .directions import KakaoDirectionsClient, RoutePlanner
from .geocode import KakaoGeocodeClient
from .kakaopay import KakaoPayClient, KakaoPayError
from .knowledge import default_knowledge_base
from .local_responder import local_model_reply, ollama_status, vllm_status
from .matching import build_pooled_order, compatible_for_pooling
from .models import (
    AddressShareCreateRequest,
    AddressShareSubmitRequest,
    AgentChatRequest,
    ApiEnvelope,
    BundleOrderRequest,
    BundleQuoteRequest,
    CallbackBody,
    CreateDeliveryRequest,
    DeliveryDraft,
    KakaoPayReadyRequest,
    LoginRequest,
    OcrImageRequest,
    RegisterRequest,
    RouteSummaryRequest,
    SandboxStatusChange,
    SmartTextRequest,
)
from .ocr import OcrUnavailableError, extract_text_from_image
from .orders import (
    cancel_order_by_id,
    get_order_status,
    get_order_steps,
    place_order,
)
from .store import MobilityStore
from .smart_input import extract_dropoff_slots
from .user_store import DuplicateEmailError, SESSION_TTL_SECONDS, UserStore
from .web import (
    ABOUT_HTML,
    ADMIN_HTML,
    FEATURES_HTML,
    HISTORY_HTML,
    INDEX_HTML,
    RECIPIENT_ADDRESS_HTML,
)


SESSION_COOKIE_NAME = "movb_session"
BRAND_HERO_PATH = Path(__file__).with_name("assets") / "movb-brand-hero.webp"
KAKAOPAY_LOGO_PATH = Path(__file__).with_name("assets") / "kakaopay-logo.png"


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
    kakaopay_client: KakaoPayClient | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    resolved_store = store or MobilityStore(resolved_settings.database_path)
    resolved_client = client or KakaoMobilityClient(resolved_settings)
    resolved_geocoder = geocoder or KakaoGeocodeClient(resolved_settings)
    resolved_directions = directions or KakaoDirectionsClient(resolved_settings)
    resolved_routes = RoutePlanner(resolved_directions)
    resolved_conversations = conversations or ConversationStore(resolved_settings.database_path)
    resolved_users = user_store or UserStore(resolved_settings.database_path)
    resolved_kakaopay = kakaopay_client or KakaoPayClient(resolved_settings)
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
    owns_kakaopay = kakaopay_client is None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if owns_client:
            await resolved_client.close()
        if owns_geocoder:
            await resolved_geocoder.close()
        if owns_directions:
            await resolved_directions.close()
        if owns_kakaopay:
            await resolved_kakaopay.close()

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
    application.state.kakaopay = resolved_kakaopay

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

    @application.exception_handler(KakaoPayError)
    async def kakaopay_error_handler(
        _: Request, exc: KakaoPayError
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
        return ABOUT_HTML

    @application.get("/order", response_class=HTMLResponse, include_in_schema=False)
    async def order_page() -> str:
        return INDEX_HTML

    @application.get(
        "/bundle",
        response_class=HTMLResponse,
        response_model=None,
        include_in_schema=False,
    )
    async def bundle_page() -> RedirectResponse:
        return RedirectResponse(url="/order#smartDelivery", status_code=307)

    @application.get(
        "/smart-delivery/form",
        response_class=HTMLResponse,
        response_model=None,
        include_in_schema=False,
    )
    async def smart_delivery_form() -> RedirectResponse:
        return RedirectResponse(url="/order#smartDelivery", status_code=307)

    @application.get("/features", response_class=HTMLResponse, include_in_schema=False)
    async def features_page() -> str:
        return FEATURES_HTML

    @application.get("/history", response_class=HTMLResponse, include_in_schema=False)
    async def history_page() -> str:
        return HISTORY_HTML

    @application.get("/about", response_class=HTMLResponse, include_in_schema=False)
    async def about_page() -> str:
        return ABOUT_HTML

    @application.get(
        "/address-request/{token}",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def recipient_address_page(token: str) -> str:
        return RECIPIENT_ADDRESS_HTML

    @application.get(
        "/assets/movb-brand-hero.webp",
        response_class=FileResponse,
        include_in_schema=False,
    )
    async def brand_hero_image() -> FileResponse:
        return FileResponse(
            BRAND_HERO_PATH,
            media_type="image/webp",
            headers={"Cache-Control": "public, max-age=604800"},
        )

    @application.get(
        "/assets/kakaopay-logo.png",
        response_class=FileResponse,
        include_in_schema=False,
    )
    async def kakaopay_logo_image() -> FileResponse:
        return FileResponse(
            KAKAOPAY_LOGO_PATH,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=604800"},
        )

    @application.post("/api/smart-input/extract", response_model=ApiEnvelope)
    async def smart_text_extract(payload: SmartTextRequest) -> ApiEnvelope:
        return ApiEnvelope(data=extract_dropoff_slots(payload.text))

    @application.post("/api/smart-input/ocr", response_model=ApiEnvelope)
    async def smart_image_ocr(payload: OcrImageRequest) -> ApiEnvelope:
        try:
            text = await asyncio.to_thread(
                extract_text_from_image,
                payload.image_base64,
                payload.content_type,
            )
        except OcrUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return ApiEnvelope(
            data={
                "text": text,
                **extract_dropoff_slots(text),
            }
        )

    def public_address_share(item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in item.items()
            if key != "senderSessionId"
        }

    @application.post("/api/address-requests", response_model=ApiEnvelope)
    async def create_address_request(
        payload: AddressShareCreateRequest,
        request: Request,
        x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
    ) -> ApiEnvelope:
        session_id = x_session_id or f"share-{uuid4().hex}"
        token = secrets.token_urlsafe(24)
        item = resolved_store.create_address_share(
            token=token,
            sender_session_id=session_id,
            recipient_name=payload.recipient_name,
            recipient_phone=payload.recipient_phone,
        )
        base_url = str(request.base_url).rstrip("/")
        return ApiEnvelope(
            data={
                **public_address_share(item),
                "url": f"{base_url}/address-request/{token}",
            }
        )

    @application.get("/api/address-requests", response_model=ApiEnvelope)
    async def list_address_requests(
        x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
    ) -> ApiEnvelope:
        if not x_session_id:
            raise HTTPException(status_code=400, detail="세션 정보가 필요합니다.")
        items = resolved_store.list_address_shares(x_session_id)
        return ApiEnvelope(data=[public_address_share(item) for item in items])

    @application.get(
        "/api/address-requests/{token}",
        response_model=ApiEnvelope,
    )
    async def get_address_request(token: str) -> ApiEnvelope:
        item = resolved_store.get_address_share(token)
        if item is None:
            raise HTTPException(status_code=404, detail="주소 요청을 찾지 못했습니다.")
        return ApiEnvelope(data=public_address_share(item))

    @application.put(
        "/api/address-requests/{token}",
        response_model=ApiEnvelope,
    )
    async def submit_address_request(
        token: str,
        payload: AddressShareSubmitRequest,
    ) -> ApiEnvelope:
        item = resolved_store.complete_address_share(
            token,
            address=payload.address,
            detail_address=payload.detail_address,
            name=payload.name,
            phone=payload.phone,
            note=payload.note,
        )
        if item is None:
            raise HTTPException(
                status_code=404,
                detail="주소 요청을 찾을 수 없거나 링크가 만료되었습니다.",
            )
        return ApiEnvelope(
            data=public_address_share(item),
            message="받는 사람의 배송지가 전달되었습니다.",
        )

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
            return RedirectResponse(url="/order?admin=1", status_code=303)
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
            "kakaoPayConfigured": resolved_settings.kakaopay_configured,
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
                "kakaoPayConfigured": resolved_settings.kakaopay_configured,
                "kakaoPayEnvironment": (
                    "development"
                    if resolved_settings.kakaopay_cid == "TC0ONETIME"
                    else "production"
                ),
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

    @application.get("/api/vllm/status", response_model=ApiEnvelope)
    async def public_model_status() -> ApiEnvelope:
        return ApiEnvelope(data=await asyncio.to_thread(vllm_status))

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

    def payment_total(provider: Any, order_type: str) -> int:
        if not isinstance(provider, dict):
            raise KakaoPayError("배송 견적에서 결제 금액을 확인하지 못했습니다.")
        direct = provider.get("totalPrice") or provider.get("totalFareAmount")
        if isinstance(direct, (int, float)) and int(direct) > 0:
            return int(direct)
        rows = provider.get("lists")
        if isinstance(rows, list):
            selected = next(
                (
                    row
                    for row in rows
                    if isinstance(row, dict) and row.get("orderType") == order_type
                ),
                None,
            )
            if isinstance(selected, dict):
                amount = selected.get("totalPrice") or selected.get("totalFareAmount")
                if isinstance(amount, (int, float)) and int(amount) > 0:
                    return int(amount)
        raise KakaoPayError("배송 견적에서 결제 금액을 확인하지 못했습니다.")

    def payment_redirect_base(request: Request) -> str:
        return (
            resolved_settings.kakaopay_redirect_base_url
            or str(request.base_url).rstrip("/")
        )

    @application.post(
        "/api/payments/kakaopay/ready",
        response_model=ApiEnvelope,
    )
    async def kakaopay_ready(
        payload: KakaoPayReadyRequest,
        request: Request,
        user: dict[str, Any] = Depends(require_current_user),
        x_client_id: str | None = Header(
            default=None,
            alias="X-Client-Id",
            min_length=8,
            max_length=100,
        ),
    ) -> ApiEnvelope:
        if not resolved_settings.kakaopay_configured:
            raise HTTPException(
                status_code=503,
                detail="카카오페이 개발 키가 아직 설정되지 않았습니다.",
            )
        token = uuid4().hex
        payment_id = f"kp-{token[:24]}"
        partner_order_id = f"movb-pay-{token[:20]}"
        delivery_order_id = f"movb-{token[:20]}"
        partner_user_id = str(user["id"])
        match_request_id: str | None = None
        if payload.match_request_id is not None:
            match_request_id = payload.match_request_id
            match_request = resolved_store.get_match_request(
                match_request_id,
                client_id=f"user:{user['id']}",
            )
            if match_request is None and x_client_id:
                match_request = resolved_store.get_match_request(
                    match_request_id,
                    client_id=f"browser:{x_client_id}",
                )
            if match_request is None:
                raise HTTPException(
                    status_code=404,
                    detail="본인의 공동배송 매칭 요청을 찾지 못했습니다.",
                )
            if match_request["status"] != "MATCHED_AWAITING_PAYMENT":
                raise HTTPException(
                    status_code=409,
                    detail="매칭과 최종 요금 확정이 끝난 주문만 결제할 수 있습니다.",
                )
            if match_request["paymentStatus"] == "PAID":
                raise HTTPException(status_code=409, detail="이미 결제한 공동배송입니다.")
            group = resolved_store.get_match_group(match_request["matchId"])
            if group is None or group["status"] not in {
                "AWAITING_PAYMENT",
                "DISPATCHING",
            }:
                raise HTTPException(
                    status_code=409,
                    detail="현재 결제할 수 없는 공동배송 상태입니다.",
                )
            prepared_order = CreateDeliveryRequest.model_validate(
                group["pooledOrder"]
            )
            quote = group["quote"]
            amount = int(match_request["finalAmount"])
            delivery_order_id = f"match-payment-{token[:20]}"
            item_name = f"MOVB 공동배송 · {prepared_order.product_name}"
        elif payload.smart_order is not None:
            try:
                quote, prepared_order = await prepare_bundle_order(
                    resolved_client,
                    resolved_geocoder,
                    payload.smart_order,
                    delivery_order_id,
                    route_planner=resolved_routes,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            amount = int(quote["bundledPrice"])
            item_name = f"MOVB 스마트 딜리버리 · {payload.smart_order.product_name}"
        else:
            assert payload.order is not None
            prepared_order = payload.order
            quote = await resolved_client.price(prepared_order)
            amount = payment_total(quote, prepared_order.order_type.value)
            item_name = f"MOVB 배송 · {prepared_order.product_name}"
        delivery_payload = prepared_order.model_dump(
            mode="json", by_alias=True, exclude_none=True
        )
        resolved_store.create_kakaopay_payment(
            payment_id=payment_id,
            partner_order_id=partner_order_id,
            partner_user_id=partner_user_id,
            delivery_order_id=delivery_order_id,
            amount=amount,
            order_payload=delivery_payload,
            match_request_id=match_request_id,
        )
        callback_base = payment_redirect_base(request)
        callback_path = f"/api/payments/kakaopay/{payment_id}"
        try:
            ready = await resolved_kakaopay.ready(
                partner_order_id=partner_order_id,
                partner_user_id=partner_user_id,
                item_name=item_name,
                quantity=1,
                total_amount=amount,
                approval_url=f"{callback_base}{callback_path}/success",
                cancel_url=f"{callback_base}{callback_path}/cancel",
                fail_url=f"{callback_base}{callback_path}/fail",
            )
            tid = ready.get("tid")
            redirect_url = ready.get("next_redirect_pc_url")
            if not isinstance(tid, str) or not isinstance(redirect_url, str):
                raise KakaoPayError("카카오페이 결제 준비 응답이 올바르지 않습니다.")
            resolved_store.mark_kakaopay_ready(
                payment_id, tid=tid, response=ready
            )
        except Exception as exc:
            resolved_store.mark_kakaopay_status(
                payment_id, "READY_FAILED", error=str(exc)
            )
            raise
        return ApiEnvelope(
            data={
                "paymentId": payment_id,
                "amount": amount,
                "nextRedirectPcUrl": redirect_url,
                "nextRedirectMobileUrl": ready.get("next_redirect_mobile_url"),
            },
            message="카카오페이 결제창을 준비했습니다.",
        )

    def payment_result_redirect(
        result: str,
        *,
        payment_id: str,
        delivery_order_id: str | None = None,
    ) -> RedirectResponse:
        query = f"payment={result}&paymentId={payment_id}"
        if delivery_order_id:
            query += f"&orderId={delivery_order_id}"
        return RedirectResponse(url=f"/order?{query}", status_code=303)

    @application.get(
        "/api/payments/kakaopay/{payment_id}/success",
        response_model=None,
        include_in_schema=False,
    )
    async def kakaopay_success(
        payment_id: str,
        pg_token: str = Query(min_length=1, max_length=500),
    ) -> RedirectResponse:
        payment = resolved_store.get_kakaopay_payment(payment_id)
        if payment is None:
            raise HTTPException(status_code=404, detail="결제 요청을 찾지 못했습니다.")
        if payment["status"] == "COMPLETED":
            result = "success"
            delivery_order_id = payment["deliveryOrderId"]
            if payment.get("matchRequestId"):
                match_request = resolved_store.get_match_request(
                    payment["matchRequestId"]
                )
                if match_request and match_request["status"] == "MATCHED":
                    delivery_order_id = match_request["partnerOrderId"]
                else:
                    result = "waiting-peer"
                    delivery_order_id = None
            return payment_result_redirect(
                result,
                payment_id=payment_id,
                delivery_order_id=delivery_order_id,
            )
        if payment["status"] != "READY" or not payment.get("tid"):
            return payment_result_redirect("fail", payment_id=payment_id)
        if not resolved_store.claim_kakaopay_approval(payment_id):
            latest = resolved_store.get_kakaopay_payment(payment_id)
            if latest and latest["status"] == "COMPLETED":
                return payment_result_redirect(
                    "success",
                    payment_id=payment_id,
                    delivery_order_id=latest["deliveryOrderId"],
                )
            return payment_result_redirect("fail", payment_id=payment_id)
        try:
            approval = await resolved_kakaopay.approve(
                tid=payment["tid"],
                partner_order_id=payment["partnerOrderId"],
                partner_user_id=payment["partnerUserId"],
                pg_token=pg_token,
            )
            resolved_store.mark_kakaopay_status(
                payment_id, "APPROVED", response=approval
            )
        except Exception as exc:
            resolved_store.mark_kakaopay_status(
                payment_id, "APPROVAL_FAILED", error=str(exc)
            )
            return payment_result_redirect("fail", payment_id=payment_id)

        delivery_order_id = payment["deliveryOrderId"]
        if payment.get("matchRequestId"):
            group = None
            try:
                group, should_dispatch = resolved_store.mark_match_participant_paid(
                    payment["matchRequestId"]
                )
                if should_dispatch:
                    delivery_request = CreateDeliveryRequest.model_validate(
                        group["pooledOrder"]
                    )
                    await place_order(
                        resolved_client,
                        resolved_store,
                        delivery_request,
                        group["providerOrderId"],
                    )
                    resolved_store.complete_paid_match(group["matchId"])
                    delivery_order_id = group["providerOrderId"]
                else:
                    delivery_order_id = None
            except Exception as exc:
                if group is not None:
                    resolved_store.fail_paid_match(group["matchId"], str(exc))
                resolved_store.mark_kakaopay_status(
                    payment_id, "DELIVERY_FAILED", response=approval, error=str(exc)
                )
                return payment_result_redirect(
                    "delivery-failed", payment_id=payment_id
                )
        else:
            delivery_request = CreateDeliveryRequest.model_validate(payment["order"])
            try:
                await place_order(
                    resolved_client,
                    resolved_store,
                    delivery_request,
                    payment["deliveryOrderId"],
                )
            except Exception as exc:
                resolved_store.mark_kakaopay_status(
                    payment_id, "DELIVERY_FAILED", response=approval, error=str(exc)
                )
                return payment_result_redirect(
                    "delivery-failed", payment_id=payment_id
                )
        resolved_store.mark_kakaopay_status(
            payment_id, "COMPLETED", response=approval
        )
        return payment_result_redirect(
            "success" if delivery_order_id else "waiting-peer",
            payment_id=payment_id,
            delivery_order_id=delivery_order_id,
        )

    @application.get(
        "/api/payments/kakaopay/{payment_id}/cancel",
        response_model=None,
        include_in_schema=False,
    )
    async def kakaopay_cancel(payment_id: str) -> RedirectResponse:
        if resolved_store.get_kakaopay_payment(payment_id):
            resolved_store.mark_kakaopay_status(payment_id, "CANCELED")
        return payment_result_redirect("cancel", payment_id=payment_id)

    @application.get(
        "/api/payments/kakaopay/{payment_id}/fail",
        response_model=None,
        include_in_schema=False,
    )
    async def kakaopay_fail(payment_id: str) -> RedirectResponse:
        if resolved_store.get_kakaopay_payment(payment_id):
            resolved_store.mark_kakaopay_status(payment_id, "FAILED")
        return payment_result_redirect("fail", payment_id=payment_id)

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

    async def prepare_delivery_match(
        *,
        first: dict[str, Any],
        second: dict[str, Any],
        match_id: str,
        partner_order_id: str,
    ) -> None:
        """Price one pooled route and freeze each user's before/after amount."""
        pooled_order = build_pooled_order(
            first["request"],
            second["request"],
            partner_order_id,
        )
        first_order = CreateDeliveryRequest.model_validate(first["request"])
        second_order = CreateDeliveryRequest.model_validate(second["request"])
        pooled_quote, first_quote, second_quote = await asyncio.gather(
            resolved_client.price(pooled_order),
            resolved_client.price(first_order),
            resolved_client.price(second_order),
        )
        pooled_total = payment_total(pooled_quote, pooled_order.order_type.value)
        first_total = payment_total(first_quote, first["request"]["orderType"])
        second_total = payment_total(second_quote, second["request"]["orderType"])
        solo_total = first_total + second_total
        first_amount = max(1, round(pooled_total * first_total / solo_total))
        second_amount = pooled_total - first_amount
        if second_amount <= 0:
            raise ValueError("공동배송 최종 결제 금액을 나누지 못했습니다.")
        resolved_store.prepare_matched_payments(
            match_id=match_id,
            request_amounts={
                first["requestId"]: first_amount,
                second["requestId"]: second_amount,
            },
            original_amounts={
                first["requestId"]: first_total,
                second["requestId"]: second_total,
            },
            provider_order_id=partner_order_id,
            pooled_order=pooled_order.model_dump(
                mode="json", by_alias=True, exclude_none=True
            ),
            quote=pooled_quote,
            total_amount=pooled_total,
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
                await prepare_delivery_match(
                    first=candidate,
                    second={"requestId": request_id, "request": request_payload},
                    match_id=match_id,
                    partner_order_id=partner_order_id,
                )
            except Exception as exc:
                resolved_store.fail_match_requests(request_ids, str(exc))
                raise
            matched = resolved_store.get_match_request(request_id)
            return ApiEnvelope(
                data={
                    "matching": public_match_request(matched or saved),
                },
                message=(
                    "다른 사용자의 배송과 매칭되어 최종 요금이 확정되었습니다. "
                    "양쪽 결제가 끝나면 배송을 접수합니다."
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

    @application.post(
        "/api/delivery-matches/{request_id}/demo-match",
        response_model=ApiEnvelope,
    )
    async def create_demo_delivery_match(
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
        """Create a clearly labelled Sandbox peer and run the real match logic."""
        owner_id = match_owner_id(x_client_id, session_token)
        current = resolved_store.get_match_request(
            request_id,
            client_id=owner_id,
        )
        if current is None:
            raise HTTPException(
                status_code=404,
                detail="본인의 매칭 요청을 찾을 수 없습니다.",
            )
        if current["status"] != "WAITING":
            raise HTTPException(
                status_code=409,
                detail="대기 중인 주문에서만 Sandbox 데모 매칭을 시작할 수 있습니다.",
            )

        demo_payload = copy.deepcopy(current["request"])
        demo_payload.pop("partnerOrderId", None)
        demo_payload["pickup"]["contact"] = {
            "name": "Sandbox 데모 발송자",
            "phone": "010-1000-0901",
        }
        demo_payload["dropoff"]["contact"] = {
            "name": "Sandbox 데모 수령인",
            "phone": "010-1000-0902",
        }
        demo_payload["productName"] = "Sandbox 데모 배송"
        demo_request_id = f"demo-peer-{uuid4().hex[:20]}"
        demo_owner_id = f"sandbox-demo:{uuid4().hex[:16]}"
        expires_at = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
        demo_request, _ = resolved_store.reserve_match_request(
            demo_request_id,
            demo_owner_id,
            demo_payload,
            expires_at,
        )
        match_id = f"demo-match-{uuid4().hex[:20]}"
        if not resolved_store.claim_match_requests(
            request_id,
            demo_request_id,
            match_id,
        ):
            raise HTTPException(
                status_code=409,
                detail="매칭 상태가 바뀌었습니다. 이용 내역에서 다시 확인해주세요.",
            )
        partner_order_id = f"smart-demo-{uuid4().hex[:18]}"
        try:
            await prepare_delivery_match(
                first=current,
                second=demo_request,
                match_id=match_id,
                partner_order_id=partner_order_id,
            )
            # 데모 상대는 결제를 마친 것으로 처리해 사용자의 결제 후 즉시 접수된다.
            resolved_store.mark_match_participant_paid(demo_request_id)
        except Exception as exc:
            resolved_store.fail_match_requests(
                (request_id, demo_request_id),
                str(exc),
            )
            raise

        matched = resolved_store.get_match_request(request_id, client_id=owner_id)
        response_matching = public_match_request(matched or current)
        response_matching["demoMatch"] = True
        return ApiEnvelope(
            data={"matching": response_matching},
            message=(
                "Sandbox 데모 주문과 실제 매칭 로직으로 묶었습니다. "
                "단독 배송가와 최종 할인 요금을 비교해보세요."
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
        cached = cached_chat_response(request.message)
        if cached is not None:
            session = resolved_conversations.get_or_create(session_id)
            resolved_conversations.append_turn(
                session_id, "user", request.message
            )
            resolved_conversations.append_turn(
                session_id, "assistant", cached.reply
            )
            return ApiEnvelope(
                data={
                    "sessionId": session_id,
                    "reply": cached.reply,
                    "stage": session["stage"],
                    "slots": session["slots"],
                    "quote": session["quote"],
                    "order": None,
                    "sources": [],
                    "actions": [
                        {
                            "label": "스마트 딜리버리",
                            "message": "스마트 딜리버리 접수 화면 열기",
                            "target": "smartDelivery",
                        },
                        {
                            "label": "배송 주문",
                            "message": "퀵 배송 주문하고 싶어",
                        },
                        {
                            "label": "배송 상태",
                            "message": "배송 상태를 확인하고 싶어",
                        },
                    ],
                    "changedSlots": {},
                    "trace": [
                        f"response_cache:{cached.match_type}:{cached.key}"
                    ],
                }
            )
        if request.mode == "local":
            reply = await asyncio.to_thread(
                local_model_reply,
                request.message,
                request.local_engine,
                request.form_snapshot,
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
        result = await resolved_agent.achat(
            session_id=session_id,
            message=request.message,
            form_snapshot=request.form_snapshot,
            input_context=request.input_context,
        )
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
