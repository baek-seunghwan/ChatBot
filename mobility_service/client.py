from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

from .auth import build_authorization
from .config import Settings
from .models import CreateDeliveryRequest, DeliveryDraft


class KakaoApiError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class KakaoMobilityClient:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=settings.request_timeout_seconds,
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        if not self.settings.configured:
            raise KakaoApiError(
                "KakaoMobility_API와 Vendor_ID 환경변수를 설정해야 합니다."
            )
        return {
            "accept": "application/json",
            "Authorization": build_authorization(self.settings.api_key),
            "Content-Type": "application/json",
            "vendor": self.settings.vendor_id,
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> Any:
        try:
            response = await self._client.request(
                method, path, headers=self._headers(), json=json
            )
        except httpx.TimeoutException as exc:
            raise KakaoApiError("카카오모빌리티 API 응답 시간이 초과됐습니다.") from exc
        except httpx.HTTPError as exc:
            raise KakaoApiError(
                f"카카오모빌리티 API 연결에 실패했습니다: {type(exc).__name__}"
            ) from exc

        try:
            body: Any = response.json() if response.content else None
        except ValueError:
            body = response.text[:1000]

        if response.is_error:
            detail = body
            if isinstance(body, dict):
                detail = (
                    body.get("message")
                    or body.get("error")
                    or body.get("detail")
                    or body
                )
            raise KakaoApiError(
                f"카카오모빌리티 API가 {response.status_code} 오류를 반환했습니다: {detail}",
                status_code=response.status_code,
                response_body=body,
            )
        return body

    async def auth_check(self) -> Any:
        return await self._request("GET", "/v1/auth/check")

    @staticmethod
    def _location(stop) -> dict[str, Any]:
        return stop.location.model_dump(
            mode="json", by_alias=True, exclude_none=True
        )

    @staticmethod
    def _stop(stop) -> dict[str, Any]:
        return stop.model_dump(mode="json", by_alias=True, exclude_none=True)

    async def estimate(self, draft: DeliveryDraft) -> Any:
        payload: dict[str, Any] = {
            "pickup": self._location(draft.pickup),
            "dropoff": self._location(draft.dropoff),
            "verticalTypes": [draft.order_type.value],
        }
        if draft.waypoints:
            payload["waypoints"] = [self._location(item) for item in draft.waypoints]
        if draft.wish_time:
            payload["wishTime"] = draft.wish_time
        return await self._request(
            "POST", "/goa-sandbox-service/api/v2/orders/estimate", json=payload
        )

    async def price(self, draft: DeliveryDraft) -> Any:
        payload: dict[str, Any] = {
            "orderType": draft.order_type.value,
            "productSize": draft.product_size.value,
            "pickup": {"location": self._location(draft.pickup)},
            "dropoff": {"location": self._location(draft.dropoff)},
        }
        if draft.waypoints:
            payload["waypoints"] = [
                {"location": self._location(item)} for item in draft.waypoints
            ]
        if draft.wish_time:
            payload["pickup"]["wishTime"] = draft.wish_time
        return await self._request(
            "POST", "/goa-sandbox-service/api/v2/orders/price", json=payload
        )

    async def create_order(
        self, request: CreateDeliveryRequest, partner_order_id: str
    ) -> Any:
        product: dict[str, Any] = {
            "name": request.product_name,
            "quantity": request.quantity,
        }
        if request.declared_value is not None:
            product["price"] = request.declared_value

        product_info: dict[str, Any] = {
            "size": request.product_size.value,
            "products": [product],
        }
        if request.declared_value is not None:
            product_info["totalPrice"] = request.declared_value

        payload: dict[str, Any] = {
            "partnerOrderId": partner_order_id,
            "orderType": request.order_type.value,
            "pickup": self._stop(request.pickup),
            "dropoff": self._stop(request.dropoff),
            "productInfo": product_info,
            "paymentInfo": {"paymentType": request.payment_type.value},
        }
        if request.waypoints:
            payload["waypoints"] = [self._stop(item) for item in request.waypoints]
        if request.wish_time:
            payload["pickup"]["wishTime"] = request.wish_time

        return await self._request(
            "POST", "/goa-sandbox-service/api/v2/orders", json=payload
        )

    async def get_order(self, partner_order_id: str) -> Any:
        safe_id = quote(partner_order_id, safe="")
        return await self._request(
            "GET", f"/goa-sandbox-service/api/v2/orders/{safe_id}"
        )

    async def get_picker(self, partner_order_id: str) -> Any:
        safe_id = quote(partner_order_id, safe="")
        return await self._request(
            "GET", f"/goa-sandbox-service/api/v2/orders/{safe_id}/picker"
        )

    async def cancel_order(self, partner_order_id: str) -> Any:
        safe_id = quote(partner_order_id, safe="")
        return await self._request(
            "PATCH", f"/goa-sandbox-service/api/v1/orders/{safe_id}/cancel"
        )

