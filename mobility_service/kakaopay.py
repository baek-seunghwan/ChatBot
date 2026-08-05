from __future__ import annotations

from typing import Any

import httpx

from .config import Settings


class KakaoPayError(RuntimeError):
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


class KakaoPayClient:
    """Server-only client for KakaoPay single-payment APIs.

    The Secret key must never be sent to the browser. The browser receives only
    the redirect URL returned by KakaoPay.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.kakaopay_base_url,
            timeout=settings.request_timeout_seconds,
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        if not self.settings.kakaopay_configured:
            raise KakaoPayError(
                "KAKAOPAY_SECRET_KEY_DEV 환경변수를 설정해야 합니다."
            )
        return {
            "Accept": "application/json",
            "Authorization": f"SECRET_KEY {self.settings.kakaopay_secret_key}",
            "Content-Type": "application/json",
        }

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(
                path,
                headers=self._headers(),
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise KakaoPayError("카카오페이 응답 시간이 초과됐습니다.") from exc
        except httpx.HTTPError as exc:
            raise KakaoPayError(
                "카카오페이 서버에 연결할 수 없습니다. 잠시 후 다시 시도해주세요."
            ) from exc

        try:
            body: Any = response.json() if response.content else {}
        except ValueError:
            body = {"error_message": response.text[:500]}

        if response.is_error or not isinstance(body, dict):
            detail = body.get("error_message") if isinstance(body, dict) else None
            raise KakaoPayError(
                detail or f"카카오페이 API가 {response.status_code} 오류를 반환했습니다.",
                status_code=response.status_code,
                response_body=body,
            )
        return body

    async def ready(
        self,
        *,
        partner_order_id: str,
        partner_user_id: str,
        item_name: str,
        quantity: int,
        total_amount: int,
        approval_url: str,
        cancel_url: str,
        fail_url: str,
    ) -> dict[str, Any]:
        return await self._post(
            "/online/v1/payment/ready",
            {
                "cid": self.settings.kakaopay_cid,
                "partner_order_id": partner_order_id,
                "partner_user_id": partner_user_id,
                "item_name": item_name[:100],
                "quantity": quantity,
                "total_amount": total_amount,
                "tax_free_amount": 0,
                "approval_url": approval_url,
                "cancel_url": cancel_url,
                "fail_url": fail_url,
            },
        )

    async def approve(
        self,
        *,
        tid: str,
        partner_order_id: str,
        partner_user_id: str,
        pg_token: str,
    ) -> dict[str, Any]:
        return await self._post(
            "/online/v1/payment/approve",
            {
                "cid": self.settings.kakaopay_cid,
                "tid": tid,
                "partner_order_id": partner_order_id,
                "partner_user_id": partner_user_id,
                "pg_token": pg_token,
            },
        )
