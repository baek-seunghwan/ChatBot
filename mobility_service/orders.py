from __future__ import annotations

import asyncio
from typing import Any

from fastapi import HTTPException

from .client import KakaoApiError, KakaoMobilityClient
from .models import CreateDeliveryRequest
from .store import MobilityStore


async def place_order(
    client: KakaoMobilityClient,
    store: MobilityStore,
    request: CreateDeliveryRequest,
    partner_order_id: str,
) -> dict[str, Any]:
    """수동 폼(`POST /api/orders`)과 채팅 에이전트가 공유하는 주문 생성 로직."""
    if not all(char.isalnum() or char in "._-" for char in partner_order_id):
        raise HTTPException(
            status_code=422,
            detail="주문 ID는 영문, 숫자, 마침표, 밑줄, 하이픈만 사용할 수 있습니다.",
        )

    request_payload = request.model_dump(mode="json", by_alias=True, exclude_none=True)
    if not store.reserve_order(partner_order_id, request_payload):
        existing = store.get_order(partner_order_id)
        return {
            "source": "existing",
            "order": existing,
            "message": "같은 주문 ID가 이미 처리되어 기존 결과를 반환했습니다.",
        }

    try:
        response = await client.create_order(request, partner_order_id)
    except KakaoApiError as exc:
        store.fail_order(partner_order_id, str(exc))
        raise
    store.complete_order(partner_order_id, response)
    return {
        "source": "created",
        "partnerOrderId": partner_order_id,
        "provider": response,
        "order": store.get_order(partner_order_id),
    }


async def get_order_status(
    client: KakaoMobilityClient,
    store: MobilityStore,
    partner_order_id: str,
    refresh: bool,
) -> dict[str, Any] | None:
    local_order = store.get_order(partner_order_id)
    if refresh:
        provider = await client.get_order(partner_order_id)
        if local_order is None:
            store.reserve_order(partner_order_id, {"restoredFromProvider": True})
        store.sync_order(partner_order_id, provider)
        local_order = store.get_order(partner_order_id)
    return local_order


async def cancel_order_by_id(
    client: KakaoMobilityClient,
    store: MobilityStore,
    partner_order_id: str,
) -> dict[str, Any]:
    response = await client.cancel_order(partner_order_id)
    store.set_status(partner_order_id, "CANCELED")
    return {"provider": response, "order": store.get_order(partner_order_id)}


def _provider_step_refs(provider: Any) -> list[dict[str, str]]:
    if not isinstance(provider, dict):
        return []
    refs: list[dict[str, str]] = []
    pickup = provider.get("pickup")
    if isinstance(pickup, dict) and pickup.get("stepId"):
        refs.append({"kind": "PICKUP", "stepId": str(pickup["stepId"])})
    for index, waypoint in enumerate(provider.get("waypoints") or [], start=1):
        if isinstance(waypoint, dict) and waypoint.get("stepId"):
            refs.append(
                {"kind": f"WAYPOINT_{index}", "stepId": str(waypoint["stepId"])}
            )
    dropoff = provider.get("dropoff")
    if isinstance(dropoff, dict) and dropoff.get("stepId"):
        refs.append({"kind": "DROPOFF", "stepId": str(dropoff["stepId"])})
    return refs


async def get_order_steps(
    client: KakaoMobilityClient,
    store: MobilityStore,
    partner_order_id: str,
    *,
    refresh_order: bool = True,
) -> dict[str, Any]:
    """주문 응답의 실제 stepId를 찾아 출발지·경유지·목적지를 각각 조회한다."""
    local_order = await get_order_status(
        client, store, partner_order_id, refresh=refresh_order
    )
    if local_order is None:
        return {"order": None, "steps": []}
    refs = _provider_step_refs(local_order.get("response"))
    if not refs:
        return {"order": local_order, "steps": []}

    results = await asyncio.gather(
        *(client.get_step(partner_order_id, item["stepId"]) for item in refs),
        return_exceptions=True,
    )
    steps: list[dict[str, Any]] = []
    for ref, result in zip(refs, results):
        if isinstance(result, Exception):
            steps.append(
                {
                    **ref,
                    "status": "UNKNOWN",
                    "error": str(result),
                }
            )
            continue
        detail = result if isinstance(result, dict) else {"raw": result}
        steps.append({**ref, **detail})
    return {"order": local_order, "steps": steps}
