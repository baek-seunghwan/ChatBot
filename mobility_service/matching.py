from __future__ import annotations

from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from typing import Any

from .models import CreateDeliveryRequest, DeliveryStop, PaymentType


MAX_PICKUP_DISTANCE_KM = 8.0
MAX_DROPOFF_DISTANCE_KM = 12.0
MAX_RESERVATION_GAP_MINUTES = 30


def _distance_km(first: dict[str, Any], second: dict[str, Any]) -> float:
    lat1 = radians(float(first["latitude"]))
    lat2 = radians(float(second["latitude"]))
    lat_delta = lat2 - lat1
    lng_delta = radians(float(second["longitude"]) - float(first["longitude"]))
    value = (
        sin(lat_delta / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(lng_delta / 2) ** 2
    )
    return 6371.0 * 2 * asin(sqrt(value))


def _location(payload: dict[str, Any], stop: str) -> dict[str, Any]:
    return payload[stop]["location"]


def _reservation_gap_minutes(first: str | None, second: str | None) -> float:
    if not first or not second:
        return 0
    try:
        return abs(
            (
                datetime.fromisoformat(first.replace("Z", "+00:00"))
                - datetime.fromisoformat(second.replace("Z", "+00:00"))
            ).total_seconds()
        ) / 60
    except (TypeError, ValueError):
        return float("inf")


def compatible_for_pooling(
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool:
    """두 일반 퀵 주문이 실제 한 경로로 묶일 수 있는지 보수적으로 판정한다."""
    if first.get("orderType") != second.get("orderType"):
        return False
    if first.get("orderType") not in {"QUICK", "QUICK_EXPRESS"}:
        return False
    if first.get("productSize") != second.get("productSize"):
        return False
    if first.get("fleetOption") != second.get("fleetOption"):
        return False
    if first.get("paymentType", "CARD") != "CARD":
        return False
    if second.get("paymentType", "CARD") != "CARD":
        return False
    if _reservation_gap_minutes(
        first.get("wishTime"),
        second.get("wishTime"),
    ) > MAX_RESERVATION_GAP_MINUTES:
        return False

    pickup_distance = _distance_km(
        _location(first, "pickup"),
        _location(second, "pickup"),
    )
    dropoff_distance = _distance_km(
        _location(first, "dropoff"),
        _location(second, "dropoff"),
    )
    if pickup_distance > MAX_PICKUP_DISTANCE_KM:
        return False
    if dropoff_distance > MAX_DROPOFF_DISTANCE_KM:
        return False

    aligned = pickup_distance + dropoff_distance
    crossed = _distance_km(
        _location(first, "pickup"),
        _location(second, "dropoff"),
    ) + _distance_km(
        _location(second, "pickup"),
        _location(first, "dropoff"),
    )
    return aligned < crossed


def _with_note(stop: DeliveryStop, prefix: str) -> DeliveryStop:
    note = f"{prefix} {stop.note or ''}".strip()
    return stop.model_copy(update={"note": note})


def _route_cost(
    first: CreateDeliveryRequest,
    second: CreateDeliveryRequest,
) -> float:
    first_payload = first.model_dump(mode="json", by_alias=True)
    second_payload = second.model_dump(mode="json", by_alias=True)
    return (
        _distance_km(
            _location(first_payload, "pickup"),
            _location(second_payload, "pickup"),
        )
        + _distance_km(
            _location(second_payload, "pickup"),
            _location(first_payload, "dropoff"),
        )
        + _distance_km(
            _location(first_payload, "dropoff"),
            _location(second_payload, "dropoff"),
        )
    )


def build_pooled_order(
    first_payload: dict[str, Any],
    second_payload: dict[str, Any],
    partner_order_id: str,
) -> CreateDeliveryRequest:
    first = CreateDeliveryRequest.model_validate(first_payload)
    second = CreateDeliveryRequest.model_validate(second_payload)
    if _route_cost(second, first) < _route_cost(first, second):
        first, second = second, first

    wish_times = [value for value in (first.wish_time, second.wish_time) if value]
    declared_values = [
        value for value in (first.declared_value, second.declared_value)
        if value is not None
    ]
    product_names = [first.product_name, second.product_name]
    return CreateDeliveryRequest(
        partnerOrderId=partner_order_id,
        orderType=first.order_type,
        productSize=first.product_size,
        pickup=_with_note(first.pickup, "[공동배송 픽업 1]"),
        waypoints=[
            _with_note(second.pickup, "[추가 픽업]"),
            _with_note(first.dropoff, "[공동배송 배송 1]"),
        ],
        dropoff=_with_note(second.dropoff, "[공동배송 배송 2]"),
        wishTime=min(wish_times) if wish_times else None,
        productName="공동배송 · " + " / ".join(product_names),
        quantity="2건",
        declaredValue=sum(declared_values) if declared_values else None,
        paymentType=PaymentType.CARD,
        fleetOption=first.fleet_option,
    )
