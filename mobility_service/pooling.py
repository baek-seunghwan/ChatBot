from __future__ import annotations

import math
from typing import Any

from .client import KakaoMobilityClient
from .models import (
    Contact,
    CreateDeliveryRequest,
    DeliveryDraft,
    DeliveryStop,
    Location,
    OrderType,
    ProductSize,
)
from .rideshare import proportional_split, road_km

# 합승 성립 조건: 두 배송의 진행 방향이 비슷하고(방위각 차),
# 묶었을 때 총 주행거리가 각자 따로 갈 때 합보다 확실히 짧아야 한다.
MAX_BEARING_DIFF_DEG = 30.0
MAX_COMBINED_RATIO = 0.85
MAX_POOL_SIZE = 2  # v1: 2건 묶음까지


def _location(stop: dict[str, Any]) -> Location:
    return Location(
        basicAddress=stop["address"],
        latitude=stop["lat"],
        longitude=stop["lng"],
    )


def bearing_deg(a: Location, b: Location) -> float:
    """a→b 진행 방위각 (0~360도)."""
    lat1, lat2 = math.radians(a.latitude), math.radians(b.latitude)
    d_lng = math.radians(b.longitude - a.longitude)
    x = math.sin(d_lng) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lng)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def combined_route(requests: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """픽업 전부 → 도착 전부 순서의 경유 동선을 최근접 이웃으로 정렬한다.

    반환: [("pickup"|"dropoff", stop_dict), ...]
    (물건을 전부 실은 뒤 내리는 동선이라 픽업이 항상 도착보다 앞선다)
    """
    pickups = [request["pickup"] for request in requests]
    dropoffs = [request["dropoff"] for request in requests]

    ordered: list[tuple[str, dict[str, Any]]] = []
    cursor: Location | None = None
    for group, kind in ((pickups, "pickup"), (dropoffs, "dropoff")):
        remaining = list(range(len(group)))
        while remaining:
            if cursor is None:
                index = remaining[0]
            else:
                index = min(
                    remaining, key=lambda i: road_km(cursor, _location(group[i]))
                )
            ordered.append((kind, group[index]))
            cursor = _location(group[index])
            remaining.remove(index)
    return ordered


def route_km(requests: list[dict[str, Any]]) -> float:
    stops = combined_route(requests)
    total = 0.0
    for (_, current), (_, nxt) in zip(stops, stops[1:]):
        total += road_km(_location(current), _location(nxt))
    return total


def is_compatible(request_a: dict[str, Any], request_b: dict[str, Any]) -> bool:
    """두 배송 요청을 한 차량에 합승시킬 실익이 있는지 판정한다."""
    bearing_a = bearing_deg(_location(request_a["pickup"]), _location(request_a["dropoff"]))
    bearing_b = bearing_deg(_location(request_b["pickup"]), _location(request_b["dropoff"]))
    diff = abs(bearing_a - bearing_b)
    if min(diff, 360 - diff) > MAX_BEARING_DIFF_DEG:
        return False

    solo_sum = sum(
        road_km(_location(r["pickup"]), _location(r["dropoff"]))
        for r in (request_a, request_b)
    )
    if solo_sum <= 0:
        return False
    return route_km([request_a, request_b]) <= solo_sum * MAX_COMBINED_RATIO


def _price_of(data: Any) -> int | None:
    if isinstance(data, dict) and isinstance(data.get("totalPrice"), (int, float)):
        return int(data["totalPrice"])
    return None


def _draft_stops(requests: list[dict[str, Any]]) -> tuple[DeliveryStop, list[DeliveryStop], DeliveryStop]:
    stops = combined_route(requests)
    delivery_stops = []
    for kind, stop in stops:
        contact = None
        if stop.get("name") and stop.get("phone"):
            contact = Contact(name=stop["name"], phone=stop["phone"])
        delivery_stops.append(DeliveryStop(location=_location(stop), contact=contact))
    return delivery_stops[0], delivery_stops[1:-1], delivery_stops[-1]


async def pool_quote(
    client: KakaoMobilityClient, requests: list[dict[str, Any]]
) -> dict[str, Any]:
    """단독 견적 합계 vs 합승(경유지 묶음) 견적을 비교하고 분담금을 계산한다."""
    size = max(
        (ProductSize(r["product"].get("productSize", "XS")) for r in requests),
        key=lambda s: list(ProductSize).index(s),
    )

    solo_prices: list[int] = []
    for request in requests:
        draft = DeliveryDraft(
            orderType=OrderType.QUICK,
            productSize=ProductSize(request["product"].get("productSize", "XS")),
            pickup=DeliveryStop(location=_location(request["pickup"])),
            dropoff=DeliveryStop(location=_location(request["dropoff"])),
        )
        price = _price_of(await client.price(draft))
        if price is None:
            raise ValueError("단독 견적 조회에 실패했어요.")
        solo_prices.append(price)

    pickup, waypoints, dropoff = _draft_stops(requests)
    pooled_draft = DeliveryDraft(
        orderType=OrderType.QUICK,
        productSize=size,
        pickup=pickup,
        dropoff=dropoff,
        waypoints=waypoints,
    )
    pooled_price = _price_of(await client.price(pooled_draft))
    if pooled_price is None:
        raise ValueError("합승 견적 조회에 실패했어요.")

    shares = proportional_split(pooled_price, solo_prices)
    return {
        "soloPrices": solo_prices,
        "soloTotal": sum(solo_prices),
        "pooledPrice": pooled_price,
        "shares": shares,
        "savings": [solo - share for solo, share in zip(solo_prices, shares)],
        "groupSaving": sum(solo_prices) - pooled_price,
        "worthIt": pooled_price < sum(solo_prices),
    }


def build_pool_order(
    requests: list[dict[str, Any]], partner_order_id: str
) -> CreateDeliveryRequest:
    """합승 주문 1건을 조립한다. 모든 정차지에 연락처가 있어야 한다 (validator가 강제)."""
    size = max(
        (ProductSize(r["product"].get("productSize", "XS")) for r in requests),
        key=lambda s: list(ProductSize).index(s),
    )
    product_names = " + ".join(
        r["product"].get("productName", "배송 물품") for r in requests
    )
    pickup, waypoints, dropoff = _draft_stops(requests)
    return CreateDeliveryRequest(
        orderType=OrderType.QUICK,
        productSize=size,
        pickup=pickup,
        dropoff=dropoff,
        waypoints=waypoints,
        productName=f"[합승] {product_names}"[:100],
        partnerOrderId=partner_order_id,
    )
