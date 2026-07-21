from __future__ import annotations

from typing import Any

from .client import KakaoApiError, KakaoMobilityClient
from .geocode import KakaoGeocodeClient
from .models import DeliveryDraft, DeliveryStop, Location, OrderType, ProductSize
from .rideshare import road_km

# 할인 자체는 카카오의 '경유지 추가 혜택'이 이미 제공한다.
# 이 모듈의 역할은 폼이 해주지 않는 판단 — "따로 보낼까 묶어 보낼까"의 자동 비교와
# 최적 경유 순서 추천 — 이다.
MAX_BUNDLE_DROPOFFS = 5


def _total_price(data: Any) -> int | None:
    if isinstance(data, dict) and isinstance(data.get("totalPrice"), (int, float)):
        return int(data["totalPrice"])
    return None


def _nearest_neighbor_order(pickup: Location, stops: list[Location]) -> list[int]:
    """출발지에서 가까운 곳부터 차례로 도는 휴리스틱 경유 순서."""
    remaining = list(range(len(stops)))
    order: list[int] = []
    cursor = pickup
    while remaining:
        nearest = min(remaining, key=lambda i: road_km(cursor, stops[i]))
        order.append(nearest)
        cursor = stops[nearest]
        remaining.remove(nearest)
    return order


async def bundle_quote(
    client: KakaoMobilityClient,
    geocoder: KakaoGeocodeClient,
    pickup_address: str,
    dropoff_addresses: list[str],
    product_size: str = "XS",
) -> dict[str, Any]:
    """여러 도착지를 '각각 따로 보낼 때'와 '한 번에 묶어 보낼 때' 요금을 비교한다.

    - 경유지는 QUICK만 지원하므로 묶음 견적은 QUICK 기준으로 계산한다.
    - 묶음 요금에는 BUNDLE_DISCOUNT_RATE 만큼 MoveOps 묶음 할인을 추가 적용한다.
    """
    if not 2 <= len(dropoff_addresses) <= MAX_BUNDLE_DROPOFFS:
        raise ValueError(f"묶음 배송은 도착지 2~{MAX_BUNDLE_DROPOFFS}곳까지 지원해요.")

    pickup = await geocoder.search_address(pickup_address)
    if pickup is None:
        raise ValueError(f"출발지 주소를 찾지 못했어요: {pickup_address}")

    dropoffs: list[Location] = []
    for address in dropoff_addresses:
        location = await geocoder.search_address(address)
        if location is None:
            raise ValueError(f"도착지 주소를 찾지 못했어요: {address}")
        dropoffs.append(location)

    size = ProductSize(product_size)
    pickup_stop = DeliveryStop(location=pickup)

    # 1) 각각 따로 보낼 때: 도착지마다 개별 QUICK 견적
    individual: list[dict[str, Any]] = []
    individual_total = 0
    for address, location in zip(dropoff_addresses, dropoffs):
        draft = DeliveryDraft(
            orderType=OrderType.QUICK,
            productSize=size,
            pickup=pickup_stop,
            dropoff=DeliveryStop(location=location),
        )
        try:
            price = _total_price(await client.price(draft))
        except KakaoApiError:
            price = None
        if price is None:
            raise ValueError(f"'{address}' 개별 견적 조회에 실패했어요.")
        individual.append({"address": location.basic_address, "price": price})
        individual_total += price

    # 2) 묶어서 보낼 때: 가까운 곳부터 도는 순서로 경유지 구성
    order = _nearest_neighbor_order(pickup, dropoffs)
    ordered = [dropoffs[i] for i in order]
    bundled_draft = DeliveryDraft(
        orderType=OrderType.QUICK,
        productSize=size,
        pickup=pickup_stop,
        dropoff=DeliveryStop(location=ordered[-1]),
        waypoints=[DeliveryStop(location=loc) for loc in ordered[:-1]],
    )
    bundled_price = _total_price(await client.price(bundled_draft))
    if bundled_price is None:
        raise ValueError("묶음 견적 조회에 실패했어요.")

    return {
        "pickup": pickup.basic_address,
        "individual": individual,
        "individualTotal": individual_total,
        "route": [ordered_stop.basic_address for ordered_stop in ordered],
        "bundledPrice": bundled_price,
        "saving": individual_total - bundled_price,
        "recommendBundle": bundled_price < individual_total,
    }
