from __future__ import annotations

from typing import Any

from .client import KakaoApiError, KakaoMobilityClient
from .directions import RoutePlanner
from .geo_math import road_km
from .geocode import KakaoGeocodeClient
from .models import (
    BundleOrderRequest,
    BundleQuoteRequest,
    CreateDeliveryRequest,
    DeliveryDraft,
    DeliveryStop,
    Location,
    OrderType,
    PaymentType,
)

MAX_BUNDLE_PICKUPS = 5
MAX_BUNDLE_DROPOFFS = 5


def _total_price(data: Any) -> int | None:
    if isinstance(data, dict) and isinstance(data.get("totalPrice"), (int, float)):
        return int(data["totalPrice"])
    return None


async def _nearest_neighbor_order(
    start: Location,
    stops: list[Location],
    route_planner: RoutePlanner | None,
) -> list[int]:
    """현재 위치에서 가까운 정차지를 하나씩 선택한다."""
    remaining = list(range(len(stops)))
    order: list[int] = []
    cursor = start
    while remaining:
        if route_planner is None:
            nearest = min(remaining, key=lambda index: road_km(cursor, stops[index]))
        else:
            distances: list[tuple[int, int]] = []
            for index in remaining:
                route = await route_planner.route_summary(cursor, stops[index])
                distances.append((route["distanceMeters"], index))
            nearest = min(distances)[1]
        order.append(nearest)
        cursor = stops[nearest]
        remaining.remove(nearest)
    return order


async def _resolve_locations(
    geocoder: KakaoGeocodeClient,
    addresses: list[str],
    *,
    label: str,
) -> list[Location]:
    locations: list[Location] = []
    for address in addresses:
        location = await geocoder.search_address(address)
        if location is None:
            raise ValueError(f"{label} 주소를 찾지 못했어요: {address}")
        locations.append(location)
    return locations


async def _bundle_route_order(
    pickups: list[Location],
    dropoffs: list[Location],
    route_planner: RoutePlanner | None,
) -> tuple[list[int], list[int]]:
    """첫 픽업에서 시작해 추가 픽업을 모두 마친 뒤 배송한다.

    물품을 싣기 전에 해당 배송지에 도착하는 일이 없도록 픽업과 배송의 순서를
    두 단계로 분리한다.
    """
    pickup_order = [0]
    if len(pickups) > 1:
        additional_order = await _nearest_neighbor_order(
            pickups[0], pickups[1:], route_planner
        )
        pickup_order.extend(index + 1 for index in additional_order)

    dropoff_order = await _nearest_neighbor_order(
        pickups[pickup_order[-1]],
        dropoffs,
        route_planner,
    )
    return pickup_order, dropoff_order


async def multi_pickup_bundle_quote(
    client: KakaoMobilityClient,
    geocoder: KakaoGeocodeClient,
    request: BundleQuoteRequest,
    *,
    route_planner: RoutePlanner | None = None,
) -> dict[str, Any]:
    """여러 픽업과 여러 배송을 개별 주문 가격과 하나의 묶음 가격으로 비교한다."""
    pickup_locations = await _resolve_locations(
        geocoder,
        [item.address for item in request.pickups],
        label="픽업",
    )
    dropoff_locations = await _resolve_locations(
        geocoder,
        [item.address for item in request.dropoffs],
        label="배송",
    )

    individual: list[dict[str, Any]] = []
    individual_total = 0
    for dropoff_request, dropoff_location in zip(
        request.dropoffs, dropoff_locations
    ):
        pickup_location = pickup_locations[dropoff_request.pickup_index]
        draft = DeliveryDraft(
            orderType=OrderType.QUICK,
            productSize=request.product_size,
            pickup=DeliveryStop(location=pickup_location),
            dropoff=DeliveryStop(location=dropoff_location),
            fleetOption=request.fleet_option,
        )
        try:
            price = _total_price(await client.price(draft))
        except KakaoApiError:
            price = None
        if price is None:
            raise ValueError(
                f"'{dropoff_request.address}' 개별 견적 조회에 실패했어요."
            )
        individual.append(
            {
                "pickupAddress": pickup_location.basic_address,
                "address": dropoff_location.basic_address,
                "pickupIndex": dropoff_request.pickup_index,
                "price": price,
            }
        )
        individual_total += price

    pickup_order, dropoff_order = await _bundle_route_order(
        pickup_locations,
        dropoff_locations,
        route_planner,
    )
    ordered_pickups = [pickup_locations[index] for index in pickup_order]
    ordered_dropoffs = [dropoff_locations[index] for index in dropoff_order]
    waypoint_locations = [*ordered_pickups[1:], *ordered_dropoffs[:-1]]

    bundled_draft = DeliveryDraft(
        orderType=OrderType.QUICK,
        productSize=request.product_size,
        pickup=DeliveryStop(location=ordered_pickups[0]),
        waypoints=[
            DeliveryStop(location=location) for location in waypoint_locations
        ],
        dropoff=DeliveryStop(location=ordered_dropoffs[-1]),
        fleetOption=request.fleet_option,
    )
    bundled_price = _total_price(await client.price(bundled_draft))
    if bundled_price is None:
        raise ValueError("스마트 딜리버리 견적 조회에 실패했어요.")

    route_info = (
        await route_planner.route_summary(
            ordered_pickups[0],
            ordered_dropoffs[-1],
            waypoints=waypoint_locations,
        )
        if route_planner
        else None
    )
    pickup_route = [location.basic_address for location in ordered_pickups]
    dropoff_route = [location.basic_address for location in ordered_dropoffs]

    return {
        "pickup": pickup_locations[0].basic_address,
        "pickups": [location.basic_address for location in pickup_locations],
        "individual": individual,
        "individualTotal": individual_total,
        "pickupRoute": pickup_route,
        "dropoffRoute": dropoff_route,
        "route": [*pickup_route, *dropoff_route],
        "routeStops": [
            *(
                {"kind": "PICKUP", "address": address}
                for address in pickup_route
            ),
            *(
                {"kind": "DROPOFF", "address": address}
                for address in dropoff_route
            ),
        ],
        "bundledPrice": bundled_price,
        "saving": individual_total - bundled_price,
        "recommendBundle": bundled_price < individual_total,
        "routeInfo": route_info,
    }


async def bundle_quote(
    client: KakaoMobilityClient,
    geocoder: KakaoGeocodeClient,
    pickup_address: str,
    dropoff_addresses: list[str],
    product_size: str = "XS",
    route_planner: RoutePlanner | None = None,
) -> dict[str, Any]:
    """AI 채팅의 한 곳 픽업 견적을 스마트 딜리버리 엔진에 연결한다."""
    request = BundleQuoteRequest.model_validate(
        {
            "pickups": [{"address": pickup_address}],
            "dropoffs": [
                {"address": address, "pickupIndex": 0}
                for address in dropoff_addresses
            ],
            "productSize": product_size,
        }
    )
    return await multi_pickup_bundle_quote(
        client,
        geocoder,
        request,
        route_planner=route_planner,
    )


def _tagged_note(kind: str, note: str | None) -> str:
    return f"[{kind}]" + (f" {note}" if note else "")


async def prepare_bundle_order(
    client: KakaoMobilityClient,
    geocoder: KakaoGeocodeClient,
    request: BundleOrderRequest,
    partner_order_id: str,
    *,
    route_planner: RoutePlanner | None = None,
) -> tuple[dict[str, Any], CreateDeliveryRequest]:
    """최신 견적을 다시 계산하고 다중 픽업 묶음 주문 payload를 만든다."""
    pickup_locations = await _resolve_locations(
        geocoder,
        [item.address for item in request.pickups],
        label="픽업",
    )
    dropoff_locations = await _resolve_locations(
        geocoder,
        [item.address for item in request.dropoffs],
        label="배송",
    )
    pickup_order, dropoff_order = await _bundle_route_order(
        pickup_locations,
        dropoff_locations,
        route_planner,
    )

    quote_request = BundleQuoteRequest.model_validate(
        {
            "pickups": [{"address": item.address} for item in request.pickups],
            "dropoffs": [
                {
                    "address": item.address,
                    "pickupIndex": item.pickup_index,
                }
                for item in request.dropoffs
            ],
            "productSize": request.product_size,
            "fleetOption": request.fleet_option,
        }
    )
    quote = await multi_pickup_bundle_quote(
        client,
        geocoder,
        quote_request,
        route_planner=route_planner,
    )

    ordered_pickup_stops = [
        DeliveryStop(
            location=pickup_locations[index],
            contact=request.pickups[index].contact,
            note=(
                request.pickups[index].note
                if position == 0
                else _tagged_note("추가 픽업", request.pickups[index].note)
            ),
        )
        for position, index in enumerate(pickup_order)
    ]
    ordered_dropoff_stops = [
        DeliveryStop(
            location=dropoff_locations[index],
            contact=request.dropoffs[index].contact,
            note=_tagged_note("배송", request.dropoffs[index].note),
        )
        for index in dropoff_order
    ]
    order_request = CreateDeliveryRequest(
        partnerOrderId=partner_order_id,
        orderType=OrderType.QUICK,
        productSize=request.product_size,
        pickup=ordered_pickup_stops[0],
        waypoints=[
            *ordered_pickup_stops[1:],
            *ordered_dropoff_stops[:-1],
        ],
        dropoff=ordered_dropoff_stops[-1],
        productName=request.product_name,
        quantity=request.quantity,
        declaredValue=request.declared_value,
        paymentType=PaymentType.CARD,
        fleetOption=request.fleet_option,
    )
    return quote, order_request
