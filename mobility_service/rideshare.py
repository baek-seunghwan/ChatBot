from __future__ import annotations

import math
from itertools import permutations
from typing import Any

from .directions import RoutePlanner
from .models import Location

# 서울 중형택시 기준 추정 요금 (기본요금 1.6km까지 4,800원 + 이후 km당 약 800원).
# 실제 미터기 요금과 다를 수 있는 "추정치"로만 안내한다.
BASE_FARE = 4800
BASE_DISTANCE_KM = 1.6
PER_KM_FARE = 800

# 직선거리 → 실제 도로 거리 보정 계수 (도심 도로 우회율 평균치)
ROAD_FACTOR = 1.3

MAX_PASSENGERS = 4


def haversine_km(a: Location, b: Location) -> float:
    lat1, lng1 = math.radians(a.latitude), math.radians(a.longitude)
    lat2, lng2 = math.radians(b.latitude), math.radians(b.longitude)
    d_lat, d_lng = lat2 - lat1, lng2 - lng1
    h = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lng / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def road_km(a: Location, b: Location) -> float:
    return haversine_km(a, b) * ROAD_FACTOR


def estimate_taxi_fare(distance_km: float) -> int:
    """거리 기반 추정 택시 요금 (100원 단위 반올림)."""
    extra_km = max(0.0, distance_km - BASE_DISTANCE_KM)
    fare = BASE_FARE + extra_km * PER_KM_FARE
    return int(round(fare / 100) * 100)


def proportional_split(total: int, solo_prices: list[int]) -> list[int]:
    """공동 요금을 각자 단독 요금에 비례해 나눈다 (100원 단위, 오차는 마지막 사람 보정)."""
    solo_total = sum(solo_prices)
    shares: list[int] = []
    for solo in solo_prices:
        raw = total * (solo / solo_total) if solo_total else total / len(solo_prices)
        shares.append(int(round(raw / 100) * 100))
    shares[-1] += total - sum(shares)
    return shares


async def carpool_plan(
    origin: Location,
    passengers: list[dict[str, Any]],
    route_planner: RoutePlanner | None = None,
) -> dict[str, Any]:
    """동승(카풀) 경유 순서와 요금 분배안을 계산한다.

    passengers: [{"name": str, "location": Location}, ...] (2~4명)

    - 경유 순서: 모든 방문 순서를 비교해 총 이동거리가 가장 짧은 순서 선택
    - 요금 분배: 각자 혼자 탔을 때의 추정 요금에 비례해 공동 요금을 나눔
      (멀리 가는 사람이 더 내되, 전원이 혼자 탈 때보다는 이득이 되는 방식)
    """
    if not 2 <= len(passengers) <= MAX_PASSENGERS:
        raise ValueError(f"동승 인원은 2~{MAX_PASSENGERS}명이어야 합니다.")

    # 1) 최적 방문 순서: 인원이 적으므로 전체 순열 비교로 충분하다
    best_order: tuple[int, ...] | None = None
    best_total_km = float("inf")
    best_route: dict[str, Any] | None = None
    for order in permutations(range(len(passengers))):
        ordered_locations = [passengers[idx]["location"] for idx in order]
        if route_planner:
            route = await route_planner.route_summary(
                origin,
                ordered_locations[-1],
                waypoints=ordered_locations[:-1],
            )
            total = float(route["distanceKm"])
        else:
            route = None
            total = 0.0
            cursor = origin
            for stop in ordered_locations:
                total += road_km(cursor, stop)
                cursor = stop
        if total < best_total_km:
            best_total_km = total
            best_order = order
            best_route = route
    assert best_order is not None

    shared_fare = (
        int(best_route.get("taxiFare") or 0)
        if best_route
        else 0
    ) or estimate_taxi_fare(best_total_km)

    # 2) 요금 분배: 혼자 탔을 때 요금에 비례
    solo_fares = []
    for passenger in passengers:
        if route_planner:
            solo_route = await route_planner.route_summary(
                origin, passenger["location"]
            )
            solo_fare = int(solo_route.get("taxiFare") or 0) or estimate_taxi_fare(
                float(solo_route["distanceKm"])
            )
        else:
            solo_fare = estimate_taxi_fare(
                road_km(origin, passenger["location"])
            )
        solo_fares.append(solo_fare)
    solo_total = sum(solo_fares)
    shares = proportional_split(shared_fare, solo_fares)

    stops = []
    cursor = origin
    cumulative_km = 0.0
    section_distances = (
        [
            float(section.get("distanceMeters") or 0) / 1000
            for section in best_route.get("sections", [])
        ]
        if best_route
        else []
    )
    for rank, idx in enumerate(best_order, start=1):
        passenger = passengers[idx]
        leg = (
            section_distances[rank - 1]
            if rank - 1 < len(section_distances)
            else road_km(cursor, passenger["location"])
        )
        cumulative_km += leg
        cursor = passenger["location"]
        stops.append(
            {
                "dropOrder": rank,
                "name": passenger.get("name") or f"탑승자{idx + 1}",
                "address": passenger["location"].basic_address,
                "legKm": round(leg, 1),
                "cumulativeKm": round(cumulative_km, 1),
                "soloFare": solo_fares[idx],
                "share": shares[idx],
                "saving": solo_fares[idx] - shares[idx],
            }
        )

    return {
        "totalKm": round(best_total_km, 1),
        "sharedFare": shared_fare,
        "soloFareTotal": solo_total,
        "groupSaving": solo_total - shared_fare,
        "stops": stops,
        "routeInfo": best_route,
        "note": (
            "카카오 실도로·예상 택시요금 기준이며 실제 미터기 요금과 다를 수 있어요."
            if best_route and best_route.get("actualRoadData")
            else "카카오 길찾기 미연결로 보정 거리 추정치를 사용했어요."
        ),
    }
