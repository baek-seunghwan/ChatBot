from __future__ import annotations

import math
from itertools import permutations
from typing import Any

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


def carpool_plan(origin: Location, passengers: list[dict[str, Any]]) -> dict[str, Any]:
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
    for order in permutations(range(len(passengers))):
        total = 0.0
        cursor = origin
        for idx in order:
            stop = passengers[idx]["location"]
            total += road_km(cursor, stop)
            cursor = stop
        if total < best_total_km:
            best_total_km = total
            best_order = order
    assert best_order is not None

    shared_fare = estimate_taxi_fare(best_total_km)

    # 2) 요금 분배: 혼자 탔을 때 요금에 비례
    solo_fares = [
        estimate_taxi_fare(road_km(origin, p["location"])) for p in passengers
    ]
    solo_total = sum(solo_fares)
    shares = proportional_split(shared_fare, solo_fares)

    stops = []
    cursor = origin
    cumulative_km = 0.0
    for rank, idx in enumerate(best_order, start=1):
        passenger = passengers[idx]
        leg = road_km(cursor, passenger["location"])
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
        "note": "직선거리 기반 추정 요금으로 실제 미터기 요금과 다를 수 있어요.",
    }
