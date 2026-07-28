from __future__ import annotations

import math

from .models import Location


# 직선거리에서 도심 도로의 우회 거리를 추정할 때 사용하는 보정 계수.
ROAD_FACTOR = 1.3


def haversine_km(a: Location, b: Location) -> float:
    """두 위·경도 좌표 사이의 대권 거리를 km로 계산한다."""
    lat1, lng1 = math.radians(a.latitude), math.radians(a.longitude)
    lat2, lng2 = math.radians(b.latitude), math.radians(b.longitude)
    d_lat, d_lng = lat2 - lat1, lng2 - lng1
    h = (
        math.sin(d_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(d_lng / 2) ** 2
    )
    return 6371 * 2 * math.asin(math.sqrt(h))


def road_km(a: Location, b: Location) -> float:
    """길찾기 API가 없을 때 사용할 보정 도로 거리 추정치."""
    return haversine_km(a, b) * ROAD_FACTOR
