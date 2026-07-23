from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import httpx

from .client import KakaoApiError
from .config import Settings
from .models import Location


def _fallback_road_km(a: Location, b: Location) -> float:
    lat1, lng1 = math.radians(a.latitude), math.radians(a.longitude)
    lat2, lng2 = math.radians(b.latitude), math.radians(b.longitude)
    d_lat, d_lng = lat2 - lat1, lng2 - lng1
    haversine = (
        math.sin(d_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(d_lng / 2) ** 2
    )
    return 6371 * 2 * math.asin(math.sqrt(haversine)) * 1.3


class KakaoDirectionsClient:
    """카카오모빌리티 길찾기 API 전용 클라이언트.

    퀵/도보 배송 Open API의 Vendor 인증과 달리 Kakao Developers REST API
    키를 사용하므로 배송 클라이언트와 분리한다.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.directions_base_url,
            timeout=settings.request_timeout_seconds,
            transport=transport,
        )

    @property
    def configured(self) -> bool:
        return self.settings.directions_configured

    async def close(self) -> None:
        await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        key = self.settings.kakao_rest_api_key.strip()
        if not key:
            raise KakaoApiError(
                "KAKAO_REST_API_KEY를 설정하면 실제 도로 거리와 ETA를 사용할 수 있습니다."
            )
        return {
            "Authorization": f"KakaoAK {key}",
            "Content-Type": "application/json",
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        try:
            response = await self._client.request(
                method,
                path,
                headers=self._headers(),
                params=params,
                json=json,
            )
        except httpx.TimeoutException as exc:
            raise KakaoApiError("카카오 길찾기 API 응답 시간이 초과됐습니다.") from exc
        except httpx.HTTPError as exc:
            raise KakaoApiError(
                f"카카오 길찾기 API 연결에 실패했습니다: {type(exc).__name__}"
            ) from exc

        try:
            body: Any = response.json() if response.content else None
        except ValueError:
            body = response.text[:1000]
        if response.is_error:
            detail = body.get("msg") if isinstance(body, dict) else body
            raise KakaoApiError(
                f"카카오 길찾기 API가 {response.status_code} 오류를 반환했습니다: {detail}",
                status_code=response.status_code,
                response_body=body,
            )
        return body

    @staticmethod
    def _point(location: Location, *, include_name: bool = False) -> str:
        point = f"{location.longitude},{location.latitude}"
        if include_name and location.basic_address:
            point += f",name={location.basic_address}"
        return point

    @staticmethod
    def _point_object(location: Location, *, name: str | None = None) -> dict[str, Any]:
        point: dict[str, Any] = {
            "x": location.longitude,
            "y": location.latitude,
        }
        if name or location.basic_address:
            point["name"] = name or location.basic_address
        return point

    @staticmethod
    def _departure_time(value: str) -> str:
        stripped = value.strip()
        if len(stripped) == 12 and stripped.isdigit():
            return stripped
        try:
            parsed = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                "예약 시간은 ISO 8601 또는 YYYYMMDDHHMM 형식이어야 합니다."
            ) from exc
        return parsed.strftime("%Y%m%d%H%M")

    @staticmethod
    def _normalized(body: Any, *, future: bool) -> dict[str, Any]:
        routes = body.get("routes") if isinstance(body, dict) else None
        if not isinstance(routes, list) or not routes:
            raise KakaoApiError("카카오 길찾기 API에서 경로를 찾지 못했습니다.")
        route = next(
            (
                item
                for item in routes
                if isinstance(item, dict) and item.get("result_code") == 0
            ),
            None,
        )
        if route is None:
            first = routes[0] if isinstance(routes[0], dict) else {}
            raise KakaoApiError(
                f"카카오 길찾기에 실패했습니다: {first.get('result_msg', '경로 없음')}",
                response_body=body,
            )
        summary = route.get("summary") if isinstance(route.get("summary"), dict) else {}
        distance = int(summary.get("distance") or 0)
        duration = int(summary.get("duration") or 0)
        fare = summary.get("fare") if isinstance(summary.get("fare"), dict) else {}
        sections = []
        for index, section in enumerate(route.get("sections") or [], start=1):
            if not isinstance(section, dict):
                continue
            sections.append(
                {
                    "index": index,
                    "distanceMeters": int(section.get("distance") or 0),
                    "durationSeconds": int(section.get("duration") or 0),
                }
            )
        return {
            "source": "kakao-directions",
            "actualRoadData": True,
            "futureTrafficApplied": future,
            "distanceMeters": distance,
            "distanceKm": round(distance / 1000, 1),
            "durationSeconds": duration,
            "durationMinutes": max(1, round(duration / 60)) if duration else 0,
            "taxiFare": int(fare.get("taxi") or 0),
            "tollFare": int(fare.get("toll") or 0),
            "sections": sections,
            "providerTransactionId": body.get("trans_id"),
        }

    async def route_summary(
        self,
        origin: Location,
        destination: Location,
        *,
        waypoints: list[Location] | None = None,
        departure_time: str | None = None,
    ) -> dict[str, Any]:
        stops = list(waypoints or [])
        if len(stops) > 30:
            raise ValueError("다중 경유지 길찾기는 경유지 30개까지 지원합니다.")

        if departure_time:
            if len(stops) > 5:
                raise ValueError("미래 운행 정보는 경유지 5개까지 지원합니다.")
            params: dict[str, Any] = {
                "origin": self._point(origin),
                "destination": self._point(destination),
                "departure_time": self._departure_time(departure_time),
                "priority": "RECOMMEND",
                "summary": "true",
            }
            if stops:
                params["waypoints"] = "|".join(self._point(item) for item in stops)
            body = await self._request(
                "GET", "/v1/future/directions", params=params
            )
            normalized = self._normalized(body, future=True)
            normalized["departureTime"] = departure_time
            return normalized

        if stops:
            body = await self._request(
                "POST",
                "/v1/waypoints/directions",
                json={
                    "origin": self._point_object(origin, name="출발지"),
                    "destination": self._point_object(destination, name="목적지"),
                    "waypoints": [
                        self._point_object(item, name=f"경유지 {index}")
                        for index, item in enumerate(stops, start=1)
                    ],
                    "priority": "RECOMMEND",
                    "alternatives": False,
                    "road_details": False,
                    "summary": True,
                },
            )
            return self._normalized(body, future=False)

        body = await self._request(
            "GET",
            "/v1/directions",
            params={
                "origin": self._point(origin),
                "destination": self._point(destination),
                "priority": "RECOMMEND",
                "summary": "true",
            },
        )
        return self._normalized(body, future=False)


class RoutePlanner:
    """실도로 길찾기를 우선 사용하고, 키/네트워크 문제 때만 추정치로 강등한다."""

    def __init__(self, client: KakaoDirectionsClient) -> None:
        self.client = client

    async def route_summary(
        self,
        origin: Location,
        destination: Location,
        *,
        waypoints: list[Location] | None = None,
        departure_time: str | None = None,
    ) -> dict[str, Any]:
        try:
            return await self.client.route_summary(
                origin,
                destination,
                waypoints=waypoints,
                departure_time=departure_time,
            )
        except (KakaoApiError, ValueError) as exc:
            stops = [origin, *(waypoints or []), destination]
            distance_km = sum(
                _fallback_road_km(current, nxt)
                for current, nxt in zip(stops, stops[1:])
            )
            duration = round(distance_km / 30 * 3600)
            return {
                "source": "heuristic-fallback",
                "actualRoadData": False,
                "futureTrafficApplied": False,
                "distanceMeters": round(distance_km * 1000),
                "distanceKm": round(distance_km, 1),
                "durationSeconds": duration,
                "durationMinutes": max(1, round(duration / 60)),
                "taxiFare": 0,
                "tollFare": 0,
                "sections": [],
                "providerTransactionId": None,
                "fallbackReason": str(exc),
            }
