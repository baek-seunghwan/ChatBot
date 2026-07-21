from __future__ import annotations

from typing import Any

import httpx

from .config import Settings
from .models import Location


class KakaoGeocodeClient:
    """카카오 로컬(Local) REST API로 주소를 좌표로 변환한다.

    KakaoMobilityClient의 HMAC 서명 인증과는 별개로, 로컬 API는
    `Authorization: KakaoAK {REST 키}` 고정 헤더만 쓰므로 별도 클라이언트로 둔다.
    지오코딩 실패는 대화 흐름에서 정상적으로 발생할 수 있는 결과이므로
    예외를 던지지 않고 항상 None을 반환한다.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._rest_api_key = settings.kakao_rest_api_key
        self._client = httpx.AsyncClient(
            base_url="https://dapi.kakao.com",
            timeout=settings.request_timeout_seconds,
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _get_documents(self, path: str, query: str) -> list[dict[str, Any]] | None:
        try:
            response = await self._client.get(
                path,
                params={"query": query},
                headers={"Authorization": f"KakaoAK {self._rest_api_key}"},
            )
            response.raise_for_status()
            body: Any = response.json()
        except (httpx.HTTPError, ValueError):
            return None
        documents = body.get("documents") if isinstance(body, dict) else None
        return documents or None

    async def search_address(self, query: str) -> Location | None:
        """주소 텍스트를 좌표로 변환한다.

        '판교역'처럼 지번/도로명 주소가 아닌 장소명은 주소 검색(address.json)이
        결과를 못 찾으므로, 실패하면 키워드(장소) 검색(keyword.json)으로 폴백한다.
        """
        query = query.strip()
        if not query or not self._rest_api_key:
            return None

        documents = await self._get_documents("/v2/local/search/address.json", query)
        if documents:
            top = documents[0]
            try:
                longitude = float(top["x"])
                latitude = float(top["y"])
            except (KeyError, TypeError, ValueError):
                documents = None
            else:
                road_address = top.get("road_address") or {}
                address = top.get("address") or {}
                basic_address = (
                    road_address.get("address_name")
                    or address.get("address_name")
                    or query
                )
                return Location(
                    basicAddress=basic_address, latitude=latitude, longitude=longitude
                )

        documents = await self._get_documents("/v2/local/search/keyword.json", query)
        if not documents:
            return None
        top = documents[0]
        try:
            longitude = float(top["x"])
            latitude = float(top["y"])
        except (KeyError, TypeError, ValueError):
            return None
        basic_address = (
            top.get("road_address_name") or top.get("address_name") or top.get("place_name") or query
        )
        return Location(basicAddress=basic_address, latitude=latitude, longitude=longitude)
