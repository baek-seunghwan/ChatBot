from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient

from mobility_service.app import create_app
from mobility_service.auth import build_authorization
from mobility_service.client import KakaoMobilityClient
from mobility_service.config import Settings
from mobility_service.models import DeliveryDraft
from mobility_service.store import MobilityStore


def settings(root: Path, *, map_key: str = "") -> Settings:
    return Settings(
        api_key="test-api-key",
        vendor_id="TEST-VENDOR",
        base_url="https://example.test",
        callback_base_url="https://callback.example.test",
        database_path=root / "test.db",
        kakao_javascript_key=map_key,
    )


def sample_order(partner_order_id: str = "test-order-001") -> dict[str, Any]:
    return {
        "partnerOrderId": partner_order_id,
        "orderType": "QUICK",
        "productSize": "XS",
        "pickup": {
            "location": {
                "basicAddress": "경기도 성남시 분당구 판교역로 152",
                "latitude": 37.3946095,
                "longitude": 127.1118735,
            },
            "contact": {"name": "테스트발송자", "phone": "010-1000-0001"},
        },
        "dropoff": {
            "location": {
                "basicAddress": "경기도 성남시 분당구 정자동 49-4",
                "latitude": 37.3595316,
                "longitude": 127.1052133,
            },
            "contact": {"name": "테스트수신자", "phone": "010-1000-0002"},
        },
        "waypoints": [],
        "productName": "테스트 서류",
        "declaredValue": 10000,
        "paymentType": "CARD",
    }


class AuthorizationTests(unittest.TestCase):
    def test_documented_sha512_authorization_format(self) -> None:
        token = build_authorization(
            "secret-key", timestamp_ms=1_700_000_000_000, nonce=121212
        )
        decoded = base64.b64decode(token).decode("utf-8")
        timestamp, nonce, signature = decoded.split("$$")

        self.assertEqual(timestamp, "1700000000000")
        self.assertEqual(nonce, "121212")
        expected = hashlib.sha512(
            b"1700000000000121212secret-key"
        ).hexdigest()
        self.assertEqual(signature, expected)


class KakaoClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_price_request_has_auth_headers_and_sandbox_payload(self) -> None:
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"totalPrice": 12000})

        with tempfile.TemporaryDirectory() as temp:
            client = KakaoMobilityClient(
                settings(Path(temp)),
                transport=httpx.MockTransport(handler),
            )
            draft = DeliveryDraft.model_validate(sample_order())
            response = await client.price(draft)
            await client.close()

        self.assertEqual(response["totalPrice"], 12000)
        self.assertEqual(
            requests[0].url.path,
            "/goa-sandbox-service/api/v2/orders/price",
        )
        self.assertEqual(requests[0].headers["vendor"], "TEST-VENDOR")
        self.assertTrue(requests[0].headers["authorization"])
        payload = json.loads(requests[0].content)
        self.assertEqual(payload["orderType"], "QUICK")
        self.assertEqual(payload["pickup"]["location"]["latitude"], 37.3946095)
        self.assertNotIn("contact", payload["pickup"])


class FakeKakaoClient:
    def __init__(self) -> None:
        self.create_calls = 0

    async def auth_check(self) -> dict[str, bool]:
        return {"authenticated": True}

    async def estimate(self, request) -> dict[str, Any]:
        return {"estimatedMinutes": 40, "orderType": request.order_type.value}

    async def price(self, request) -> dict[str, Any]:
        return {"totalPrice": 12000, "orderType": request.order_type.value}

    async def create_order(self, request, partner_order_id: str) -> dict[str, Any]:
        self.create_calls += 1
        return {
            "partnerOrderId": partner_order_id,
            "receipt": {"status": "MATCHING", "totalPrice": 12000},
        }

    async def get_order(self, partner_order_id: str) -> dict[str, Any]:
        return {
            "partnerOrderId": partner_order_id,
            "receipt": {"status": "MATCHED"},
        }

    async def get_picker(self, partner_order_id: str) -> dict[str, Any]:
        return {"partnerOrderId": partner_order_id, "picker": None}

    async def cancel_order(self, partner_order_id: str) -> dict[str, Any]:
        return {"partnerOrderId": partner_order_id, "status": "CANCELED"}


class MobilityApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.fake = FakeKakaoClient()
        self.store = MobilityStore(root / "mobility.db")
        self.app = create_app(
            settings=settings(root, map_key="public-javascript-key"),
            client=self.fake,  # type: ignore[arg-type]
            store=self.store,
        )
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.client.close()
        self.temporary.cleanup()

    def test_create_order_is_idempotent(self) -> None:
        payload = sample_order("same-order")

        first = self.client.post("/api/orders", json=payload)
        second = self.client.post("/api/orders", json=payload)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(self.fake.create_calls, 1)
        self.assertEqual(second.json()["data"]["source"], "existing")

    def test_callback_is_deduplicated_and_old_state_does_not_win(self) -> None:
        self.client.post("/api/orders", json=sample_order("callback-order"))

        matched = self.client.put(
            "/api/v1/callback/orders/callback-order/matched",
            json={"pickerId": "picker-1"},
        )
        picked_up = self.client.put(
            "/api/v1/callback/orders/callback-order/pickupCompleted",
            json={"pickerId": "picker-1"},
        )
        duplicate = self.client.put(
            "/api/v1/callback/orders/callback-order/pickupCompleted",
            json={"pickerId": "picker-1"},
        )
        late_matched = self.client.put(
            "/api/v1/callback/orders/callback-order/matched",
            json={"pickerId": "picker-2"},
        )
        order = self.client.get("/api/orders/callback-order")

        self.assertTrue(matched.json()["data"]["applied"])
        self.assertTrue(picked_up.json()["data"]["applied"])
        self.assertTrue(duplicate.json()["data"]["duplicate"])
        self.assertFalse(late_matched.json()["data"]["applied"])
        self.assertEqual(order.json()["data"]["status"], "PICKUP_COMPLETED")

    def test_price_endpoint_uses_normalized_request(self) -> None:
        response = self.client.post(
            "/api/deliveries/price", json=sample_order("quote-only")
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["totalPrice"], 12000)

    def test_public_config_only_exposes_javascript_map_key(self) -> None:
        response = self.client.get("/api/config")
        data = response.json()["data"]

        self.assertEqual(response.status_code, 200)
        self.assertTrue(data["mapConfigured"])
        self.assertEqual(data["kakaoJavascriptKey"], "public-javascript-key")
        self.assertNotIn("apiKey", data)
        self.assertNotIn("vendorId", data)


if __name__ == "__main__":
    unittest.main()
