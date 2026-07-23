from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from mobility_service.app import create_app
from mobility_service.auth import build_authorization
from mobility_service.client import KakaoMobilityClient
from mobility_service.config import Settings
from mobility_service.models import DeliveryDraft
from mobility_service.my_model import own_model_reply
from mobility_service.store import MobilityStore


def settings(
    root: Path,
    *,
    map_key: str = "",
    rest_key: str = "",
    admin_username: str = "",
    admin_password: str = "",
) -> Settings:
    return Settings(
        api_key="test-api-key",
        vendor_id="TEST-VENDOR",
        base_url="https://example.test",
        callback_base_url="https://callback.example.test",
        database_path=root / "test.db",
        kakao_javascript_key=map_key,
        kakao_rest_api_key=rest_key,
        admin_username=admin_username,
        admin_password=admin_password,
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
        self.sandbox_status_calls: list[tuple[str, str, str | None]] = []

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
            "pickup": {"stepId": "pickup-step"},
            "waypoints": [{"stepId": "waypoint-step"}],
            "dropoff": {"stepId": "dropoff-step"},
        }

    async def get_picker(self, partner_order_id: str) -> dict[str, Any]:
        return {"partnerOrderId": partner_order_id, "picker": None}

    async def cancel_order(self, partner_order_id: str) -> dict[str, Any]:
        return {"partnerOrderId": partner_order_id, "status": "CANCELED"}

    async def get_step(
        self, partner_order_id: str, step_id: str
    ) -> dict[str, Any]:
        return {
            "stepId": step_id,
            "status": "waiting",
            "estimatedEndedAt": "2026-07-23T18:00:00+09:00",
            "location": {"basicAddress": f"{step_id} 주소"},
        }

    async def change_sandbox_status(
        self,
        partner_order_id: str,
        order_status: str,
        *,
        cancel_by: str | None = None,
    ) -> dict[str, Any]:
        self.sandbox_status_calls.append(
            (partner_order_id, order_status, cancel_by)
        )
        return {"changed": True}


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

    def test_feature_intro_page_is_linked_from_home(self) -> None:
        home = self.client.get("/")
        features = self.client.get("/features")

        self.assertEqual(home.status_code, 200)
        self.assertIn('href="/features"', home.text)
        self.assertLess(
            home.text.index('href="/features"'),
            home.text.index("택시 접수하기"),
        )
        self.assertEqual(features.status_code, 200)
        self.assertIn("새로 추가한 기능은 두 가지입니다", features.text)
        self.assertIn("기능 소개", features.text)

    def test_home_has_ollama_toggle_for_local_chat(self) -> None:
        home = self.client.get("/")

        self.assertEqual(home.status_code, 200)
        self.assertIn('id="ollamaSwitch"', home.text)
        self.assertIn('role="switch"', home.text)
        self.assertIn('ollamaAvailable && ollamaEnabled ? "ollama" : "own"', home.text)

    def test_local_chat_status_reports_server_side_ollama_state(self) -> None:
        with patch(
            "mobility_service.app.ollama_status",
            return_value={
                "available": False,
                "model": "gemma4:e2b",
                "message": "이 서버에서 Ollama에 연결할 수 없습니다.",
            },
        ):
            response = self.client.get("/api/local-chat/status")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["data"]["available"])
        self.assertEqual(response.json()["data"]["model"], "gemma4:e2b")

    def test_quick_request_has_guided_booking_flow(self) -> None:
        home = self.client.get("/")

        self.assertEqual(home.status_code, 200)
        self.assertIn("1. 배송 경로", home.text)
        self.assertIn("받는 분이 보내는 분과 같아요", home.text)
        self.assertIn('id="quickReview"', home.text)
        self.assertIn('id="quoteButton"', home.text)
        self.assertIn('id="orderButton"', home.text)
        self.assertIn("주소 확인이 필요해요", home.text)

    def test_customer_pages_include_accessible_typography_and_controls(self) -> None:
        home = self.client.get("/")
        taxi = self.client.get("/taxi")
        features = self.client.get("/features")

        for response in (home, taxi, features):
            self.assertEqual(response.status_code, 200)
            self.assertIn("system-ui, -apple-system", response.text)
            self.assertIn("focus-visible", response.text)

        self.assertIn("min-height: 44px", home.text)
        self.assertIn("font-size: 16px", home.text)
        self.assertIn("color: #707070", home.text)
        self.assertIn("min-height: 44px", taxi.text)
        self.assertIn("color: #707070", taxi.text)

    def test_local_chat_uses_selected_own_engine(self) -> None:
        with patch(
            "mobility_service.app.local_model_reply",
            return_value="자체 QA 응답",
        ) as responder:
            response = self.client.post(
                "/api/agent/chat",
                json={
                    "message": "MOVB가 뭐야?",
                    "mode": "local",
                    "localEngine": "own",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["reply"], "자체 QA 응답")
        self.assertEqual(response.json()["data"]["trace"], ["local_model:own"])
        responder.assert_called_once_with("MOVB가 뭐야?", "own")

    def test_own_model_handles_basic_deployed_chat(self) -> None:
        self.assertIn("안녕하세요", own_model_reply("헬로"))
        self.assertIn("안녕하세요", own_model_reply("ㅇㅇ"))
        self.assertIn("MOVB", own_model_reply("MOVB가 뭐야"))
        self.assertIn("연결 상태", own_model_reply("지금은 켰어"))
        self.assertIn("연결 상태", own_model_reply("라마"))

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

    def test_register_creates_session_and_logout_revokes_it(self) -> None:
        registered = self.client.post(
            "/api/auth/register",
            json={
                "name": "홍길동",
                "email": "USER@example.com",
                "password": "safe-pass-1234",
            },
        )

        self.assertEqual(registered.status_code, 201)
        self.assertEqual(registered.json()["data"]["user"]["email"], "user@example.com")
        self.assertNotIn("password", registered.text)
        self.assertIn("HttpOnly", registered.headers["set-cookie"])

        me = self.client.get("/api/auth/me")
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["data"]["user"]["name"], "홍길동")

        logged_out = self.client.post("/api/auth/logout")
        self.assertEqual(logged_out.status_code, 200)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)

    def test_login_validates_credentials_and_duplicate_email(self) -> None:
        payload = {
            "name": "테스트 사용자",
            "email": "test@example.com",
            "password": "correct-pass-1234",
        }
        self.assertEqual(
            self.client.post("/api/auth/register", json=payload).status_code,
            201,
        )
        self.assertEqual(
            self.client.post("/api/auth/register", json=payload).status_code,
            409,
        )
        self.client.cookies.clear()

        wrong = self.client.post(
            "/api/auth/login",
            json={"email": payload["email"], "password": "wrong-pass-1234"},
        )
        self.assertEqual(wrong.status_code, 401)

        correct = self.client.post(
            "/api/auth/login",
            json={"email": payload["email"], "password": payload["password"]},
        )
        self.assertEqual(correct.status_code, 200)
        self.assertEqual(correct.json()["data"]["user"]["email"], payload["email"])

    def test_admin_bootstrap_and_role_protected_endpoints(self) -> None:
        root = Path(self.temporary.name) / "admin"
        admin_app = create_app(
            settings=settings(
                root,
                admin_username="admin",
                admin_password="admin-test-pass-1234",
            ),
            client=FakeKakaoClient(),  # type: ignore[arg-type]
            store=MobilityStore(root / "test.db"),
        )
        with TestClient(admin_app) as admin_client:
            self.assertEqual(
                admin_client.get("/api/admin/summary").status_code,
                401,
            )

            logged_in = admin_client.post(
                "/api/auth/login",
                json={"identifier": "admin", "password": "admin-test-pass-1234"},
            )
            self.assertEqual(logged_in.status_code, 200)
            self.assertEqual(logged_in.json()["data"]["user"]["role"], "ADMIN")
            self.assertEqual(admin_client.get("/admin").status_code, 200)

            summary = admin_client.get("/api/admin/summary")
            self.assertEqual(summary.status_code, 200)
            self.assertEqual(summary.json()["data"]["users"]["admins"], 1)

            admin_client.post("/api/auth/logout")
            admin_client.post(
                "/api/auth/register",
                json={
                    "name": "일반 사용자",
                    "email": "regular@example.com",
                    "password": "regular-pass-1234",
                },
            )
            self.assertEqual(
                admin_client.get("/api/admin/users").status_code,
                403,
            )
            self.assertEqual(
                admin_client.get("/admin", follow_redirects=False).status_code,
                303,
            )


if __name__ == "__main__":
    unittest.main()
