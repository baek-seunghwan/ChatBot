from __future__ import annotations

import base64
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import httpx
from fastapi.testclient import TestClient

from mobility_service.app import create_app
from mobility_service.agent import DeliveryAgent
from mobility_service.auth import build_authorization
from mobility_service.chat_cache import GREETING_REPLY
from mobility_service.client import KakaoMobilityClient
from mobility_service.config import Settings
from mobility_service.kakaopay import KakaoPayClient
from mobility_service.models import DeliveryDraft
from mobility_service.my_model import load_qa_index, own_model_reply
from mobility_service.providers import LLMRouter
from mobility_service.smart_input import (
    address_is_specific_enough,
    extract_dropoff_slots,
)
from mobility_service.store import MobilityStore


def settings(
    root: Path,
    *,
    map_key: str = "",
    rest_key: str = "",
    admin_username: str = "",
    admin_password: str = "",
    kakaopay_secret_key: str = "",
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
        kakaopay_secret_key=kakaopay_secret_key,
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


class KakaoPayClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_dev_secret_stays_in_server_authorization_header(self) -> None:
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "tid": "T1234567890123456789",
                    "next_redirect_pc_url": "https://pay.example/redirect",
                },
            )

        with tempfile.TemporaryDirectory() as temp:
            client = KakaoPayClient(
                settings(Path(temp), kakaopay_secret_key="fresh-dev-key"),
                transport=httpx.MockTransport(handler),
            )
            await client.ready(
                partner_order_id="payment-order-1",
                partner_user_id="user-1",
                item_name="MOVB 배송",
                quantity=1,
                total_amount=12000,
                approval_url="https://service.example/success",
                cancel_url="https://service.example/cancel",
                fail_url="https://service.example/fail",
            )
            await client.close()

        self.assertEqual(
            requests[0].headers["Authorization"],
            "SECRET_KEY fresh-dev-key",
        )
        self.assertEqual(
            requests[0].url.path,
            "/online/v1/payment/ready",
        )
        self.assertEqual(json.loads(requests[0].content)["total_amount"], 12000)


class VllmProviderTests(unittest.TestCase):
    def test_open_weight_vllm_can_generate_for_ai_agent(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": "주소를 알려주세요."}}]
        }
        router = LLMRouter(
            primary_provider="vllm",
            fallback_provider=None,
            vllm_base_url="https://model.example/v1",
            vllm_api_key="test-secret",
            vllm_model="Qwen/Qwen3-4B-Instruct-2507",
        )

        with patch("mobility_service.providers.httpx.post", return_value=response) as post:
            result = router.generate(
                "퀵 접수하고 싶어",
                system="배송 도우미",
                max_tokens=100,
                temperature=0.0,
            )

        self.assertEqual(result.provider, "vllm")
        self.assertEqual(result.text, "주소를 알려주세요.")
        self.assertEqual(
            post.call_args.args[0],
            "https://model.example/v1/chat/completions",
        )
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            "Bearer test-secret",
        )


class FakeKakaoClient:
    def __init__(self) -> None:
        self.create_calls = 0
        self.price_calls = 0
        self.sandbox_status_calls: list[tuple[str, str, str | None]] = []

    async def auth_check(self) -> dict[str, bool]:
        return {"authenticated": True}

    async def estimate(self, request) -> dict[str, Any]:
        return {"estimatedMinutes": 40, "orderType": request.order_type.value}

    async def price(self, request) -> dict[str, Any]:
        self.price_calls += 1
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


class FakeKakaoPayClient:
    def __init__(self) -> None:
        self.ready_calls: list[dict[str, Any]] = []
        self.approve_calls: list[dict[str, Any]] = []

    async def ready(self, **payload: Any) -> dict[str, Any]:
        self.ready_calls.append(payload)
        return {
            "tid": "T1234567890123456789",
            "next_redirect_pc_url": "https://mockup-pg-web.kakao.com/pay",
            "next_redirect_mobile_url": "https://mockup-pg-web.kakao.com/mobile",
        }

    async def approve(self, **payload: Any) -> dict[str, Any]:
        self.approve_calls.append(payload)
        return {
            "aid": "A1234567890123456789",
            "tid": payload["tid"],
            "amount": {"total": 12000},
        }


class KakaoPayFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.mobility = FakeKakaoClient()
        self.kakaopay = FakeKakaoPayClient()
        self.store = MobilityStore(root / "mobility.db")
        app = create_app(
            settings=settings(root, kakaopay_secret_key="fresh-dev-key"),
            client=self.mobility,  # type: ignore[arg-type]
            kakaopay_client=self.kakaopay,  # type: ignore[arg-type]
            store=self.store,
        )
        self.client = TestClient(app)
        registered = self.client.post(
            "/api/auth/register",
            json={
                "name": "결제연습",
                "email": "payment@example.com",
                "password": "practice-password-123",
            },
        )
        self.assertEqual(registered.status_code, 201)

    def tearDown(self) -> None:
        self.client.close()
        self.temporary.cleanup()

    def test_ready_approves_payment_then_creates_delivery(self) -> None:
        ready = self.client.post(
            "/api/payments/kakaopay/ready",
            json={"order": sample_order()},
        )

        self.assertEqual(ready.status_code, 200)
        data = ready.json()["data"]
        self.assertEqual(data["amount"], 12000)
        self.assertEqual(
            data["nextRedirectPcUrl"],
            "https://mockup-pg-web.kakao.com/pay",
        )
        self.assertEqual(self.kakaopay.ready_calls[0]["total_amount"], 12000)
        self.assertIn(
            f"/{data['paymentId']}/success",
            self.kakaopay.ready_calls[0]["approval_url"],
        )

        approved = self.client.get(
            f"/api/payments/kakaopay/{data['paymentId']}/success",
            params={"pg_token": "test-pg-token"},
            follow_redirects=False,
        )

        self.assertEqual(approved.status_code, 303)
        self.assertIn("payment=success", approved.headers["location"])
        self.assertEqual(self.kakaopay.approve_calls[0]["pg_token"], "test-pg-token")
        self.assertEqual(self.mobility.create_calls, 1)
        payment = self.store.get_kakaopay_payment(data["paymentId"])
        self.assertIsNotNone(payment)
        self.assertEqual(payment["status"], "COMPLETED")

        duplicate = self.client.get(
            f"/api/payments/kakaopay/{data['paymentId']}/success",
            params={"pg_token": "same-token-again"},
            follow_redirects=False,
        )
        self.assertEqual(duplicate.status_code, 303)
        self.assertIn("payment=success", duplicate.headers["location"])
        self.assertEqual(len(self.kakaopay.approve_calls), 1)
        self.assertEqual(self.mobility.create_calls, 1)


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
        home = self.client.get("/order")
        features = self.client.get("/features")

        self.assertEqual(home.status_code, 200)
        self.assertIn('href="/features"', home.text)
        self.assertIn("스마트 딜리버리", home.text)
        self.assertEqual(features.status_code, 200)
        self.assertIn("여러 곳에서 픽업하고", features.text)
        self.assertIn('href="/order#smartDelivery"', features.text)
        self.assertIn("기능 소개", features.text)

    def test_brand_intro_has_hero_image(self) -> None:
        about = self.client.get("/about")
        hero = self.client.get("/assets/movb-brand-hero.webp")

        self.assertEqual(about.status_code, 200)
        self.assertIn('url("/assets/movb-brand-hero.webp")', about.text)
        self.assertIn("MOVB BRAND", about.text)
        self.assertIn(
            'href="/order?booking=scheduled"',
            about.text,
        )
        self.assertEqual(hero.status_code, 200)
        self.assertEqual(hero.headers["content-type"], "image/webp")
        self.assertGreater(len(hero.content), 100_000)

    def test_smart_delivery_is_integrated_into_home(self) -> None:
        home = self.client.get("/order")
        legacy = self.client.get("/bundle", follow_redirects=False)
        old_form = self.client.get("/smart-delivery/form", follow_redirects=False)

        self.assertEqual(home.status_code, 200)
        self.assertNotIn('src="/smart-delivery/form"', home.text)
        self.assertIn('id="addSmartPickup"', home.text)
        self.assertIn('id="addSmartDropoff"', home.text)
        self.assertIn("function isSmartDelivery()", home.text)
        self.assertIn("다른 사용자의 배송과 자동으로 묶기", home.text)
        self.assertIn("/api/delivery-matches", home.text)
        self.assertIn("스마트 딜리버리", home.text)
        self.assertEqual(legacy.status_code, 307)
        self.assertEqual(legacy.headers["location"], "/order#smartDelivery")
        self.assertEqual(old_form.status_code, 307)
        self.assertEqual(old_form.headers["location"], "/order#smartDelivery")
        self.assertIn("/api/smart-delivery/quote", home.text)
        self.assertIn("/api/smart-delivery/orders", home.text)
        self.assertIn("grid-column: 1 / -1", home.text)

    def test_home_keeps_chat_shortcuts_and_links_to_separate_history(self) -> None:
        home = self.client.get("/order")
        history = self.client.get("/history")

        self.assertIn("배송 주문", home.text)
        self.assertIn("차량 선택", home.text)
        self.assertIn("예약 ETA", home.text)
        self.assertIn("배송 상태", home.text)
        self.assertIn("서비스 안내", home.text)
        self.assertNotIn('id="history"', home.text)
        self.assertIn('href="/history"', home.text)
        self.assertIn('id="smartPickupList"', home.text)
        self.assertIn('id="smartDropoffList"', home.text)
        self.assertEqual(history.status_code, 200)
        self.assertIn("<h1>이용 내역</h1>", history.text)
        self.assertIn('id="orderList"', history.text)
        self.assertIn('id="orderDetail"', history.text)
        self.assertIn("/api/orders?limit=50", history.text)
        self.assertIn("단일 퀵으로 다시 이용하기", history.text)
        self.assertIn('id="retryDialog"', history.text)
        self.assertIn("기사 이동 경로", history.text)
        self.assertIn("/picker", history.text)
        self.assertIn("Sandbox 경로 시뮬레이션", history.text)

    def test_customer_pages_use_the_exact_same_shared_header(self) -> None:
        pages = [
            self.client.get("/order"),
            self.client.get("/features"),
        ]
        headers: list[str] = []

        for response in pages:
            self.assertEqual(response.status_code, 200)
            match = re.search(
                r'<header class="movb-site-header">.*?</header>',
                response.text,
                re.DOTALL,
            )
            self.assertIsNotNone(match)
            headers.append(match.group(0))

        self.assertEqual(headers[0], headers[1])
        self.assertIn("Move Better", headers[0])
        self.assertNotIn("일반 퀵 접수", headers[0])
        self.assertIn("이용 내역", headers[0])
        self.assertIn("로그인", headers[0])

    def test_history_page_preserves_shared_brand_and_header_position(self) -> None:
        home = self.client.get("/order")
        history = self.client.get("/history")

        self.assertEqual(history.status_code, 200)
        brand_pattern = r'<a class="movb-brand".*?</a>'
        home_brand = re.search(brand_pattern, home.text, re.DOTALL)
        history_brand = re.search(brand_pattern, history.text, re.DOTALL)
        self.assertIsNotNone(home_brand)
        self.assertIsNotNone(history_brand)
        self.assertEqual(home_brand.group(0), history_brand.group(0))
        self.assertIn('id="movb-shared-header-styles"', history.text)
        self.assertIn(
            'class="movb-nav-item active" data-nav="history"',
            history.text,
        )
        self.assertIn('href="/history"', history.text)

    def test_home_uses_single_ai_chat_without_local_toggle(self) -> None:
        home = self.client.get("/order")

        self.assertEqual(home.status_code, 200)
        self.assertNotIn('id="ollamaSwitch"', home.text)
        self.assertNotIn('class="chat-mode-toggle"', home.text)
        self.assertNotIn("function selectedLocalEngine()", home.text)
        self.assertIn('mode: "ai"', home.text)

    def test_chat_can_apply_message_photo_and_shared_address_to_form(self) -> None:
        home = self.client.get("/order")

        self.assertEqual(home.status_code, 200)
        self.assertIn('id="chatImageInput"', home.text)
        self.assertIn("문자 붙여넣기", home.text)
        self.assertIn("사진 읽기", home.text)
        self.assertIn("주소 요청 링크", home.text)
        self.assertIn("function applyAgentSlots", home.text)
        self.assertIn("/api/smart-input/ocr", home.text)
        self.assertIn("/api/address-requests", home.text)
        self.assertIn('mode: "ai"', home.text)
        self.assertNotIn('localEngine: selectedLocalEngine()', home.text)
        self.assertIn('sessionStorage.getItem("moveops_session_id")', home.text)
        self.assertIn('item.status === "COMPLETED" && !item.appliedAt', home.text)

    def test_smart_text_extracts_recipient_fields(self) -> None:
        response = self.client.post(
            "/api/smart-input/extract",
            json={
                "text": (
                    "받는 사람: 김민수\n"
                    "연락처: 010-1234-5678\n"
                    "배송지: 대전광역시 서구 둔산로 100 3층\n"
                    "물품: 서류 봉투"
                )
            },
        )

        self.assertEqual(response.status_code, 200)
        slots = response.json()["data"]["slots"]
        self.assertEqual(slots["dropoffName"], "김민수")
        self.assertEqual(slots["dropoffPhone"], "010-1234-5678")
        self.assertEqual(slots["dropoffAddress"], "대전광역시 서구 둔산로 100")
        self.assertEqual(slots["dropoffDetailAddress"], "3층")
        self.assertEqual(slots["productName"], "서류 봉투")

        unlabeled = extract_dropoff_slots(
            "여기로 보내줘\n경기도 성남시 분당구 판교역로 152, 13층\n"
            "김서연 010 2222 3333"
        )
        self.assertEqual(unlabeled["slots"]["dropoffName"], "김서연")
        self.assertEqual(unlabeled["slots"]["dropoffPhone"], "010-2222-3333")
        agent_slots = DeliveryAgent._heuristic_slots(
            "받는 사람: 김민수\n010-1234-5678\n"
            "배송지: 대전광역시 서구 둔산로 100 3층"
        )
        self.assertEqual(agent_slots["dropoffAddress"], "대전광역시 서구 둔산로 100")

    def test_unlabelled_short_address_stays_pending_until_user_confirms(self) -> None:
        message = "백승환 와우로 85 토마토오피스텔1동 408호"
        extracted = extract_dropoff_slots(message)
        agent_slots = DeliveryAgent._heuristic_slots(message)

        self.assertEqual(extracted["slots"]["dropoffName"], "백승환")
        self.assertEqual(extracted["slots"]["dropoffAddress"], "와우로 85")
        self.assertFalse(extracted["addressSpecificEnough"])
        self.assertFalse(address_is_specific_enough("와우로 85"))
        self.assertTrue(address_is_specific_enough("경기 화성시 와우로 85"))
        self.assertNotIn("dropoffAddress", agent_slots)
        self.assertNotIn("pickupAddress", agent_slots)
        self.assertEqual(agent_slots["_pendingAddress"], "와우로 85")
        self.assertEqual(agent_slots["_pendingContactName"], "백승환")
        self.assertTrue(agent_slots["_addressRoleAmbiguous"])
        self.assertTrue(agent_slots["_addressRegionAmbiguous"])

        pasted_message_slots = DeliveryAgent._heuristic_slots(
            message,
            input_context="recipient_message",
        )
        self.assertNotIn("dropoffAddress", pasted_message_slots)
        self.assertNotIn("pickupAddress", pasted_message_slots)
        self.assertTrue(pasted_message_slots["_addressRoleAmbiguous"])

        labelled = DeliveryAgent._heuristic_slots(
            "도착지: 경기 화성시 와우로 85 토마토오피스텔1동 408호"
        )
        self.assertEqual(labelled["dropoffAddress"], "경기 화성시 와우로 85")
        self.assertNotIn("_pendingAddress", labelled)

    def test_ocr_text_uses_same_recipient_extractor(self) -> None:
        with patch(
            "mobility_service.app.extract_text_from_image",
            return_value=(
                "수령인: 박서연\n"
                "010-9876-5432\n"
                "주소: 서울특별시 중구 세종대로 110 5층"
            ),
        ):
            response = self.client.post(
                "/api/smart-input/ocr",
                json={
                    "imageBase64": "data:image/jpeg;base64," + ("A" * 40),
                    "contentType": "image/jpeg",
                },
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertIn("서울특별시", data["text"])
        self.assertEqual(data["slots"]["dropoffName"], "박서연")
        self.assertEqual(data["slots"]["dropoffPhone"], "010-9876-5432")

    def test_recipient_address_link_round_trip(self) -> None:
        created = self.client.post(
            "/api/address-requests",
            headers={"X-Session-Id": "sender-session"},
            json={
                "recipientName": "받는사람",
                "recipientPhone": "010-1000-0002",
            },
        )

        self.assertEqual(created.status_code, 200)
        token = created.json()["data"]["token"]
        self.assertIn(f"/address-request/{token}", created.json()["data"]["url"])
        self.assertNotIn("senderSessionId", created.json()["data"])

        page = self.client.get(f"/address-request/{token}")
        self.assertEqual(page.status_code, 200)
        self.assertIn("받으실 곳을 알려주세요", page.text)

        submitted = self.client.put(
            f"/api/address-requests/{token}",
            json={
                "name": "받는사람",
                "phone": "010-1000-0002",
                "address": "대전광역시 서구 둔산로 100",
                "detailAddress": "3층",
                "note": "도착 전 연락",
            },
        )
        self.assertEqual(submitted.status_code, 200)

        completed = self.client.get(f"/api/address-requests/{token}")
        self.assertEqual(completed.status_code, 200)
        result = completed.json()["data"]
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["address"], "대전광역시 서구 둔산로 100")
        self.assertEqual(result["detailAddress"], "3층")
        self.assertNotIn("senderSessionId", result)

        sender_requests = self.client.get(
            "/api/address-requests",
            headers={"X-Session-Id": "sender-session"},
        )
        self.assertEqual(sender_requests.status_code, 200)
        listed = sender_requests.json()["data"]
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["token"], token)
        self.assertEqual(listed[0]["status"], "COMPLETED")
        self.assertNotIn("senderSessionId", listed[0])

        missing_session = self.client.get("/api/address-requests")
        self.assertEqual(missing_session.status_code, 400)

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
        home = self.client.get("/order")

        self.assertEqual(home.status_code, 200)
        self.assertIn(
            "필요한 정보만 입력하면 자동차 실제 도로 기준 예상 시간과 요금을 바로 확인할 수 있어요.",
            home.text,
        )
        self.assertNotIn('class="booking-steps"', home.text)
        self.assertIn("보내는 분 연락처와 동일", home.text)
        self.assertIn("받는 사람 이름", home.text)
        self.assertIn('id="quickReview"', home.text)
        self.assertIn('id="quoteButton"', home.text)
        self.assertIn('id="orderButton"', home.text)
        self.assertIn('id="reviewEmpty"', home.text)
        self.assertIn('id="paymentStepScreen"', home.text)
        self.assertIn('onclick="openPaymentStep()"', home.text)
        self.assertIn('id="finalOrderButton"', home.text)
        self.assertIn("주소 확인이 필요해요", home.text)
        self.assertIn("출발지 미설정", home.text)
        self.assertIn("도착지 미설정", home.text)
        self.assertIn('id="wishDate"', home.text)
        self.assertIn('id="pickupSchedule"', home.text)
        self.assertIn("level: 5", home.text)

    def test_customer_pages_include_accessible_typography_and_controls(self) -> None:
        home = self.client.get("/order")
        features = self.client.get("/features")
        history = self.client.get("/history")

        for response in (home, features, history):
            self.assertEqual(response.status_code, 200)
            self.assertIn("system-ui, -apple-system", response.text)
            self.assertIn("focus-visible", response.text)

        self.assertIn("min-height: 44px", home.text)
        self.assertIn("font-size: 16px", home.text)
        self.assertIn("color: #707070", home.text)

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
                    "formSnapshot": {"pickupAddress": "서울역"},
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["reply"], "자체 QA 응답")
        self.assertEqual(response.json()["data"]["trace"], ["local_model:own"])
        responder.assert_called_once_with(
            "MOVB가 뭐야?",
            "own",
            {"pickupAddress": "서울역"},
        )

    def test_own_model_handles_basic_deployed_chat(self) -> None:
        greeting = GREETING_REPLY
        self.assertEqual(own_model_reply("헬로"), greeting)
        self.assertEqual(own_model_reply("ㅇㅇ"), greeting)
        self.assertIn("MOVB", own_model_reply("MOVB가 뭐야"))
        self.assertIn("연결 상태", own_model_reply("지금은 켰어"))
        self.assertIn("연결 상태", own_model_reply("라마"))
        self.assertGreaterEqual(len(load_qa_index()), 80)

    def test_own_model_routes_service_actions_without_hallucinating(self) -> None:
        order_reply = own_model_reply("배송 주문해줘")
        difference_reply = own_model_reply("퀵과 도보 배송 차이가 뭐야?")

        self.assertIn("AI 채팅", order_reply)
        self.assertNotIn("인증 키", order_reply)
        self.assertIn("차량", difference_reply)
        self.assertIn("도보", difference_reply)
        self.assertNotIn("택시 동승", difference_reply)

    def test_own_model_answers_quick_customer_support_questions(self) -> None:
        self.assertIn("15%", own_model_reply("퀵 취소 수수료가 얼마야?"))
        self.assertIn("140cm", own_model_reply("보내면 안 되는 물건 알려줘"))
        self.assertIn("2,000원", own_model_reply("기사 대기료가 얼마야?"))
        self.assertIn("픽업 출발", own_model_reply("기사님 위치는 언제 보여?"))
        self.assertIn("서비스 문의", own_model_reply("고객센터 어디야?"))

    def test_product_size_uses_measurable_api_limits(self) -> None:
        answer = own_model_reply("물품 크기는 어떤 게 있어")
        order_page = (
            Path(__file__).resolve().parents[1] / "mobility_service" / "index.html"
        ).read_text(encoding="utf-8")

        self.assertIn("80cm·2kg 이하", answer)
        self.assertIn("140cm·20kg 이하", answer)
        self.assertIn("가로+세로+높이 100cm 이하 · 5kg 이하", order_page)
        self.assertNotIn("작은 택배상자", order_page)
        self.assertNotIn("보통 크기 상자", order_page)

    def test_own_model_answers_general_chat_without_ollama(self) -> None:
        self.assertIn("힘드셨겠어요", own_model_reply("오늘 기분이 안 좋아"))
        self.assertIn("실시간 날씨", own_model_reply("오늘 날씨 어때?"))
        self.assertIn("답을 찾지 못했어요", own_model_reply("서울에서 부산까지 얼마나 걸려?"))
        self.assertIn("답을 찾지 못했어요", own_model_reply("파이썬이 뭐야?"))
        self.assertNotIn("프로그래밍 언어", own_model_reply("파이썬이 뭐야?"))

    def test_own_model_uses_bundled_daily_and_movb_qa(self) -> None:
        self.assertGreaterEqual(len(load_qa_index()), 80)
        self.assertIn("AI 채팅", own_model_reply("배송 주문해줘"))
        self.assertIn("김치찌개", own_model_reply("뭐 먹을까?"))

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
