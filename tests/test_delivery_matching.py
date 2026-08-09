from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from mobility_service.app import create_app
from mobility_service.store import MobilityStore
from tests.test_mobility_service import (
    FakeKakaoClient,
    sample_order,
    settings,
)


def nearby_order() -> dict:
    payload = copy.deepcopy(sample_order())
    payload.pop("partnerOrderId", None)
    payload["pickup"]["location"] = {
        "basicAddress": "경기도 성남시 분당구 판교역로 160",
        "latitude": 37.397,
        "longitude": 127.114,
    }
    payload["pickup"]["contact"] = {
        "name": "두 번째 발송자",
        "phone": "010-1000-0003",
    }
    payload["dropoff"]["location"] = {
        "basicAddress": "경기도 성남시 분당구 정자일로 100",
        "latitude": 37.361,
        "longitude": 127.107,
    }
    payload["dropoff"]["contact"] = {
        "name": "두 번째 수령인",
        "phone": "010-1000-0004",
    }
    payload["productName"] = "두 번째 서류"
    return payload


class DeliveryMatchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.fake = FakeKakaoClient()
        self.store = MobilityStore(root / "matching.db")
        self.client = TestClient(
            create_app(
                settings=settings(root),
                client=self.fake,  # type: ignore[arg-type]
                store=self.store,
            )
        )

    def tearDown(self) -> None:
        self.client.close()
        self.temporary.cleanup()

    def test_two_computers_freeze_one_pooled_quote_before_payment(self) -> None:
        first_payload = sample_order()
        first_payload.pop("partnerOrderId", None)
        first = self.client.post(
            "/api/delivery-matches",
            headers={
                "X-Client-Id": "computer-one",
                "Idempotency-Key": "waiting-order-one",
            },
            json=first_payload,
        )

        self.assertEqual(first.status_code, 202)
        self.assertEqual(first.json()["data"]["matching"]["status"], "WAITING")
        self.assertEqual(self.fake.create_calls, 0)

        second = self.client.post(
            "/api/delivery-matches",
            headers={
                "X-Client-Id": "computer-two",
                "Idempotency-Key": "waiting-order-two",
            },
            json=nearby_order(),
        )

        self.assertEqual(second.status_code, 202)
        matching = second.json()["data"]["matching"]
        self.assertEqual(matching["status"], "MATCHED_AWAITING_PAYMENT")
        self.assertTrue(matching["partnerOrderId"].startswith("smart-pool-"))
        self.assertEqual(matching["paymentStatus"], "PENDING")
        self.assertEqual(matching["finalAmount"], 6000)
        self.assertEqual(self.fake.create_calls, 0)

        saved = self.store.get_order(matching["partnerOrderId"])
        self.assertIsNone(saved)

        first_history = self.client.get(
            "/api/delivery-matches",
            headers={"X-Client-Id": "computer-one"},
        )
        first_match = first_history.json()["data"][0]
        self.assertEqual(first_match["status"], "MATCHED_AWAITING_PAYMENT")
        self.assertEqual(first_match["finalAmount"], 6000)
        self.assertEqual(
            first_match["partnerOrderId"],
            matching["partnerOrderId"],
        )

    def test_same_computer_does_not_match_itself(self) -> None:
        first_payload = sample_order()
        first_payload.pop("partnerOrderId", None)
        for key, payload in (
            ("same-client-one", first_payload),
            ("same-client-two", nearby_order()),
        ):
            response = self.client.post(
                "/api/delivery-matches",
                headers={
                    "X-Client-Id": "same-computer",
                    "Idempotency-Key": key,
                },
                json=payload,
            )
            self.assertEqual(response.status_code, 202)
            self.assertEqual(
                response.json()["data"]["matching"]["status"],
                "WAITING",
            )
        self.assertEqual(self.fake.create_calls, 0)

    def test_same_logged_in_account_on_two_computers_does_not_self_match(self) -> None:
        registered = self.client.post(
            "/api/auth/register",
            json={
                "name": "공동배송 사용자",
                "email": "matching@example.com",
                "password": "safe-password",
            },
        )
        self.assertEqual(registered.status_code, 201)
        first_payload = sample_order()
        first_payload.pop("partnerOrderId", None)
        for client_id, key, payload in (
            ("logged-computer-one", "logged-order-one", first_payload),
            ("logged-computer-two", "logged-order-two", nearby_order()),
        ):
            response = self.client.post(
                "/api/delivery-matches",
                headers={
                    "X-Client-Id": client_id,
                    "Idempotency-Key": key,
                },
                json=payload,
            )
            self.assertEqual(response.status_code, 202)
            self.assertEqual(
                response.json()["data"]["matching"]["status"],
                "WAITING",
            )
        self.assertEqual(self.fake.create_calls, 0)

    def test_different_computers_with_unrelated_routes_stay_waiting(self) -> None:
        first_payload = sample_order()
        first_payload.pop("partnerOrderId", None)
        far_payload = nearby_order()
        far_payload["pickup"]["location"] = {
            "basicAddress": "서울특별시 중구 세종대로 110",
            "latitude": 37.5663,
            "longitude": 126.9779,
        }
        far_payload["dropoff"]["location"] = {
            "basicAddress": "서울특별시 마포구 월드컵북로 400",
            "latitude": 37.5796,
            "longitude": 126.8898,
        }
        for client_id, key, payload in (
            ("route-client-one", "route-order-one", first_payload),
            ("route-client-two", "route-order-two", far_payload),
        ):
            response = self.client.post(
                "/api/delivery-matches",
                headers={
                    "X-Client-Id": client_id,
                    "Idempotency-Key": key,
                },
                json=payload,
            )
            self.assertEqual(response.status_code, 202)
            self.assertEqual(
                response.json()["data"]["matching"]["status"],
                "WAITING",
            )
        self.assertEqual(self.fake.create_calls, 0)

    def test_waiting_match_can_be_canceled_by_its_computer(self) -> None:
        payload = sample_order()
        payload.pop("partnerOrderId", None)
        self.client.post(
            "/api/delivery-matches",
            headers={
                "X-Client-Id": "cancel-computer",
                "Idempotency-Key": "cancel-waiting-order",
            },
            json=payload,
        )

        denied = self.client.patch(
            "/api/delivery-matches/cancel-waiting-order/cancel",
            headers={"X-Client-Id": "another-computer"},
        )
        self.assertEqual(denied.status_code, 409)

        canceled = self.client.patch(
            "/api/delivery-matches/cancel-waiting-order/cancel",
            headers={"X-Client-Id": "cancel-computer"},
        )
        self.assertEqual(canceled.status_code, 200)
        self.assertEqual(
            canceled.json()["data"]["matching"]["status"],
            "CANCELED",
        )

    def test_canceled_match_can_retry_as_one_idempotent_quick_order(self) -> None:
        payload = sample_order()
        payload.pop("partnerOrderId", None)
        self.client.post(
            "/api/delivery-matches",
            headers={
                "X-Client-Id": "single-retry-computer",
                "Idempotency-Key": "single-retry-request",
            },
            json=payload,
        )
        self.client.patch(
            "/api/delivery-matches/single-retry-request/cancel",
            headers={"X-Client-Id": "single-retry-computer"},
        )

        first = self.client.post(
            "/api/delivery-matches/single-retry-request/single-order",
            headers={"X-Client-Id": "single-retry-computer"},
        )
        second = self.client.post(
            "/api/delivery-matches/single-retry-request/single-order",
            headers={"X-Client-Id": "single-retry-computer"},
        )

        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 201)
        self.assertEqual(self.fake.create_calls, 1)
        first_result = first.json()["data"]["orderResult"]
        second_result = second.json()["data"]["orderResult"]
        self.assertTrue(first_result["partnerOrderId"].startswith("quick-retry-"))
        self.assertEqual(second_result["source"], "existing")

    def test_waiting_match_cannot_create_a_duplicate_single_order(self) -> None:
        payload = sample_order()
        payload.pop("partnerOrderId", None)
        self.client.post(
            "/api/delivery-matches",
            headers={
                "X-Client-Id": "still-waiting-computer",
                "Idempotency-Key": "still-waiting-request",
            },
            json=payload,
        )

        response = self.client.post(
            "/api/delivery-matches/still-waiting-request/single-order",
            headers={"X-Client-Id": "still-waiting-computer"},
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.fake.create_calls, 0)
