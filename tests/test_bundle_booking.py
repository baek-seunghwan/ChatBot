from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from mobility_service.app import create_app
from mobility_service.models import Location
from mobility_service.store import MobilityStore
from tests.test_mobility_service import FakeKakaoClient, settings


class FakeGeocoder:
    locations = {
        "판교역": Location(
            basicAddress="경기 성남시 분당구 판교역",
            latitude=37.3947,
            longitude=127.1112,
        ),
        "정자역": Location(
            basicAddress="경기 성남시 분당구 정자역",
            latitude=37.3661,
            longitude=127.1080,
        ),
        "서현역": Location(
            basicAddress="경기 성남시 분당구 서현역",
            latitude=37.3851,
            longitude=127.1230,
        ),
        "야탑역": Location(
            basicAddress="경기 성남시 분당구 야탑역",
            latitude=37.4113,
            longitude=127.1287,
        ),
    }

    async def search_address(self, query: str) -> Location | None:
        return self.locations.get(query)


def bundle_payload(*, consent: bool = True) -> dict:
    return {
        "pickupAddress": "판교역",
        "pickupContact": {
            "name": "테스트 발송자",
            "phone": "010-1000-0001",
        },
        "pickupNote": "1번 출구 앞",
        "dropoffs": [
            {
                "address": "정자역",
                "contact": {
                    "name": "첫 번째 수령인",
                    "phone": "010-1000-0002",
                },
                "note": "역무실 앞",
            },
            {
                "address": "서현역",
                "contact": {
                    "name": "두 번째 수령인",
                    "phone": "010-1000-0003",
                },
            },
        ],
        "productSize": "XS",
        "productName": "테스트 서류",
        "quantity": "2",
        "declaredValue": 20000,
        "consent": consent,
    }


class BundleBookingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.fake = FakeKakaoClient()
        self.store = MobilityStore(root / "mobility.db")
        self.client = TestClient(
            create_app(
                settings=settings(root),
                client=self.fake,  # type: ignore[arg-type]
                store=self.store,
                geocoder=FakeGeocoder(),  # type: ignore[arg-type]
            )
        )

    def tearDown(self) -> None:
        self.client.close()
        self.temporary.cleanup()

    def test_bundle_page_has_quote_consent_and_order_flow(self) -> None:
        response = self.client.get("/bundle")

        self.assertEqual(response.status_code, 200)
        self.assertIn("목적지 2곳부터 시작", response.text)
        self.assertIn('id="quoteButton"', response.text)
        self.assertIn('id="confirmOrder"', response.text)
        self.assertIn('id="orderButton"', response.text)
        self.assertIn('id="receipt"', response.text)
        self.assertNotIn("택시합승", response.text)

    def test_quote_compares_individual_and_bundled_prices(self) -> None:
        response = self.client.post(
            "/api/bundle/quote",
            json={
                "pickupAddress": "판교역",
                "dropoffAddresses": ["정자역", "서현역"],
                "productSize": "XS",
            },
        )

        self.assertEqual(response.status_code, 200)
        quote = response.json()["data"]
        self.assertEqual(quote["individualTotal"], 24000)
        self.assertEqual(quote["bundledPrice"], 12000)
        self.assertEqual(quote["saving"], 12000)
        self.assertTrue(quote["recommendBundle"])
        self.assertEqual(len(quote["route"]), 2)

    def test_order_recomputes_quote_and_saves_all_contacts(self) -> None:
        response = self.client.post(
            "/api/bundle/orders",
            headers={"Idempotency-Key": "bundle-test-001"},
            json=bundle_payload(),
        )

        self.assertEqual(response.status_code, 201)
        data = response.json()["data"]
        self.assertEqual(data["orderResult"]["source"], "created")
        self.assertEqual(data["quote"]["bundledPrice"], 12000)

        saved = self.store.get_order("bundle-test-001")
        self.assertIsNotNone(saved)
        request = saved["request"]
        self.assertEqual(request["pickup"]["contact"]["name"], "테스트 발송자")
        self.assertEqual(len(request["waypoints"]), 1)
        receiver_names = {
            request["waypoints"][0]["contact"]["name"],
            request["dropoff"]["contact"]["name"],
        }
        self.assertEqual(
            receiver_names,
            {"첫 번째 수령인", "두 번째 수령인"},
        )

    def test_same_idempotency_key_does_not_create_twice(self) -> None:
        headers = {"Idempotency-Key": "bundle-same-001"}

        first = self.client.post(
            "/api/bundle/orders", headers=headers, json=bundle_payload()
        )
        second = self.client.post(
            "/api/bundle/orders", headers=headers, json=bundle_payload()
        )

        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 201)
        self.assertEqual(self.fake.create_calls, 1)
        self.assertEqual(
            second.json()["data"]["orderResult"]["source"],
            "existing",
        )

    def test_order_requires_explicit_consent(self) -> None:
        response = self.client.post(
            "/api/bundle/orders",
            json=bundle_payload(consent=False),
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("동의", response.text)

    def test_order_rejects_duplicate_destination_addresses(self) -> None:
        payload = bundle_payload()
        payload["dropoffs"][1]["address"] = "정자역"

        response = self.client.post("/api/bundle/orders", json=payload)

        self.assertEqual(response.status_code, 422)
        self.assertIn("서로 다른 주소", response.text)


if __name__ == "__main__":
    unittest.main()
