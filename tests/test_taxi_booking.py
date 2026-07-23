from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from mobility_service.app import create_app
from mobility_service.config import Settings
from mobility_service.models import Location
from mobility_service.store import MobilityStore


class FakeGeocoder:
    locations = {
        "강남역": Location(
            basicAddress="서울 강남구 강남역", latitude=37.4979, longitude=127.0276
        ),
        "잠실역": Location(
            basicAddress="서울 송파구 잠실역", latitude=37.5133, longitude=127.1001
        ),
        "건대입구역": Location(
            basicAddress="서울 광진구 건대입구역", latitude=37.5404, longitude=127.0692
        ),
    }

    async def search_address(self, query: str) -> Location | None:
        return self.locations.get(query)


def test_settings(root: Path) -> Settings:
    return Settings(
        api_key="test-key",
        vendor_id="test-vendor",
        base_url="https://example.test",
        callback_base_url="https://callback.example.test",
        database_path=root / "mobility.db",
    )


def booking_payload(*, consent: bool = True) -> dict:
    return {
        "originAddress": "강남역",
        "passengers": [
            {"name": "A", "address": "잠실역"},
            {"name": "B", "address": "건대입구역"},
        ],
        "requesterName": "홍길동",
        "requesterPhone": "010-1234-5678",
        "departureMode": "now",
        "departureAt": None,
        "taxiType": "standard",
        "hasLuggage": False,
        "note": "3번 출구 앞",
        "consent": consent,
    }


class TaxiBookingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.store = MobilityStore(root / "mobility.db")
        self.client = TestClient(
            create_app(
                settings=test_settings(root),
                store=self.store,
                geocoder=FakeGeocoder(),  # type: ignore[arg-type]
            )
        )

    def tearDown(self) -> None:
        self.client.close()
        self.temporary.cleanup()

    def test_taxi_page_has_guided_booking_and_final_submit(self) -> None:
        response = self.client.get("/taxi")

        self.assertEqual(response.status_code, 200)
        self.assertIn("경로 입력", response.text)
        self.assertIn("탑승 조건", response.text)
        self.assertIn('id="requesterPhone"', response.text)
        self.assertIn("이 내용으로 택시 합승 접수하기", response.text)
        self.assertIn('id="receiptResult"', response.text)
        self.assertIn("실제 택시 배차와 결제는 진행되지", response.text)

    def test_booking_is_recomputed_saved_and_retrievable(self) -> None:
        response = self.client.post("/api/carpool/requests", json=booking_payload())

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["data"]["status"], "RECEIVED")
        self.assertTrue(body["data"]["requestId"].startswith("taxi-"))
        self.assertEqual(len(body["data"]["plan"]["stops"]), 2)
        self.assertIn("실제 택시 배차가 아닌", body["message"])

        request_id = body["data"]["requestId"]
        saved = self.client.get(f"/api/carpool/requests/{request_id}")
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.json()["data"]["request"]["requesterName"], "홍길동")
        self.assertEqual(saved.json()["data"]["plan"]["sharedFare"], body["data"]["plan"]["sharedFare"])

    def test_booking_requires_explicit_notice_consent(self) -> None:
        response = self.client.post(
            "/api/carpool/requests", json=booking_payload(consent=False)
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
