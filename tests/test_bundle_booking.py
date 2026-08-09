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
        "대전역": Location(
            basicAddress="대전 동구 중앙로 215",
            latitude=36.3324,
            longitude=127.4342,
        ),
        "대전시청": Location(
            basicAddress="대전 서구 둔산로 100",
            latitude=36.3504,
            longitude=127.3848,
        ),
        "유성온천역": Location(
            basicAddress="대전 유성구 계룡로 97",
            latitude=36.3537,
            longitude=127.3414,
        ),
    }

    async def search_address(self, query: str) -> Location | None:
        return self.locations.get(query)


def bundle_payload(*, consent: bool = True) -> dict:
    return {
        "pickups": [
            {
                "address": "판교역",
                "contact": {
                    "name": "테스트 발송자",
                    "phone": "010-1000-0001",
                },
                "note": "1번 출구 앞",
            }
        ],
        "dropoffs": [
            {
                "address": "정자역",
                "contact": {
                    "name": "첫 번째 수령인",
                    "phone": "010-1000-0002",
                },
                "note": "역무실 앞",
                "pickupIndex": 0,
            },
            {
                "address": "서현역",
                "contact": {
                    "name": "두 번째 수령인",
                    "phone": "010-1000-0003",
                },
                "pickupIndex": 0,
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

    def test_home_form_has_quick_and_automatic_smart_delivery_flow(self) -> None:
        response = self.client.get("/order")

        self.assertEqual(response.status_code, 200)
        self.assertIn("+ 보내는 사람", response.text)
        self.assertIn('id="addSmartDropoff"', response.text)
        self.assertIn("경유지 추가", response.text)
        self.assertIn('id="roundTrip"', response.text)
        self.assertIn('data-order-type="SMART"', response.text)
        self.assertIn('data-fleet="MOTORCYCLE"', response.text)
        self.assertIn('name="pickupHandling"', response.text)
        self.assertIn('id="driverNote"', response.text)
        self.assertIn('name="paymentType"', response.text)
        self.assertIn("카드 선결제", response.text)
        self.assertNotIn("🚗 자동차 실제 도로", response.text)
        self.assertNotIn("🚗 자동차 도로 약", response.text)
        self.assertNotIn("가격 조회 선택값", response.text)
        self.assertIn("function isSmartDelivery()", response.text)
        self.assertIn("스마트 딜리버리로 자동 전환", response.text)
        section_order = [
            'class="form-section address-section"',
            'class="form-section vehicle-section"',
            'class="form-section delivery-option-section"',
            'class="form-section product-section"',
            'class="form-section handling-section"',
            'class="form-section driver-note-section"',
        ]
        section_positions = [response.text.index(label) for label in section_order]
        self.assertEqual(section_positions, sorted(section_positions))
        self.assertGreater(
            response.text.index('id="paymentStepScreen"'),
            section_positions[-1],
        )
        self.assertGreater(
            response.text.index('class="result result-prominent"'),
            response.text.index('class="order-entry-layout"'),
        )
        self.assertLess(
            response.text.index('class="result result-prominent"'),
            response.text.index('id="paymentStepScreen"'),
        )
        self.assertIn('id="quoteButton"', response.text)
        self.assertIn('id="orderButton"', response.text)
        self.assertIn("배송 방법을 선택하세요", response.text)
        self.assertIn("이 배송으로 결제하기", response.text)
        self.assertIn("전체 배송 옵션 보기", response.text)
        self.assertIn("fare-recommendation-card", response.text)
        self.assertIn('{item: current, label: value.autoRecommended ? "추천" : "현재 선택"}', response.text)
        self.assertNotIn("CHECKOUT · 2단계", response.text)
        self.assertNotIn("QUICK DELIVERY", response.text)
        self.assertNotIn("지도에서 출발지와 도착지를 바로 지정할 수 있어요.", response.text)
        self.assertNotIn("다마스·라보·1톤 이용 시 기사님이 물품 픽업과 상·하차를 도와드리는 옵션입니다.", response.text)
        self.assertIn("#quickRequest.match-stage .order-entry-layout", response.text)
        self.assertIn('byId("quickRequest").classList.add("match-stage")', response.text)
        self.assertNotIn("ORDER SUMMARY", response.text)
        self.assertNotIn("PAYMENT METHOD", response.text)
        self.assertIn('src="/assets/kakaopay-logo.png?v=20260810"', response.text)
        self.assertNotIn("priceRanks", response.text)
        self.assertIn('value="" selected disabled>물품 크기를 선택해주세요', response.text)
        self.assertNotIn('byId("productSize").value = "L";', response.text)
        self.assertIn('["DAMAS", "LABO", "TON"].includes(byId("fleet").value)', response.text)
        self.assertIn("fare-vehicle-group", response.text)
        self.assertIn('await callPrice(byId("quoteButton"))', response.text)
        self.assertIn("스마트 딜리버리 매칭 시작", response.text)
        self.assertIn("MATCHED_AWAITING_PAYMENT", response.text)
        self.assertIn("startMatchedKakaoPay", response.text)
        self.assertIn("배송이 함께 묶였어요", response.text)
        self.assertIn("matched-checkout-card", response.text)
        self.assertNotIn("Sandbox 데모 매칭 완료", response.text)
        self.assertIn("/api/smart-delivery/quote", response.text)
        self.assertIn("/api/smart-delivery/orders", response.text)
        self.assertNotIn("택시합승", response.text)

    def test_quote_compares_individual_and_bundled_prices(self) -> None:
        response = self.client.post(
            "/api/smart-delivery/quote",
            json={
                "pickups": [{"address": "판교역"}],
                "dropoffs": [
                    {"address": "정자역", "pickupIndex": 0},
                    {"address": "서현역", "pickupIndex": 0},
                ],
                "productSize": "XS",
            },
        )

        self.assertEqual(response.status_code, 200)
        quote = response.json()["data"]
        self.assertEqual(quote["individualTotal"], 24000)
        self.assertEqual(quote["bundledPrice"], 12000)
        self.assertEqual(quote["saving"], 12000)
        self.assertTrue(quote["recommendBundle"])
        self.assertEqual(quote["pickupRoute"], ["경기 성남시 분당구 판교역"])

    def test_smart_delivery_accepts_daejeon_addresses(self) -> None:
        response = self.client.post(
            "/api/smart-delivery/quote",
            json={
                "pickups": [{"address": "대전역"}],
                "dropoffs": [
                    {"address": "대전시청", "pickupIndex": 0},
                    {"address": "유성온천역", "pickupIndex": 0},
                ],
                "productSize": "XS",
            },
        )

        self.assertEqual(response.status_code, 200)
        quote = response.json()["data"]
        self.assertEqual(quote["pickup"], "대전 동구 중앙로 215")
        self.assertEqual(len(quote["dropoffRoute"]), 2)
        self.assertEqual(len(quote["route"]), 3)
        self.assertEqual(len(quote["route"]), 3)

    def test_order_recomputes_quote_and_saves_all_contacts(self) -> None:
        response = self.client.post(
            "/api/smart-delivery/orders",
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
        history = self.client.get("/api/orders").json()["data"]
        self.assertEqual(history[0]["serviceType"], "SMART_DELIVERY")
        self.assertIn("경유 1곳", history[0]["routeSummary"])

    def test_multiple_senders_are_picked_up_before_any_delivery(self) -> None:
        payload = bundle_payload()
        payload["pickups"].append(
            {
                "address": "야탑역",
                "contact": {
                    "name": "두 번째 발송자",
                    "phone": "010-1000-0004",
                },
                "note": "2번 출구 앞",
            }
        )
        payload["dropoffs"][1]["pickupIndex"] = 1

        response = self.client.post(
            "/api/smart-delivery/orders",
            headers={"Idempotency-Key": "bundle-multi-pickup-001"},
            json=payload,
        )

        self.assertEqual(response.status_code, 201)
        data = response.json()["data"]
        self.assertEqual(len(data["quote"]["pickupRoute"]), 2)
        saved = self.store.get_order("bundle-multi-pickup-001")
        request = saved["request"]
        self.assertEqual(request["pickup"]["contact"]["name"], "테스트 발송자")
        self.assertEqual(
            request["waypoints"][0]["contact"]["name"],
            "두 번째 발송자",
        )
        self.assertTrue(
            request["waypoints"][0]["note"].startswith("[추가 픽업]")
        )

    def test_same_idempotency_key_does_not_create_twice(self) -> None:
        headers = {"Idempotency-Key": "bundle-same-001"}

        first = self.client.post(
            "/api/smart-delivery/orders", headers=headers, json=bundle_payload()
        )
        second = self.client.post(
            "/api/smart-delivery/orders", headers=headers, json=bundle_payload()
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
            "/api/smart-delivery/orders",
            json=bundle_payload(consent=False),
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("동의", response.text)

    def test_order_rejects_duplicate_destination_addresses(self) -> None:
        payload = bundle_payload()
        payload["dropoffs"][1]["address"] = "정자역"

        response = self.client.post("/api/smart-delivery/orders", json=payload)

        self.assertEqual(response.status_code, 422)
        self.assertIn("배송 주소는 서로 달라야", response.text)

    def test_every_sender_must_be_connected_to_a_receiver(self) -> None:
        payload = bundle_payload()
        payload["pickups"].append(
            {
                "address": "야탑역",
                "contact": {
                    "name": "연결 안 된 발송자",
                    "phone": "010-1000-0004",
                },
            }
        )

        response = self.client.post("/api/smart-delivery/orders", json=payload)

        self.assertEqual(response.status_code, 422)
        self.assertIn("모든 보내는 사람", response.text)


if __name__ == "__main__":
    unittest.main()
