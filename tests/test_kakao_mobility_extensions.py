from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient
from pydantic import ValidationError

from mobility_service.app import create_app
from mobility_service.client import KakaoMobilityClient
from mobility_service.directions import KakaoDirectionsClient
from mobility_service.models import DeliveryDraft, Location
from mobility_service.store import MobilityStore
from tests.test_mobility_service import (
    FakeKakaoClient,
    sample_order,
    settings,
)


def directions_response(
    *,
    distance: int = 12_340,
    duration: int = 1_800,
) -> dict[str, Any]:
    return {
        "trans_id": "route-request-1",
        "routes": [
            {
                "result_code": 0,
                "result_msg": "길찾기 성공",
                "summary": {
                    "distance": distance,
                    "duration": duration,
                    "fare": {"taxi": 18_000, "toll": 0},
                },
                "sections": [
                    {"distance": 5_000, "duration": 700},
                    {"distance": 7_340, "duration": 1_100},
                ],
            }
        ],
    }


class DirectionsClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_multi_waypoint_uses_official_endpoint_and_kakaoak_auth(self) -> None:
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=directions_response())

        with tempfile.TemporaryDirectory() as temp:
            client = KakaoDirectionsClient(
                settings(Path(temp), rest_key="rest-api-key"),
                transport=httpx.MockTransport(handler),
            )
            origin = Location(
                basicAddress="판교역", latitude=37.3946, longitude=127.1118
            )
            waypoint = Location(
                basicAddress="정자역", latitude=37.366, longitude=127.108
            )
            destination = Location(
                basicAddress="서현역", latitude=37.385, longitude=127.123
            )
            result = await client.route_summary(
                origin, destination, waypoints=[waypoint]
            )
            await client.close()

        self.assertEqual(requests[0].url.path, "/v1/waypoints/directions")
        self.assertEqual(
            requests[0].headers["authorization"], "KakaoAK rest-api-key"
        )
        payload = json.loads(requests[0].content)
        self.assertEqual(payload["waypoints"][0]["name"], "경유지 1")
        self.assertTrue(payload["summary"])
        self.assertEqual(result["distanceKm"], 12.3)
        self.assertTrue(result["actualRoadData"])

    async def test_future_route_normalizes_iso_departure_time(self) -> None:
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=directions_response())

        with tempfile.TemporaryDirectory() as temp:
            client = KakaoDirectionsClient(
                settings(Path(temp), rest_key="rest-api-key"),
                transport=httpx.MockTransport(handler),
            )
            origin = Location(
                basicAddress="판교역", latitude=37.3946, longitude=127.1118
            )
            destination = Location(
                basicAddress="서현역", latitude=37.385, longitude=127.123
            )
            result = await client.route_summary(
                origin,
                destination,
                departure_time="2026-07-24T10:30:00+09:00",
            )
            await client.close()

        self.assertEqual(requests[0].url.path, "/v1/future/directions")
        self.assertEqual(
            requests[0].url.params["departure_time"], "202607241030"
        )
        self.assertTrue(result["futureTrafficApplied"])


class FleetAndLogisticsClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_fleet_option_is_sent_to_price_and_create_order(self) -> None:
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json=(
                    {"totalPrice": 24_000}
                    if request.url.path.endswith("/price")
                    else {
                        "partnerOrderId": "fleet-order",
                        "receipt": {"status": "MATCHING"},
                    }
                ),
            )

        with tempfile.TemporaryDirectory() as temp:
            client = KakaoMobilityClient(
                settings(Path(temp)),
                transport=httpx.MockTransport(handler),
            )
            payload = sample_order("fleet-order")
            payload["fleetOption"] = {"fleet": "DAMAS", "type": "REQUIRED"}
            draft = DeliveryDraft.model_validate(payload)
            await client.price(draft)
            from mobility_service.models import CreateDeliveryRequest

            order = CreateDeliveryRequest.model_validate(payload)
            await client.create_order(order, "fleet-order")
            await client.close()

        bodies = [json.loads(request.content) for request in requests]
        self.assertEqual(
            bodies[0]["fleetOption"],
            {"fleet": "DAMAS", "type": "REQUIRED"},
        )
        self.assertEqual(bodies[1]["fleetOption"]["fleet"], "DAMAS")

    async def test_step_and_sandbox_status_paths_match_official_docs(self) -> None:
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"ok": True})

        with tempfile.TemporaryDirectory() as temp:
            client = KakaoMobilityClient(
                settings(Path(temp)),
                transport=httpx.MockTransport(handler),
            )
            await client.get_step("order/unsafe", "step/unsafe")
            await client.change_sandbox_status(
                "order-1", "CANCEL", cancel_by="ADMIN"
            )
            await client.close()

        self.assertEqual(
            requests[0].url.raw_path.decode(),
            "/goa-sandbox-service/api/v2/orders/order%2Funsafe/steps/step%2Funsafe",
        )
        self.assertEqual(
            requests[1].url.path,
            "/goa-sandbox-service/api/v1/developers/orders/order-1/status",
        )
        self.assertEqual(
            json.loads(requests[1].content),
            {"orderStatus": "CANCEL", "cancelBy": "ADMIN"},
        )

    def test_large_quick_requires_explicit_fleet(self) -> None:
        payload = sample_order()
        payload["productSize"] = "L"
        with self.assertRaises(ValidationError):
            DeliveryDraft.model_validate(payload)


class MobilityExtensionApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.fake = FakeKakaoClient()
        self.store = MobilityStore(root / "mobility.db")
        self.app = create_app(
            settings=settings(
                root,
                admin_username="admin",
                admin_password="admin-test-pass-1234",
            ),
            client=self.fake,  # type: ignore[arg-type]
            store=self.store,
        )
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.client.close()
        self.temporary.cleanup()

    def test_order_steps_returns_pickup_waypoint_and_dropoff(self) -> None:
        self.client.post("/api/orders", json=sample_order("step-order"))
        response = self.client.get("/api/orders/step-order/steps")
        data = response.json()["data"]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [step["kind"] for step in data["steps"]],
            ["PICKUP", "WAYPOINT_1", "DROPOFF"],
        )
        self.assertTrue(all(step["status"] == "waiting" for step in data["steps"]))

    def test_admin_can_drive_sandbox_lifecycle(self) -> None:
        self.client.post("/api/orders", json=sample_order("lifecycle-order"))
        login = self.client.post(
            "/api/auth/login",
            json={"identifier": "admin", "password": "admin-test-pass-1234"},
        )
        self.assertEqual(login.status_code, 200)

        response = self.client.patch(
            "/api/admin/orders/lifecycle-order/sandbox-status",
            json={"orderStatus": "MATCH_PICKER"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["order"]["status"], "MATCHED")
        self.assertEqual(
            self.fake.sandbox_status_calls[-1],
            ("lifecycle-order", "MATCH_PICKER", None),
        )

    def test_route_summary_falls_back_transparently_without_rest_key(self) -> None:
        response = self.client.post(
            "/api/routes/summary",
            json={
                "origin": {
                    "basicAddress": "판교역",
                    "latitude": 37.3946,
                    "longitude": 127.1118,
                },
                "destination": {
                    "basicAddress": "정자역",
                    "latitude": 37.366,
                    "longitude": 127.108,
                },
            },
        )
        data = response.json()["data"]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["source"], "heuristic-fallback")
        self.assertFalse(data["actualRoadData"])


if __name__ == "__main__":
    unittest.main()
