from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from mobility_service.agent import DeliveryAgent
from mobility_service.app import create_app
from mobility_service.conversation_store import ConversationStore
from mobility_service.geocode import KakaoGeocodeClient
from mobility_service.knowledge import default_knowledge_base
from mobility_service.pool_store import PoolStore
from mobility_service.store import MobilityStore
from tests.test_mobility_service import FakeKakaoClient, settings


class OfflineRouter:
    def generate(self, *args, **kwargs):
        raise RuntimeError("테스트에서는 외부 LLM을 호출하지 않습니다.")


class MobilityKnowledgeTests(unittest.TestCase):
    def test_retrieval_finds_expected_domain_documents(self) -> None:
        knowledge = default_knowledge_base()
        cases = {
            "MOVB는 어떤 서비스야?": "01-service-overview",
            "퀵과 도보 배송의 차이는 뭐야?": "02-delivery-options",
            "주문 상태는 어떻게 확인해?": "03-order-lifecycle",
            "합승 요금은 어떻게 나눠?": "04-bundle-pool-carpool",
            "Sandbox에서 실제 결제가 돼?": "05-sandbox-and-safety",
        }

        for question, expected_source in cases.items():
            with self.subTest(question=question):
                results = knowledge.search(question, limit=3)
                self.assertTrue(results)
                self.assertTrue(
                    any(result.chunk_id.startswith(expected_source) for result in results)
                )

    def test_unrelated_question_does_not_force_a_knowledge_match(self) -> None:
        self.assertEqual(
            default_knowledge_base().search("오늘 부산 날씨 어때?"),
            [],
        )

    def test_extractive_fallback_hides_internal_evidence_title(self) -> None:
        knowledge = default_knowledge_base()
        results = knowledge.search("합승 요금은 어떻게 나눠?")
        answer = knowledge.fallback_answer(results)

        self.assertNotIn("근거 문서", answer)
        self.assertNotIn("[퀵 합승", answer)


class AgentKnowledgeRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        app_settings = settings(root)
        client = FakeKakaoClient()
        store = MobilityStore(root / "mobility.db")
        conversations = ConversationStore(root / "mobility.db")
        geocoder = KakaoGeocodeClient(app_settings)
        agent = DeliveryAgent(
            client,  # type: ignore[arg-type]
            geocoder,
            store,
            conversations,
            router=OfflineRouter(),  # type: ignore[arg-type]
            pools=PoolStore(root / "mobility.db"),
        )
        self.client = TestClient(
            create_app(
                settings=app_settings,
                client=client,  # type: ignore[arg-type]
                store=store,
                geocoder=geocoder,
                conversations=conversations,
                agent=agent,
            )
        )

    def tearDown(self) -> None:
        self.client.close()
        self.temporary.cleanup()

    def test_service_question_uses_knowledge_route_without_llm(self) -> None:
        response = self.client.post(
            "/api/agent/chat",
            json={"message": "퀵이랑 도보 배송은 뭐가 달라?"},
        )
        data = response.json()["data"]

        self.assertEqual(response.status_code, 200)
        self.assertTrue(data["sources"])
        self.assertTrue(
            any(source["id"].startswith("02-delivery-options") for source in data["sources"])
        )
        self.assertTrue(
            any(item.startswith("knowledge_qa:extractive") for item in data["trace"])
        )
        self.assertNotIn("출발지 주소", data["reply"])

    def test_knowledge_search_endpoint_exposes_ranked_evidence(self) -> None:
        response = self.client.get(
            "/api/knowledge/search",
            params={"q": "Sandbox에서 실제 결제가 돼?", "limit": 2},
        )
        data = response.json()["data"]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["query"], "Sandbox에서 실제 결제가 돼?")
        self.assertLessEqual(len(data["results"]), 2)
        self.assertTrue(
            data["results"][0]["id"].startswith("05-sandbox-and-safety")
        )

    def test_pool_fee_question_is_not_misread_as_registration(self) -> None:
        response = self.client.post(
            "/api/agent/chat",
            json={"message": "합승 요금은 어떻게 나눠?"},
        )
        data = response.json()["data"]

        self.assertTrue(data["sources"])
        self.assertNotIn("근거 문서", data["reply"])
        self.assertNotIn("합승 등록에 아래 정보", data["reply"])

    def test_greeting_has_useful_offline_response(self) -> None:
        response = self.client.post(
            "/api/agent/chat",
            json={"message": "안녕하세요"},
        )
        data = response.json()["data"]

        self.assertIn("MOVB", data["reply"])
        self.assertIn("묶음배송", data["reply"])
        self.assertNotIn("필요한 정보가 더", data["reply"])

    def test_vehicle_choice_is_interactive_and_saved_without_llm(self) -> None:
        menu = self.client.post(
            "/api/agent/chat",
            headers={"X-Session-Id": "vehicle-session"},
            json={"message": "차량 선택지를 보여줘"},
        ).json()["data"]

        self.assertEqual(
            [action["label"] for action in menu["actions"]],
            ["오토바이", "다마스", "라보", "1톤"],
        )

        selected = self.client.post(
            "/api/agent/chat",
            headers={"X-Session-Id": "vehicle-session"},
            json={"message": "다마스로 퀵 배송할래"},
        ).json()["data"]

        self.assertEqual(selected["slots"]["fleet"], "DAMAS")
        self.assertEqual(selected["slots"]["orderType"], "QUICK")
        self.assertIn("출발지 주소", selected["reply"])

    def test_reservation_flow_asks_for_time_and_offers_a_choice(self) -> None:
        response = self.client.post(
            "/api/agent/chat",
            headers={"X-Session-Id": "reservation-session"},
            json={"message": "예약 배송을 시작하고 싶어"},
        )
        data = response.json()["data"]

        self.assertEqual(response.status_code, 200)
        self.assertIn("픽업 예약 시간", data["reply"])
        self.assertEqual(data["actions"][0]["label"], "내일 15시")
        self.assertTrue(data["slots"]["_reservationRequested"])


if __name__ == "__main__":
    unittest.main()
