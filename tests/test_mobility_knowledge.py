from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
from fastapi.testclient import TestClient

from mobility_service.agent import DeliveryAgent
from mobility_service.app import create_app
from mobility_service.conversation_store import ConversationStore
from mobility_service.geocode import KakaoGeocodeClient
from mobility_service.knowledge import default_knowledge_base
from mobility_service.models import Location
from mobility_service import local_responder
from mobility_service.site_crawler import (
    CrawledPage,
    extract_visible_sections,
    render_knowledge_snapshot,
)
from mobility_service.store import MobilityStore
from tests.test_mobility_service import FakeKakaoClient, settings


class OfflineRouter:
    def generate(self, *args, **kwargs):
        raise RuntimeError("테스트에서는 외부 LLM을 호출하지 않습니다.")


class DeterministicGeocoder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search_address(self, query: str) -> Location | None:
        self.queries.append(query)
        locations = {
            "서울 동작구 사당로2가길 10-3": Location(
                basicAddress="서울 동작구 사당로2가길 10-3",
                latitude=37.48,
                longitude=126.97,
            ),
            "경기 화성시 와우로 85": Location(
                basicAddress="경기 화성시 봉담읍 와우로 85",
                latitude=37.21,
                longitude=126.95,
            ),
        }
        return locations.get(query)

    async def close(self) -> None:
        return None


class MobilityKnowledgeTests(unittest.TestCase):
    def test_retrieval_finds_expected_domain_documents(self) -> None:
        knowledge = default_knowledge_base()
        cases = {
            "MOVB는 어떤 서비스야?": "01-service-overview",
            "퀵과 도보 배송의 차이는 뭐야?": "02-delivery-options",
            "주문 상태는 어떻게 확인해?": "03-order-lifecycle",
            "묶음퀵 요금은 어떻게 계산해?": "04-bundle-quick",
            "Sandbox에서 실제 결제가 돼?": "05-sandbox-and-safety",
            "퀵 취소 수수료가 얼마야?": "07-quick-customer-support",
            "기사님 위치는 언제 보여?": "07-quick-customer-support",
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
        results = knowledge.search("묶음퀵 요금은 어떻게 계산해?")
        answer = knowledge.fallback_answer(results)

        self.assertNotIn("근거 문서", answer)
        self.assertNotIn("[묶음퀵", answer)

    def test_homepage_crawler_extracts_visible_service_copy(self) -> None:
        sections = extract_visible_sections(
            """
            <style>.hidden { display: none; }</style>
            <h1>스마트 딜리버리</h1>
            <p>대전에서도 주소를 입력할 수 있습니다.</p>
            <script>secret = "not knowledge"</script>
            """
        )
        snapshot = render_knowledge_snapshot(
            [
                CrawledPage(
                    url="https://movb.example/",
                    sections=sections,
                )
            ],
            crawled_at=datetime(2026, 7, 28, tzinfo=timezone.utc),
        )

        self.assertIn("스마트 딜리버리", snapshot)
        self.assertIn("대전", snapshot)
        self.assertNotIn("secret", snapshot)

    def test_smart_delivery_question_uses_homepage_knowledge(self) -> None:
        results = default_knowledge_base().search(
            "대전에서도 스마트 딜리버리 이용할 수 있어?"
        )

        self.assertTrue(results)
        self.assertTrue(
            any(result.chunk_id.startswith("06-homepage-crawl") for result in results)
        )

    def test_quick_customer_support_knowledge_covers_core_policies(self) -> None:
        knowledge = default_knowledge_base()
        for question in (
            "퀵에서 보낼 수 없는 물건은 뭐야?",
            "기사 배정이 안 되면 어떻게 해?",
            "과적료와 대기료가 얼마야?",
            "카카오 퀵 고객센터는 어디야?",
        ):
            with self.subTest(question=question):
                results = knowledge.search(question, limit=3)
                self.assertTrue(
                    any(
                        result.chunk_id.startswith("07-quick-customer-support")
                        for result in results
                    )
                )


class LocalResponderTests(unittest.TestCase):
    def test_greetings_do_not_trigger_knowledge_search(self) -> None:
        knowledge = default_knowledge_base()
        with patch.object(
            knowledge,
            "search",
            wraps=knowledge.search,
        ) as mocked_search, patch(
            "mobility_service.local_responder.httpx.post",
            side_effect=httpx.ConnectError("offline"),
        ):
            reply = local_responder.local_model_reply("안녕", engine="ollama")

        self.assertIn("안녕하세요", reply)
        mocked_search.assert_not_called()

    def test_own_local_chat_can_read_current_form_snapshot(self) -> None:
        reply = local_responder.local_model_reply(
            "지금 화면에 입력된 배송 정보 알려줘",
            engine="own",
            form_snapshot={
                "pickupAddress": "서울역",
                "dropoffAddress": "대전역",
                "productName": "계약서",
                "unknownField": "모델에 보내면 안 되는 값",
            },
        )

        self.assertIn("출발지: 서울역", reply)
        self.assertIn("도착지: 대전역", reply)
        self.assertIn("물품명: 계약서", reply)
        self.assertNotIn("unknownField", reply)
        self.assertNotIn("모델에 보내면 안 되는 값", reply)

    def test_vllm_receives_retrieved_knowledge_and_current_form(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": "현재 입력 내용을 확인했어요."}}]
        }

        with patch.object(
            local_responder,
            "VLLM_BASE_URL",
            "https://model.example/v1",
        ), patch.object(
            local_responder,
            "VLLM_API_KEY",
            "test-secret",
        ), patch(
            "mobility_service.local_responder.httpx.post",
            return_value=response,
        ) as post:
            reply = local_responder.local_model_reply(
                "MOVB 스마트 딜리버리로 접수하려면 어떻게 해?",
                engine="vllm",
                form_snapshot={
                    "pickupAddress": "서울역",
                    "dropoffAddress": "대전역",
                },
            )

        self.assertEqual(reply, "현재 입력 내용을 확인했어요.")
        self.assertEqual(
            post.call_args.args[0],
            "https://model.example/v1/chat/completions",
        )
        system_prompt = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("[검색된 MOVB 근거]", system_prompt)
        self.assertIn("[현재 화면에 입력된 배송 정보]", system_prompt)
        self.assertIn("출발지: 서울역", system_prompt)
        self.assertIn("도착지: 대전역", system_prompt)
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            "Bearer test-secret",
        )


class AgentKnowledgeRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        app_settings = settings(root)
        client = FakeKakaoClient()
        store = MobilityStore(root / "mobility.db")
        conversations = ConversationStore(root / "mobility.db")
        geocoder = DeterministicGeocoder()
        agent = DeliveryAgent(
            client,  # type: ignore[arg-type]
            geocoder,  # type: ignore[arg-type]
            store,
            conversations,
            router=OfflineRouter(),  # type: ignore[arg-type]
        )
        self.client = TestClient(
            create_app(
                settings=app_settings,
                client=client,  # type: ignore[arg-type]
                store=store,
                geocoder=geocoder,  # type: ignore[arg-type]
                conversations=conversations,
                agent=agent,
            )
        )
        self.fake = client
        self.conversations = conversations
        self.geocoder = geocoder

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

    def test_bundle_fee_question_is_answered_from_knowledge(self) -> None:
        response = self.client.post(
            "/api/agent/chat",
            json={"message": "묶음퀵 요금은 어떻게 계산해?"},
        )
        data = response.json()["data"]

        self.assertTrue(data["sources"])
        self.assertNotIn("근거 문서", data["reply"])
        self.assertIn("견적", data["reply"])

    def test_smart_delivery_chat_opens_integrated_form(self) -> None:
        response = self.client.post(
            "/api/agent/chat",
            json={"message": "스마트 딜리버리 접수하고 싶어"},
        )
        data = response.json()["data"]

        self.assertEqual(response.status_code, 200)
        self.assertIn("대전", data["reply"])
        self.assertEqual(data["actions"][0]["target"], "smartDelivery")

    def test_greeting_has_useful_offline_response(self) -> None:
        response = self.client.post(
            "/api/agent/chat",
            json={"message": "안녕하세요"},
        )
        data = response.json()["data"]

        self.assertEqual(
            data["reply"],
            "안녕하세요! MOVB 서비스에 대해 물어보세요 🙂",
        )

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

    def test_ambiguous_address_asks_before_touching_form_or_quoting(self) -> None:
        session_id = "ambiguous-address-session"
        self.conversations.get_or_create(session_id)
        self.conversations.save_slots(
            session_id,
            {
                "pickupAddress": "서울 동작구 사당로2가길 10-3",
                "pickupLat": 37.48,
                "pickupLng": 126.97,
                "pickupAddressGeocoded": "서울 동작구 사당로2가길 10-3",
                "pickupName": "기존 발송자",
                "pickupPhone": "010-1000-0203",
                "dropoffAddress": "서울 동작구 기존로 1",
                "dropoffLat": 37.49,
                "dropoffLng": 126.98,
                "dropoffAddressGeocoded": "서울 동작구 기존로 1",
                "dropoffName": "ㅇ",
                "dropoffPhone": "010-1002-3091",
                "productName": "서류",
            },
            "collecting",
        )

        first = self.client.post(
            "/api/agent/chat",
            headers={"X-Session-Id": session_id},
            json={
                "message": "백승환 와우로 85 토마토오피스텔1동 408호",
                "formSnapshot": {
                    "pickupAddress": "서울 동작구 사당로2가길 10-3",
                    "pickupName": "기존 발송자",
                    "pickupPhone": "010-1000-0203",
                    "dropoffAddress": "서울 동작구 기존로 1",
                    "dropoffName": "ㅇ",
                    "dropoffPhone": "010-1002-3091",
                    "productName": "서류",
                },
            },
        ).json()["data"]

        self.assertEqual(self.fake.price_calls, 0)
        self.assertIsNone(first["quote"])
        self.assertEqual(first["changedSlots"], {})
        self.assertNotIn("서울 동작구 와우로 85", str(first["slots"]))
        self.assertIn("출발지·보내는 사람", first["reply"])
        self.assertIn("경기 화성시 와우로 85", first["reply"])
        self.assertEqual(
            [item["label"] for item in first["actions"][:2]],
            ["출발지예요", "도착지예요"],
        )

        role = self.client.post(
            "/api/agent/chat",
            headers={"X-Session-Id": session_id},
            json={"message": "도착지예요"},
        ).json()["data"]
        self.assertEqual(self.fake.price_calls, 0)
        self.assertIn("시·군·구", role["reply"])

        corrected = self.client.post(
            "/api/agent/chat",
            headers={"X-Session-Id": session_id},
            json={"message": "경기 화성시 와우로 85"},
        ).json()["data"]
        self.assertEqual(self.fake.price_calls, 0)
        self.assertEqual(
            corrected["changedSlots"]["dropoffAddress"],
            "경기 화성시 봉담읍 와우로 85",
        )
        self.assertEqual(corrected["changedSlots"]["dropoffName"], "백승환")
        self.assertEqual(corrected["changedSlots"]["dropoffPhone"], "")
        self.assertIn("받는 분 이름과 연락처", corrected["reply"])

        contact = self.client.post(
            "/api/agent/chat",
            headers={"X-Session-Id": session_id},
            json={"message": "받는 사람 연락처: 010-1000-0456"},
        ).json()["data"]
        self.assertEqual(self.fake.price_calls, 0)
        self.assertIn("이 내용으로 예상 시간과 요금을 확인할까요", contact["reply"])

        quoted = self.client.post(
            "/api/agent/chat",
            headers={"X-Session-Id": session_id},
            json={"message": "이 내용으로 견적 확인해줘"},
        ).json()["data"]
        self.assertEqual(self.fake.price_calls, 1)
        self.assertIsNotNone(quoted["quote"])

    def test_failed_new_geocode_clears_old_coordinates_and_blocks_quote(self) -> None:
        session_id = "failed-geocode-session"
        self.conversations.get_or_create(session_id)
        self.conversations.save_slots(
            session_id,
            {
                "pickupAddress": "서울 동작구 사당로2가길 10-3",
                "pickupLat": 37.48,
                "pickupLng": 126.97,
                "pickupAddressGeocoded": "서울 동작구 사당로2가길 10-3",
                "pickupName": "발송자",
                "pickupPhone": "010-1000-0001",
                "dropoffAddress": "서울 동작구 기존로 1",
                "dropoffLat": 37.49,
                "dropoffLng": 126.98,
                "dropoffAddressGeocoded": "서울 동작구 기존로 1",
                "dropoffName": "수령인",
                "dropoffPhone": "010-1000-0002",
                "productName": "서류",
            },
            "collecting",
        )

        response = self.client.post(
            "/api/agent/chat",
            headers={"X-Session-Id": session_id},
            json={
                "message": "도착지: 경기 화성시 없는로 999로 바꾸고 견적 확인해줘"
            },
        ).json()["data"]

        self.assertEqual(self.fake.price_calls, 0)
        self.assertNotIn("dropoffLat", response["slots"])
        self.assertNotIn("dropoffLng", response["slots"])
        self.assertEqual(
            response["slots"]["dropoffAddressLookupFailed"],
            "경기 화성시 없는로 999로 바꾸고 견적 확인해줘",
        )
        self.assertIn("정확히 찾지 못했어요", response["reply"])


if __name__ == "__main__":
    unittest.main()
