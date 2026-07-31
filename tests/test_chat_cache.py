from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from mobility_service.app import create_app
from mobility_service.chat_cache import GREETING_REPLY, cached_chat_response
from mobility_service.store import MobilityStore
from tests.test_mobility_service import FakeKakaoClient, settings


class ChatCacheUnitTests(unittest.TestCase):
    def test_exact_greeting_uses_fixed_response(self) -> None:
        cached = cached_chat_response("안녕하세요")

        self.assertIsNotNone(cached)
        self.assertEqual(cached.match_type, "exact")
        self.assertEqual(cached.reply, GREETING_REPLY)

    def test_greeting_variants_use_semantic_response(self) -> None:
        for message in ("안녕", "안뇽?", "안녕!", "하이", "ㅎㅇ", "반가워요"):
            with self.subTest(message=message):
                cached = cached_chat_response(message)
                self.assertIsNotNone(cached)
                self.assertEqual(cached.match_type, "semantic")
                self.assertEqual(cached.reply, GREETING_REPLY)

    def test_greeting_prefix_does_not_hide_delivery_request(self) -> None:
        self.assertIsNone(cached_chat_response("안녕하세요 퀵 배송 접수해줘"))


class ChatCacheApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.app = create_app(
            settings=settings(root),
            client=FakeKakaoClient(),  # type: ignore[arg-type]
            store=MobilityStore(root / "mobility.db"),
        )
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.client.close()
        self.temporary.cleanup()

    def test_exact_and_semantic_cache_skip_ai_and_local_models(self) -> None:
        with patch.object(
            self.app.state.agent,
            "achat",
            new_callable=AsyncMock,
            side_effect=AssertionError("AI agent must not be called"),
        ) as agent_chat, patch(
            "mobility_service.app.local_model_reply",
            side_effect=AssertionError("local model must not be called"),
        ) as local_chat:
            exact = self.client.post(
                "/api/agent/chat",
                json={"message": "안녕하세요", "mode": "ai"},
            )
            semantic = self.client.post(
                "/api/agent/chat",
                json={"message": "안뇽?", "mode": "ai"},
            )
            local = self.client.post(
                "/api/agent/chat",
                json={
                    "message": "안녕!",
                    "mode": "local",
                    "localEngine": "vllm",
                },
            )

        self.assertEqual(exact.status_code, 200)
        self.assertEqual(semantic.status_code, 200)
        self.assertEqual(local.status_code, 200)
        self.assertEqual(
            exact.json()["data"]["trace"],
            ["response_cache:exact:greeting"],
        )
        self.assertEqual(
            semantic.json()["data"]["trace"],
            ["response_cache:semantic:greeting"],
        )
        self.assertEqual(
            local.json()["data"]["trace"],
            ["response_cache:semantic:greeting"],
        )
        agent_chat.assert_not_awaited()
        local_chat.assert_not_called()

    def test_cached_greeting_keeps_existing_delivery_session(self) -> None:
        session_id = "cached-greeting-session"
        conversations = self.app.state.conversations
        conversations.get_or_create(session_id)
        conversations.save_slots(
            session_id,
            {"pickupAddress": "서울역"},
            "confirming",
        )

        response = self.client.post(
            "/api/agent/chat",
            headers={"X-Session-Id": session_id},
            json={"message": "안녕하세요"},
        )
        data = response.json()["data"]

        self.assertEqual(data["stage"], "confirming")
        self.assertEqual(data["slots"]["pickupAddress"], "서울역")
