from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .store import utc_now


class ConversationStore:
    """배송 도우미 챗봇의 세션(대화) 상태를 저장한다.

    MobilityStore와 같은 sqlite 파일을 쓰지만, '주문 생명주기'와
    '대화 상태'는 의미가 다르므로 delivery_orders 테이블을 오버로드하지 않고
    별도 테이블/클래스로 둔다.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    session_id TEXT PRIMARY KEY,
                    stage TEXT NOT NULL DEFAULT 'collecting',
                    slots_json TEXT NOT NULL DEFAULT '{}',
                    turns_json TEXT NOT NULL DEFAULT '[]',
                    quote_json TEXT,
                    quote_hash TEXT,
                    partner_order_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def get_or_create(self, session_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                now = utc_now()
                connection.execute(
                    """
                    INSERT INTO agent_sessions (session_id, created_at, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (session_id, now, now),
                )
                return {
                    "session_id": session_id,
                    "stage": "collecting",
                    "slots": {},
                    "turns": [],
                    "quote": None,
                    "quote_hash": None,
                    "partner_order_id": None,
                }
            return self._row_to_dict(row)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "session_id": row["session_id"],
            "stage": row["stage"],
            "slots": json.loads(row["slots_json"]),
            "turns": json.loads(row["turns_json"]),
            "quote": json.loads(row["quote_json"]) if row["quote_json"] else None,
            "quote_hash": row["quote_hash"],
            "partner_order_id": row["partner_order_id"],
        }

    def save_slots(self, session_id: str, slots: dict[str, Any], stage: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE agent_sessions
                SET slots_json = ?, stage = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (json.dumps(slots, ensure_ascii=False), stage, utc_now(), session_id),
            )

    def save_quote(self, session_id: str, quote: Any, quote_hash: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE agent_sessions
                SET quote_json = ?, quote_hash = ?, stage = 'confirming', updated_at = ?
                WHERE session_id = ?
                """,
                (json.dumps(quote, ensure_ascii=False), quote_hash, utc_now(), session_id),
            )

    def set_partner_order_id(self, session_id: str, partner_order_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE agent_sessions
                SET partner_order_id = ?, stage = 'placed', updated_at = ?
                WHERE session_id = ?
                """,
                (partner_order_id, utc_now(), session_id),
            )

    def reset_draft(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE agent_sessions
                SET stage = 'collecting', slots_json = '{}', quote_json = NULL,
                    quote_hash = NULL, updated_at = ?
                WHERE session_id = ?
                """,
                (utc_now(), session_id),
            )

    def append_turn(
        self, session_id: str, role: str, content: str, cap: int = 12
    ) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT turns_json FROM agent_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            turns = json.loads(row["turns_json"]) if row else []
            turns.append({"role": role, "content": content, "at": utc_now()})
            turns = turns[-cap:]
            connection.execute(
                """
                UPDATE agent_sessions
                SET turns_json = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (json.dumps(turns, ensure_ascii=False), utc_now(), session_id),
            )
