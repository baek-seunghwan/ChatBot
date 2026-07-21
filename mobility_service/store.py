from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVENT_TO_STATUS = {
    "matched": "MATCHED",
    "canceled": "CANCELED",
    "pickupCompleted": "PICKUP_COMPLETED",
    "dropoffCompleted": "DROPOFF_COMPLETED",
    "completed": "COMPLETED",
    "matchingFailed": "MATCHING_FAILED",
    "aborted": "ABORTED",
    "rematching": "MATCHING",
}

STATUS_RANK = {
    "PENDING": 0,
    "MATCHING": 10,
    "MATCHED": 20,
    "PICKUP_WAITING": 25,
    "PICKUP_STARTED": 30,
    "PICKUP_COMPLETED": 40,
    "DROPOFF_WAITING": 45,
    "DROPOFF_STARTED": 50,
    "DROPOFF_COMPLETED": 55,
    "COMPLETED": 60,
}

TERMINAL_STATUSES = {"COMPLETED", "CANCELED", "MATCHING_FAILED", "ABORTED"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MobilityStore:
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
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS delivery_orders (
                    partner_order_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS callback_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_key TEXT NOT NULL UNIQUE,
                    partner_order_id TEXT NOT NULL,
                    event TEXT NOT NULL,
                    body_json TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    applied INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_callbacks_order
                ON callback_events(partner_order_id, received_at);
                """
            )

    def reserve_order(
        self, partner_order_id: str, request_payload: dict[str, Any]
    ) -> bool:
        now = utc_now()
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO delivery_orders (
                        partner_order_id, status, request_json,
                        created_at, updated_at
                    ) VALUES (?, 'PENDING', ?, ?, ?)
                    """,
                    (
                        partner_order_id,
                        json.dumps(request_payload, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def complete_order(self, partner_order_id: str, response: Any) -> None:
        status = self._status_from_response(response) or "MATCHING"
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE delivery_orders
                SET status = ?, response_json = ?, error = NULL, updated_at = ?
                WHERE partner_order_id = ?
                """,
                (
                    status,
                    json.dumps(response, ensure_ascii=False),
                    utc_now(),
                    partner_order_id,
                ),
            )

    def fail_order(self, partner_order_id: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE delivery_orders
                SET status = 'REQUEST_FAILED', error = ?, updated_at = ?
                WHERE partner_order_id = ?
                """,
                (error[:1000], utc_now(), partner_order_id),
            )

    def sync_order(self, partner_order_id: str, response: Any) -> None:
        status = self._status_from_response(response)
        with self._connect() as connection:
            if status:
                connection.execute(
                    """
                    UPDATE delivery_orders
                    SET status = ?, response_json = ?, updated_at = ?
                    WHERE partner_order_id = ?
                    """,
                    (
                        status,
                        json.dumps(response, ensure_ascii=False),
                        utc_now(),
                        partner_order_id,
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE delivery_orders
                    SET response_json = ?, updated_at = ?
                    WHERE partner_order_id = ?
                    """,
                    (
                        json.dumps(response, ensure_ascii=False),
                        utc_now(),
                        partner_order_id,
                    ),
                )

    def set_status(self, partner_order_id: str, status: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE delivery_orders
                SET status = ?, updated_at = ?
                WHERE partner_order_id = ?
                """,
                (status, utc_now(), partner_order_id),
            )

    @staticmethod
    def _status_from_response(response: Any) -> str | None:
        if not isinstance(response, dict):
            return None
        receipt = response.get("receipt")
        if isinstance(receipt, dict) and isinstance(receipt.get("status"), str):
            return receipt["status"]
        if isinstance(response.get("status"), str):
            return response["status"]
        return None

    def get_order(self, partner_order_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT partner_order_id, status, request_json, response_json,
                       error, created_at, updated_at
                FROM delivery_orders
                WHERE partner_order_id = ?
                """,
                (partner_order_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "partnerOrderId": row["partner_order_id"],
            "status": row["status"],
            "request": json.loads(row["request_json"]),
            "response": (
                json.loads(row["response_json"]) if row["response_json"] else None
            ),
            "error": row["error"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
            "callbacks": self.list_callbacks(partner_order_id),
        }

    def list_orders(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT partner_order_id, status, error, created_at, updated_at
                FROM delivery_orders
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "partnerOrderId": row["partner_order_id"],
                "status": row["status"],
                "error": row["error"],
                "createdAt": row["created_at"],
                "updatedAt": row["updated_at"],
            }
            for row in rows
        ]

    def record_callback(
        self, partner_order_id: str, event: str, body: dict[str, Any]
    ) -> dict[str, bool]:
        body_json = json.dumps(body, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(
            f"{partner_order_id}|{event}|{body_json}".encode("utf-8")
        ).hexdigest()

        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO callback_events (
                        event_key, partner_order_id, event, body_json, received_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (digest, partner_order_id, event, body_json, utc_now()),
                )
        except sqlite3.IntegrityError:
            return {"duplicate": True, "applied": False}

        applied = self._apply_event(partner_order_id, event)
        with self._connect() as connection:
            connection.execute(
                "UPDATE callback_events SET applied = ? WHERE event_key = ?",
                (1 if applied else 0, digest),
            )
        return {"duplicate": False, "applied": applied}

    def _apply_event(self, partner_order_id: str, event: str) -> bool:
        next_status = EVENT_TO_STATUS.get(event)
        if next_status is None:
            return False

        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM delivery_orders WHERE partner_order_id = ?",
                (partner_order_id,),
            ).fetchone()
            if row is None:
                return False

            current_status = row["status"]
            if current_status in TERMINAL_STATUSES:
                return False
            if next_status not in TERMINAL_STATUSES and (
                STATUS_RANK.get(next_status, -1) < STATUS_RANK.get(current_status, -1)
            ):
                return False

            connection.execute(
                """
                UPDATE delivery_orders
                SET status = ?, updated_at = ?
                WHERE partner_order_id = ?
                """,
                (next_status, utc_now(), partner_order_id),
            )
        return True

    def list_callbacks(self, partner_order_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event, body_json, received_at, applied
                FROM callback_events
                WHERE partner_order_id = ?
                ORDER BY id ASC
                """,
                (partner_order_id,),
            ).fetchall()
        return [
            {
                "event": row["event"],
                "body": json.loads(row["body_json"]),
                "receivedAt": row["received_at"],
                "applied": bool(row["applied"]),
            }
            for row in rows
        ]
