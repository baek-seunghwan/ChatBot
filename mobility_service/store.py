from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
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

                CREATE TABLE IF NOT EXISTS delivery_match_requests (
                    request_id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    match_id TEXT,
                    provider_order_id TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_delivery_match_status
                ON delivery_match_requests(status, expires_at, created_at);

                CREATE INDEX IF NOT EXISTS idx_delivery_match_client
                ON delivery_match_requests(client_id, created_at);

                CREATE TABLE IF NOT EXISTS address_share_requests (
                    token TEXT PRIMARY KEY,
                    sender_session_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'WAITING',
                    recipient_name TEXT,
                    recipient_phone TEXT,
                    address TEXT,
                    detail_address TEXT,
                    note TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_address_share_sender
                ON address_share_requests(sender_session_id, created_at);
                """
            )

    def create_address_share(
        self,
        *,
        token: str,
        sender_session_id: str,
        recipient_name: str | None = None,
        recipient_phone: str | None = None,
        ttl_hours: int = 24,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=ttl_hours)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO address_share_requests (
                    token, sender_session_id, recipient_name, recipient_phone,
                    created_at, updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    token,
                    sender_session_id,
                    recipient_name,
                    recipient_phone,
                    now.isoformat(),
                    now.isoformat(),
                    expires_at.isoformat(),
                ),
            )
        result = self.get_address_share(token)
        if result is None:
            raise RuntimeError("주소 요청 링크를 저장하지 못했습니다.")
        return result

    def get_address_share(self, token: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT token, sender_session_id, status, recipient_name,
                       recipient_phone, address, detail_address, note,
                       created_at, updated_at, expires_at
                FROM address_share_requests
                WHERE token = ?
                """,
                (token,),
            ).fetchone()
        if row is None:
            return None

        expired = datetime.fromisoformat(row["expires_at"]) <= datetime.now(timezone.utc)
        status = "EXPIRED" if expired and row["status"] == "WAITING" else row["status"]
        return {
            "token": row["token"],
            "senderSessionId": row["sender_session_id"],
            "status": status,
            "recipientName": row["recipient_name"],
            "recipientPhone": row["recipient_phone"],
            "address": row["address"],
            "detailAddress": row["detail_address"],
            "note": row["note"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
            "expiresAt": row["expires_at"],
        }

    def list_address_shares(
        self,
        sender_session_id: str,
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT token
                FROM address_share_requests
                WHERE sender_session_id = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (sender_session_id, limit),
            ).fetchall()
        items = [self.get_address_share(row["token"]) for row in rows]
        return [item for item in items if item is not None]

    def complete_address_share(
        self,
        token: str,
        *,
        address: str,
        detail_address: str | None,
        name: str,
        phone: str,
        note: str | None,
    ) -> dict[str, Any] | None:
        existing = self.get_address_share(token)
        if existing is None or existing["status"] == "EXPIRED":
            return None
        if existing["status"] == "COMPLETED":
            return existing
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE address_share_requests
                SET status = 'COMPLETED', address = ?, detail_address = ?,
                    recipient_name = ?, recipient_phone = ?, note = ?,
                    updated_at = ?
                WHERE token = ?
                """,
                (
                    address,
                    detail_address,
                    name,
                    phone,
                    note,
                    utc_now(),
                    token,
                ),
            )
        return self.get_address_share(token)

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
                SELECT partner_order_id, status, request_json, error,
                       created_at, updated_at
                FROM delivery_orders
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        orders: list[dict[str, Any]] = []
        for row in rows:
            request = json.loads(row["request_json"])
            waypoints = request.get("waypoints") or []
            is_smart = (
                row["partner_order_id"].startswith(("smart-", "bundle-"))
                or any(
                    str(item.get("note") or "").startswith("[추가 픽업]")
                    for item in waypoints
                    if isinstance(item, dict)
                )
            )
            pickup = request.get("pickup", {}).get("location", {}).get(
                "basicAddress", ""
            )
            dropoff = request.get("dropoff", {}).get("location", {}).get(
                "basicAddress", ""
            )
            route_summary = (
                f"{pickup} → 경유 {len(waypoints)}곳 → {dropoff}"
                if waypoints
                else f"{pickup} → {dropoff}"
            )
            orders.append(
                {
                    "partnerOrderId": row["partner_order_id"],
                    "status": row["status"],
                    "serviceType": (
                        "SMART_DELIVERY" if is_smart else request.get("orderType")
                    ),
                    "routeSummary": route_summary,
                    "error": row["error"],
                    "createdAt": row["created_at"],
                    "updatedAt": row["updated_at"],
                }
            )
        return orders

    def order_counts(self) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM delivery_orders
                GROUP BY status
                """
            ).fetchall()
        by_status = {row["status"]: row["count"] for row in rows}
        return {"total": sum(by_status.values()), "byStatus": by_status}

    @staticmethod
    def _match_request_from_row(
        row: sqlite3.Row,
        *,
        include_payload: bool = True,
    ) -> dict[str, Any]:
        request = json.loads(row["request_json"])
        pickup = request.get("pickup", {}).get("location", {}).get(
            "basicAddress", ""
        )
        dropoff = request.get("dropoff", {}).get("location", {}).get(
            "basicAddress", ""
        )
        result = {
            "requestId": row["request_id"],
            "clientId": row["client_id"],
            "status": row["status"],
            "matchId": row["match_id"],
            "partnerOrderId": row["provider_order_id"],
            "routeSummary": f"{pickup} → {dropoff}",
            "error": row["error"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
            "expiresAt": row["expires_at"],
        }
        if include_payload:
            result["request"] = request
        return result

    def reserve_match_request(
        self,
        request_id: str,
        client_id: str,
        request_payload: dict[str, Any],
        expires_at: str,
    ) -> tuple[dict[str, Any], bool]:
        now = utc_now()
        created = False
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO delivery_match_requests (
                        request_id, client_id, request_json, status,
                        created_at, updated_at, expires_at
                    ) VALUES (?, ?, ?, 'WAITING', ?, ?, ?)
                    """,
                    (
                        request_id,
                        client_id,
                        json.dumps(request_payload, ensure_ascii=False),
                        now,
                        now,
                        expires_at,
                    ),
                )
            created = True
        except sqlite3.IntegrityError:
            pass
        existing = self.get_match_request(request_id)
        if existing is None:
            raise RuntimeError("공동배송 매칭 요청을 저장하지 못했습니다.")
        return existing, created

    def get_match_request(
        self,
        request_id: str,
        *,
        client_id: str | None = None,
    ) -> dict[str, Any] | None:
        query = """
            SELECT request_id, client_id, request_json, status, match_id,
                   provider_order_id, error, created_at, updated_at, expires_at
            FROM delivery_match_requests
            WHERE request_id = ?
        """
        params: list[Any] = [request_id]
        if client_id is not None:
            query += " AND client_id = ?"
            params.append(client_id)
        with self._connect() as connection:
            row = connection.execute(query, params).fetchone()
        if row is None:
            return None
        return self._match_request_from_row(row)

    def list_pending_match_requests(
        self,
        *,
        exclude_request_id: str,
        exclude_client_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE delivery_match_requests
                SET status = 'EXPIRED', updated_at = ?
                WHERE status = 'WAITING' AND expires_at <= ?
                """,
                (now, now),
            )
            rows = connection.execute(
                """
                SELECT request_id, client_id, request_json, status, match_id,
                       provider_order_id, error, created_at, updated_at, expires_at
                FROM delivery_match_requests
                WHERE status = 'WAITING'
                  AND expires_at > ?
                  AND request_id != ?
                  AND client_id != ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (now, exclude_request_id, exclude_client_id, limit),
            ).fetchall()
        return [self._match_request_from_row(row) for row in rows]

    def claim_match_requests(
        self,
        first_request_id: str,
        second_request_id: str,
        match_id: str,
    ) -> bool:
        now = utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE delivery_match_requests
                SET status = 'MATCHING', match_id = ?, updated_at = ?
                WHERE request_id IN (?, ?)
                  AND status = 'WAITING'
                  AND expires_at > ?
                """,
                (
                    match_id,
                    now,
                    first_request_id,
                    second_request_id,
                    now,
                ),
            )
            if cursor.rowcount != 2:
                connection.rollback()
                return False
        return True

    def complete_match_requests(
        self,
        request_ids: tuple[str, str],
        provider_order_id: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE delivery_match_requests
                SET status = 'MATCHED', provider_order_id = ?,
                    error = NULL, updated_at = ?
                WHERE request_id IN (?, ?)
                """,
                (provider_order_id, utc_now(), *request_ids),
            )

    def fail_match_requests(
        self,
        request_ids: tuple[str, str],
        error: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE delivery_match_requests
                SET status = 'FAILED', error = ?, updated_at = ?
                WHERE request_id IN (?, ?)
                """,
                (error[:1000], utc_now(), *request_ids),
            )

    def list_match_requests(
        self,
        *,
        client_id: str,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE delivery_match_requests
                SET status = 'EXPIRED', updated_at = ?
                WHERE status = 'WAITING' AND expires_at <= ?
                """,
                (now, now),
            )
            rows = connection.execute(
                """
                SELECT request_id, client_id, request_json, status, match_id,
                       provider_order_id, error, created_at, updated_at, expires_at
                FROM delivery_match_requests
                WHERE client_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (client_id, limit),
            ).fetchall()
        return [
            self._match_request_from_row(row, include_payload=False)
            for row in rows
        ]

    def cancel_match_request(
        self,
        request_id: str,
        *,
        client_id: str,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE delivery_match_requests
                SET status = 'CANCELED', updated_at = ?
                WHERE request_id = ? AND client_id = ? AND status = 'WAITING'
                """,
                (utc_now(), request_id, client_id),
            )
        if cursor.rowcount != 1:
            return None
        return self.get_match_request(request_id, client_id=client_id)

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
