from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .store import utc_now


class PoolStore:
    """퀵 합승(패키지 카풀) 대기 요청을 저장한다."""

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
                CREATE TABLE IF NOT EXISTS pool_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    pickup_json TEXT NOT NULL,
                    dropoff_json TEXT NOT NULL,
                    product_json TEXT NOT NULL DEFAULT '{}',
                    solo_price INTEGER,
                    share_price INTEGER,
                    pool_id TEXT,
                    auto_consent INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "sessionId": row["session_id"],
            "status": row["status"],
            "pickup": json.loads(row["pickup_json"]),
            "dropoff": json.loads(row["dropoff_json"]),
            "product": json.loads(row["product_json"]),
            "soloPrice": row["solo_price"],
            "sharePrice": row["share_price"],
            "poolId": row["pool_id"],
            "autoConsent": bool(row["auto_consent"]),
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }

    def create_request(
        self,
        session_id: str,
        pickup: dict[str, Any],
        dropoff: dict[str, Any],
        product: dict[str, Any],
        solo_price: int | None,
        auto_consent: bool = False,
    ) -> dict[str, Any]:
        now = utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO pool_requests (
                    session_id, pickup_json, dropoff_json, product_json,
                    solo_price, auto_consent, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    json.dumps(pickup, ensure_ascii=False),
                    json.dumps(dropoff, ensure_ascii=False),
                    json.dumps(product, ensure_ascii=False),
                    solo_price,
                    1 if auto_consent else 0,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM pool_requests WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        return self._row_to_dict(row)

    def get_request(self, request_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM pool_requests WHERE id = ?", (request_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_open(self, exclude_session: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM pool_requests WHERE status = 'open'"
        params: tuple[Any, ...] = ()
        if exclude_session:
            query += " AND session_id != ?"
            params = (exclude_session,)
        query += " ORDER BY created_at ASC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_by_session(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM pool_requests
                WHERE session_id = ? AND status != 'canceled'
                ORDER BY created_at DESC
                """,
                (session_id,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def set_auto_consent(self, request_id: int, consent: bool) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE pool_requests SET auto_consent = ?, updated_at = ? WHERE id = ?",
                (1 if consent else 0, utc_now(), request_id),
            )

    def mark_ordered(
        self, request_ids: list[int], pool_id: str, shares: dict[int, int]
    ) -> None:
        now = utc_now()
        with self._connect() as connection:
            for request_id in request_ids:
                connection.execute(
                    """
                    UPDATE pool_requests
                    SET status = 'ordered', pool_id = ?, share_price = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (pool_id, shares.get(request_id), now, request_id),
                )

    def cancel_request(self, request_id: int, session_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE pool_requests
                SET status = 'canceled', updated_at = ?
                WHERE id = ? AND session_id = ? AND status = 'open'
                """,
                (utc_now(), request_id, session_id),
            )
        return cursor.rowcount > 0

    def cancel_open_by_session(self, session_id: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE pool_requests
                SET status = 'canceled', updated_at = ?
                WHERE session_id = ? AND status = 'open'
                """,
                (utc_now(), session_id),
            )
        return cursor.rowcount
