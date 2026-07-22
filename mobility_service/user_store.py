from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


PASSWORD_ITERATIONS = 600_000
SESSION_TTL_SECONDS = 60 * 60 * 24 * 30


class DuplicateEmailError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class UserStore:
    """SQLite-backed users and opaque login sessions."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
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
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at INTEGER NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_user_sessions_user
                ON user_sessions(user_id);
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(users)").fetchall()
            }
            if "username" not in columns:
                connection.execute("ALTER TABLE users ADD COLUMN username TEXT")
            if "role" not in columns:
                connection.execute(
                    "ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'USER'"
                )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username
                ON users(username COLLATE NOCASE)
                WHERE username IS NOT NULL
                """
            )

    @staticmethod
    def _normalize_email(email: str) -> str:
        return email.strip().lower()

    @staticmethod
    def _hash_password(password: str, salt: bytes) -> str:
        return hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            PASSWORD_ITERATIONS,
        ).hex()

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _public_user(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "email": row["email"],
            "name": row["name"],
            "username": row["username"],
            "role": row["role"],
            "createdAt": row["created_at"],
        }

    def create_user(self, *, name: str, email: str, password: str) -> dict[str, Any]:
        user_id = f"user-{uuid4().hex}"
        normalized_email = self._normalize_email(email)
        salt = secrets.token_bytes(16)
        password_hash = self._hash_password(password, salt)
        created_at = utc_now()
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO users (
                        id, email, name, username, role,
                        password_hash, password_salt, created_at
                    ) VALUES (?, ?, ?, NULL, 'USER', ?, ?, ?)
                    """,
                    (
                        user_id,
                        normalized_email,
                        name.strip(),
                        password_hash,
                        salt.hex(),
                        created_at,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateEmailError("이미 가입된 이메일입니다.") from exc
        return {
            "id": user_id,
            "email": normalized_email,
            "name": name.strip(),
            "username": None,
            "role": "USER",
            "createdAt": created_at,
        }

    def authenticate(self, *, identifier: str, password: str) -> dict[str, Any] | None:
        normalized_identifier = identifier.strip().lower()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, email, name, username, role,
                       password_hash, password_salt, created_at
                FROM users
                WHERE lower(email) = ? OR lower(username) = ?
                """,
                (normalized_identifier, normalized_identifier),
            ).fetchone()
        if row is None:
            return None
        actual_hash = self._hash_password(password, bytes.fromhex(row["password_salt"]))
        if not hmac.compare_digest(actual_hash, row["password_hash"]):
            return None
        return self._public_user(row)

    def ensure_admin(self, *, username: str, password: str) -> dict[str, Any]:
        normalized_username = username.strip().lower()
        if not 3 <= len(normalized_username) <= 40 or not all(
            char.isalnum() or char in "._-" for char in normalized_username
        ):
            raise ValueError("관리자 아이디는 영문, 숫자, 마침표, 밑줄, 하이픈만 사용할 수 있습니다.")
        if len(password) < 8:
            raise ValueError("관리자 비밀번호는 8자 이상이어야 합니다.")

        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT id, email, name, username, role,
                       password_hash, password_salt, created_at
                FROM users WHERE lower(username) = ?
                """,
                (normalized_username,),
            ).fetchone()

            if existing is not None:
                supplied_hash = self._hash_password(
                    password, bytes.fromhex(existing["password_salt"])
                )
                password_changed = not hmac.compare_digest(
                    supplied_hash, existing["password_hash"]
                )
                if password_changed:
                    salt = secrets.token_bytes(16)
                    connection.execute(
                        """
                        UPDATE users
                        SET password_hash = ?, password_salt = ?, role = 'ADMIN'
                        WHERE id = ?
                        """,
                        (self._hash_password(password, salt), salt.hex(), existing["id"]),
                    )
                    connection.execute(
                        "DELETE FROM user_sessions WHERE user_id = ?",
                        (existing["id"],),
                    )
                elif existing["role"] != "ADMIN":
                    connection.execute(
                        "UPDATE users SET role = 'ADMIN' WHERE id = ?",
                        (existing["id"],),
                    )
                row = connection.execute(
                    """
                    SELECT id, email, name, username, role, created_at
                    FROM users WHERE id = ?
                    """,
                    (existing["id"],),
                ).fetchone()
                return self._public_user(row)

            admin_id = f"user-{uuid4().hex}"
            salt = secrets.token_bytes(16)
            created_at = utc_now()
            connection.execute(
                """
                INSERT INTO users (
                    id, email, name, username, role,
                    password_hash, password_salt, created_at
                ) VALUES (?, ?, '관리자', ?, 'ADMIN', ?, ?, ?)
                """,
                (
                    admin_id,
                    f"{normalized_username}@admin.movb.local",
                    normalized_username,
                    self._hash_password(password, salt),
                    salt.hex(),
                    created_at,
                ),
            )
        return {
            "id": admin_id,
            "email": f"{normalized_username}@admin.movb.local",
            "name": "관리자",
            "username": normalized_username,
            "role": "ADMIN",
            "createdAt": created_at,
        }

    def list_users(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, email, name, username, role, created_at
                FROM users
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._public_user(row) for row in rows]

    def user_counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT role, COUNT(*) AS count FROM users GROUP BY role"
            ).fetchall()
        counts = {row["role"]: row["count"] for row in rows}
        return {
            "total": sum(counts.values()),
            "admins": counts.get("ADMIN", 0),
            "users": counts.get("USER", 0),
        }

    def create_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO user_sessions (
                    token_hash, user_id, created_at, expires_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    self._token_hash(token),
                    user_id,
                    utc_now(),
                    int(time.time()) + SESSION_TTL_SECONDS,
                ),
            )
        return token

    def get_user_by_session(self, token: str) -> dict[str, Any] | None:
        now = int(time.time())
        with self._connect() as connection:
            connection.execute("DELETE FROM user_sessions WHERE expires_at <= ?", (now,))
            row = connection.execute(
                """
                SELECT users.id, users.email, users.name, users.username,
                       users.role, users.created_at
                FROM user_sessions
                JOIN users ON users.id = user_sessions.user_id
                WHERE user_sessions.token_hash = ?
                  AND user_sessions.expires_at > ?
                """,
                (self._token_hash(token), now),
            ).fetchone()
        return self._public_user(row) if row else None

    def revoke_session(self, token: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM user_sessions WHERE token_hash = ?",
                (self._token_hash(token),),
            )
