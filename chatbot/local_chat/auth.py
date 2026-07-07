# 📝 회원가입 / 로그인 / 세션 토큰을 담당하는 파일
# 📝 비밀번호는 절대 원문으로 저장하지 않고 PBKDF2 해시로 저장함
from __future__ import annotations

import hashlib
import secrets

from .db import get_conn

# 📝 해시 반복 횟수: 높을수록 무차별 대입 공격에 강해짐
_ITERATIONS = 200_000


def _hash_password(password: str, salt: str) -> str:
    # 📝 pbkdf2_hmac: 비밀번호 + salt를 여러 번 반복 해싱하는 표준 방식
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), _ITERATIONS
    )
    return digest.hex()


def create_user(username: str, password: str) -> int:
    """회원가입. 성공하면 user id, 중복이면 ValueError."""
    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)
    with get_conn() as conn:
        try:
            cursor = conn.execute(
                "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
                (username, password_hash, salt),
            )
        except Exception as exc:
            if "UNIQUE" in str(exc):
                raise ValueError("이미 존재하는 아이디입니다.") from exc
            raise
        return int(cursor.lastrowid)


def verify_user(username: str, password: str) -> int | None:
    """로그인 검증. 성공하면 user id, 실패하면 None."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, password_hash, salt FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if row is None:
        return None
    # 📝 compare_digest: 문자열 비교 시간 차이로 해시를 유추하는 공격을 막음
    if secrets.compare_digest(row["password_hash"], _hash_password(password, row["salt"])):
        return int(row["id"])
    return None


def create_session(user_id: int) -> str:
    """로그인 성공 시 발급하는 랜덤 토큰."""
    token = secrets.token_hex(32)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id) VALUES (?, ?)", (token, user_id)
        )
    return token


def get_user_by_token(token: str) -> dict | None:
    """토큰으로 로그인한 사용자를 찾음. 없으면 None."""
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT users.id, users.username
            FROM sessions JOIN users ON users.id = sessions.user_id
            WHERE sessions.token = ?
            """,
            (token,),
        ).fetchone()
    return dict(row) if row else None


def delete_session(token: str) -> None:
    """로그아웃: 토큰을 삭제함."""
    with get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
