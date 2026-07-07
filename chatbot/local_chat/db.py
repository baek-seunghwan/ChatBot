# 📝 SQLite 데이터베이스 연결과 테이블 관리를 담당하는 파일
# 📝 SQLite는 별도 서버 없이 파일 하나로 동작하는 DB라 로컬 앱에 적합함
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

# 📝 DB 파일 위치: ChatBot/local_chat.db (환경변수로 변경 가능)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = Path(os.getenv("LOCAL_CHAT_DB", BASE_DIR / "local_chat.db"))

# 📝 테이블 정의
# 📝 users: 회원 정보 / sessions: 로그인 토큰 / messages: 질문과 단일 챗봇 답변
_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    responder TEXT,
    answer TEXT,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
"""

# 📝 이미 만들어진 DB에 나중에 추가된 컬럼을 넣어주는 마이그레이션 목록
_MIGRATIONS = (
    "responder",
    "answer",
    "error",
)


@contextmanager
def get_conn():
    """요청마다 새 연결을 열고 끝나면 자동으로 닫음."""
    conn = sqlite3.connect(DB_PATH)
    # 📝 row_factory를 지정하면 결과를 dict처럼 컬럼 이름으로 접근할 수 있음
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """앱 시작 시 테이블이 없으면 만들고, 빠진 컬럼이 있으면 추가함."""
    with get_conn() as conn:
        conn.executescript(_SCHEMA)
        # 📝 기존 DB에 새 컬럼이 없으면 ALTER TABLE로 추가 (데이터는 그대로 유지)
        existing = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(messages)").fetchall()
        }
        for column in _MIGRATIONS:
            if column not in existing:
                conn.execute(f"ALTER TABLE messages ADD COLUMN {column} TEXT")
