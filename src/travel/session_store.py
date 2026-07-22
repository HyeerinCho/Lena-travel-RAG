"""Persistent travel chat sessions with message history."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import TRAVEL_DIR

SESSION_DB_PATH = TRAVEL_DIR / "sessions.db"
HISTORY_TURN_LIMIT = 8  # recent user+assistant pairs to feed the LLM


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or SESSION_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_session_db(db_path: Path | None = None) -> None:
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                destination TEXT,
                days INTEGER,
                budget INTEGER,
                language TEXT,
                preferences TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                payload TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, created_at);
            """
        )
        conn.commit()


def _row_to_session(row: sqlite3.Row) -> dict[str, Any]:
    prefs_raw = row["preferences"]
    preferences: list[str] = []
    if prefs_raw:
        try:
            preferences = json.loads(prefs_raw)
        except json.JSONDecodeError:
            preferences = []
    return {
        "id": row["id"],
        "title": row["title"],
        "destination": row["destination"],
        "days": row["days"],
        "budget": row["budget"],
        "language": row["language"] or "ko",
        "preferences": preferences,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _row_to_message(row: sqlite3.Row) -> dict[str, Any]:
    payload = None
    if row["payload"]:
        try:
            payload = json.loads(row["payload"])
        except json.JSONDecodeError:
            payload = None
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "role": row["role"],
        "content": row["content"],
        "payload": payload,
        "created_at": row["created_at"],
    }


class TravelSessionStore:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or SESSION_DB_PATH
        init_session_db(self.db_path)

    def create_session(self, title: str | None = None) -> dict[str, Any]:
        session_id = str(uuid.uuid4())
        now = _utcnow()
        title = (title or "").strip() or "새 여행"
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    id, title, destination, days, budget, language, preferences,
                    created_at, updated_at
                ) VALUES (?, ?, NULL, NULL, NULL, 'ko', '[]', ?, ?)
                """,
                (session_id, title, now, now),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return _row_to_session(row)

    def list_sessions(self) -> list[dict[str, Any]]:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT s.*,
                       (
                         SELECT COUNT(*) FROM messages m
                         WHERE m.session_id = s.id
                       ) AS message_count
                FROM sessions s
                ORDER BY s.updated_at DESC
                """
            ).fetchall()
        out = []
        for row in rows:
            item = _row_to_session(row)
            item["message_count"] = row["message_count"]
            out.append(item)
        return out

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return _row_to_session(row) if row else None

    def get_session_detail(self, session_id: str) -> dict[str, Any] | None:
        session = self.get_session(session_id)
        if not session:
            return None
        session["messages"] = self.list_messages(session_id)
        return session

    def delete_session(self, session_id: str) -> bool:
        with _connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE id = ?", (session_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def update_session_meta(
        self,
        session_id: str,
        *,
        title: str | None = None,
        destination: str | None = None,
        days: int | None = None,
        budget: int | None = None,
        language: str | None = None,
        preferences: list[str] | None = None,
    ) -> dict[str, Any] | None:
        session = self.get_session(session_id)
        if not session:
            return None

        new_title = title if title is not None else session["title"]
        new_destination = (
            destination if destination is not None else session["destination"]
        )
        new_days = days if days is not None else session["days"]
        new_budget = budget if budget is not None else session["budget"]
        new_language = language if language is not None else session["language"]
        new_preferences = (
            preferences if preferences is not None else session["preferences"]
        )
        now = _utcnow()

        # Auto-title from first destination if still default
        if (
            title is None
            and session["title"] in {"새 여행", "New trip"}
            and new_destination
        ):
            new_title = f"{new_destination} 여행"

        with _connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE sessions
                SET title = ?, destination = ?, days = ?, budget = ?,
                    language = ?, preferences = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    new_title,
                    new_destination,
                    new_days,
                    new_budget,
                    new_language,
                    json.dumps(new_preferences, ensure_ascii=False),
                    now,
                    session_id,
                ),
            )
            conn.commit()
        return self.get_session(session_id)

    def list_messages(self, session_id: str) -> list[dict[str, Any]]:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ?
                ORDER BY created_at ASC, rowid ASC
                """,
                (session_id,),
            ).fetchall()
        return [_row_to_message(row) for row in rows]

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        message_id = str(uuid.uuid4())
        now = _utcnow()
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO messages (id, session_id, role, content, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    session_id,
                    role,
                    content,
                    json.dumps(payload, ensure_ascii=False) if payload else None,
                    now,
                ),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM messages WHERE id = ?", (message_id,)
            ).fetchone()
        return _row_to_message(row)

    def recent_history_text(
        self,
        session_id: str,
        *,
        turn_limit: int = HISTORY_TURN_LIMIT,
    ) -> str:
        messages = self.list_messages(session_id)
        if not messages:
            return ""
        # Keep the last N*2 messages (approx N turns)
        trimmed = messages[-(turn_limit * 2) :]
        lines: list[str] = []
        for msg in trimmed:
            label = "사용자" if msg["role"] == "user" else "어시스턴트"
            lines.append(f"{label}: {msg['content']}")
        return "\n".join(lines)
