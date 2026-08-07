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


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row["name"] for row in rows}


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
        cols = _table_columns(conn, "sessions")
        migrations = [
            ("owner_id", "TEXT"),
            ("start_date", "TEXT"),
            ("excluded_places", "TEXT"),
            ("shown_poi_ids", "TEXT"),
            ("shown_place_names", "TEXT"),
            ("last_target_day", "INTEGER"),
        ]
        for name, col_type in migrations:
            if name not in cols:
                conn.execute(f"ALTER TABLE sessions ADD COLUMN {name} {col_type}")
        conn.commit()


def _json_list(raw: Any) -> list[Any]:
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _row_to_session(row: sqlite3.Row) -> dict[str, Any]:
    keys = row.keys()
    return {
        "id": row["id"],
        "title": row["title"],
        "destination": row["destination"],
        "days": row["days"],
        "budget": row["budget"],
        "language": row["language"] or "ko",
        "preferences": _json_list(row["preferences"] if "preferences" in keys else None),
        "owner_id": row["owner_id"] if "owner_id" in keys else None,
        "start_date": row["start_date"] if "start_date" in keys else None,
        "excluded_places": _json_list(
            row["excluded_places"] if "excluded_places" in keys else None
        ),
        "shown_poi_ids": _json_list(
            row["shown_poi_ids"] if "shown_poi_ids" in keys else None
        ),
        "shown_place_names": _json_list(
            row["shown_place_names"] if "shown_place_names" in keys else None
        ),
        "last_target_day": row["last_target_day"] if "last_target_day" in keys else None,
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

    def create_session(
        self,
        title: str | None = None,
        *,
        owner_id: str | None = None,
    ) -> dict[str, Any]:
        session_id = str(uuid.uuid4())
        now = _utcnow()
        title = (title or "").strip() or "새 여행"
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    id, title, destination, days, budget, language, preferences,
                    owner_id, start_date, excluded_places, shown_poi_ids, shown_place_names,
                    created_at, updated_at
                ) VALUES (?, ?, NULL, NULL, NULL, 'ko', '[]', ?, NULL, '[]', '[]', '[]', ?, ?)
                """,
                (session_id, title, owner_id, now, now),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return _row_to_session(row)

    def list_sessions(self, owner_id: str | None = None) -> list[dict[str, Any]]:
        with _connect(self.db_path) as conn:
            if owner_id:
                rows = conn.execute(
                    """
                    SELECT s.*,
                           (
                             SELECT COUNT(*) FROM messages m
                             WHERE m.session_id = s.id
                           ) AS message_count
                    FROM sessions s
                    WHERE s.owner_id = ? OR s.owner_id IS NULL
                    ORDER BY s.updated_at DESC
                    """,
                    (owner_id,),
                ).fetchall()
                # 미소유 세션은 아직 메시지가 없는(즉, 실제 대화 기록이 없는) 경우에만
                # 자동으로 이 클라이언트 소유로 넘긴다. 메시지가 있는 레거시 미소유
                # 세션까지 통째로 넘기면 먼저 목록을 조회한 클라이언트가 다른 사람의
                # 대화 기록을 그대로 가져가 버리는 세션 유출이 될 수 있다.
                filtered = []
                for row in rows:
                    item = _row_to_session(row)
                    item["message_count"] = row["message_count"]
                    if item.get("owner_id") is None and item["message_count"] == 0:
                        conn.execute(
                            "UPDATE sessions SET owner_id = ? WHERE id = ? AND owner_id IS NULL",
                            (owner_id, item["id"]),
                        )
                        item["owner_id"] = owner_id
                    if item.get("owner_id") == owner_id:
                        filtered.append(item)
                conn.commit()
                return filtered

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

    def ensure_owner(self, session_id: str, owner_id: str | None) -> dict[str, Any] | None:
        session = self.get_session(session_id)
        if not session:
            return None
        if not owner_id:
            return session
        if session.get("owner_id") is None:
            with _connect(self.db_path) as conn:
                has_messages = conn.execute(
                    "SELECT 1 FROM messages WHERE session_id = ? LIMIT 1", (session_id,)
                ).fetchone()
                if has_messages:
                    # 메시지가 있는 미소유 세션은 세션 ID만 알면(추측/유출) 누구나
                    # 자동으로 가져갈 수 있으므로 자동 클레임하지 않는다.
                    return None
                conn.execute(
                    "UPDATE sessions SET owner_id = ? WHERE id = ? AND owner_id IS NULL",
                    (owner_id, session_id),
                )
                conn.commit()
            session["owner_id"] = owner_id
            return session
        if session.get("owner_id") != owner_id:
            return None
        return session

    def get_session_detail(
        self,
        session_id: str,
        *,
        owner_id: str | None = None,
    ) -> dict[str, Any] | None:
        session = self.ensure_owner(session_id, owner_id) if owner_id else self.get_session(session_id)
        if not session:
            return None
        if owner_id and session.get("owner_id") and session["owner_id"] != owner_id:
            return None
        session["messages"] = self.list_messages(session_id)
        return session

    def delete_session(self, session_id: str, *, owner_id: str | None = None) -> bool:
        session = self.ensure_owner(session_id, owner_id) if owner_id else self.get_session(session_id)
        if not session:
            return False
        if owner_id and session.get("owner_id") and session["owner_id"] != owner_id:
            return False
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
        start_date: str | None = None,
        excluded_places: list[str] | None = None,
        shown_poi_ids: list[str] | None = None,
        shown_place_names: list[str] | None = None,
        last_target_day: int | None = None,
        owner_id: str | None = None,
    ) -> dict[str, Any] | None:
        session = self.ensure_owner(session_id, owner_id) if owner_id else self.get_session(session_id)
        if not session:
            return None
        if owner_id and session.get("owner_id") and session["owner_id"] != owner_id:
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
        new_start = start_date if start_date is not None else session.get("start_date")
        new_excluded = (
            excluded_places
            if excluded_places is not None
            else session.get("excluded_places") or []
        )
        new_shown_ids = (
            shown_poi_ids
            if shown_poi_ids is not None
            else session.get("shown_poi_ids") or []
        )
        new_shown_names = (
            shown_place_names
            if shown_place_names is not None
            else session.get("shown_place_names") or []
        )
        # 0을 넘기면 명시적으로 초기화(마지막으로 손댄 일차 없음)한다는 뜻.
        new_last_target_day = (
            last_target_day if last_target_day is not None else session.get("last_target_day")
        )
        new_last_target_day = new_last_target_day or None
        now = _utcnow()

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
                    language = ?, preferences = ?, start_date = ?,
                    excluded_places = ?, shown_poi_ids = ?, shown_place_names = ?,
                    last_target_day = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    new_title,
                    new_destination,
                    new_days,
                    new_budget,
                    new_language,
                    json.dumps(new_preferences, ensure_ascii=False),
                    new_start,
                    json.dumps(new_excluded, ensure_ascii=False),
                    json.dumps(new_shown_ids, ensure_ascii=False),
                    json.dumps(new_shown_names, ensure_ascii=False),
                    new_last_target_day,
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
        trimmed = messages[-(turn_limit * 2) :]
        lines: list[str] = []
        for msg in trimmed:
            label = "사용자" if msg["role"] == "user" else "어시스턴트"
            lines.append(f"{label}: {msg['content']}")
        return "\n".join(lines)
