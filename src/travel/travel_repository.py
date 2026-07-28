"""SQLite repository for structured travel search."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.config import TRAVEL_DB_PATH, TRAVEL_NORMALIZED_DIR


SCHEMA = """
CREATE TABLE IF NOT EXISTS places (
    poi_id TEXT PRIMARY KEY,
    travel_type TEXT,
    name_ko TEXT,
    name_en TEXT,
    name_norm TEXT,
    region TEXT,
    city TEXT,
    address_ko TEXT,
    address_en TEXT,
    phone TEXT,
    hours_ko TEXT,
    hours_en TEXT,
    closed_ko TEXT,
    fee_ko TEXT,
    fee_en TEXT,
    parking_ko TEXT,
    menu_ko TEXT,
    homepage TEXT,
    checkin TEXT,
    checkout TEXT,
    source TEXT,
    page_content TEXT
);

CREATE TABLE IF NOT EXISTS courses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    package_id TEXT,
    title TEXT,
    city TEXT,
    duration_raw TEXT,
    duration_days INTEGER,
    duration_nights INTEGER,
    departure_city TEXT,
    departure_date TEXT,
    places TEXT,
    price_ref INTEGER,
    price_raw TEXT,
    source TEXT,
    page_content TEXT
);

CREATE INDEX IF NOT EXISTS idx_places_city ON places(city);
CREATE INDEX IF NOT EXISTS idx_places_region ON places(region);
CREATE INDEX IF NOT EXISTS idx_places_type ON places(travel_type);
CREATE INDEX IF NOT EXISTS idx_places_name_norm ON places(name_norm);
CREATE INDEX IF NOT EXISTS idx_courses_city ON courses(city);
CREATE INDEX IF NOT EXISTS idx_courses_days ON courses(duration_days);
CREATE INDEX IF NOT EXISTS idx_courses_price ON courses(price_ref);
"""


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else TRAVEL_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_database(
    *,
    db_path: Path | str | None = None,
    normalized_dir: Path | str | None = None,
) -> Path:
    out = Path(db_path) if db_path else TRAVEL_DB_PATH
    norm = Path(normalized_dir) if normalized_dir else TRAVEL_NORMALIZED_DIR
    pois = _load_jsonl(norm / "pois.jsonl")
    courses = _load_jsonl(norm / "courses.jsonl")

    if out.exists():
        out.unlink()

    conn = connect(out)
    try:
        init_db(conn)
        conn.executemany(
            """
            INSERT OR REPLACE INTO places (
                poi_id, travel_type, name_ko, name_en, name_norm, region, city,
                address_ko, address_en, phone, hours_ko, hours_en, closed_ko,
                fee_ko, fee_en, parking_ko, menu_ko, homepage, checkin, checkout,
                source, page_content
            ) VALUES (
                :poi_id, :travel_type, :name_ko, :name_en, :name_norm, :region, :city,
                :address_ko, :address_en, :phone, :hours_ko, :hours_en, :closed_ko,
                :fee_ko, :fee_en, :parking_ko, :menu_ko, :homepage, :checkin, :checkout,
                :source, :page_content
            )
            """,
            [
                {
                    "poi_id": r.get("poi_id"),
                    "travel_type": r.get("travel_type"),
                    "name_ko": r.get("name_ko"),
                    "name_en": r.get("name_en"),
                    "name_norm": r.get("name_norm"),
                    "region": r.get("region"),
                    "city": r.get("city"),
                    "address_ko": r.get("address_ko"),
                    "address_en": r.get("address_en"),
                    "phone": r.get("phone"),
                    "hours_ko": r.get("hours_ko"),
                    "hours_en": r.get("hours_en"),
                    "closed_ko": r.get("closed_ko"),
                    "fee_ko": r.get("fee_ko"),
                    "fee_en": r.get("fee_en"),
                    "parking_ko": r.get("parking_ko"),
                    "menu_ko": r.get("menu_ko"),
                    "homepage": r.get("homepage"),
                    "checkin": r.get("checkin"),
                    "checkout": r.get("checkout"),
                    "source": r.get("source"),
                    "page_content": r.get("page_content"),
                }
                for r in pois
            ],
        )
        conn.executemany(
            """
            INSERT INTO courses (
                package_id, title, city, duration_raw, duration_days, duration_nights,
                departure_city, departure_date, places, price_ref, price_raw, source,
                page_content
            ) VALUES (
                :package_id, :title, :city, :duration_raw, :duration_days, :duration_nights,
                :departure_city, :departure_date, :places, :price_ref, :price_raw, :source,
                :page_content
            )
            """,
            [
                {
                    "package_id": r.get("package_id"),
                    "title": r.get("title"),
                    "city": r.get("city"),
                    "duration_raw": r.get("duration_raw"),
                    "duration_days": r.get("duration_days"),
                    "duration_nights": r.get("duration_nights"),
                    "departure_city": r.get("departure_city"),
                    "departure_date": r.get("departure_date"),
                    "places": r.get("places"),
                    "price_ref": r.get("price_ref"),
                    "price_raw": r.get("price_raw"),
                    "source": r.get("source"),
                    "page_content": r.get("page_content"),
                }
                for r in courses
            ],
        )
        conn.commit()
    finally:
        conn.close()

    print(f"SQLite 저장 완료: places={len(pois)}, courses={len(courses)} -> {out}")
    return out


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}


# Destination aliases used for structured city filters
CITY_ALIASES: dict[str, list[str]] = {
    "제주": ["제주", "제주시", "서귀포시", "제주특별자치도", "Jeju"],
    "서울": ["서울", "서울특별시", "Seoul"],
    "부산": ["부산", "부산광역시", "Busan"],
    "인천": ["인천", "인천광역시", "Incheon"],
    "대구": ["대구", "대구광역시", "Daegu"],
    "광주": ["광주", "광주광역시", "Gwangju"],
    "대전": ["대전", "대전광역시", "Daejeon"],
    "울산": ["울산", "울산광역시", "Ulsan"],
    "강릉": ["강릉", "강릉시"],
    "전주": ["전주", "전주시"],
    "경주": ["경주", "경주시"],
    "여수": ["여수", "여수시"],
}


def city_match_values(city: str) -> list[str]:
    text = city.strip()
    aliases = CITY_ALIASES.get(text, [])
    values = [text, *aliases]
    # de-dupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


class TravelRepository:
    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else TRAVEL_DB_PATH
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"여행 DB가 없습니다: {self.db_path}. "
                "먼저 `python scripts/build_travel_index.py` 를 실행하세요."
            )

    def search_places(
        self,
        *,
        query: str | None = None,
        city: str | None = None,
        region: str | None = None,
        types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        if city:
            aliases = city_match_values(city)
            city_parts = []
            for alias in aliases:
                city_parts.append(
                    "(city = ? OR city LIKE ? OR region = ? OR region LIKE ? "
                    "OR address_ko LIKE ? OR address_ko LIKE ?)"
                )
                params.extend(
                    [
                        alias,
                        f"{alias}%",
                        alias,
                        f"{alias}%",
                        f"{alias} %",
                        f"{alias}%",
                    ]
                )
            clauses.append("(" + " OR ".join(city_parts) + ")")
        if region:
            clauses.append("region LIKE ?")
            params.append(f"%{region}%")
        if types:
            placeholders = ",".join("?" for _ in types)
            clauses.append(f"travel_type IN ({placeholders})")
            params.extend(types)
        if query:
            # Prefer short keyword matches; long free-text is handled by FAISS
            token = query.strip()
            if len(token) <= 30:
                clauses.append(
                    "(name_ko LIKE ? OR name_en LIKE ? OR page_content LIKE ?)"
                )
                like = f"%{token}%"
                params.extend([like, like, like])

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT poi_id, travel_type, name_ko, name_en, region, city,
                   address_ko, address_en, phone, hours_ko, hours_en, closed_ko,
                   fee_ko, fee_en, menu_ko, source, page_content
            FROM places
            {where}
            ORDER BY
                CASE travel_type
                    WHEN '관광지' THEN 0
                    WHEN '문화시설' THEN 1
                    WHEN '음식점' THEN 2
                    WHEN '숙박' THEN 3
                    ELSE 9
                END,
                name_ko
            LIMIT ?
        """
        params.append(limit)

        with connect(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_place(self, poi_id: str) -> dict[str, Any] | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM places WHERE poi_id = ?", (poi_id,)
            ).fetchone()
        return _row_to_dict(row) if row else None

    def search_courses(
        self,
        *,
        city: str | None = None,
        days: int | None = None,
        budget: int | None = None,
        query: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        if city:
            aliases = city_match_values(city)
            parts = []
            for alias in aliases:
                parts.append("(city LIKE ? OR places LIKE ? OR title LIKE ?)")
                like = f"%{alias}%"
                params.extend([like, like, like])
            clauses.append("(" + " OR ".join(parts) + ")")
        if days is not None:
            clauses.append("(duration_days = ? OR duration_days IS NULL)")
            params.append(days)
        if budget is not None:
            clauses.append("(price_ref IS NULL OR price_ref <= ?)")
            params.append(budget)
        if query:
            clauses.append("(title LIKE ? OR places LIKE ? OR page_content LIKE ?)")
            like = f"%{query}%"
            params.extend([like, like, like])

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT package_id, title, city, duration_raw, duration_days, duration_nights,
                   departure_city, departure_date, places, price_ref, price_raw, source,
                   page_content
            FROM courses
            {where}
            ORDER BY
                CASE WHEN price_ref IS NULL THEN 1 ELSE 0 END,
                price_ref ASC,
                title
            LIMIT ?
        """
        params.append(limit)

        with connect(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(r) for r in rows]
