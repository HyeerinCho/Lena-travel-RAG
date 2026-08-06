"""Hybrid search helpers over SQLite + travel FAISS."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote as _url_quote

from src.config import TRAVEL_DB_PATH, TRAVEL_VECTORSTORE_PATH
from src.travel.travel_repository import TravelRepository
from src.travel.travel_vectorstore import _index_exists


class TravelSearchService:
    def __init__(
        self,
        *,
        db_path=None,
        vectorstore_path=None,
        vectorstore=None,
    ):
        self.repo = TravelRepository(db_path or TRAVEL_DB_PATH)
        self.vectorstore_path = vectorstore_path or TRAVEL_VECTORSTORE_PATH
        self._vectorstore = vectorstore

    @property
    def vectorstore(self):
        if self._vectorstore is None and _index_exists(self.vectorstore_path):
            from src.travel.travel_vectorstore import get_travel_vectorstore

            self._vectorstore = get_travel_vectorstore(self.vectorstore_path)
        return self._vectorstore

    def search_places(
        self,
        query: str | None = None,
        city: str | None = None,
        types: list[str] | None = None,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        # Structured filter uses city/types; free-text query is for semantic re-rank
        sql_query = None
        if query:
            # keep only compact preference tokens for SQL LIKE
            tokens = [t for t in query.replace(",", " ").split() if 1 < len(t) <= 12]
            # Avoid dumping the whole user question into SQL AND filters
            if len(tokens) == 1:
                sql_query = tokens[0]
        sql_hits = self.repo.search_places(
            query=sql_query,
            city=city,
            types=types,
            limit=limit * 3,
        )
        if len(sql_hits) < max(3, limit // 2) and sql_query:
            sql_hits = self.repo.search_places(
                query=None,
                city=city,
                types=types,
                limit=limit * 3,
            )

        semantic_hits: list[dict[str, Any]] = []
        vs = self.vectorstore
        if vs is not None and query:
            docs = vs.similarity_search(query, k=limit * 2)
            for doc in docs:
                meta = dict(doc.metadata)
                if meta.get("source") not in (None, "poi"):
                    continue
                if city:
                    blob = " ".join(
                        str(meta.get(k) or "")
                        for k in ("city", "region", "address_ko", "page_content")
                    )
                    if city not in blob and city.lower() not in blob.lower():
                        # also check document content
                        if city not in doc.page_content:
                            continue
                if types and meta.get("travel_type") not in types:
                    continue
                item = {
                    "poi_id": meta.get("poi_id"),
                    "travel_type": meta.get("travel_type"),
                    "name_ko": meta.get("name_ko"),
                    "name_en": meta.get("name_en"),
                    "region": meta.get("region"),
                    "city": meta.get("city"),
                    "address_ko": meta.get("address_ko"),
                    "hours_ko": meta.get("hours_ko"),
                    "fee_ko": meta.get("fee_ko"),
                    "source": meta.get("source", "poi"),
                    "page_content": doc.page_content,
                    "match": "semantic",
                }
                semantic_hits.append(item)

        merged = _merge_by_key(sql_hits, semantic_hits, key="poi_id", limit=limit)
        for item in merged:
            item.setdefault("match", "sql")
        return merged

    def search_courses(
        self,
        city: str | None = None,
        days: int | None = None,
        budget: int | None = None,
        query: str | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        sql_hits = self.repo.search_courses(
            city=city,
            days=days,
            budget=budget,
            query=query,
            limit=limit * 2,
        )

        semantic_hits: list[dict[str, Any]] = []
        vs = self.vectorstore
        if vs is not None and (query or city):
            q = query or f"{city} travel course"
            docs = vs.similarity_search(q, k=limit * 2)
            for doc in docs:
                meta = dict(doc.metadata)
                if meta.get("source") != "course":
                    continue
                if city:
                    blob = " ".join(
                        str(meta.get(k) or "") for k in ("city", "title", "places")
                    )
                    if city not in blob and city not in doc.page_content:
                        continue
                if days is not None and meta.get("duration_days") not in (None, days, str(days)):
                    try:
                        if int(meta.get("duration_days")) != days:
                            continue
                    except (TypeError, ValueError):
                        pass
                if budget is not None and meta.get("price_ref") not in (None, ""):
                    try:
                        if int(meta["price_ref"]) > budget:
                            continue
                    except (TypeError, ValueError):
                        pass
                semantic_hits.append(
                    {
                        "package_id": meta.get("package_id"),
                        "title": meta.get("title"),
                        "city": meta.get("city"),
                        "duration_raw": meta.get("duration_raw"),
                        "duration_days": meta.get("duration_days"),
                        "places": meta.get("places"),
                        "price_ref": meta.get("price_ref"),
                        "source": "course",
                        "page_content": doc.page_content,
                        "match": "semantic",
                    }
                )

        merged = _merge_by_key(
            sql_hits,
            semantic_hits,
            key="package_id",
            limit=limit,
            fallback_key="title",
        )
        for item in merged:
            item.setdefault("match", "sql")
            item["price_note"] = "과거 상품 참고 가격 (실시간 아님)"
        return merged

    def get_place_details(self, poi_id: str) -> dict[str, Any] | None:
        place = self.repo.get_place(poi_id)
        if not place:
            return None
        homepage = (place.get("homepage") or "").strip()
        if not homepage:
            # TourAPI detailCommon homepage 보강 시도
            try:
                from src.travel.tour_api import kor_service

                common = kor_service.detail_common(str(poi_id))
                if isinstance(common, dict):
                    for key in ("homepage", "hmpg", "hmpgurl"):
                        val = str(common.get(key) or "").strip()
                        if val.startswith("http"):
                            homepage = val
                            place["homepage"] = homepage
                            break
                    # HTML anchor sometimes nested
                    if not homepage:
                        raw = str(common.get("homepage") or "")
                        if "http" in raw:
                            import re as _re

                            m = _re.search(r"https?://[^\s\"'<>]+", raw)
                            if m:
                                place["homepage"] = m.group(0)
                                homepage = m.group(0)
            except Exception:
                pass
        name = place.get("name_ko") or place.get("name_en") or ""
        region = " ".join(
            x for x in (place.get("region"), place.get("city")) if x
        )
        query = f"{name} {region}".strip()
        place["search_url"] = (
            f"https://search.naver.com/search.naver?query={_url_quote(query)}"
            if query
            else None
        )
        place["has_homepage"] = bool(
            homepage and str(homepage).startswith("http")
        )
        return place


def _merge_by_key(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    *,
    key: str,
    limit: int,
    fallback_key: str | None = None,
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []

    def _id(item: dict[str, Any]) -> str | None:
        value = item.get(key)
        if value:
            return str(value)
        if fallback_key and item.get(fallback_key):
            return f"fb:{item.get(fallback_key)}"
        return None

    for item in secondary + primary:
        ident = _id(item)
        if ident and ident in seen:
            continue
        if ident:
            seen.add(ident)
        out.append(item)
        if len(out) >= limit:
            break
    return out
