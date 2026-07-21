"""LangGraph for conversational Korea travel planning."""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from src.config import LLM_MODEL
from src.prompts import TRAVEL_EXTRACT_PROMPT, TRAVEL_ITINERARY_PROMPT
from src.travel.travel_tools import TravelSearchService


class TravelState(TypedDict, total=False):
    query: str
    destination: str | None
    days: int | None
    budget: int | None
    preferences: list[str]
    language: str
    place_types: list[str]
    places: list[dict[str, Any]]
    courses: list[dict[str, Any]]
    itinerary: list[dict[str, Any]]
    warnings: list[str]
    sources: list[dict[str, Any]]
    answer: str


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _heuristic_extract(query: str) -> dict[str, Any]:
    language = "en" if re.search(r"[A-Za-z]", query) and not re.search(r"[가-힣]", query) else "ko"
    days = None
    m = re.search(r"(\d+)\s*박\s*(\d+)\s*일", query)
    if m:
        days = int(m.group(2))
    else:
        m = re.search(r"(\d+)\s*일", query)
        if m:
            days = int(m.group(1))
        else:
            m = re.search(r"(\d+)\s*days?", query, re.I)
            if m:
                days = int(m.group(1))

    budget = None
    m = re.search(r"(\d+)\s*만원", query)
    if m:
        budget = int(m.group(1)) * 10_000
    else:
        m = re.search(r"예산\s*(\d[\d,]*)", query)
        if m:
            budget = int(m.group(1).replace(",", ""))

    city_aliases = [
        ("제주", "제주"),
        ("서울", "서울"),
        ("부산", "부산"),
        ("강릉", "강릉"),
        ("전주", "전주"),
        ("경주", "경주"),
        ("인천", "인천"),
        ("대구", "대구"),
        ("광주", "광주"),
        ("여수", "여수"),
        ("Jeju", "제주"),
        ("Seoul", "서울"),
        ("Busan", "부산"),
    ]
    destination = None
    for needle, canon in city_aliases:
        if needle.lower() in query.lower():
            destination = canon
            break

    preferences = []
    pref_map = {
        "자연": "자연",
        "경관": "경관",
        "실내": "실내",
        "문화": "문화",
        "가족": "가족",
        "맛집": "음식",
        "음식": "음식",
        "family": "family",
        "nature": "nature",
        "museum": "culture",
        "indoor": "indoor",
    }
    for key, label in pref_map.items():
        if key.lower() in query.lower():
            preferences.append(label)

    place_types = ["관광지", "문화시설"]
    if any(p in preferences for p in ("음식", "맛집")):
        place_types.append("음식점")
    if "실내" in preferences or "문화" in preferences or "culture" in preferences:
        if "문화시설" not in place_types:
            place_types.append("문화시설")

    return {
        "destination": destination,
        "days": days,
        "budget": budget,
        "preferences": preferences,
        "language": language,
        "place_types": place_types,
    }


def _compact_places(places: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    keys = (
        "poi_id",
        "name_ko",
        "name_en",
        "travel_type",
        "city",
        "region",
        "address_ko",
        "hours_ko",
        "fee_ko",
        "source",
    )
    out = []
    for p in places[:limit]:
        out.append({k: p.get(k) for k in keys if p.get(k) is not None})
    return out


def _compact_courses(courses: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    keys = (
        "package_id",
        "title",
        "city",
        "duration_raw",
        "duration_days",
        "places",
        "price_ref",
        "source",
    )
    out = []
    for c in courses[:limit]:
        item = {k: c.get(k) for k in keys if c.get(k) is not None}
        item["price_note"] = "과거 상품 참고 가격"
        out.append(item)
    return out


def _fallback_itinerary(state: TravelState) -> dict[str, Any]:
    days = state.get("days") or 1
    places = state.get("places") or []
    warnings = list(state.get("warnings") or [])
    if not places:
        warnings.append("조건에 맞는 장소를 찾지 못했습니다. 도시나 선호를 더 구체적으로 알려주세요.")
        answer = (
            "추천 가능한 장소 데이터가 부족합니다. 목적지(예: 제주/서울)와 관심사(자연, 문화시설 등)를 알려주세요."
            if state.get("language") != "en"
            else "I could not find enough matching places. Please specify a destination and preferences."
        )
        return {
            "itinerary": [],
            "warnings": warnings,
            "answer": answer,
            "sources": [],
        }

    itinerary = []
    idx = 0
    slots_order = ("morning", "afternoon", "evening")
    for day in range(1, days + 1):
        slots = []
        for slot in slots_order:
            if idx >= len(places):
                break
            p = places[idx]
            idx += 1
            slots.append(
                {
                    "time": slot,
                    "place_name": p.get("name_ko") or p.get("name_en") or "Unknown",
                    "poi_id": p.get("poi_id"),
                    "note": p.get("travel_type") or "",
                }
            )
        itinerary.append({"day": day, "theme": state.get("destination") or "Korea", "slots": slots})

    warnings.append("가격·운영시간은 데이터에 있는 값만 참고하세요. 실시간 정보는 별도 확인이 필요합니다.")
    lines = []
    for day in itinerary:
        lines.append(f"Day {day['day']}:")
        for slot in day["slots"]:
            lines.append(f"- {slot['time']}: {slot['place_name']}")
    answer = "\n".join(lines)
    sources = [
        {"type": "poi", "id": p.get("poi_id"), "name": p.get("name_ko")}
        for p in places[: idx or len(places)]
        if p.get("poi_id")
    ]
    return {
        "itinerary": itinerary,
        "warnings": warnings,
        "answer": answer,
        "sources": sources,
    }


def build_travel_graph(search: TravelSearchService):
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)

    def extract_requirements(state: TravelState) -> TravelState:
        base = _heuristic_extract(state["query"])
        # Allow pre-filled API fields to win
        for key in ("destination", "days", "budget", "language", "preferences", "place_types"):
            if state.get(key) not in (None, [], ""):
                base[key] = state[key]

        try:
            raw = (TRAVEL_EXTRACT_PROMPT | llm).invoke({"question": state["query"]})
            content = getattr(raw, "content", raw)
            parsed = _extract_json(str(content))
        except Exception:
            parsed = {}

        merged = {**base}
        for key in ("destination", "days", "budget", "language", "preferences", "place_types"):
            if parsed.get(key) not in (None, [], ""):
                # keep explicit state overrides
                if state.get(key) in (None, [], ""):
                    merged[key] = parsed[key]

        if not merged.get("preferences"):
            merged["preferences"] = []
        if not merged.get("place_types"):
            merged["place_types"] = ["관광지", "문화시설"]
        if not merged.get("language"):
            merged["language"] = "ko"
        return merged

    def search_candidates(state: TravelState) -> TravelState:
        prefs = " ".join(state.get("preferences") or [])
        query = " ".join(x for x in [state.get("query"), prefs] if x)
        destination = state.get("destination")
        types = state.get("place_types") or ["관광지", "문화시설"]

        places = search.search_places(
            query=query,
            city=destination,
            types=types,
            limit=12,
        )
        # If too few, relax types
        if len(places) < 4 and destination:
            places = search.search_places(
                query=query,
                city=destination,
                types=None,
                limit=12,
            )

        courses = search.search_courses(
            city=destination,
            days=state.get("days"),
            budget=state.get("budget"),
            query=query,
            limit=8,
        )

        warnings = []
        if not places:
            warnings.append("조건에 맞는 POI가 충분하지 않습니다.")
        if state.get("budget") is not None and courses:
            warnings.append("코스 가격은 과거 상품 참고값입니다.")

        return {"places": places, "courses": courses, "warnings": warnings}

    def build_itinerary(state: TravelState) -> TravelState:
        places = _compact_places(state.get("places") or [])
        courses = _compact_courses(state.get("courses") or [])
        language = state.get("language") or "ko"

        if not places and not courses:
            return _fallback_itinerary(state)

        try:
            raw = (TRAVEL_ITINERARY_PROMPT | llm).invoke(
                {
                    "question": state.get("query"),
                    "destination": state.get("destination"),
                    "days": state.get("days") or 1,
                    "budget": state.get("budget"),
                    "preferences": ", ".join(state.get("preferences") or []),
                    "language": language,
                    "places": json.dumps(places, ensure_ascii=False),
                    "courses": json.dumps(courses, ensure_ascii=False),
                }
            )
            content = getattr(raw, "content", raw)
            parsed = _extract_json(str(content))
        except Exception:
            parsed = {}

        if not parsed.get("answer"):
            fallback = _fallback_itinerary(state)
            return fallback

        warnings = list(state.get("warnings") or [])
        warnings.extend(parsed.get("warnings") or [])
        sources = [
            {"type": "poi", "id": p.get("poi_id"), "name": p.get("name_ko") or p.get("name_en")}
            for p in (state.get("places") or [])
            if p.get("poi_id")
        ]
        sources += [
            {
                "type": "course",
                "id": c.get("package_id"),
                "name": c.get("title"),
            }
            for c in (state.get("courses") or [])
            if c.get("package_id") or c.get("title")
        ]

        return {
            "itinerary": parsed.get("itinerary") or [],
            "warnings": warnings,
            "answer": parsed["answer"],
            "sources": sources,
        }

    graph = StateGraph(TravelState)
    graph.add_node("extract_requirements", extract_requirements)
    graph.add_node("search_candidates", search_candidates)
    graph.add_node("build_itinerary", build_itinerary)
    graph.set_entry_point("extract_requirements")
    graph.add_edge("extract_requirements", "search_candidates")
    graph.add_edge("search_candidates", "build_itinerary")
    graph.add_edge("build_itinerary", END)
    return graph.compile()
