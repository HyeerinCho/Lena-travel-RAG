"""LangGraph for conversational Korea travel planning."""

from __future__ import annotations

import json
import re
from typing import Any, Iterator, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from src.config import LLM_MODEL
from src.prompts import (
    TRAVEL_CITY_LIST_PROMPT,
    TRAVEL_EXTRACT_PROMPT,
    TRAVEL_ITINERARY_PROMPT,
    TRAVEL_REWRITE_DAY_PROMPT,
)
from src.travel.travel_tools import TravelSearchService


class TravelState(TypedDict, total=False):
    query: str
    history: str
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
    rewrite_day: int | None
    previous_itinerary: list[dict[str, Any]]
    intent: str
    cities: list[dict[str, Any]]


_ORDINAL_KO = {
    "첫": 1,
    "첫째": 1,
    "두": 2,
    "둘째": 2,
    "세": 3,
    "셋째": 3,
    "네": 4,
    "넷째": 4,
    "다섯": 5,
    "다섯째": 5,
    "여섯": 6,
    "여섯째": 6,
    "일곱": 7,
    "일곱째": 7,
}


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


def _heuristic_rewrite_day(query: str) -> int | None:
    patterns = [
        r"(\d+)\s*일차\s*만",
        r"day\s*(\d+)\s*only",
        r"only\s*day\s*(\d+)",
        r"(\d+)\s*일\s*만\s*(바꿔|수정|다시|재작성)",
        r"(\d+)\s*일차\s*(바꿔|수정|다시|재작성)",
    ]
    for pattern in patterns:
        m = re.search(pattern, query, re.I)
        if m:
            return int(m.group(1))

    for word, day in _ORDINAL_KO.items():
        if re.search(rf"{word}\s*날\s*만", query):
            return day
        if re.search(rf"{word}\s*날\s*(바꿔|수정|다시)", query):
            return day
    return None


def _detect_city_list_intent(query: str) -> bool:
    """True when the user wants a list of cities, not a day-by-day itinerary."""
    if re.search(r"일정|코스|며칠|\d+\s*박|\d+\s*일차|day\s*\d", query, re.I):
        return False
    patterns = [
        r"도시\s*(들|만|목록|리스트)",
        r"(도시|지역).{0,6}(뽑아|추천|골라|알려|정리)",
        r"(뽑아|추천|골라|알려|정리).{0,6}(도시|지역)",
        r"(리스트|목록)\s*로\s*(뽑|만들|정리)",
        r"어느\s*도시|어떤\s*도시|어디가?\s*좋",
        r"city\s*list|list\s*of\s*cities",
    ]
    return any(re.search(p, query, re.I) for p in patterns)


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
        "rewrite_day": _heuristic_rewrite_day(query),
        "intent": "city_list" if _detect_city_list_intent(query) else "itinerary",
    }


def _focus_single_region(
    places: list[dict[str, Any]],
    destination: str | None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Keep candidates within a single province/region so the itinerary stays local.

    When ``destination`` is empty (e.g. "아무곳이나 골라줘"), the SQLite/FAISS search
    returns POIs from all over the country. Feeding those straight to the LLM produces
    scattered plans (대구/부산/광주 mixed). We pick the region with the most candidates
    and drop everything else, then surface that region as the destination.
    """
    if not places:
        return places, destination

    def _region_of(p: dict[str, Any]) -> str | None:
        value = p.get("region") or p.get("city")
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    if destination:
        matched = [p for p in places if destination in " ".join(
            str(p.get(k) or "") for k in ("region", "city", "address_ko")
        )]
        # Only narrow when we still have enough to build a plan.
        if len(matched) >= 3:
            return matched, destination
        return places, destination

    counts: dict[str, int] = {}
    for p in places:
        region = _region_of(p)
        if region:
            counts[region] = counts.get(region, 0) + 1
    if not counts:
        return places, destination

    dominant = max(counts, key=lambda r: counts[r])
    focused = [p for p in places if _region_of(p) == dominant]
    return (focused or places), dominant


def _group_places_by_city(
    places: list[dict[str, Any]],
    max_cities: int = 6,
    max_per_city: int = 3,
) -> list[dict[str, Any]]:
    """Group candidate POIs by city for the city-list intent.

    Order of cities follows first appearance (search relevance). Each city keeps up
    to ``max_per_city`` place names, so the LLM can only pick from real candidates.
    """
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for p in places:
        city = (p.get("city") or p.get("region") or "").strip()
        if not city:
            continue
        name = p.get("name_ko") or p.get("name_en")
        if not name:
            continue
        if city not in grouped:
            grouped[city] = {
                "city": city,
                "region": p.get("region"),
                "places": [],
            }
            order.append(city)
        bucket = grouped[city]["places"]
        if len(bucket) < max_per_city and all(
            existing.get("poi_id") != p.get("poi_id") for existing in bucket
        ):
            bucket.append({"place_name": name, "poi_id": p.get("poi_id")})
    return [grouped[c] for c in order[:max_cities]]


def _build_city_list(
    state: TravelState,
    parsed: dict[str, Any],
) -> TravelState:
    """Merge LLM merits/answer with code-grouped city candidates (safe place names)."""
    grouped = state.get("cities") or []
    by_city = {g["city"]: g for g in grouped}

    cities: list[dict[str, Any]] = []
    for item in parsed.get("cities") or []:
        city = (item.get("city") or "").strip()
        source = by_city.get(city)
        if source is None:
            # Only trust cities we actually have candidates for.
            continue
        cities.append(
            {
                "city": city,
                "region": source.get("region"),
                "merit": item.get("merit") or "",
                "places": source.get("places") or [],
            }
        )

    # Fallback: if the model returned nothing usable, list grouped cities as-is.
    if not cities:
        cities = [
            {
                "city": g["city"],
                "region": g.get("region"),
                "merit": "",
                "places": g.get("places") or [],
            }
            for g in grouped
        ]

    warnings = list(state.get("warnings") or [])
    warnings.extend(parsed.get("warnings") or [])
    answer = parsed.get("answer")
    if not answer:
        lines = []
        for c in cities:
            names = ", ".join(p["place_name"] for p in c["places"]) or "-"
            merit = f" — {c['merit']}" if c.get("merit") else ""
            lines.append(f"- {c['city']}{merit} (추천: {names})")
        answer = "\n".join(lines) or "조건에 맞는 도시를 찾지 못했습니다."

    return {
        "cities": cities,
        "warnings": warnings,
        "answer": answer,
        "sources": _build_sources(state),
        "intent": "city_list",
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


def _merge_day_into_itinerary(
    previous: list[dict[str, Any]],
    day_num: int,
    new_day: dict[str, Any],
) -> list[dict[str, Any]]:
    merged = [dict(d) for d in previous]
    replacement = {
        "day": day_num,
        "theme": new_day.get("theme") or f"Day {day_num}",
        "slots": new_day.get("slots") or [],
    }
    replaced = False
    for i, day in enumerate(merged):
        if int(day.get("day") or 0) == day_num:
            merged[i] = replacement
            replaced = True
            break
    if not replaced:
        merged.append(replacement)
        merged.sort(key=lambda d: int(d.get("day") or 0))
    return merged


def _build_sources(state: TravelState) -> list[dict[str, Any]]:
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
    return sources


def _llm_content(raw: Any) -> str:
    content = getattr(raw, "content", raw)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(getattr(block, "text", block)))
        return "".join(parts)
    return str(content)


def build_travel_graph(search: TravelSearchService):
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)

    def extract_requirements(state: TravelState) -> TravelState:
        base = _heuristic_extract(state["query"])
        history = state.get("history") or "(없음)"
        for key in (
            "destination",
            "days",
            "budget",
            "language",
            "preferences",
            "place_types",
            "rewrite_day",
        ):
            if state.get(key) not in (None, [], ""):
                base[key] = state[key]

        try:
            raw = (TRAVEL_EXTRACT_PROMPT | llm).invoke(
                {
                    "question": state["query"],
                    "history": history,
                    "session_destination": state.get("destination") or base.get("destination"),
                    "session_days": state.get("days") or base.get("days"),
                    "session_budget": state.get("budget") or base.get("budget"),
                    "session_preferences": ", ".join(
                        state.get("preferences") or base.get("preferences") or []
                    )
                    or "(없음)",
                }
            )
            parsed = _extract_json(_llm_content(raw))
        except Exception:
            parsed = {}

        merged = {**base}
        for key in (
            "destination",
            "days",
            "budget",
            "language",
            "preferences",
            "place_types",
            "rewrite_day",
        ):
            if parsed.get(key) not in (None, [], ""):
                if state.get(key) in (None, [], ""):
                    merged[key] = parsed[key]

        for key in ("destination", "days", "budget", "language"):
            if merged.get(key) in (None, "", []) and state.get(key) not in (None, "", []):
                merged[key] = state[key]
        if not merged.get("preferences") and state.get("preferences"):
            merged["preferences"] = state["preferences"]
        if state.get("rewrite_day") is not None:
            merged["rewrite_day"] = state["rewrite_day"]
        elif merged.get("rewrite_day") is None:
            merged["rewrite_day"] = _heuristic_rewrite_day(state["query"])

        # city_list wins if either the heuristic or the LLM detects it.
        merged["intent"] = (
            "city_list"
            if "city_list" in (base.get("intent"), parsed.get("intent"))
            else "itinerary"
        )

        if not merged.get("preferences"):
            merged["preferences"] = []
        if not merged.get("place_types"):
            merged["place_types"] = ["관광지", "문화시설"]
        if not merged.get("language"):
            merged["language"] = "ko"
        if "previous_itinerary" in state:
            merged["previous_itinerary"] = state["previous_itinerary"]
        return merged

    def search_candidates(state: TravelState) -> TravelState:
        prefs = " ".join(state.get("preferences") or [])
        query = " ".join(x for x in [state.get("query"), prefs] if x)
        destination = state.get("destination")
        types = state.get("place_types") or ["관광지", "문화시설"]

        # City-list intent: gather a wide spread across many cities (no region focus).
        if state.get("intent") == "city_list":
            places = search.search_places(
                query=query,
                city=destination,
                types=types,
                limit=60,
            )
            if len(places) < 12:
                places = search.search_places(
                    query=query,
                    city=destination,
                    types=None,
                    limit=60,
                )
            cities = _group_places_by_city(places)
            warnings = []
            if not cities:
                warnings.append("조건에 맞는 도시를 충분히 찾지 못했습니다.")
            return {"places": places, "courses": [], "warnings": warnings, "cities": cities}

        places = search.search_places(
            query=query,
            city=destination,
            types=types,
            limit=12,
        )
        if len(places) < 4 and destination:
            places = search.search_places(
                query=query,
                city=destination,
                types=None,
                limit=12,
            )

        places, focused_destination = _focus_single_region(places, destination)
        if focused_destination and not destination:
            destination = focused_destination

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

        result: TravelState = {
            "places": places,
            "courses": courses,
            "warnings": warnings,
        }
        if focused_destination:
            result["destination"] = focused_destination
        return result

    def build_itinerary(state: TravelState) -> TravelState:
        if state.get("intent") == "city_list":
            grouped = state.get("cities") or []
            if not grouped:
                return {
                    "cities": [],
                    "warnings": list(state.get("warnings") or []),
                    "answer": "조건에 맞는 도시를 찾지 못했습니다. 조건을 조금 바꿔서 다시 물어봐 주세요.",
                    "sources": _build_sources(state),
                    "intent": "city_list",
                }
            try:
                raw = (TRAVEL_CITY_LIST_PROMPT | llm).invoke(
                    {
                        "question": state.get("query"),
                        "history": state.get("history") or "(없음)",
                        "preferences": ", ".join(state.get("preferences") or []),
                        "language": state.get("language") or "ko",
                        "candidates": json.dumps(grouped, ensure_ascii=False),
                    }
                )
                parsed = _extract_json(_llm_content(raw))
            except Exception:
                parsed = {}
            return _build_city_list(state, parsed)

        places = _compact_places(state.get("places") or [])
        courses = _compact_courses(state.get("courses") or [])
        language = state.get("language") or "ko"
        rewrite_day = state.get("rewrite_day")
        previous = list(state.get("previous_itinerary") or [])

        if not places and not courses:
            return _fallback_itinerary(state)

        try:
            if rewrite_day and previous:
                raw = (TRAVEL_REWRITE_DAY_PROMPT | llm).invoke(
                    {
                        "question": state.get("query"),
                        "history": state.get("history") or "(없음)",
                        "destination": state.get("destination"),
                        "budget": state.get("budget"),
                        "preferences": ", ".join(state.get("preferences") or []),
                        "language": language,
                        "rewrite_day": rewrite_day,
                        "previous_itinerary": json.dumps(previous, ensure_ascii=False),
                        "places": json.dumps(places, ensure_ascii=False),
                        "courses": json.dumps(courses, ensure_ascii=False),
                    }
                )
                parsed = _extract_json(_llm_content(raw))
                new_day = parsed.get("day") or {}
                warnings = list(state.get("warnings") or [])
                if not new_day.get("slots"):
                    warnings.append(f"{rewrite_day}일차 재작성에 실패해 기존 일정을 유지합니다.")
                    return {
                        "itinerary": previous,
                        "warnings": warnings,
                        "answer": parsed.get("answer")
                        or f"{rewrite_day}일차를 바꾸지 못했습니다. 다른 표현으로 다시 요청해 주세요.",
                        "sources": _build_sources(state),
                        "rewrite_day": rewrite_day,
                    }
                itinerary = _merge_day_into_itinerary(previous, int(rewrite_day), new_day)
                warnings.extend(parsed.get("warnings") or [])
                answer = parsed.get("answer") or f"{rewrite_day}일차 일정을 수정했습니다."
                return {
                    "itinerary": itinerary,
                    "warnings": warnings,
                    "answer": answer,
                    "sources": _build_sources(state),
                    "rewrite_day": rewrite_day,
                }

            raw = (TRAVEL_ITINERARY_PROMPT | llm).invoke(
                {
                    "question": state.get("query"),
                    "history": state.get("history") or "(없음)",
                    "destination": state.get("destination"),
                    "days": state.get("days") or 1,
                    "budget": state.get("budget"),
                    "preferences": ", ".join(state.get("preferences") or []),
                    "language": language,
                    "places": json.dumps(places, ensure_ascii=False),
                    "courses": json.dumps(courses, ensure_ascii=False),
                }
            )
            parsed = _extract_json(_llm_content(raw))
        except Exception:
            parsed = {}

        if not parsed.get("answer"):
            return _fallback_itinerary(state)

        warnings = list(state.get("warnings") or [])
        warnings.extend(parsed.get("warnings") or [])
        return {
            "itinerary": parsed.get("itinerary") or [],
            "warnings": warnings,
            "answer": parsed["answer"],
            "sources": _build_sources(state),
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


def stream_build_itinerary_tokens(
    search: TravelSearchService,
    state: TravelState,
) -> Iterator[tuple[str, Any]]:
    """Yield (event_type, payload) for SSE. Runs extract+search sync, streams LLM."""
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)
    result_so_far: TravelState = dict(state)

    yield ("status", {"stage": "extracting"})
    base = _heuristic_extract(result_so_far["query"])
    for key in (
        "destination",
        "days",
        "budget",
        "language",
        "preferences",
        "place_types",
        "rewrite_day",
    ):
        if result_so_far.get(key) not in (None, [], ""):
            base[key] = result_so_far[key]
    try:
        raw = (TRAVEL_EXTRACT_PROMPT | llm).invoke(
            {
                "question": result_so_far["query"],
                "history": result_so_far.get("history") or "(없음)",
                "session_destination": result_so_far.get("destination") or base.get("destination"),
                "session_days": result_so_far.get("days") or base.get("days"),
                "session_budget": result_so_far.get("budget") or base.get("budget"),
                "session_preferences": ", ".join(
                    result_so_far.get("preferences") or base.get("preferences") or []
                )
                or "(없음)",
            }
        )
        parsed = _extract_json(_llm_content(raw))
    except Exception:
        parsed = {}
    merged = {**base}
    for key in (
        "destination",
        "days",
        "budget",
        "language",
        "preferences",
        "place_types",
        "rewrite_day",
    ):
        if parsed.get(key) not in (None, [], ""):
            if result_so_far.get(key) in (None, [], ""):
                merged[key] = parsed[key]
    for key in ("destination", "days", "budget", "language"):
        if merged.get(key) in (None, "", []) and result_so_far.get(key) not in (None, "", []):
            merged[key] = result_so_far[key]
    if not merged.get("preferences") and result_so_far.get("preferences"):
        merged["preferences"] = result_so_far["preferences"]
    if result_so_far.get("rewrite_day") is not None:
        merged["rewrite_day"] = result_so_far["rewrite_day"]
    elif merged.get("rewrite_day") is None:
        merged["rewrite_day"] = _heuristic_rewrite_day(result_so_far["query"])
    merged["intent"] = (
        "city_list"
        if "city_list" in (base.get("intent"), parsed.get("intent"))
        else "itinerary"
    )
    if not merged.get("preferences"):
        merged["preferences"] = []
    if not merged.get("place_types"):
        merged["place_types"] = ["관광지", "문화시설"]
    if not merged.get("language"):
        merged["language"] = "ko"
    if "previous_itinerary" in result_so_far:
        merged["previous_itinerary"] = result_so_far["previous_itinerary"]
    result_so_far.update(merged)

    yield ("status", {"stage": "searching"})
    prefs = " ".join(result_so_far.get("preferences") or [])
    query = " ".join(x for x in [result_so_far.get("query"), prefs] if x)
    destination = result_so_far.get("destination")
    types = result_so_far.get("place_types") or ["관광지", "문화시설"]

    if result_so_far.get("intent") == "city_list":
        places = search.search_places(query=query, city=destination, types=types, limit=60)
        if len(places) < 12:
            places = search.search_places(query=query, city=destination, types=None, limit=60)
        grouped = _group_places_by_city(places)
        warnings = [] if grouped else ["조건에 맞는 도시를 충분히 찾지 못했습니다."]
        result_so_far["places"] = places
        result_so_far["courses"] = []
        result_so_far["cities"] = grouped
        result_so_far["warnings"] = warnings

        yield ("status", {"stage": "writing"})
        if not grouped:
            result_so_far.update(
                {
                    "cities": [],
                    "answer": "조건에 맞는 도시를 찾지 못했습니다. 조건을 조금 바꿔서 다시 물어봐 주세요.",
                    "sources": _build_sources(result_so_far),
                }
            )
            yield ("done", _result_payload(result_so_far))
            return

        inputs = {
            "question": result_so_far.get("query"),
            "history": result_so_far.get("history") or "(없음)",
            "preferences": ", ".join(result_so_far.get("preferences") or []),
            "language": result_so_far.get("language") or "ko",
            "candidates": json.dumps(grouped, ensure_ascii=False),
        }
        accumulated = ""
        try:
            for chunk in (TRAVEL_CITY_LIST_PROMPT | llm).stream(inputs):
                piece = _llm_content(chunk)
                if not piece:
                    continue
                accumulated += piece
                yield ("token", {"text": piece})
            parsed = _extract_json(accumulated)
        except Exception:
            parsed = {}
        result_so_far.update(_build_city_list(result_so_far, parsed))
        yield ("done", _result_payload(result_so_far))
        return

    places = search.search_places(query=query, city=destination, types=types, limit=12)
    if len(places) < 4 and destination:
        places = search.search_places(query=query, city=destination, types=None, limit=12)

    places, focused_destination = _focus_single_region(places, destination)
    if focused_destination:
        destination = focused_destination
        result_so_far["destination"] = focused_destination

    courses = search.search_courses(
        city=destination,
        days=result_so_far.get("days"),
        budget=result_so_far.get("budget"),
        query=query,
        limit=8,
    )
    warnings: list[str] = []
    if not places:
        warnings.append("조건에 맞는 POI가 충분하지 않습니다.")
    if result_so_far.get("budget") is not None and courses:
        warnings.append("코스 가격은 과거 상품 참고값입니다.")
    result_so_far["places"] = places
    result_so_far["courses"] = courses
    result_so_far["warnings"] = warnings

    yield ("status", {"stage": "writing"})

    compact_places = _compact_places(places)
    compact_courses = _compact_courses(courses)
    language = result_so_far.get("language") or "ko"
    rewrite_day = result_so_far.get("rewrite_day")
    previous = list(result_so_far.get("previous_itinerary") or [])

    if not compact_places and not compact_courses:
        fallback = _fallback_itinerary(result_so_far)
        result_so_far.update(fallback)
        yield ("done", _result_payload(result_so_far))
        return

    if rewrite_day and previous:
        prompt = TRAVEL_REWRITE_DAY_PROMPT
        inputs = {
            "question": result_so_far.get("query"),
            "history": result_so_far.get("history") or "(없음)",
            "destination": result_so_far.get("destination"),
            "budget": result_so_far.get("budget"),
            "preferences": ", ".join(result_so_far.get("preferences") or []),
            "language": language,
            "rewrite_day": rewrite_day,
            "previous_itinerary": json.dumps(previous, ensure_ascii=False),
            "places": json.dumps(compact_places, ensure_ascii=False),
            "courses": json.dumps(compact_courses, ensure_ascii=False),
        }
    else:
        prompt = TRAVEL_ITINERARY_PROMPT
        inputs = {
            "question": result_so_far.get("query"),
            "history": result_so_far.get("history") or "(없음)",
            "destination": result_so_far.get("destination"),
            "days": result_so_far.get("days") or 1,
            "budget": result_so_far.get("budget"),
            "preferences": ", ".join(result_so_far.get("preferences") or []),
            "language": language,
            "places": json.dumps(compact_places, ensure_ascii=False),
            "courses": json.dumps(compact_courses, ensure_ascii=False),
        }

    accumulated = ""
    try:
        for chunk in (prompt | llm).stream(inputs):
            piece = _llm_content(chunk)
            if not piece:
                continue
            accumulated += piece
            yield ("token", {"text": piece})
        parsed = _extract_json(accumulated)
    except Exception:
        parsed = {}

    if rewrite_day and previous:
        new_day = parsed.get("day") or {}
        w = list(result_so_far.get("warnings") or [])
        if not new_day.get("slots"):
            w.append(f"{rewrite_day}일차 재작성에 실패해 기존 일정을 유지합니다.")
            result_so_far.update(
                {
                    "itinerary": previous,
                    "warnings": w,
                    "answer": parsed.get("answer")
                    or f"{rewrite_day}일차를 바꾸지 못했습니다. 다른 표현으로 다시 요청해 주세요.",
                    "sources": _build_sources(result_so_far),
                    "rewrite_day": rewrite_day,
                }
            )
        else:
            itinerary = _merge_day_into_itinerary(previous, int(rewrite_day), new_day)
            w.extend(parsed.get("warnings") or [])
            result_so_far.update(
                {
                    "itinerary": itinerary,
                    "warnings": w,
                    "answer": parsed.get("answer") or f"{rewrite_day}일차 일정을 수정했습니다.",
                    "sources": _build_sources(result_so_far),
                    "rewrite_day": rewrite_day,
                }
            )
    elif not parsed.get("answer"):
        fallback = _fallback_itinerary(result_so_far)
        result_so_far.update(fallback)
    else:
        w = list(result_so_far.get("warnings") or [])
        w.extend(parsed.get("warnings") or [])
        result_so_far.update(
            {
                "itinerary": parsed.get("itinerary") or [],
                "warnings": w,
                "answer": parsed["answer"],
                "sources": _build_sources(result_so_far),
            }
        )

    yield ("done", _result_payload(result_so_far))


def _result_payload(state: TravelState) -> dict[str, Any]:
    return {
        "destination": state.get("destination"),
        "days": state.get("days"),
        "budget": state.get("budget"),
        "preferences": state.get("preferences") or [],
        "language": state.get("language") or "ko",
        "itinerary": state.get("itinerary") or [],
        "cities": state.get("cities") or [],
        "intent": state.get("intent") or "itinerary",
        "places": state.get("places") or [],
        "courses": state.get("courses") or [],
        "warnings": state.get("warnings") or [],
        "sources": state.get("sources") or [],
        "answer": state.get("answer") or "",
        "rewrite_day": state.get("rewrite_day"),
    }
