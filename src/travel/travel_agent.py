"""Facade for the travel planning agent."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterator

from src.travel.session_store import TravelSessionStore
from src.travel.travel_graph import (
    _dedupe_itinerary_slots,
    _merge_day_into_itinerary,
    _simple_itinerary_answer,
    build_travel_graph,
    stream_build_itinerary_tokens,
)
from src.travel.travel_tools import TravelSearchService


@lru_cache
def get_travel_agent():
    search = TravelSearchService()
    return build_travel_graph(search)


@lru_cache
def get_search_service() -> TravelSearchService:
    return TravelSearchService()


@lru_cache
def get_session_store() -> TravelSessionStore:
    return TravelSessionStore()


def _latest_itinerary(session_id: str) -> list[dict[str, Any]]:
    store = get_session_store()
    messages = store.list_messages(session_id)
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        payload = message.get("payload") or {}
        itinerary = payload.get("itinerary")
        if itinerary:
            return list(itinerary)
    return []


def _collect_shown_from_result(result: dict[str, Any]) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    names: list[str] = []

    def _add(poi_id: Any, name: Any) -> None:
        if poi_id and str(poi_id) not in ids:
            ids.append(str(poi_id))
        if name and str(name) not in names:
            names.append(str(name))

    for day in result.get("itinerary") or []:
        for slot in day.get("slots") or []:
            if isinstance(slot, dict):
                _add(slot.get("poi_id"), slot.get("place_name"))
    for place in result.get("recommended_places") or []:
        if isinstance(place, dict):
            _add(place.get("poi_id"), place.get("place_name"))
    for city in result.get("cities") or []:
        for p in city.get("places") or []:
            if isinstance(p, dict):
                _add(p.get("poi_id"), p.get("place_name"))
    for it in result.get("itineraries") or []:
        for day in it.get("days") or []:
            for slot in day.get("slots") or []:
                if isinstance(slot, dict):
                    _add(slot.get("poi_id"), slot.get("place_name"))
    return ids, names


def _apply_confirm_day(
    session: dict[str, Any],
    previous_itinerary: list[dict[str, Any]],
    confirm_day: dict[str, Any],
) -> dict[str, Any] | None:
    """'이 날만 다시' 대안 카드 확정을 LLM/정규식 재해석 없이 직접 반영한다.

    day_options는 이미 구조화된 slots를 갖고 있으므로, 사용자가 대안을 고르면
    그 slots를 그대로 previous_itinerary에 병합한다 (자연어로 다시 만들지 않음).
    """
    try:
        day_num = int(confirm_day.get("day"))
    except (TypeError, ValueError):
        return None
    slots = confirm_day.get("slots")
    if not day_num or not isinstance(slots, list) or not slots:
        return None

    new_day = {"day": day_num, "theme": confirm_day.get("theme") or "", "slots": slots}
    itinerary = _merge_day_into_itinerary(previous_itinerary, day_num, new_day)
    itinerary = _dedupe_itinerary_slots(itinerary)
    return {
        "question": "",
        "answer": f"{day_num}일차를 선택하신 대안으로 확정했어요.\n\n" + _simple_itinerary_answer(itinerary),
        "destination": session.get("destination"),
        "days": session.get("days"),
        "budget": session.get("budget"),
        "preferences": session.get("preferences") or [],
        "language": session.get("language") or "ko",
        "start_date": session.get("start_date"),
        "itinerary": itinerary,
        "itineraries": [],
        "accommodations": [],
        "itinerary_count": None,
        "cities": [],
        "recommended_places": [],
        "day_options": [],
        "intent": "edit",
        "places": [],
        "courses": [],
        "warnings": [],
        "sources": [],
        "rewrite_day": day_num,
        "target_day": day_num,
        "exclude_places": session.get("excluded_places") or [],
        "realtime": {},
        "external": {},
    }


def ask_travel(
    question: str,
    *,
    destination: str | None = None,
    days: int | None = None,
    budget: int | None = None,
    language: str | None = None,
    preferences: list[str] | None = None,
    history: str | None = None,
    rewrite_day: int | None = None,
    previous_itinerary: list[dict[str, Any]] | None = None,
    start_date: str | None = None,
    exclude_places: list[str] | None = None,
    shown_poi_ids: list[str] | None = None,
    shown_place_names: list[str] | None = None,
    last_target_day: int | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {"query": question, "history": history or ""}
    if destination:
        state["destination"] = destination
    if days is not None:
        state["days"] = days
    if budget is not None:
        state["budget"] = budget
    if language:
        state["language"] = language
    if preferences:
        state["preferences"] = preferences
    if rewrite_day is not None:
        state["rewrite_day"] = rewrite_day
    if previous_itinerary:
        state["previous_itinerary"] = previous_itinerary
    if start_date:
        state["start_date"] = start_date
    if exclude_places:
        state["exclude_places"] = exclude_places
    if shown_poi_ids is not None:
        state["shown_poi_ids"] = shown_poi_ids
    if shown_place_names is not None:
        state["shown_place_names"] = shown_place_names
    if last_target_day:
        state["last_target_day"] = last_target_day

    result = get_travel_agent().invoke(state)
    return {
        "question": question,
        "answer": result.get("answer") or "",
        "destination": result.get("destination"),
        "days": result.get("days"),
        "budget": result.get("budget"),
        "preferences": result.get("preferences") or [],
        "language": result.get("language") or "ko",
        "start_date": result.get("start_date"),
        "itinerary": result.get("itinerary") or [],
        "itineraries": result.get("itineraries") or [],
        "accommodations": result.get("accommodations") or [],
        "itinerary_count": result.get("itinerary_count"),
        "cities": result.get("cities") or [],
        "recommended_places": result.get("recommended_places") or [],
        "day_options": result.get("day_options") or [],
        "intent": result.get("intent") or "itinerary",
        "places": result.get("places") or [],
        "courses": result.get("courses") or [],
        "warnings": result.get("warnings") or [],
        "sources": result.get("sources") or [],
        "rewrite_day": result.get("rewrite_day"),
        "target_day": result.get("target_day"),
        "exclude_places": result.get("exclude_places") or exclude_places or [],
        "realtime": result.get("realtime") or {},
        "external": result.get("external") or {},
    }


def _next_last_target_day(result: dict[str, Any]) -> int | None:
    """다음 턴에서 참조할 '마지막으로 손댄 일차'를 계산한다.

    - 특정 일차만 다시 짜거나(rewrite_day) 그 일차에 장소를 넣었으면(edit+target_day) 그 값을 기록.
    - 완전히 새로운 전체 일정을 만들었으면 0을 반환해 명시적으로 초기화.
    - 그 외(메타/QA/장소 목록 등, 일정과 무관한 턴)는 None을 반환해 기존 값을 유지.
    """
    if result.get("rewrite_day"):
        return int(result["rewrite_day"])
    if result.get("intent") == "edit" and result.get("target_day"):
        return int(result["target_day"])
    if result.get("intent") == "itinerary" and not result.get("rewrite_day"):
        return 0
    return None


def _persist_session_result(session_id: str, question: str, result: dict[str, Any]) -> dict[str, Any]:
    store = get_session_store()
    session = store.get_session(session_id) or {}
    new_ids, new_names = _collect_shown_from_result(result)
    shown_ids = list(session.get("shown_poi_ids") or [])
    shown_names = list(session.get("shown_place_names") or [])
    for i in new_ids:
        if i not in shown_ids:
            shown_ids.append(i)
    for n in new_names:
        if n not in shown_names:
            shown_names.append(n)
    excl = list(session.get("excluded_places") or [])
    for n in result.get("exclude_places") or []:
        if n and n not in excl:
            excl.append(n)

    store.add_message(
        session_id,
        "assistant",
        result["answer"],
        payload={
            "destination": result.get("destination"),
            "days": result.get("days"),
            "budget": result.get("budget"),
            "preferences": result.get("preferences"),
            "start_date": result.get("start_date"),
            "itinerary": result.get("itinerary"),
            "itineraries": result.get("itineraries"),
            "accommodations": result.get("accommodations"),
            "itinerary_count": result.get("itinerary_count"),
            "cities": result.get("cities"),
            "recommended_places": result.get("recommended_places"),
            "day_options": result.get("day_options"),
            "intent": result.get("intent"),
            "warnings": result.get("warnings"),
            "sources": result.get("sources"),
            "realtime": result.get("realtime"),
            "external": result.get("external"),
        },
    )
    updated = store.update_session_meta(
        session_id,
        destination=result.get("destination"),
        days=result.get("days"),
        budget=result.get("budget"),
        language=result.get("language"),
        preferences=result.get("preferences") or [],
        start_date=result.get("start_date"),
        excluded_places=excl,
        shown_poi_ids=shown_ids,
        shown_place_names=shown_names,
        last_target_day=_next_last_target_day(result),
    )
    result["session_id"] = session_id
    result["session"] = updated
    return result


def ask_travel_in_session(
    session_id: str,
    question: str,
    *,
    destination: str | None = None,
    days: int | None = None,
    budget: int | None = None,
    language: str | None = None,
    preferences: list[str] | None = None,
    rewrite_day: int | None = None,
    start_date: str | None = None,
    confirm_day: dict[str, Any] | None = None,
) -> dict[str, Any]:
    store = get_session_store()
    session = store.get_session(session_id)
    if not session:
        raise KeyError(f"session not found: {session_id}")

    history = store.recent_history_text(session_id)
    previous_itinerary = _latest_itinerary(session_id)
    store.add_message(session_id, "user", question)

    if confirm_day and previous_itinerary:
        applied = _apply_confirm_day(session, previous_itinerary, confirm_day)
        if applied is not None:
            applied["question"] = question
            return _persist_session_result(session_id, question, applied)

    result = ask_travel(
        question,
        destination=destination or session.get("destination"),
        days=days if days is not None else session.get("days"),
        budget=budget if budget is not None else session.get("budget"),
        language=language or session.get("language"),
        preferences=preferences or session.get("preferences") or None,
        history=history,
        rewrite_day=rewrite_day,
        previous_itinerary=previous_itinerary or None,
        start_date=start_date or session.get("start_date"),
        exclude_places=session.get("excluded_places") or [],
        shown_poi_ids=session.get("shown_poi_ids") or [],
        shown_place_names=session.get("shown_place_names") or [],
        last_target_day=session.get("last_target_day"),
    )
    return _persist_session_result(session_id, question, result)


def stream_travel_in_session(
    session_id: str,
    question: str,
    *,
    destination: str | None = None,
    days: int | None = None,
    budget: int | None = None,
    language: str | None = None,
    preferences: list[str] | None = None,
    rewrite_day: int | None = None,
    start_date: str | None = None,
    confirm_day: dict[str, Any] | None = None,
) -> Iterator[tuple[str, Any]]:
    store = get_session_store()
    session = store.get_session(session_id)
    if not session:
        raise KeyError(f"session not found: {session_id}")

    history = store.recent_history_text(session_id)
    previous_itinerary = _latest_itinerary(session_id)
    store.add_message(session_id, "user", question)

    if confirm_day and previous_itinerary:
        applied = _apply_confirm_day(session, previous_itinerary, confirm_day)
        if applied is not None:
            applied["question"] = question
            persisted = _persist_session_result(session_id, question, applied)
            yield ("status", {"stage": "writing"})
            yield ("token", {"text": persisted.get("answer") or ""})
            yield ("done", persisted)
            return

    state: dict[str, Any] = {
        "query": question,
        "history": history or "",
        "destination": destination or session.get("destination"),
        "days": days if days is not None else session.get("days"),
        "budget": budget if budget is not None else session.get("budget"),
        "language": language or session.get("language") or "ko",
        "preferences": preferences or session.get("preferences") or [],
        "start_date": start_date or session.get("start_date"),
        "exclude_places": session.get("excluded_places") or [],
        "shown_poi_ids": session.get("shown_poi_ids") or [],
        "shown_place_names": session.get("shown_place_names") or [],
        "last_target_day": session.get("last_target_day"),
    }
    if rewrite_day is not None:
        state["rewrite_day"] = rewrite_day
    if previous_itinerary:
        state["previous_itinerary"] = previous_itinerary

    for event_type, payload in stream_build_itinerary_tokens(get_search_service(), state):
        if event_type == "done":
            result = {
                "question": question,
                **payload,
            }
            persisted = _persist_session_result(session_id, question, result)
            yield ("done", persisted)
        else:
            yield (event_type, payload)
