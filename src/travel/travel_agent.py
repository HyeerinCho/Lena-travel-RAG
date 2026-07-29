"""Facade for the travel planning agent."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterator

from src.travel.session_store import TravelSessionStore
from src.travel.travel_graph import build_travel_graph, stream_build_itinerary_tokens
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

    result = get_travel_agent().invoke(state)
    return {
        "question": question,
        "answer": result.get("answer") or "",
        "destination": result.get("destination"),
        "days": result.get("days"),
        "budget": result.get("budget"),
        "preferences": result.get("preferences") or [],
        "language": result.get("language") or "ko",
        "itinerary": result.get("itinerary") or [],
        "itineraries": result.get("itineraries") or [],
        "itinerary_count": result.get("itinerary_count"),
        "cities": result.get("cities") or [],
        "intent": result.get("intent") or "itinerary",
        "places": result.get("places") or [],
        "courses": result.get("courses") or [],
        "warnings": result.get("warnings") or [],
        "sources": result.get("sources") or [],
        "rewrite_day": result.get("rewrite_day"),
        "realtime": result.get("realtime") or {},
        "external": result.get("external") or {},
    }


def _persist_session_result(session_id: str, question: str, result: dict[str, Any]) -> dict[str, Any]:
    store = get_session_store()
    store.add_message(
        session_id,
        "assistant",
        result["answer"],
        payload={
            "destination": result.get("destination"),
            "days": result.get("days"),
            "budget": result.get("budget"),
            "preferences": result.get("preferences"),
            "itinerary": result.get("itinerary"),
            "itineraries": result.get("itineraries"),
            "itinerary_count": result.get("itinerary_count"),
            "cities": result.get("cities"),
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
) -> dict[str, Any]:
    store = get_session_store()
    session = store.get_session(session_id)
    if not session:
        raise KeyError(f"session not found: {session_id}")

    history = store.recent_history_text(session_id)
    previous_itinerary = _latest_itinerary(session_id)
    store.add_message(session_id, "user", question)

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
) -> Iterator[tuple[str, Any]]:
    store = get_session_store()
    session = store.get_session(session_id)
    if not session:
        raise KeyError(f"session not found: {session_id}")

    history = store.recent_history_text(session_id)
    previous_itinerary = _latest_itinerary(session_id)
    store.add_message(session_id, "user", question)

    state: dict[str, Any] = {
        "query": question,
        "history": history or "",
        "destination": destination or session.get("destination"),
        "days": days if days is not None else session.get("days"),
        "budget": budget if budget is not None else session.get("budget"),
        "language": language or session.get("language") or "ko",
        "preferences": preferences or session.get("preferences") or [],
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
