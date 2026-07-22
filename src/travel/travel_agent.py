"""Facade for the travel planning agent."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.travel.session_store import TravelSessionStore
from src.travel.travel_graph import build_travel_graph
from src.travel.travel_tools import TravelSearchService


@lru_cache
def get_travel_agent():
    search = TravelSearchService()
    return build_travel_graph(search)


@lru_cache
def get_session_store() -> TravelSessionStore:
    return TravelSessionStore()


def ask_travel(
    question: str,
    *,
    destination: str | None = None,
    days: int | None = None,
    budget: int | None = None,
    language: str | None = None,
    preferences: list[str] | None = None,
    history: str | None = None,
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
        "places": result.get("places") or [],
        "courses": result.get("courses") or [],
        "warnings": result.get("warnings") or [],
        "sources": result.get("sources") or [],
    }


def ask_travel_in_session(
    session_id: str,
    question: str,
    *,
    destination: str | None = None,
    days: int | None = None,
    budget: int | None = None,
    language: str | None = None,
    preferences: list[str] | None = None,
) -> dict[str, Any]:
    store = get_session_store()
    session = store.get_session(session_id)
    if not session:
        raise KeyError(f"session not found: {session_id}")

    history = store.recent_history_text(session_id)
    store.add_message(session_id, "user", question)

    result = ask_travel(
        question,
        destination=destination or session.get("destination"),
        days=days if days is not None else session.get("days"),
        budget=budget if budget is not None else session.get("budget"),
        language=language or session.get("language"),
        preferences=preferences or session.get("preferences") or None,
        history=history,
    )

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
            "warnings": result.get("warnings"),
            "sources": result.get("sources"),
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
