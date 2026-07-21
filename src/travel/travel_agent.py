"""Facade for the travel planning agent."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.travel.travel_graph import build_travel_graph
from src.travel.travel_tools import TravelSearchService


@lru_cache
def get_travel_agent():
    search = TravelSearchService()
    return build_travel_graph(search)


def ask_travel(
    question: str,
    *,
    destination: str | None = None,
    days: int | None = None,
    budget: int | None = None,
    language: str | None = None,
    preferences: list[str] | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {"query": question}
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
