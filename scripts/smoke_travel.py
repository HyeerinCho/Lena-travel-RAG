#!/usr/bin/env python3
"""Smoke-test the travel agent with a few representative queries."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.travel.travel_agent import ask_travel  # noqa: E402


CASES = [
    {
        "question": "제주 2박3일, 자연 경관 위주, 예산 50만원",
        "destination": "제주",
        "days": 3,
        "budget": 500_000,
        "language": "ko",
    },
    {
        "question": "서울에서 비 오는 날 실내 문화시설 추천해줘",
        "destination": "서울",
        "days": 1,
        "language": "ko",
    },
    {
        "question": "Family-friendly places in Busan for 2 days",
        "destination": "부산",
        "days": 2,
        "language": "en",
    },
]


def main() -> None:
    for i, case in enumerate(CASES, 1):
        print("=" * 60)
        print(f"CASE {i}: {case['question']}")
        result = ask_travel(**case)
        assert result.get("answer"), "empty answer"
        print(f"places={len(result.get('places') or [])} "
              f"courses={len(result.get('courses') or [])} "
              f"itinerary_days={len(result.get('itinerary') or [])}")
        print((result.get("answer") or "")[:500])
    print("SMOKE OK")


if __name__ == "__main__":
    main()
