"""LangGraph for conversational Korea travel planning."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from src.config import LLM_MODEL
from src.prompts import (
    TRAVEL_CITY_LIST_PROMPT,
    TRAVEL_EDIT_PROMPT,
    TRAVEL_EXTRACT_PROMPT,
    TRAVEL_ITINERARY_PROMPT,
    TRAVEL_META_PROMPT,
    TRAVEL_MULTI_ITINERARY_PROMPT,
    TRAVEL_PLACE_LIST_PROMPT,
    TRAVEL_QA_PROMPT,
    TRAVEL_REWRITE_DAY_PROMPT,
)
from src.travel.tour_enrich import build_tour_context
from src.travel.travel_realtime import build_realtime_context
from src.travel.travel_repository import city_match_values
from src.travel.travel_tools import TravelSearchService

MAX_ITINERARIES = 6
KST = timezone(timedelta(hours=9))
META_FALLBACK_KO = (
    "안녕하세요! 저는 LENA예요. 한국 여행 장소와 일정을 추천해 드리는 여행 플래너입니다. "
    "가고 싶은 지역·관심사·기간을 알려주시면 장소를 고르거나 일정을 짜 드릴 수 있어요."
)


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
    stays: list[dict[str, Any]]
    accommodations: list[dict[str, Any]]
    itinerary: list[dict[str, Any]]
    warnings: list[str]
    sources: list[dict[str, Any]]
    answer: str
    rewrite_day: int | None
    previous_itinerary: list[dict[str, Any]]
    intent: str
    cities: list[dict[str, Any]]
    exclude_cities: list[str]
    exclude_places: list[str]
    shown_poi_ids: list[str]
    shown_place_names: list[str]
    realtime: dict[str, Any]
    external: dict[str, Any]
    external_text: str
    itinerary_count: int | None
    itineraries: list[dict[str, Any]]
    start_date: str | None
    insert_place: str | None
    target_day: int | None
    want_alternatives: bool
    recommended_places: list[dict[str, Any]]
    day_options: list[dict[str, Any]]


_ORDINAL_KO = {
    "첫": 1, "첫째": 1, "두": 2, "둘째": 2, "세": 3, "셋째": 3,
    "네": 4, "넷째": 4, "다섯": 5, "다섯째": 5, "여섯": 6, "여섯째": 6,
    "일곱": 7, "일곱째": 7,
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


def _heuristic_target_day(query: str) -> int | None:
    m = re.search(r"(\d+)\s*일차\s*에", query)
    if m:
        return int(m.group(1))
    for word, day in _ORDINAL_KO.items():
        if re.search(rf"{word}\s*날\s*에", query):
            return day
    return None


def _heuristic_insert_place(query: str) -> str | None:
    m = re.search(
        r"(.+?)\s*(을|를)\s*(?:(?:\d+)\s*일차|첫째\s*날|둘째\s*날|셋째\s*날|[^\s]*\s*날)"
        r"\s*에\s*(?:넣어|추가|배치|넣)",
        query,
    )
    if m:
        name = m.group(1).strip(" \"'「」『』")
        name = re.sub(r"^(그리고|또|그럼|그다음에?)\s*", "", name)
        if 1 < len(name) <= 40:
            return name
    m = re.search(r"(.+?)\s*(을|를)\s*(넣어|추가해|배치해)", query)
    if m:
        name = m.group(1).strip(" \"'「」『』")
        if 1 < len(name) <= 40 and "일정" not in name:
            return name
    return None


def _heuristic_exclude_places(query: str) -> list[str]:
    out: list[str] = []
    patterns = [
        r"(.+?)\s*(싫어|별로|맘에\s*안|안\s*좋아)",
        r"(.+?)\s*(빼고|제외|말고|빼줘|빼\s*줘)",
        r"일정\s*속\s*(.+?)\s*(싫어|빼고|제외)",
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, query):
            name = m.group(1).strip(" \"'「」『』,.!?")
            name = re.sub(r"^(일정|코스|여행|그|저|이)\s*(속|안|에서)?\s*", "", name)
            name = re.sub(r"\s*(은|는|이|가|을|를)$", "", name)
            if 1 < len(name) <= 40 and name not in out:
                # Avoid catching whole sentences
                if " " in name and len(name) > 20:
                    continue
                out.append(name)
    return out


def _heuristic_start_date(query: str, *, now: datetime | None = None) -> str | None:
    now = now or datetime.now(KST)
    m = re.search(r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})", query)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).strftime("%Y-%m-%d")
        except ValueError:
            pass
    m = re.search(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", query)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        year = now.year
        try:
            dt = datetime(year, month, day, tzinfo=KST)
            if dt.date() < now.date() and (now.month - month) > 6:
                dt = datetime(year + 1, month, day, tzinfo=KST)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
    if re.search(r"내일부터|내일\s*출발", query):
        return (now + timedelta(days=1)).strftime("%Y-%m-%d")
    if re.search(r"모레부터|모레\s*출발", query):
        return (now + timedelta(days=2)).strftime("%Y-%m-%d")
    return None


def _capped_count(count: int | None) -> int:
    if not count or count < 1:
        return 1
    return min(count, MAX_ITINERARIES)


def _place_limit_for(state: "TravelState") -> int:
    if state.get("intent") not in ("multi_itinerary", "place_list"):
        return 18 if state.get("want_alternatives") else 12
    if state.get("intent") == "place_list":
        return 24
    capped = _capped_count(state.get("itinerary_count"))
    days = state.get("days") or 1
    return max(18, min(60, capped * max(3, days) * 3))


def _first_valid_count(*values: Any) -> int | None:
    for value in values:
        try:
            n = int(value)
        except (TypeError, ValueError):
            continue
        if n >= 2:
            return n
    return None


def _heuristic_itinerary_count(query: str) -> int | None:
    plan_word = r"(?:일정|코스|플랜|플렌|루트|여행|안|버전)"
    patterns = [
        rf"(\d+)\s*(?:개|가지|종류)\s*(?:의|짜리)?\s*{plan_word}",
        rf"{plan_word}\s*(?:를|을)?\s*(\d+)\s*(?:개|가지|종류)",
        r"(\d+)\s*군데",
        rf"(\d+)\s*(?:different\s*)?(?:itinerar(?:y|ies)|plans?|courses?|routes?|options?)",
    ]
    for pattern in patterns:
        m = re.search(pattern, query, re.I)
        if m:
            try:
                n = int(m.group(1))
            except (TypeError, ValueError):
                continue
            if n >= 2:
                return n
    return None


def _detect_meta_intent(query: str) -> bool:
    patterns = [
        r"뭐\s*하는",
        r"뭔\s*데|뭔데|뭐야|무엇입니까|무엇인가요",
        r"자기\s*소개",
        r"누구야|누구세요|너\s*는\s*누구",
        r"이\s*(서비스|앱|봇|채팅|챗봇|모델|건)",
        r"소개\s*해",
        r"how\s+do\s+you\s+work|what\s+are\s+you|who\s+are\s+you",
        r"lena\s*(가|이|은|는)?\s*뭐",
        r"할\s*수\s*있는\s*일|기능\s*이\s*뭐|어떻게\s*써|사용\s*법",
    ]
    return any(re.search(p, query, re.I) for p in patterns)


def _detect_city_list_intent(query: str) -> bool:
    if re.search(r"일정|코스|며칠|\d+\s*박|\d+\s*일차|day\s*\d", query, re.I):
        return False
    patterns = [
        r"도시\s*(들|만|목록|리스트)",
        r"(도시|지역).{0,6}(뽑아|추천|골라|알려|정리)",
        r"(뽑아|추천|골라|알려|정리).{0,6}(도시|지역)",
        r"(리스트|목록)\s*로\s*(뽑|만들|정리)",
        r"어느\s*도시|어떤\s*도시",
        r"city\s*list|list\s*of\s*cities",
    ]
    return any(re.search(p, query, re.I) for p in patterns)


def _detect_place_list_intent(query: str) -> bool:
    # Explicit plan request wins over place-list.
    if re.search(
        r"일정|코스|며칠|\d+\s*박|\d+\s*일\s*차|day\s*\d|"
        r"짜\s*줘|짜줘|만들어\s*줘|세워\s*줘|플랜\s*짜",
        query,
        re.I,
    ):
        return False
    if re.search(r"\d+\s*일(?:\s*일정|\s*코스|\s*여행)?", query) and re.search(
        r"짜|만들|세워|추천", query
    ):
        # "3일 일정 추천" is itinerary; bare "3일" may still be place.
        if re.search(r"일정|코스|여행", query):
            return False
    patterns = [
        r"여행\s*가\s*기\s*좋",
        r"(여행지|장소|곳|명소|관광지).{0,6}(추천|알려|골라|어디)",
        r"(추천|알려|골라).{0,6}(여행지|장소|곳|명소)",
        r"어디\s*(갈|가)\s*(까|지|면)",
        r"가\s*볼\s*만",
        r"다른\s*(곳|장소|데)",
        r"place\s*recommend|where\s+to\s+go",
    ]
    return any(re.search(p, query, re.I) for p in patterns)


def _detect_want_alternatives(query: str) -> bool:
    return bool(
        re.search(
            r"다른\s*(곳|장소|데|코스|일정)|재추천|또\s*없|없어\s*\?|없을까|말고\s*다른",
            query,
        )
    )


def _detect_edit_intent(query: str, previous: list[dict[str, Any]] | None) -> bool:
    if not previous:
        return False
    if re.search(r"(\d+)\s*일차\s*에\s*(넣어|추가|배치|넣)", query):
        return True
    if _heuristic_insert_place(query) and re.search(r"넣어|추가|배치", query):
        return True
    if re.search(r"(빼줘|빼\s*줘|제외해|삭제해|지워)", query) and _heuristic_exclude_places(query):
        return True
    if re.search(r"바꿔\s*줘|교체|대신", query) and previous:
        # Whole rewrite day is separate
        if _heuristic_rewrite_day(query) is not None:
            return False
        return True
    return False


_ITINERARY_REQUEST_HINTS = (
    "짜줘", "짜 줘", "짜주", "짜볼", "짜 볼", "만들어", "만들어줘", "세워줘",
    "세워 줘", "다시 짜", "새로 짜", "다시 만들", "새로 만들", "바꿔", "변경해",
    "변경하", "수정해", "수정하", "추가해", "추가로 넣", "넣어줘", "빼줘", "빼 줘",
    "빼고 짜", "교체", "제외해", "recommend", "plan a", "make me", "build",
)
# "추천해" alone is NOT itinerary — used by place_list
_ITINERARY_PLAN_HINTS = (
    "일정", "코스", "플랜", "N박", "박", "일 여행",
)

_QA_HINTS = (
    "가능해", "가능한가", "가능할까", "가능한지", "가능?", "가능한가요", "되나",
    "되나요", "될까", "돼?", "돼요", "되니", "할 수 있", "해도 돼", "해도 되",
    "괜찮을까", "괜찮나", "괜찮아", "무리일까", "무리인가", "무리 아닐까",
    "맞아?", "맞나요", "인가요", "일까?", "가도 돼", "가도 되", "없이도",
    "충분해", "충분할까", "충분한가", "포함돼", "포함되나", "들어가나",
    "있나요", "있어?", "어려울까", "어렵나", "고마워", "감사", "충분?",
    "설명해", "어떻게 해",
)


def _detect_qa_intent(query: str) -> bool:
    q = query or ""
    if _heuristic_rewrite_day(q) is not None:
        return False
    if _detect_meta_intent(q):
        return False
    if any(h in q for h in _ITINERARY_REQUEST_HINTS):
        return False
    return any(h in q for h in _QA_HINTS)


def _resolve_intent(
    base_intent: str | None,
    parsed_intent: str | None,
    count: int | None,
    rewrite_day: int | None,
    *,
    has_previous: bool = False,
) -> str:
    intents = (base_intent, parsed_intent)
    if "meta" in intents:
        return "meta"
    if "city_list" in intents:
        return "city_list"
    if "place_list" in intents:
        return "place_list"
    if count:
        return "multi_itinerary"
    if "edit" in intents and has_previous:
        return "edit"
    if rewrite_day is not None:
        return "itinerary"
    if "qa" in intents:
        return "qa"
    return "itinerary"


def _heuristic_extract(query: str) -> dict[str, Any]:
    language = (
        "en"
        if re.search(r"[A-Za-z]", query) and not re.search(r"[가-힣]", query)
        else "ko"
    )
    days = None
    if re.search(r"당일치기|당일\s*여행|day\s*trip", query, re.I):
        days = 1
    else:
        m = re.search(r"(\d+)\s*박\s*(\d+)\s*일", query)
        if m:
            days = int(m.group(2))
        else:
            m = re.search(r"(\d+)\s*일", query)
            if m and re.search(r"일정|코스|여행|일정", query):
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
        ("제주", "제주"), ("서울", "서울"), ("부산", "부산"), ("강릉", "강릉"),
        ("전주", "전주"), ("경주", "경주"), ("인천", "인천"), ("대구", "대구"),
        ("광주", "광주"), ("여수", "여수"), ("Jeju", "제주"), ("Seoul", "서울"),
        ("Busan", "부산"),
    ]
    lower = query.lower()
    exclude_cities: list[str] = []
    destination = None
    neg = re.search(r"(말고|빼고|제외|이외|except|other\s+than)", query, re.I)
    if neg:
        neg_pos = neg.start()
        for needle, canon in city_aliases:
            idx = lower.find(needle.lower())
            if idx == -1:
                continue
            if idx < neg_pos:
                if canon not in exclude_cities:
                    exclude_cities.append(canon)
            elif destination is None:
                destination = canon
    else:
        for needle, canon in city_aliases:
            if needle.lower() in lower:
                destination = canon
                break

    preferences = []
    pref_map = {
        "자연": "자연", "경관": "경관", "실내": "실내", "문화": "문화",
        "가족": "가족", "맛집": "음식", "음식": "음식", "family": "family",
        "nature": "nature", "museum": "culture", "indoor": "indoor",
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

    if _detect_meta_intent(query):
        intent = "meta"
    elif _detect_city_list_intent(query):
        intent = "city_list"
    elif _detect_place_list_intent(query) or _detect_want_alternatives(query):
        intent = "place_list"
    elif _detect_qa_intent(query):
        intent = "qa"
    else:
        intent = "itinerary"

    return {
        "destination": destination,
        "days": days,
        "budget": budget,
        "preferences": preferences,
        "language": language,
        "place_types": place_types,
        "rewrite_day": _heuristic_rewrite_day(query),
        "intent": intent,
        "exclude_cities": exclude_cities,
        "exclude_places": _heuristic_exclude_places(query),
        "itinerary_count": _heuristic_itinerary_count(query),
        "start_date": _heuristic_start_date(query),
        "insert_place": _heuristic_insert_place(query),
        "target_day": _heuristic_target_day(query),
        "want_alternatives": _detect_want_alternatives(query),
    }


def _drop_excluded(
    places: list[dict[str, Any]],
    exclude_cities: list[str] | None,
) -> list[dict[str, Any]]:
    if not exclude_cities:
        return places
    needles: set[str] = set()
    for city in exclude_cities:
        for alias in city_match_values(city):
            alias = alias.strip()
            if alias:
                needles.add(alias)
                needles.add(alias.lower())
    if not needles:
        return places
    out: list[dict[str, Any]] = []
    for p in places:
        blob = " ".join(str(p.get(k) or "") for k in ("city", "region", "address_ko"))
        blob_lower = blob.lower()
        if any(n in blob or n in blob_lower for n in needles):
            continue
        out.append(p)
    return out


def _norm_name(name: Any) -> str:
    return re.sub(r"\s+", "", str(name or "")).lower()


def _drop_excluded_places(
    places: list[dict[str, Any]],
    exclude_places: list[str] | None,
    shown_poi_ids: list[str] | None = None,
    shown_place_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    ban_names = {_norm_name(n) for n in (exclude_places or []) if n}
    ban_names |= {_norm_name(n) for n in (shown_place_names or []) if n}
    ban_ids = {str(i) for i in (shown_poi_ids or []) if i}
    if not ban_names and not ban_ids:
        return places
    out: list[dict[str, Any]] = []
    for p in places:
        pid = str(p.get("poi_id") or "")
        name = _norm_name(p.get("name_ko") or p.get("name_en") or p.get("place_name"))
        if pid and pid in ban_ids:
            continue
        if name and any(b and (b in name or name in b) for b in ban_names):
            continue
        out.append(p)
    return out


def _places_from_itinerary(itinerary: list[dict[str, Any]] | None) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    names: list[str] = []
    for day in itinerary or []:
        for slot in day.get("slots") or []:
            if not isinstance(slot, dict):
                continue
            if slot.get("poi_id"):
                ids.append(str(slot["poi_id"]))
            if slot.get("place_name"):
                names.append(str(slot["place_name"]))
    return ids, names


def _dedupe_itinerary_slots(
    itinerary: list[dict[str, Any]],
    candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    spare = list(candidates or [])
    spare_i = 0
    result: list[dict[str, Any]] = []
    for day in itinerary:
        new_slots = []
        for slot in day.get("slots") or []:
            if not isinstance(slot, dict):
                continue
            key = str(slot.get("poi_id") or "") or _norm_name(slot.get("place_name"))
            if key and key in seen:
                replaced = False
                while spare_i < len(spare):
                    cand = spare[spare_i]
                    spare_i += 1
                    ckey = str(cand.get("poi_id") or "") or _norm_name(
                        cand.get("name_ko") or cand.get("name_en")
                    )
                    if ckey in seen:
                        continue
                    seen.add(ckey)
                    new_slots.append(
                        {
                            "time": slot.get("time"),
                            "place_name": cand.get("name_ko") or cand.get("name_en"),
                            "poi_id": cand.get("poi_id"),
                            "note": slot.get("note") or cand.get("travel_type") or "",
                        }
                    )
                    replaced = True
                    break
                if not replaced:
                    continue
            else:
                if key:
                    seen.add(key)
                new_slots.append(slot)
        result.append({**day, "slots": new_slots})
    return result


def _patch_place_into_day(
    previous: list[dict[str, Any]],
    day_num: int,
    place: dict[str, Any],
    *,
    time_slot: str | None = None,
) -> list[dict[str, Any]]:
    merged = [dict(d) for d in previous]
    name = place.get("name_ko") or place.get("name_en") or place.get("place_name")
    slot_obj = {
        "time": time_slot or "afternoon",
        "place_name": name,
        "poi_id": place.get("poi_id"),
        "note": place.get("travel_type") or place.get("note") or "추가",
    }
    for i, day in enumerate(merged):
        if int(day.get("day") or 0) != day_num:
            continue
        slots = list(day.get("slots") or [])
        times = [s.get("time") for s in slots]
        prefer = time_slot or next(
            (t for t in ("morning", "afternoon", "evening") if t not in times),
            "afternoon",
        )
        slot_obj["time"] = prefer
        # Replace existing slot with same time, or append.
        replaced = False
        for j, s in enumerate(slots):
            if s.get("time") == prefer:
                slots[j] = slot_obj
                replaced = True
                break
        if not replaced:
            slots.append(slot_obj)
        order = {"morning": 0, "afternoon": 1, "evening": 2}
        slots.sort(key=lambda s: order.get(s.get("time") or "", 9))
        merged[i] = {**day, "slots": slots}
        return merged
    # Day missing — append
    merged.append({"day": day_num, "theme": "", "slots": [slot_obj]})
    merged.sort(key=lambda d: int(d.get("day") or 0))
    return merged


def _remove_places_from_itinerary(
    previous: list[dict[str, Any]],
    ban_names: list[str],
) -> list[dict[str, Any]]:
    bans = {_norm_name(n) for n in ban_names if n}
    if not bans:
        return previous
    result = []
    for day in previous:
        slots = []
        for s in day.get("slots") or []:
            name = _norm_name(s.get("place_name"))
            if any(b and (b in name or name in b) for b in bans):
                continue
            slots.append(s)
        result.append({**day, "slots": slots})
    return result


def _focus_single_region(
    places: list[dict[str, Any]],
    destination: str | None,
) -> tuple[list[dict[str, Any]], str | None]:
    if not places:
        return places, destination

    def _region_of(p: dict[str, Any]) -> str | None:
        value = p.get("region") or p.get("city")
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    if destination:
        matched = [
            p
            for p in places
            if destination
            in " ".join(str(p.get(k) or "") for k in ("region", "city", "address_ko"))
        ]
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
            grouped[city] = {"city": city, "region": p.get("region"), "places": []}
            order.append(city)
        bucket = grouped[city]["places"]
        if len(bucket) < max_per_city and all(
            existing.get("poi_id") != p.get("poi_id") for existing in bucket
        ):
            bucket.append({"place_name": name, "poi_id": p.get("poi_id")})
    return [grouped[c] for c in order[:max_cities]]


def _build_city_list(state: TravelState, parsed: dict[str, Any]) -> TravelState:
    grouped = state.get("cities") or []
    by_city = {g["city"]: g for g in grouped}
    cities: list[dict[str, Any]] = []
    for item in parsed.get("cities") or []:
        city = (item.get("city") or "").strip()
        source = by_city.get(city)
        if source is None:
            continue
        cities.append(
            {
                "city": city,
                "region": source.get("region"),
                "merit": item.get("merit") or "",
                "places": source.get("places") or [],
            }
        )
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


def _build_place_list(state: TravelState, parsed: dict[str, Any]) -> TravelState:
    candidates = state.get("places") or []
    by_id = {str(p.get("poi_id")): p for p in candidates if p.get("poi_id")}
    by_name = {
        _norm_name(p.get("name_ko") or p.get("name_en")): p for p in candidates
    }
    recommended: list[dict[str, Any]] = []
    for item in parsed.get("recommended_places") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("place_name") or "").strip()
        poi_id = item.get("poi_id")
        src = None
        if poi_id and str(poi_id) in by_id:
            src = by_id[str(poi_id)]
        elif name and _norm_name(name) in by_name:
            src = by_name[_norm_name(name)]
        if src is None and name:
            # keep name-only if LLM invented? drop if not in candidates when we have any
            if candidates:
                continue
        recommended.append(
            {
                "place_name": (src or {}).get("name_ko") or (src or {}).get("name_en") or name,
                "poi_id": (src or {}).get("poi_id") or poi_id,
                "merit": item.get("merit") or "",
                "city": (src or {}).get("city") or item.get("city"),
            }
        )
    if not recommended:
        for p in candidates[:8]:
            recommended.append(
                {
                    "place_name": p.get("name_ko") or p.get("name_en"),
                    "poi_id": p.get("poi_id"),
                    "merit": p.get("travel_type") or "",
                    "city": p.get("city"),
                }
            )
    warnings = list(state.get("warnings") or [])
    warnings.extend(parsed.get("warnings") or [])
    answer = parsed.get("answer")
    if not answer:
        lines = [f"{i}. {r['place_name']} — {r.get('merit') or ''}" for i, r in enumerate(recommended, 1)]
        lines.append("")
        lines.append(
            "원하시면 ① 이 중 한 곳으로 일정 짜기 ② 다른 장소 더 보기 ③ 조건 알려주기 중 골라 주세요."
        )
        answer = "\n".join(lines) or "추천할 장소를 찾지 못했습니다."
    return {
        "recommended_places": recommended,
        "cities": [],
        "itinerary": [],
        "itineraries": [],
        "warnings": warnings,
        "answer": answer,
        "sources": _build_sources(state),
        "intent": "place_list",
    }


_LODGING_TYPE = "숙박"
_LODGING_NAME_RE = re.compile(
    r"숙소|숙박|호텔|게스트\s*하우스|펜션|리조트|모텔|민박|호스텔|"
    r"\bhotel\b|\bpension\b|\bresort\b|\bmotel\b|\bhostel\b|\bguesthouse\b",
    re.I,
)


def _is_lodging(place: dict[str, Any]) -> bool:
    if place.get("travel_type") == _LODGING_TYPE:
        return True
    name = str(place.get("name_ko") or place.get("name_en") or place.get("place_name") or "")
    return bool(_LODGING_NAME_RE.search(name))


def _drop_lodging(places: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [p for p in places if not _is_lodging(p)]


def _external_experience_places(external: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not external:
        return []
    out: list[dict[str, Any]] = []
    for kind, label in (("activity", "액티비티·레포츠"), ("festival", "축제·행사")):
        for item in external.get(kind) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("title") or "").strip()
            if not name:
                continue
            row: dict[str, Any] = {
                "name_ko": name,
                "travel_type": label,
                "source": "관광공사 API",
            }
            addr = str(item.get("addr1") or "").strip()
            if addr:
                row["address_ko"] = addr
            start = str(item.get("eventstartdate") or "").strip()
            end = str(item.get("eventenddate") or "").strip()
            if start:
                period = start if not end or end == start else f"{start}~{end}"
                row["hours_ko"] = f"행사기간 {period}"
            out.append(row)
    return out


def _merge_experience_places(
    places: list[dict[str, Any]],
    external: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    extras = _external_experience_places(external)
    if not extras:
        return places
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for p in extras + list(places):
        key = str(p.get("name_ko") or p.get("name_en") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(p)
    return merged


def _compact_places(places: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    keys = (
        "poi_id", "name_ko", "name_en", "travel_type", "city", "region",
        "address_ko", "hours_ko", "closed_ko", "fee_ko", "source",
    )
    out = []
    for p in places[:limit]:
        out.append({k: p.get(k) for k in keys if p.get(k) is not None})
    return out


def _compact_courses(courses: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    keys = ("package_id", "title", "city", "duration_raw", "duration_days", "places", "price_ref", "source")
    out = []
    for c in courses[:limit]:
        item = {k: c.get(k) for k in keys if c.get(k) is not None}
        item["price_note"] = "과거 상품 참고 가격"
        out.append(item)
    return out


def _compact_stays(stays: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    keys = ("poi_id", "name_ko", "name_en", "travel_type", "city", "region", "address_ko")
    out = []
    for s in stays[:limit]:
        out.append({k: s.get(k) for k in keys if s.get(k) is not None})
    return out


def _sanitize_accommodations(
    raw_list: Any,
    stays: list[dict[str, Any]] | None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    candidates = stays or []
    valid_names = {
        str(s.get("name_ko") or s.get("name_en") or "").strip()
        for s in candidates
        if (s.get("name_ko") or s.get("name_en"))
    }
    valid_ids = {str(s.get("poi_id")) for s in candidates if s.get("poi_id")}
    out: list[dict[str, Any]] = []
    for item in raw_list or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("place_name") or "").strip()
        poi_id = item.get("poi_id")
        if not name:
            continue
        if valid_names and name not in valid_names and str(poi_id) not in valid_ids:
            continue
        out.append(
            {
                "place_name": name,
                "poi_id": poi_id,
                "area": item.get("area") or "",
                "note": item.get("note") or "",
            }
        )
        if len(out) >= limit:
            break
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
        return {"itinerary": [], "warnings": warnings, "answer": answer, "sources": []}

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
    blocks = []
    for day in itinerary:
        theme = day.get("theme")
        header = f"Day {day['day']} · {theme}" if theme else f"Day {day['day']}"
        block_lines = [header]
        for slot in day["slots"]:
            block_lines.append(f"- {slot['time']}: {slot['place_name']}")
        blocks.append("\n".join(block_lines))
    answer = "\n\n".join(blocks)
    sources = [
        {"type": "poi", "id": p.get("poi_id"), "name": p.get("name_ko")}
        for p in places[: idx or len(places)]
        if p.get("poi_id")
    ]
    return {"itinerary": itinerary, "warnings": warnings, "answer": answer, "sources": sources}


_SLOT_LABEL_KO = {"morning": "오전", "afternoon": "오후", "evening": "저녁"}


def _sanitize_itineraries(raw_list: Any, days: int, cap: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in raw_list or []:
        if not isinstance(item, dict):
            continue
        norm_days: list[dict[str, Any]] = []
        for d in item.get("days") or []:
            if not isinstance(d, dict):
                continue
            slots = []
            for s in d.get("slots") or []:
                if not isinstance(s, dict) or not s.get("place_name"):
                    continue
                if _is_lodging({"place_name": s.get("place_name"), "travel_type": s.get("note") or ""}):
                    continue
                slots.append(s)
            if not slots:
                continue
            norm_days.append(
                {
                    "day": int(d.get("day") or (len(norm_days) + 1)),
                    "theme": d.get("theme") or "",
                    "slots": slots,
                }
            )
        if not norm_days:
            continue
        out.append({"title": item.get("title") or f"추천 {len(out) + 1}", "days": norm_days})
        if len(out) >= cap:
            break
    return out


def _chunk_places_into_itineraries(
    places: list[dict[str, Any]], days: int, cap: int
) -> list[dict[str, Any]]:
    slots_order = ("morning", "afternoon", "evening")
    itineraries: list[dict[str, Any]] = []
    idx = 0
    for n in range(cap):
        day_objs: list[dict[str, Any]] = []
        for day in range(1, (days or 1) + 1):
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
            if slots:
                day_objs.append({"day": day, "theme": "", "slots": slots})
        if not day_objs:
            break
        itineraries.append({"title": f"코스 {n + 1}", "days": day_objs})
    return itineraries


def _multi_fallback_answer(itineraries: list[dict[str, Any]]) -> str:
    if not itineraries:
        return "조건에 맞는 일정을 만들지 못했습니다."
    blocks: list[str] = []
    for i, it in enumerate(itineraries, 1):
        title = it.get("title") or f"추천 {i}"
        lines = [f"추천 {i} · {title}"]
        multi_day = len(it.get("days") or []) > 1
        for day in it.get("days") or []:
            if multi_day:
                theme = day.get("theme") or ""
                lines.append(f"[Day {day['day']}]{f' {theme}' if theme else ''}")
            for slot in day.get("slots") or []:
                label = _SLOT_LABEL_KO.get(slot.get("time"), slot.get("time") or "")
                note = f" — {slot['note']}" if slot.get("note") else ""
                lines.append(f"- {label}: {slot.get('place_name', '')}{note}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _multi_count_warnings(
    requested: int, cap: int, actual: int, place_count: int, destination: str | None
) -> list[str]:
    msgs: list[str] = []
    region = destination or "해당 지역"
    if actual < cap:
        if place_count <= 0:
            msgs.append(f"{region}에 등록된 장소가 없어서 일정을 만들지 못했어요.")
        else:
            msgs.append(
                f"{region}에 등록된 장소가 {place_count}곳뿐이라 서로 다른 일정을 "
                f"{actual}개까지만 만들 수 있었어요. (요청: {requested}개)"
            )
    elif requested > MAX_ITINERARIES:
        msgs.append(
            f"일정은 한 번에 최대 {MAX_ITINERARIES}개까지만 추천해요. "
            f"요청하신 {requested}개는 너무 많아 {MAX_ITINERARIES}개로 정리했어요."
        )
    return msgs


def _build_multi_itinerary(
    state: TravelState, parsed: dict[str, Any], realtime: dict[str, Any]
) -> dict[str, Any]:
    requested = int(state.get("itinerary_count") or 0)
    cap = _capped_count(requested)
    days = state.get("days") or 1
    itineraries = _sanitize_itineraries(parsed.get("itineraries"), days, cap)
    answer = parsed.get("answer")
    if not itineraries:
        itineraries = _chunk_places_into_itineraries(state.get("places") or [], days, cap)
        answer = None
    place_count = len(state.get("places") or [])
    warnings = list(state.get("warnings") or [])
    warnings.extend(parsed.get("warnings") or [])
    for w in _multi_count_warnings(
        requested, cap, len(itineraries), place_count, state.get("destination")
    ):
        if w not in warnings:
            warnings.append(w)
    if not answer:
        answer = _multi_fallback_answer(itineraries)
    first_days = itineraries[0]["days"] if itineraries else []
    return {
        "itineraries": itineraries,
        "itinerary": first_days,
        "accommodations": [],
        "warnings": warnings,
        "answer": answer,
        "sources": _build_sources(state),
        "intent": "multi_itinerary",
        "itinerary_count": requested or None,
        "realtime": realtime,
    }


def _merge_day_into_itinerary(
    previous: list[dict[str, Any]], day_num: int, new_day: dict[str, Any]
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
        {"type": "course", "id": c.get("package_id"), "name": c.get("title")}
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


def _merge_lists(*lists: list[str] | None) -> list[str]:
    out: list[str] = []
    for lst in lists:
        for item in lst or []:
            if item and item not in out:
                out.append(item)
    return out


def _run_extract(state: TravelState, llm: ChatGoogleGenerativeAI) -> TravelState:
    base = _heuristic_extract(state["query"])
    history = state.get("history") or "(없음)"
    for key in (
        "destination", "days", "budget", "language", "preferences",
        "place_types", "rewrite_day", "start_date",
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
                "session_start_date": state.get("start_date") or base.get("start_date") or "(없음)",
            }
        )
        parsed = _extract_json(_llm_content(raw))
    except Exception:
        parsed = {}

    merged = {**base}
    for key in (
        "destination", "days", "budget", "language", "preferences",
        "place_types", "rewrite_day", "start_date", "insert_place", "target_day",
    ):
        if parsed.get(key) not in (None, [], ""):
            if state.get(key) in (None, [], "") or key in ("insert_place", "target_day", "rewrite_day"):
                merged[key] = parsed[key]

    for key in ("destination", "days", "budget", "language", "start_date"):
        if merged.get(key) in (None, "", []) and state.get(key) not in (None, "", []):
            merged[key] = state[key]
    if not merged.get("preferences") and state.get("preferences"):
        merged["preferences"] = state["preferences"]
    if state.get("rewrite_day") is not None:
        merged["rewrite_day"] = state["rewrite_day"]
    elif merged.get("rewrite_day") is None:
        merged["rewrite_day"] = _heuristic_rewrite_day(state["query"])

    if not merged.get("insert_place"):
        merged["insert_place"] = _heuristic_insert_place(state["query"])
    if merged.get("target_day") is None:
        merged["target_day"] = _heuristic_target_day(state["query"])
    if not merged.get("start_date"):
        merged["start_date"] = _heuristic_start_date(state["query"])

    count = _first_valid_count(parsed.get("itinerary_count"), base.get("itinerary_count"))
    merged["itinerary_count"] = count

    # edit detection after insert heuristics
    previous = list(state.get("previous_itinerary") or [])
    if merged.get("intent") != "meta":
        if _detect_edit_intent(state["query"], previous) or (
            previous and (merged.get("insert_place") or parsed.get("intent") == "edit")
        ):
            if not (merged.get("rewrite_day") and not merged.get("insert_place")):
                base["intent"] = "edit"
                if parsed.get("intent") != "meta":
                    parsed["intent"] = parsed.get("intent") or "edit"

    # place_list when wanting alternatives without explicit plan
    if (
        (base.get("want_alternatives") or merged.get("want_alternatives") or parsed.get("want_alternatives"))
        and not previous
        and merged.get("intent") not in ("meta", "city_list")
    ):
        base["intent"] = "place_list"

    if previous and _heuristic_exclude_places(state["query"]) and not _detect_edit_intent(
        state["query"], previous
    ):
        # "XX 싫어해" with existing itinerary → edit
        if not merged.get("rewrite_day"):
            base["intent"] = "edit"

    merged["want_alternatives"] = bool(
        base.get("want_alternatives")
        or parsed.get("want_alternatives")
        or state.get("want_alternatives")
    )

    merged["intent"] = _resolve_intent(
        base.get("intent"),
        parsed.get("intent"),
        count,
        merged.get("rewrite_day"),
        has_previous=bool(previous),
    )

    # Force meta if heuristic says so
    if _detect_meta_intent(state["query"]):
        merged["intent"] = "meta"

    # Force edit for insert into day
    if previous and merged.get("insert_place") and merged.get("target_day"):
        merged["intent"] = "edit"

    excluded = list(base.get("exclude_cities") or [])
    for c in parsed.get("exclude_cities") or []:
        if c and c not in excluded:
            excluded.append(c)
    merged["exclude_cities"] = excluded
    if merged.get("destination") and merged["destination"] in excluded:
        merged["destination"] = None

    excl_places = _merge_lists(
        state.get("exclude_places"),
        base.get("exclude_places"),
        parsed.get("exclude_places") if isinstance(parsed.get("exclude_places"), list) else None,
    )
    merged["exclude_places"] = excl_places

    if not merged.get("preferences"):
        merged["preferences"] = []
    if not merged.get("place_types"):
        merged["place_types"] = ["관광지", "문화시설"]
    if not merged.get("language"):
        merged["language"] = "ko"
    if "previous_itinerary" in state:
        merged["previous_itinerary"] = state["previous_itinerary"]
    if state.get("shown_poi_ids") is not None:
        merged["shown_poi_ids"] = list(state.get("shown_poi_ids") or [])
    if state.get("shown_place_names") is not None:
        merged["shown_place_names"] = list(state.get("shown_place_names") or [])
    return merged


def _search_for_state(state: TravelState, search: TravelSearchService) -> TravelState:
    prefs = " ".join(state.get("preferences") or [])
    query = " ".join(x for x in [state.get("query"), prefs] if x)
    destination = state.get("destination")
    types = state.get("place_types") or ["관광지", "문화시설"]
    exclude_cities = state.get("exclude_cities") or []
    # For alternatives: exclude previously shown so we get fresh POIs
    exclude_places = list(state.get("exclude_places") or [])
    shown_ids = list(state.get("shown_poi_ids") or [])
    shown_names = list(state.get("shown_place_names") or [])
    if state.get("want_alternatives") or state.get("intent") == "place_list":
        pass  # keep shown in filter
    elif state.get("intent") == "itinerary" and not state.get("rewrite_day"):
        # new full plan can reuse region places unless alternatives
        shown_ids = [] if not state.get("want_alternatives") else shown_ids
        shown_names = [] if not state.get("want_alternatives") else shown_names

    if state.get("intent") == "city_list":
        places = search.search_places(query=query, city=destination, types=types, limit=60)
        if len(places) < 12:
            places = search.search_places(query=query, city=destination, types=None, limit=60)
        places = _drop_excluded(places, exclude_cities)
        places = _drop_excluded_places(places, exclude_places, shown_ids, shown_names)
        cities = _group_places_by_city(places)
        warnings = [] if cities else ["조건에 맞는 도시를 충분히 찾지 못했습니다."]
        return {"places": places, "courses": [], "warnings": warnings, "cities": cities}

    place_limit = _place_limit_for(state)
    # For rewrite: gather more candidates and exclude used places
    prev = state.get("previous_itinerary") or []
    used_ids, used_names = _places_from_itinerary(prev)
    if state.get("rewrite_day") or state.get("intent") == "edit":
        exclude_places = _merge_lists(exclude_places, used_names)
        shown_ids = _merge_lists(shown_ids, used_ids)
        shown_names = _merge_lists(shown_names, used_names)
        place_limit = max(place_limit, 24)

    places = search.search_places(query=query, city=destination, types=types, limit=place_limit)
    if len(places) < 4 and destination:
        places = search.search_places(query=query, city=destination, types=None, limit=place_limit)

    places = _drop_excluded(places, exclude_cities)
    places = _drop_excluded_places(places, exclude_places, shown_ids, shown_names)
    places, focused_destination = _focus_single_region(places, destination)
    places = _drop_lodging(places)
    if focused_destination and not destination:
        destination = focused_destination

    courses = search.search_courses(
        city=destination,
        days=state.get("days"),
        budget=state.get("budget"),
        query=query,
        limit=8,
    )

    is_multi = state.get("intent") == "multi_itinerary"
    if is_multi or state.get("intent") == "place_list":
        stays: list[dict[str, Any]] = []
    else:
        stays = search.search_places(query=query, city=destination, types=["숙박"], limit=8)
        stays = _drop_excluded(stays, exclude_cities)

    warnings: list[str] = []
    if not places:
        warnings.append("조건에 맞는 POI가 충분하지 않습니다.")
    if state.get("budget") is not None and courses:
        warnings.append("코스 가격은 과거 상품 참고값입니다.")
    if (state.get("rewrite_day") or state.get("want_alternatives")) and len(places) < 3:
        warnings.append(
            f"대체로 쓸 수 있는 장소가 {len(places)}곳뿐이라 대안이 부족할 수 있어요."
        )

    result: TravelState = {
        "places": places,
        "courses": courses,
        "stays": stays,
        "warnings": warnings,
    }
    if focused_destination:
        result["destination"] = focused_destination

    if state.get("intent") not in ("place_list", "meta", "qa"):
        tour = build_tour_context(
            state.get("query") or "",
            destination,
            language=state.get("language") or "ko",
            limit=8 if is_multi else 5,
        )
        if tour:
            result["external"] = tour.get("data") or {}
            result["external_text"] = tour.get("text") or ""
            result["places"] = _merge_experience_places(places, result["external"])
            result["places"] = _drop_excluded_places(
                result["places"], exclude_places, shown_ids, shown_names
            )
    return result


def build_travel_graph(search: TravelSearchService):
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)

    def extract_requirements(state: TravelState) -> TravelState:
        return _run_extract(state, llm)

    def search_candidates(state: TravelState) -> TravelState:
        if state.get("intent") in ("meta", "qa"):
            return {}
        return _search_for_state(state, search)

    def build_itinerary(state: TravelState) -> TravelState:
        return _build_response(state, llm, search)

    graph = StateGraph(TravelState)
    graph.add_node("extract_requirements", extract_requirements)
    graph.add_node("search_candidates", search_candidates)
    graph.add_node("build_itinerary", build_itinerary)
    graph.set_entry_point("extract_requirements")
    graph.add_edge("extract_requirements", "search_candidates")
    graph.add_edge("search_candidates", "build_itinerary")
    graph.add_edge("build_itinerary", END)
    return graph.compile()


def _build_response(
    state: TravelState,
    llm: ChatGoogleGenerativeAI,
    search: TravelSearchService,
) -> TravelState:
    intent = state.get("intent") or "itinerary"
    language = state.get("language") or "ko"
    previous = list(state.get("previous_itinerary") or [])
    exclude_places = state.get("exclude_places") or []
    exclude_str = ", ".join(exclude_places) if exclude_places else "(없음)"

    if intent == "meta":
        try:
            raw = (TRAVEL_META_PROMPT | llm).invoke(
                {"question": state.get("query"), "language": language}
            )
            answer = _llm_content(raw).strip()
        except Exception:
            answer = ""
        return {
            "answer": answer or META_FALLBACK_KO,
            "itinerary": [],
            "itineraries": [],
            "cities": [],
            "recommended_places": [],
            "warnings": [],
            "sources": [],
            "intent": "meta",
        }

    if intent == "qa":
        try:
            raw = (TRAVEL_QA_PROMPT | llm).invoke(
                {
                    "question": state.get("query"),
                    "history": state.get("history") or "(없음)",
                    "language": language,
                    "previous_itinerary": (
                        json.dumps(previous, ensure_ascii=False) if previous else "(없음)"
                    ),
                }
            )
            answer = _llm_content(raw).strip()
        except Exception:
            answer = ""
        return {
            "answer": answer or "죄송해요, 답변을 만들지 못했어요. 다시 물어봐 주세요.",
            "itinerary": previous if previous else [],
            "itineraries": [],
            "cities": [],
            "warnings": [],
            "sources": [],
            "intent": "qa",
        }

    if intent == "city_list":
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
                    "language": language,
                    "candidates": json.dumps(grouped, ensure_ascii=False),
                }
            )
            parsed = _extract_json(_llm_content(raw))
        except Exception:
            parsed = {}
        return _build_city_list(state, parsed)

    if intent == "place_list":
        places = _compact_places(state.get("places") or [], limit=_place_limit_for(state))
        if not places:
            return {
                "recommended_places": [],
                "warnings": list(state.get("warnings") or []),
                "answer": "조건에 맞는 장소를 찾지 못했습니다. 지역이나 관심사를 알려주시면 다시 찾아볼게요.",
                "sources": [],
                "intent": "place_list",
                "itinerary": [],
            }
        try:
            raw = (TRAVEL_PLACE_LIST_PROMPT | llm).invoke(
                {
                    "question": state.get("query"),
                    "history": state.get("history") or "(없음)",
                    "preferences": ", ".join(state.get("preferences") or []),
                    "destination": state.get("destination"),
                    "language": language,
                    "exclude_places": exclude_str,
                    "candidates": json.dumps(places, ensure_ascii=False),
                }
            )
            parsed = _extract_json(_llm_content(raw))
        except Exception:
            parsed = {}
        return _build_place_list(state, parsed)

    places = _compact_places(state.get("places") or [], limit=_place_limit_for(state))
    courses = _compact_courses(state.get("courses") or [])
    stays = _compact_stays(state.get("stays") or [])
    stays_json = json.dumps(stays, ensure_ascii=False)
    rewrite_day = state.get("rewrite_day")
    weather_days = state.get("days") or (len(previous) if previous else None) or 1
    realtime = build_realtime_context(
        state.get("destination"),
        state.get("places") or [],
        weather_days,
        start_date=state.get("start_date"),
    )
    realtime_text = realtime.get("text") or "(제공 가능한 실시간 정보 없음)"
    external_text = state.get("external_text") or ""
    if external_text:
        realtime_text = f"{realtime_text}\n{external_text}"

    if intent == "edit" and previous:
        insert_place = state.get("insert_place")
        target_day = state.get("target_day")
        # Code-path patch for insert when we can resolve the place
        if insert_place and target_day:
            pool = state.get("places") or []
            match = None
            needle = _norm_name(insert_place)
            for p in pool:
                n = _norm_name(p.get("name_ko") or p.get("name_en"))
                if n and (needle in n or n in needle):
                    match = p
                    break
            if match is None:
                # Search once more by name
                try:
                    hits = search.search_places(
                        query=insert_place,
                        city=state.get("destination"),
                        types=None,
                        limit=5,
                    )
                    for p in hits:
                        n = _norm_name(p.get("name_ko") or p.get("name_en"))
                        if n and (needle in n or n in needle):
                            match = p
                            break
                    if match is None and hits:
                        match = hits[0]
                except Exception:
                    match = None
            if match:
                itinerary = _patch_place_into_day(previous, int(target_day), match)
                itinerary = _dedupe_itinerary_slots(itinerary, state.get("places"))
                name = match.get("name_ko") or match.get("name_en") or insert_place
                return {
                    "itinerary": itinerary,
                    "warnings": list(state.get("warnings") or []),
                    "answer": (
                        f"{int(target_day)}일차에 **{name}**을(를) 넣었어요.\n\n"
                        + _simple_itinerary_answer(itinerary)
                    ),
                    "sources": _build_sources(state),
                    "intent": "edit",
                    "realtime": realtime,
                }

        # Remove disliked places via code
        if exclude_places and not insert_place:
            itinerary = _remove_places_from_itinerary(previous, exclude_places)
            spare = state.get("places") or []
            itinerary = _dedupe_itinerary_slots(itinerary, spare)
            # fill empty slot days lightly
            try:
                raw = (TRAVEL_EDIT_PROMPT | llm).invoke(
                    {
                        "question": state.get("query"),
                        "history": state.get("history") or "(없음)",
                        "destination": state.get("destination"),
                        "days": state.get("days") or len(previous),
                        "preferences": ", ".join(state.get("preferences") or []),
                        "language": language,
                        "previous_itinerary": json.dumps(itinerary, ensure_ascii=False),
                        "exclude_places": exclude_str,
                        "insert_place": insert_place or "(없음)",
                        "target_day": target_day or "(없음)",
                        "places": json.dumps(places, ensure_ascii=False),
                    }
                )
                parsed = _extract_json(_llm_content(raw))
                if parsed.get("itinerary"):
                    itinerary = _dedupe_itinerary_slots(
                        parsed["itinerary"], state.get("places")
                    )
                    answer = parsed.get("answer") or f"요청하신 장소를 일정에서 빼 반영했어요."
                    return {
                        "itinerary": itinerary,
                        "warnings": list(state.get("warnings") or [])
                        + list(parsed.get("warnings") or []),
                        "answer": answer,
                        "sources": _build_sources(state),
                        "intent": "edit",
                        "realtime": realtime,
                    }
            except Exception:
                pass
            return {
                "itinerary": itinerary,
                "warnings": list(state.get("warnings") or []),
                "answer": f"요청하신 장소({', '.join(exclude_places)})를 일정에서 뺐어요.\n\n"
                + _simple_itinerary_answer(itinerary),
                "sources": _build_sources(state),
                "intent": "edit",
                "realtime": realtime,
            }

        try:
            raw = (TRAVEL_EDIT_PROMPT | llm).invoke(
                {
                    "question": state.get("query"),
                    "history": state.get("history") or "(없음)",
                    "destination": state.get("destination"),
                    "days": state.get("days") or len(previous),
                    "preferences": ", ".join(state.get("preferences") or []),
                    "language": language,
                    "previous_itinerary": json.dumps(previous, ensure_ascii=False),
                    "exclude_places": exclude_str,
                    "insert_place": insert_place or "(없음)",
                    "target_day": target_day or "(없음)",
                    "places": json.dumps(places, ensure_ascii=False),
                }
            )
            parsed = _extract_json(_llm_content(raw))
        except Exception:
            parsed = {}
        itinerary = parsed.get("itinerary") or previous
        itinerary = _dedupe_itinerary_slots(itinerary, state.get("places"))
        return {
            "itinerary": itinerary,
            "warnings": list(state.get("warnings") or []) + list(parsed.get("warnings") or []),
            "answer": parsed.get("answer") or "일정을 수정했어요.",
            "sources": _build_sources(state),
            "intent": "edit",
            "realtime": realtime,
        }

    if intent == "multi_itinerary":
        cap = _capped_count(state.get("itinerary_count"))
        parsed: dict[str, Any] = {}
        if places:
            try:
                raw = (TRAVEL_MULTI_ITINERARY_PROMPT | llm).invoke(
                    {
                        "question": state.get("query"),
                        "history": state.get("history") or "(없음)",
                        "destination": state.get("destination"),
                        "days": state.get("days") or 1,
                        "budget": state.get("budget"),
                        "preferences": ", ".join(state.get("preferences") or []),
                        "language": language,
                        "requested_count": state.get("itinerary_count") or cap,
                        "max_count": cap,
                        "place_count": len(state.get("places") or []),
                        "places": json.dumps(places, ensure_ascii=False),
                        "courses": json.dumps(courses, ensure_ascii=False),
                        "realtime": realtime_text,
                        "exclude_places": exclude_str,
                    }
                )
                parsed = _extract_json(_llm_content(raw))
            except Exception:
                parsed = {}
        return _build_multi_itinerary(state, parsed, realtime)

    if not places and not courses:
        fb = _fallback_itinerary(state)
        fb["realtime"] = realtime
        return fb

    used_ids, used_names = _places_from_itinerary(previous)
    used_str = ", ".join(used_names) if used_names else "(없음)"

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
                    "realtime": realtime_text,
                    "exclude_places": exclude_str,
                    "used_places": used_str,
                }
            )
            parsed = _extract_json(_llm_content(raw))
            new_day = parsed.get("day") or {}
            warnings = list(state.get("warnings") or [])
            day_options = []
            for opt in parsed.get("day_options") or []:
                if isinstance(opt, dict) and opt.get("slots"):
                    day_options.append(
                        {
                            "day": int(rewrite_day),
                            "theme": opt.get("theme") or "",
                            "slots": opt.get("slots") or [],
                        }
                    )
            if new_day.get("slots") and not day_options:
                day_options = [
                    {
                        "day": int(rewrite_day),
                        "theme": new_day.get("theme") or "",
                        "slots": new_day.get("slots"),
                    }
                ]
            if not new_day.get("slots") and not day_options:
                warnings.append(
                    f"{rewrite_day}일차를 바꿀 대체 장소가 부족해 기존 일정을 유지합니다."
                )
                return {
                    "itinerary": previous,
                    "day_options": [],
                    "warnings": warnings,
                    "answer": parsed.get("answer")
                    or f"{rewrite_day}일차 대안을 찾지 못했어요. 조건을 바꿔 다시 요청해 주세요.",
                    "sources": _build_sources(state),
                    "rewrite_day": rewrite_day,
                    "realtime": realtime,
                }
            if not new_day.get("slots") and day_options:
                new_day = day_options[0]
            itinerary = _merge_day_into_itinerary(previous, int(rewrite_day), new_day)
            itinerary = _dedupe_itinerary_slots(itinerary, state.get("places"))
            warnings.extend(parsed.get("warnings") or [])
            if len(places) < 3:
                warnings.append("등록된 대체 장소가 적어 선택지가 제한적일 수 있어요.")
            answer = parsed.get("answer") or f"{rewrite_day}일차 일정을 수정했습니다."
            return {
                "itinerary": itinerary,
                "day_options": day_options,
                "warnings": warnings,
                "answer": answer,
                "sources": _build_sources(state),
                "rewrite_day": rewrite_day,
                "realtime": realtime,
            }

        raw = (TRAVEL_ITINERARY_PROMPT | llm).invoke(
            {
                "question": state.get("query"),
                "history": state.get("history") or "(없음)",
                "destination": state.get("destination"),
                "days": state.get("days") or 1,
                "start_date": state.get("start_date") or "(오늘)",
                "budget": state.get("budget"),
                "preferences": ", ".join(state.get("preferences") or []),
                "language": language,
                "places": json.dumps(places, ensure_ascii=False),
                "courses": json.dumps(courses, ensure_ascii=False),
                "accommodations": stays_json,
                "realtime": realtime_text,
                "previous_itinerary": json.dumps(previous, ensure_ascii=False) if previous else "[]",
                "exclude_places": exclude_str,
            }
        )
        parsed = _extract_json(_llm_content(raw))
    except Exception:
        parsed = {}

    if not parsed.get("answer"):
        fb = _fallback_itinerary(state)
        fb["realtime"] = realtime
        return fb

    warnings = list(state.get("warnings") or [])
    warnings.extend(parsed.get("warnings") or [])
    itinerary = _dedupe_itinerary_slots(
        parsed.get("itinerary") or [], state.get("places")
    )
    return {
        "itinerary": itinerary,
        "accommodations": _sanitize_accommodations(
            parsed.get("accommodations"), state.get("stays")
        ),
        "warnings": warnings,
        "answer": parsed["answer"],
        "sources": _build_sources(state),
        "realtime": realtime,
        "intent": "itinerary",
    }


def _simple_itinerary_answer(itinerary: list[dict[str, Any]]) -> str:
    blocks = []
    for day in itinerary:
        lines = [f"Day {day.get('day')} · {day.get('theme') or ''}"]
        for slot in day.get("slots") or []:
            label = _SLOT_LABEL_KO.get(slot.get("time"), slot.get("time") or "")
            lines.append(f"- {label}: {slot.get('place_name', '')}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def stream_build_itinerary_tokens(
    search: TravelSearchService,
    state: TravelState,
) -> Iterator[tuple[str, Any]]:
    """Yield (event_type, payload) for SSE. extract+search sync, stream LLM when useful."""
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)
    result_so_far: TravelState = dict(state)

    yield ("status", {"stage": "extracting"})
    merged = _run_extract(result_so_far, llm)
    result_so_far.update(merged)
    intent = result_so_far.get("intent") or "itinerary"

    if intent == "meta":
        yield ("status", {"stage": "answering"})
        answer = META_FALLBACK_KO
        accumulated = ""
        try:
            for chunk in (TRAVEL_META_PROMPT | llm).stream(
                {
                    "question": result_so_far.get("query"),
                    "language": result_so_far.get("language") or "ko",
                }
            ):
                piece = _llm_content(chunk)
                if not piece:
                    continue
                accumulated += piece
                yield ("token", {"text": piece})
            answer = accumulated.strip() or META_FALLBACK_KO
        except Exception:
            yield ("token", {"text": answer})
        result_so_far.update(
            {
                "answer": answer,
                "itinerary": [],
                "itineraries": [],
                "cities": [],
                "recommended_places": [],
                "warnings": [],
                "sources": [],
                "intent": "meta",
            }
        )
        yield ("done", _result_payload(result_so_far))
        return

    if intent == "qa":
        yield ("status", {"stage": "answering"})
        prev = result_so_far.get("previous_itinerary") or []
        inputs = {
            "question": result_so_far.get("query"),
            "history": result_so_far.get("history") or "(없음)",
            "language": result_so_far.get("language") or "ko",
            "previous_itinerary": (
                json.dumps(prev, ensure_ascii=False) if prev else "(없음)"
            ),
        }
        accumulated = ""
        try:
            for chunk in (TRAVEL_QA_PROMPT | llm).stream(inputs):
                piece = _llm_content(chunk)
                if not piece:
                    continue
                accumulated += piece
                yield ("token", {"text": piece})
        except Exception:
            pass
        result_so_far.update(
            {
                "answer": accumulated.strip()
                or "죄송해요, 답변을 만들지 못했어요. 다시 물어봐 주세요.",
                "itinerary": prev if prev else [],
                "itineraries": [],
                "cities": [],
                "warnings": [],
                "sources": [],
                "intent": "qa",
            }
        )
        yield ("done", _result_payload(result_so_far))
        return

    yield ("status", {"stage": "searching"})
    searched = _search_for_state(result_so_far, search)
    result_so_far.update(searched)

    yield ("status", {"stage": "writing" if intent not in ("place_list",) else "writing"})

    # Reuse non-stream builder for consistency on complex branches
    if intent in ("edit", "city_list", "place_list", "multi_itinerary") or (
        result_so_far.get("rewrite_day") and result_so_far.get("previous_itinerary")
    ):
        built = _build_response(result_so_far, llm, search)
        # Optionally stream the answer as tokens for UX
        answer = built.get("answer") or ""
        if answer:
            # chunk lightly for UI
            step = 80
            for i in range(0, len(answer), step):
                yield ("token", {"text": answer[i : i + step]})
        result_so_far.update(built)
        yield ("done", _result_payload(result_so_far))
        return

    # Full itinerary with streaming LLM JSON
    places = _compact_places(result_so_far.get("places") or [], limit=_place_limit_for(result_so_far))
    courses = _compact_courses(result_so_far.get("courses") or [])
    stays = _compact_stays(result_so_far.get("stays") or [])
    stays_json = json.dumps(stays, ensure_ascii=False)
    language = result_so_far.get("language") or "ko"
    previous = list(result_so_far.get("previous_itinerary") or [])
    exclude_str = ", ".join(result_so_far.get("exclude_places") or []) or "(없음)"
    weather_days = result_so_far.get("days") or 1
    realtime = build_realtime_context(
        result_so_far.get("destination"),
        result_so_far.get("places") or [],
        weather_days,
        start_date=result_so_far.get("start_date"),
    )
    realtime_text = realtime.get("text") or "(제공 가능한 실시간 정보 없음)"
    external_text = result_so_far.get("external_text") or ""
    if external_text:
        realtime_text = f"{realtime_text}\n{external_text}"
    result_so_far["realtime"] = realtime

    if not places and not courses:
        fallback = _fallback_itinerary(result_so_far)
        fallback["realtime"] = realtime
        result_so_far.update(fallback)
        yield ("done", _result_payload(result_so_far))
        return

    inputs = {
        "question": result_so_far.get("query"),
        "history": result_so_far.get("history") or "(없음)",
        "destination": result_so_far.get("destination"),
        "days": result_so_far.get("days") or 1,
        "start_date": result_so_far.get("start_date") or "(오늘)",
        "budget": result_so_far.get("budget"),
        "preferences": ", ".join(result_so_far.get("preferences") or []),
        "language": language,
        "places": json.dumps(places, ensure_ascii=False),
        "courses": json.dumps(courses, ensure_ascii=False),
        "accommodations": stays_json,
        "realtime": realtime_text,
        "previous_itinerary": json.dumps(previous, ensure_ascii=False) if previous else "[]",
        "exclude_places": exclude_str,
    }
    accumulated = ""
    try:
        for chunk in (TRAVEL_ITINERARY_PROMPT | llm).stream(inputs):
            piece = _llm_content(chunk)
            if not piece:
                continue
            accumulated += piece
            yield ("token", {"text": piece})
        parsed = _extract_json(accumulated)
    except Exception:
        parsed = {}

    if not parsed.get("answer"):
        fallback = _fallback_itinerary(result_so_far)
        fallback["realtime"] = realtime
        result_so_far.update(fallback)
    else:
        w = list(result_so_far.get("warnings") or [])
        w.extend(parsed.get("warnings") or [])
        result_so_far.update(
            {
                "itinerary": _dedupe_itinerary_slots(
                    parsed.get("itinerary") or [], result_so_far.get("places")
                ),
                "accommodations": _sanitize_accommodations(
                    parsed.get("accommodations"), result_so_far.get("stays")
                ),
                "warnings": w,
                "answer": parsed["answer"],
                "sources": _build_sources(result_so_far),
                "intent": "itinerary",
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
        "start_date": state.get("start_date"),
        "itinerary": state.get("itinerary") or [],
        "itineraries": state.get("itineraries") or [],
        "accommodations": state.get("accommodations") or [],
        "itinerary_count": state.get("itinerary_count"),
        "cities": state.get("cities") or [],
        "recommended_places": state.get("recommended_places") or [],
        "day_options": state.get("day_options") or [],
        "intent": state.get("intent") or "itinerary",
        "places": state.get("places") or [],
        "courses": state.get("courses") or [],
        "warnings": state.get("warnings") or [],
        "sources": state.get("sources") or [],
        "answer": state.get("answer") or "",
        "rewrite_day": state.get("rewrite_day"),
        "exclude_places": state.get("exclude_places") or [],
        "realtime": state.get("realtime") or {},
        "external": state.get("external") or {},
    }
