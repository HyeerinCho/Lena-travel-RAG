"""여행 질문을 한국관광공사(data.go.kr) 오픈 API로 보강(enrich)하는 모듈.

여행 그래프(build_itinerary)가 쓰는 '실시간 참고 정보'에 덧붙일 컨텍스트를
만듭니다. Open-Meteo 날씨와 동일하게 **네트워크/키 실패 시 조용히 빈 결과**를
반환해 기존 챗봇 동작을 해치지 않습니다.

감지 의도:
  - pet       : 반려동물 동반 (KorPetTourService2)
  - barrier   : 무장애 여행 (KorWithService2)
  - trail     : 걷기여행길/둘레길 (Durunubi)
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from src.travel.tour_api import (
    TourAPIError,
    durunubi,
    kor_pet_tour,
    kor_service,
    kor_with_service,
)

# TourAPI 콘텐츠 타입: 레포츠(액티비티/체험)
_LEPORTS_CONTENT_TYPE_ID = "28"

# 대표 도시/권역명 -> TourAPI 지역코드(areaCode). 광역시도 기준.
_AREA_CODE: dict[str, int] = {
    "서울": 1,
    "인천": 2,
    "대전": 3,
    "대구": 4,
    "광주": 5,
    "부산": 6,
    "울산": 7,
    "세종": 8,
    "경기": 31, "수원": 31,
    "강원": 32, "춘천": 32, "강릉": 32, "속초": 32,
    "충북": 33, "청주": 33,
    "충남": 34, "천안": 34,
    "경북": 35, "경주": 35, "포항": 35, "안동": 35,
    "경남": 36, "창원": 36, "통영": 36, "진주": 36,
    "전북": 37, "전주": 37,
    "전남": 38, "목포": 38, "여수": 38, "순천": 38,
    "제주": 39, "서귀포": 39,
}

# 의도 감지 키워드
_PET_PAT = re.compile(
    r"반려\s*동물|반려견|반려묘|애견|애완|강아지|고양이|펫|멍멍이|\bpet\b|\bdog\b|\bcat\b",
    re.I,
)
_BARRIER_PAT = re.compile(
    r"무장애|배리어\s*프리|휠체어|장애인|고령자|노약자|유모차|유아차|"
    r"barrier[\s-]?free|wheelchair|accessib",
    re.I,
)
_TRAIL_PAT = re.compile(
    r"둘레길|걷기\s*여행|걷기여행|도보\s*여행|트레킹|트래킹|산책로|올레|"
    r"해파랑길|남파랑길|서해랑길|코리아둘레길|trekking|trail|walking\s*course",
    re.I,
)


def detect_kinds(query: str) -> list[str]:
    """질문에서 보강 의도 목록을 반환. 없으면 빈 리스트."""
    kinds: list[str] = []
    if _PET_PAT.search(query):
        kinds.append("pet")
    if _BARRIER_PAT.search(query):
        kinds.append("barrier")
    if _TRAIL_PAT.search(query):
        kinds.append("trail")
    return kinds


def resolve_area_code(destination: str | None) -> int | None:
    if not destination:
        return None
    dest = str(destination).strip()
    if dest in _AREA_CODE:
        return _AREA_CODE[dest]
    for name, code in _AREA_CODE.items():
        if name in dest or dest in name:
            return code
    return None


def _place_line(item: dict[str, Any]) -> str | None:
    name = item.get("title")
    if not name:
        return None
    addr = item.get("addr1") or ""
    addr = str(addr).strip()
    tel = str(item.get("tel") or "").strip()
    parts = [str(name).strip()]
    if addr:
        parts.append(f"({addr})")
    if tel:
        parts.append(f"☎ {tel}")
    return " ".join(parts)


def _trail_line(item: dict[str, Any]) -> str | None:
    name = item.get("crsKorNm") or item.get("crsNm")
    if not name:
        return None
    bits: list[str] = [str(name).strip()]
    dist = item.get("crsDstnc")
    if dist:
        bits.append(f"{dist}km")
    level = item.get("crsLevel")
    if level:
        bits.append(f"난이도 {level}")
    hour = item.get("crsTotlRqrmHour")
    if hour:
        bits.append(f"약 {hour}분")
    summary = item.get("crsSummary") or item.get("crsContents")
    line = " · ".join(bits)
    if summary:
        line += f" — {str(summary).strip()[:60]}"
    return line


def _festival_line(item: dict[str, Any]) -> str | None:
    name = item.get("title")
    if not name:
        return None
    parts = [str(name).strip()]
    start = str(item.get("eventstartdate") or "").strip()
    end = str(item.get("eventenddate") or "").strip()
    if start:
        period = _fmt_date(start)
        if end and end != start:
            period += f"~{_fmt_date(end)}"
        parts.append(f"({period})")
    addr = str(item.get("addr1") or "").strip()
    if addr:
        parts.append(addr)
    return " ".join(parts)


def _fmt_date(yyyymmdd: str) -> str:
    digits = re.sub(r"[^\d]", "", yyyymmdd)
    if len(digits) == 8:
        return f"{digits[4:6]}.{digits[6:8]}"
    return yyyymmdd


def _item_matches(item: dict[str, Any], needle: str) -> bool:
    blob = " ".join(str(v) for v in item.values() if v is not None)
    return needle in blob


def _fetch_pet(destination: str | None, area_code: int | None, limit: int) -> list[dict[str, Any]]:
    try:
        if area_code is not None:
            res = kor_pet_tour.area_based_list(area_code=str(area_code), num_of_rows=limit)
        elif destination:
            res = kor_pet_tour.search_keyword(destination, num_of_rows=limit)
        else:
            return []
        return (res.get("items") or [])[:limit]
    except TourAPIError:
        return []


def _fetch_barrier(destination: str | None, area_code: int | None, limit: int) -> list[dict[str, Any]]:
    try:
        if area_code is not None:
            res = kor_with_service.area_based_list(area_code=str(area_code), num_of_rows=limit)
        elif destination:
            res = kor_with_service.search_keyword(destination, num_of_rows=limit)
        else:
            return []
        return (res.get("items") or [])[:limit]
    except TourAPIError:
        return []


def _fetch_trails(destination: str | None, limit: int) -> list[dict[str, Any]]:
    try:
        res = durunubi.course_list(num_of_rows=50)
        items = res.get("items") or []
    except TourAPIError:
        return []
    if destination:
        matched = [it for it in items if _item_matches(it, str(destination).strip())]
        if matched:
            return matched[:limit]
    return items[:limit]


def _fetch_activities(area_code: int | None, limit: int) -> list[dict[str, Any]]:
    if area_code is None:
        return []
    try:
        res = kor_service.area_based_list(
            area_code=str(area_code),
            content_type_id=_LEPORTS_CONTENT_TYPE_ID,
            num_of_rows=limit,
        )
        return (res.get("items") or [])[:limit]
    except TourAPIError:
        return []


def _fetch_festivals(area_code: int | None, limit: int) -> list[dict[str, Any]]:
    if area_code is None:
        return []
    try:
        today = datetime.now().strftime("%Y%m%d")
        res = kor_service.search_festival(
            event_start_date=today,
            area_code=str(area_code),
            num_of_rows=limit,
        )
        return (res.get("items") or [])[:limit]
    except TourAPIError:
        return []


def build_tour_context(
    query: str,
    destination: str | None,
    *,
    language: str = "ko",
    limit: int = 5,
) -> dict[str, Any]:
    """관광공사 API 보강 컨텍스트.

    액티비티(레포츠)·행사는 목적지 지역코드만 있으면 항상 함께 붙여
    일정에 체험/축제를 최대한 녹일 수 있게 합니다. 반려동물/무장애/걷기길은
    질문에서 의도가 감지될 때만 붙입니다. 붙일 게 없으면 {}.
    """
    kinds = detect_kinds(query) if query else []

    en = language == "en"
    area_code = resolve_area_code(destination)
    where = f" · {destination}" if destination else ""

    data: dict[str, list[dict[str, Any]]] = {}
    sections: list[str] = []

    # 액티비티(레포츠) — 목적지 지역코드가 있으면 항상 시도
    activities = _fetch_activities(area_code, limit)
    if activities:
        data["activity"] = activities
        header = (
            f"[Activities · leisure sports{where} (KTO API)]"
            if en
            else f"[액티비티·레포츠·체험{where} · 관광공사 API]"
        )
        lines = [f"- {ln}" for it in activities if (ln := _place_line(it))]
        sections.append(header + "\n" + "\n".join(lines))

    # 행사·축제 — 오늘 이후 진행 중/예정 행사
    festivals = _fetch_festivals(area_code, limit)
    if festivals:
        data["festival"] = festivals
        header = (
            f"[Festivals & events{where} (KTO API)]"
            if en
            else f"[축제·공연·행사{where} · 관광공사 API]"
        )
        lines = [f"- {ln}" for it in festivals if (ln := _festival_line(it))]
        sections.append(header + "\n" + "\n".join(lines))

    if "pet" in kinds:
        items = _fetch_pet(destination, area_code, limit)
        if items:
            data["pet"] = items
            header = (
                f"[Pet-friendly places{where} (KTO API)]"
                if en
                else f"[반려동물 동반 가능 장소{where} · 관광공사 API]"
            )
            lines = [f"- {ln}" for it in items if (ln := _place_line(it))]
            sections.append(header + "\n" + "\n".join(lines))

    if "barrier" in kinds:
        items = _fetch_barrier(destination, area_code, limit)
        if items:
            data["barrier"] = items
            header = (
                f"[Barrier-free travel spots{where} (KTO API)]"
                if en
                else f"[무장애 여행 정보{where} · 관광공사 API]"
            )
            lines = [f"- {ln}" for it in items if (ln := _place_line(it))]
            sections.append(header + "\n" + "\n".join(lines))

    if "trail" in kinds:
        items = _fetch_trails(destination, limit)
        if items:
            data["trail"] = items
            header = (
                "[Walking trails (Durunubi/KTO API)]"
                if en
                else "[걷기여행길·둘레길 · 두루누비 관광공사 API]"
            )
            lines = [f"- {ln}" for it in items if (ln := _trail_line(it))]
            sections.append(header + "\n" + "\n".join(lines))

    if not sections:
        return {}

    return {
        "kinds": list(data.keys()),
        "data": data,
        "text": "\n".join(sections),
    }
