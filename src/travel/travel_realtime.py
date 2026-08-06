"""Real-time context (Open-Meteo weather + operating status) for travel itineraries.

Everything here degrades gracefully: when the network call fails, the functions
return empty results so itinerary generation still works. Open-Meteo needs no API key.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any

from src.config import (
    OPEN_METEO_URL,
    WEATHER_DEFAULT_DAYS,
    WEATHER_FORECAST_MAX_DAYS,
    WEATHER_REQUEST_TIMEOUT_SEC,
)

KST = timezone(timedelta(hours=9))
_WEEKDAY_KO = "월화수목금토일"

# 앱에서 쓰는 대표 도시/권역명 -> (위도, 경도). Open-Meteo 조회용.
CITY_LATLON: dict[str, tuple[float, float]] = {
    "서울": (37.5665, 126.9780),
    "인천": (37.4563, 126.7052),
    "수원": (37.2636, 127.0286),
    "경기": (37.2911, 127.0089),
    "춘천": (37.8813, 127.7298),
    "강원": (37.8228, 128.1555),
    "강릉": (37.7519, 128.8761),
    "속초": (38.2070, 128.5918),
    "대전": (36.3504, 127.3845),
    "세종": (36.4800, 127.2890),
    "청주": (36.6424, 127.4890),
    "충북": (36.6357, 127.4917),
    "충남": (36.6588, 126.6728),
    "천안": (36.8151, 127.1139),
    "전주": (35.8242, 127.1480),
    "전북": (35.8203, 127.1088),
    "광주": (35.1595, 126.8526),
    "전남": (34.8679, 126.9910),
    "목포": (34.8118, 126.3922),
    "여수": (34.7604, 127.6622),
    "순천": (34.9506, 127.4872),
    "부산": (35.1796, 129.0756),
    "울산": (35.5384, 129.3114),
    "경남": (35.2280, 128.6811),
    "창원": (35.2280, 128.6811),
    "통영": (34.8544, 128.4331),
    "진주": (35.1800, 128.1076),
    "대구": (35.8714, 128.6014),
    "경북": (36.5684, 128.7294),
    "경주": (35.8562, 129.2247),
    "포항": (36.0190, 129.3435),
    "안동": (36.5684, 128.7294),
    "제주": (33.4996, 126.5312),
    "서귀포": (33.2541, 126.5600),
}

# WMO weather code -> (한국어 상태, 강수여부)
_WMO_CODE: dict[int, tuple[str, bool]] = {
    0: ("맑음", False),
    1: ("대체로 맑음", False),
    2: ("구름많음", False),
    3: ("흐림", False),
    45: ("안개", False),
    48: ("안개", False),
    51: ("이슬비", True),
    53: ("이슬비", True),
    55: ("이슬비", True),
    56: ("어는 이슬비", True),
    57: ("어는 이슬비", True),
    61: ("비", True),
    63: ("비", True),
    65: ("강한 비", True),
    66: ("어는 비", True),
    67: ("어는 비", True),
    71: ("눈", True),
    73: ("눈", True),
    75: ("많은 눈", True),
    77: ("싸락눈", True),
    80: ("소나기", True),
    81: ("소나기", True),
    82: ("강한 소나기", True),
    85: ("눈 소나기", True),
    86: ("눈 소나기", True),
    95: ("뇌우", True),
    96: ("뇌우/우박", True),
    99: ("뇌우/우박", True),
}


def _latlon_for_city(city: str | None) -> tuple[float, float] | None:
    if not city:
        return None
    city = str(city).strip()
    if city in CITY_LATLON:
        return CITY_LATLON[city]
    for name, coords in CITY_LATLON.items():
        if name in city or city in name:
            return coords
    return None


def _to_int(value: Any) -> int | None:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _parse_date(value: str | None, *, now: datetime) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=KST)
        except ValueError:
            continue
    return None


def resolve_start_date(
    start_date: str | None,
    *,
    now: datetime | None = None,
) -> datetime:
    """Return KST midnight of trip start. Defaults to today."""
    now = now or datetime.now(KST)
    parsed = _parse_date(start_date, now=now)
    if parsed is not None:
        return parsed.replace(hour=0, minute=0, second=0, microsecond=0)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def fetch_weather(
    city: str | None,
    days: int | None = None,
    *,
    start_date: str | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """도시별 일자 요약 예보 리스트. start_date 기준으로 Day1..N 정렬. 실패 시 []."""
    coords = _latlon_for_city(city)
    if coords is None:
        return []

    now = now or datetime.now(KST)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    trip_start = resolve_start_date(start_date, now=now)
    horizon = max(1, min(days or WEATHER_DEFAULT_DAYS, WEATHER_FORECAST_MAX_DAYS))

    # Open-Meteo forecast is relative to today; need enough days to cover trip end.
    days_until_start = (trip_start.date() - today.date()).days
    if days_until_start < 0:
        # Past start date → treat as starting today and still return trip-relative days.
        trip_start = today
        days_until_start = 0

    end_offset = days_until_start + horizon
    if days_until_start >= WEATHER_FORECAST_MAX_DAYS:
        return []
    forecast_days = min(WEATHER_FORECAST_MAX_DAYS, max(1, end_offset))

    params = {
        "latitude": f"{coords[0]:.4f}",
        "longitude": f"{coords[1]:.4f}",
        "daily": (
            "weathercode,temperature_2m_max,temperature_2m_min,"
            "precipitation_probability_max"
        ),
        "timezone": "Asia/Seoul",
        "forecast_days": str(forecast_days),
    }
    url = f"{OPEN_METEO_URL}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=WEATHER_REQUEST_TIMEOUT_SEC) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []

    daily = payload.get("daily") or {}
    times = daily.get("time") or []
    if not times:
        return []
    codes = daily.get("weathercode") or []
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    pops = daily.get("precipitation_probability_max") or []

    by_date: dict[str, int] = {str(d): i for i, d in enumerate(times)}

    def _at(seq: list[Any], i: int) -> Any:
        return seq[i] if i < len(seq) else None

    out: list[dict[str, Any]] = []
    for day_index in range(1, horizon + 1):
        target = trip_start + timedelta(days=day_index - 1)
        date_str = target.strftime("%Y-%m-%d")
        api_i = by_date.get(date_str)
        if api_i is None:
            # Outside forecast window: still emit a shell so Day N maps correctly.
            try:
                weekday = _WEEKDAY_KO[target.weekday()]
            except Exception:
                weekday = ""
            out.append(
                {
                    "day": day_index,
                    "date": date_str,
                    "weekday": weekday,
                    "condition": "예보 없음",
                    "temp_min": None,
                    "temp_max": None,
                    "pop": None,
                    "rain": False,
                    "summary": "예보 범위 밖",
                }
            )
            continue
        out.append(
            _summarize_day(
                day_index,
                date_str,
                _at(codes, api_i),
                _at(tmax, api_i),
                _at(tmin, api_i),
                _at(pops, api_i),
            )
        )
    return out


def _summarize_day(
    day_index: int,
    date_str: str,
    code: Any,
    temp_max: Any,
    temp_min: Any,
    pop: Any,
) -> dict[str, Any]:
    condition, code_rain = _WMO_CODE.get(
        _to_int(code) if code is not None else -1, ("정보없음", False)
    )
    temp_min_i = _to_int(temp_min)
    temp_max_i = _to_int(temp_max)
    pop_i = _to_int(pop)
    rain = bool(code_rain or (pop_i is not None and pop_i >= 60))

    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        weekday = _WEEKDAY_KO[dt.weekday()]
    except ValueError:
        weekday = ""

    parts = [condition]
    if temp_min_i is not None and temp_max_i is not None:
        parts.append(f"{temp_min_i}~{temp_max_i}℃")
    if pop_i is not None:
        parts.append(f"강수확률 {pop_i}%")
    return {
        "day": day_index,
        "date": date_str,
        "weekday": weekday,
        "condition": condition,
        "temp_min": temp_min_i,
        "temp_max": temp_max_i,
        "pop": pop_i,
        "rain": rain,
        "summary": ", ".join(parts),
    }


def _closed_today(closed_ko: str | None, when: datetime) -> bool:
    if not closed_ko:
        return False
    text = str(closed_ko)
    if any(k in text for k in ("연중무휴", "무휴", "없음")):
        return False
    today = _WEEKDAY_KO[when.weekday()]
    if f"{today}요일" in text or f"매주 {today}" in text:
        return True
    if f"{today}" in text and "요일" in text:
        return True
    return False


def operating_notes(
    places: list[dict[str, Any]],
    *,
    when: datetime | None = None,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """오늘 휴무로 추정되는 후보 장소 노트."""
    when = when or datetime.now(KST)
    notes: list[dict[str, Any]] = []
    for p in places[:limit]:
        if _closed_today(p.get("closed_ko"), when):
            notes.append(
                {
                    "poi_id": p.get("poi_id"),
                    "name": p.get("name_ko") or p.get("name_en"),
                    "closed_ko": p.get("closed_ko"),
                }
            )
    return notes


def build_realtime_context(
    destination: str | None,
    places: list[dict[str, Any]],
    days: int | None,
    *,
    start_date: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """프롬프트/응답에 쓸 실시간 컨텍스트(dict). text 필드는 프롬프트 주입용."""
    now = now or datetime.now(KST)
    trip_days = max(1, days or WEATHER_DEFAULT_DAYS)
    # Align weather length with itinerary days when provided.
    weather = fetch_weather(
        destination,
        trip_days if days else WEATHER_DEFAULT_DAYS,
        start_date=start_date,
        now=now,
    )
    closed_today = operating_notes(places, when=now)

    lines: list[str] = []
    if weather:
        start = resolve_start_date(start_date, now=now).strftime("%Y-%m-%d")
        lines.append(f"[날씨 예보 · {destination or '여행지'} · 시작 {start}]")
        for w in weather:
            tail = " → 실내 위주 권장" if w.get("rain") else ""
            lines.append(
                f"- Day{w['day']} {w['date']}({w['weekday']}): {w['summary']}{tail}"
            )
        if any(w.get("condition") == "예보 없음" for w in weather):
            lines.append("- 일부 날짜는 Open-Meteo 예보 범위(최대 16일) 밖입니다.")
    if closed_today:
        lines.append(f"[오늘({_WEEKDAY_KO[now.weekday()]}) 휴무 추정]")
        for c in closed_today:
            lines.append(f"- {c['name']}: 휴무일 {c['closed_ko']}")

    text = "\n".join(lines) if lines else "(제공 가능한 실시간 정보 없음)"
    return {
        "generated_at": now.strftime("%Y-%m-%d %H:%M KST"),
        "start_date": resolve_start_date(start_date, now=now).strftime("%Y-%m-%d"),
        "weather": weather,
        "closed_today": closed_today,
        "text": f"제공 시각: {now.strftime('%Y-%m-%d %H:%M KST')}\n{text}",
    }
