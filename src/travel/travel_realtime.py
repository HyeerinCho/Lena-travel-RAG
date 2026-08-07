"""Real-time context (Open-Meteo weather + operating status) for travel itineraries.

Everything here degrades gracefully: when the network call fails, the functions
return empty results so itinerary generation still works. Open-Meteo needs no API key.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

from src.config import (
    CLIMATE_AVG_WINDOW_DAYS,
    CLIMATE_AVG_YEARS,
    OPEN_METEO_ARCHIVE_URL,
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


def _fetch_climate_normal(
    coords: tuple[float, float],
    target: datetime,
    *,
    years: int = CLIMATE_AVG_YEARS,
    window_days: int = CLIMATE_AVG_WINDOW_DAYS,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """예보 범위 밖 날짜용 참고치: 최근 N년간 같은 날짜(±window) 평균 기온/날씨.

    Open-Meteo Historical Weather API(과거 실측치)를 연도별로 조회해 평균낸다.
    실패하거나 데이터가 없으면 None (호출부에서 '예보 없음'으로 대체).
    """
    now = now or datetime.now(KST)
    tmax_values: list[float] = []
    tmin_values: list[float] = []
    codes: list[int] = []
    used_years: list[int] = []

    for back in range(1, years + 1):
        year = target.year - back
        try:
            center = target.replace(year=year)
        except ValueError:
            continue  # 2/29처럼 해당 연도에 없는 날짜
        start = center - timedelta(days=window_days)
        end = center + timedelta(days=window_days)
        if end.date() >= now.date():
            end = now - timedelta(days=1)
        if start.date() > end.date():
            continue
        params = {
            "latitude": f"{coords[0]:.4f}",
            "longitude": f"{coords[1]:.4f}",
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "daily": "weathercode,temperature_2m_max,temperature_2m_min",
            "timezone": "Asia/Seoul",
        }
        url = f"{OPEN_METEO_ARCHIVE_URL}?{urllib.parse.urlencode(params)}"
        try:
            with urllib.request.urlopen(url, timeout=WEATHER_REQUEST_TIMEOUT_SEC) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception:
            continue
        daily = payload.get("daily") or {}
        year_tmax = [v for v in (daily.get("temperature_2m_max") or []) if v is not None]
        year_tmin = [v for v in (daily.get("temperature_2m_min") or []) if v is not None]
        year_codes = [v for v in (daily.get("weathercode") or []) if v is not None]
        if not year_tmax and not year_tmin:
            continue
        tmax_values.extend(year_tmax)
        tmin_values.extend(year_tmin)
        codes.extend(year_codes)
        used_years.append(year)

    if not tmax_values or not tmin_values:
        return None

    avg_tmax = round(sum(tmax_values) / len(tmax_values))
    avg_tmin = round(sum(tmin_values) / len(tmin_values))
    condition = "정보없음"
    if codes:
        common_code = Counter(_to_int(c) for c in codes).most_common(1)[0][0]
        condition, _ = _WMO_CODE.get(common_code if common_code is not None else -1, ("정보없음", False))

    year_label = "·".join(str(y) for y in sorted(used_years))
    return {
        "years_used": used_years,
        "temp_min": avg_tmin,
        "temp_max": avg_tmax,
        "condition": condition,
        "summary": (
            f"(예보 범위 밖 참고치) {year_label}년 {target.strftime('%m/%d')} 전후 평균 "
            f"{condition}, {avg_tmin}~{avg_tmax}℃"
        ),
    }


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
            # Outside forecast window: fall back to a historical-average estimate
            # so we still emit a shell and Day N maps correctly.
            try:
                weekday = _WEEKDAY_KO[target.weekday()]
            except Exception:
                weekday = ""
            normal = _fetch_climate_normal(coords, target, now=now)
            if normal:
                out.append(
                    {
                        "day": day_index,
                        "date": date_str,
                        "weekday": weekday,
                        "condition": normal["condition"],
                        "temp_min": normal["temp_min"],
                        "temp_max": normal["temp_max"],
                        "pop": None,
                        "rain": False,
                        "summary": normal["summary"],
                        "source": "climate_avg",
                    }
                )
            else:
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
                        "source": "unavailable",
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
        "source": "forecast",
    }


def _closed_on(closed_ko: str | None, when: datetime) -> bool:
    if not closed_ko:
        return False
    text = str(closed_ko)
    if any(k in text for k in ("연중무휴", "무휴", "없음")):
        return False
    weekday = _WEEKDAY_KO[when.weekday()]
    if f"{weekday}요일" in text or f"매주 {weekday}" in text:
        return True
    if f"{weekday}" in text and "요일" in text:
        return True
    return False


def operating_notes(
    places: list[dict[str, Any]],
    *,
    when: datetime | None = None,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """주어진 날짜 기준으로 휴무로 추정되는 후보 장소 노트."""
    when = when or datetime.now(KST)
    notes: list[dict[str, Any]] = []
    for p in places[:limit]:
        if _closed_on(p.get("closed_ko"), when):
            notes.append(
                {
                    "poi_id": p.get("poi_id"),
                    "name": p.get("name_ko") or p.get("name_en"),
                    "closed_ko": p.get("closed_ko"),
                }
            )
    return notes


def operating_notes_by_day(
    places: list[dict[str, Any]],
    *,
    trip_start: datetime,
    num_days: int,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """여행 시작일 기준 Day1..N 각각의 실제 날짜에 맞춰 휴무 추정 장소를 계산한다.

    '오늘'(서버 시각) 요일이 아니라 각 Day가 실제로 해당하는 요일을 사용해야
    미래 출발 일정에서 휴무 안내가 일정 날짜와 어긋나지 않는다.
    """
    out: list[dict[str, Any]] = []
    for day_index in range(1, max(1, num_days) + 1):
        target = trip_start + timedelta(days=day_index - 1)
        notes = operating_notes(places, when=target, limit=limit)
        if notes:
            out.append(
                {
                    "day": day_index,
                    "date": target.strftime("%Y-%m-%d"),
                    "weekday": _WEEKDAY_KO[target.weekday()],
                    "places": notes,
                }
            )
    return out


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
    trip_start = resolve_start_date(start_date, now=now)
    closed_by_day = operating_notes_by_day(places, trip_start=trip_start, num_days=trip_days)
    closed_by_day_map = {c["day"]: c for c in closed_by_day}

    lines: list[str] = []
    if weather:
        lines.append(f"[날씨 예보 · {destination or '여행지'} · 시작 {trip_start.strftime('%Y-%m-%d')}]")
        for w in weather:
            tail = " → 실내 위주 권장" if w.get("rain") else ""
            lines.append(
                f"- Day{w['day']} {w['date']}({w['weekday']}): {w['summary']}{tail}"
            )
            closed = closed_by_day_map.get(w["day"])
            if closed:
                names = ", ".join(p["name"] for p in closed["places"] if p.get("name"))
                lines.append(f"  · 휴무 추정({closed['weekday']}요일): {names}")
        if any(w.get("source") == "climate_avg" for w in weather):
            lines.append(
                f"- (참고치) 표시된 날짜는 Open-Meteo 예보 범위(최대 {WEATHER_FORECAST_MAX_DAYS}일) 밖이라 "
                "실제 예보 대신 최근 수년간 같은 시기의 평균 기온/날씨입니다. 정확한 예보가 아니니 "
                "출발일이 가까워지면 다시 확인하라고 안내하세요."
            )
        if any(w.get("source") == "unavailable" for w in weather):
            lines.append("- 일부 날짜는 예보/평균 데이터를 모두 가져오지 못했습니다.")
    elif closed_by_day:
        for closed in closed_by_day:
            names = ", ".join(p["name"] for p in closed["places"] if p.get("name"))
            lines.append(
                f"[Day{closed['day']} {closed['date']}({closed['weekday']}) 휴무 추정]: {names}"
            )

    text = "\n".join(lines) if lines else "(제공 가능한 실시간 정보 없음)"
    # closed_today: 하위 호환용(실제 서버 시각 기준). 신규 로직은 closed_by_day를 사용.
    closed_today = operating_notes(places, when=now)
    return {
        "generated_at": now.strftime("%Y-%m-%d %H:%M KST"),
        "start_date": trip_start.strftime("%Y-%m-%d"),
        "weather": weather,
        "closed_today": closed_today,
        "closed_by_day": closed_by_day,
        "text": f"제공 시각: {now.strftime('%Y-%m-%d %H:%M KST')}\n{text}",
    }
