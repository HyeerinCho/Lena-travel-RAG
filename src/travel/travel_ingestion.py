"""Normalize tourism POI JSON and travel course CSV into JSONL + SQLite-ready rows."""

from __future__ import annotations

import csv
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from langchain_core.documents import Document

from src.config import (
    TRAVEL_NORMALIZED_DIR,
    TRAVEL_POI_FULL_TYPES,
    TRAVEL_POI_SAMPLE_PER_REGION,
    TRAVEL_POI_SAMPLED_TYPES,
    course_csv_path,
    poi_data_root,
)

NAME_COLUMNS = ("관광지명", "시설명", "업소명", "음식점명")
ADDRESS_COLUMNS = ("주소(도로명)", "주소(지번주소)", "주소")
HOURS_COLUMNS = ("이용시간", "영업시간")
FEE_COLUMNS = ("입장료", "시설이용료")
NULLISH = {"", "null", "none", "없음", "Null", "None", "NULL"}

# Low-value venues frequently mislabeled as tourist attractions in the source data
SKIP_NAME_KEYWORDS = (
    "교회",
    "성당",
    "사찰",
    "약국",
    "대학원",
    "기숙사",
    "아이스링크",
    "편의점",
    "GS25",
    "CU ",
    "CU점",
    "세븐일레븐",
    "이마트24",
    "헬스장",
    "피트니스",
    "사우나",
    "목욕탕",
    "찜질방",
    "주유소",
    "은행",
    "우체국",
    "주민센터",
    "아파트",
    "공인중개",
    "노래방",
    "노래연습장",
    "코인노래",
    "PC방",
    "피시방",
    "당구장",
    "스크린골프",
)

REGION_FROM_CODE = {
    "su": "서울",
    "gg": "경기",
    "ic": "인천",
    "dj": "대전",
    "jj": "제주",
    "bs": "부산",
    "dg": "대구",
    "gj": "광주",
    "us": "울산",
    "sj": "세종",
    "gw": "강원",
    "cb": "충북",
    "cn": "충남",
    "jb": "전북",
    "jn": "전남",
    "gb": "경북",
    "gn": "경남",
}


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def clean_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in NULLISH or text.lower() in {"null", "none"}:
        return None
    return text


def parse_price(raw: str | None) -> int | None:
    text = clean_value(raw)
    if text is None:
        return None
    digits = re.sub(r"[^\d]", "", text)
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def parse_duration_days(raw: str | None) -> int | None:
    text = clean_value(raw)
    if text is None:
        return None
    # "2박 3일", "0박 1일", "3일", "6시간"
    nights = re.search(r"(\d+)\s*박", text)
    days = re.search(r"(\d+)\s*일", text)
    hours = re.search(r"(\d+)\s*시간", text)
    if days:
        return int(days.group(1))
    if nights:
        return int(nights.group(1)) + 1
    if hours:
        return 1
    return None


def parse_duration_nights(raw: str | None) -> int | None:
    text = clean_value(raw)
    if text is None:
        return None
    nights = re.search(r"(\d+)\s*박", text)
    if nights:
        return int(nights.group(1))
    days = parse_duration_days(text)
    if days is None:
        return None
    return max(days - 1, 0)


def normalize_name(name: str | None) -> str | None:
    text = clean_value(name)
    if text is None:
        return None
    # Drop trailing " (English)" style suffixes
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    text = re.sub(r"\s+", "", text)
    return text.casefold()


def _ann_map(annotations: list[dict]) -> dict[str, dict[str, str | None]]:
    result: dict[str, dict[str, str | None]] = {}
    for ann in annotations:
        key = clean_value(ann.get("k_column"))
        if not key:
            continue
        result[key] = {
            "ko": clean_value(ann.get("k_context")),
            "en": clean_value(ann.get("t_context")),
        }
    return result


def _first_field(
    fields: dict[str, dict[str, str | None]], columns: tuple[str, ...]
) -> dict[str, str | None]:
    for col in columns:
        if col in fields:
            return fields[col]
    return {"ko": None, "en": None}


def region_from_filename(path: Path) -> str | None:
    # en_{POI}_{권역}_{지역}_{유형}_{시설명}_{n}.json
    parts = path.stem.split("_")
    if len(parts) < 5:
        return None
    code = parts[3]
    return REGION_FROM_CODE.get(code, code)


def city_from_address(address: str | None, region: str | None) -> str | None:
    if not address:
        return region
    # "서울 중구 ..." / "제주특별자치도 제주시 ..."
    tokens = address.replace(",", " ").split()
    if not tokens:
        return region
    first = tokens[0]
    if first.endswith(("특별시", "광역시", "특별자치시", "특별자치도", "도")):
        if len(tokens) >= 2 and tokens[1].endswith(("시", "군", "구")):
            return tokens[1]
        return first
    if first.endswith(("시", "군", "구")):
        return first
    return region or first


def parse_poi_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    data = payload.get("data") or {}
    annotations = payload.get("annotations") or []
    if not data or not annotations:
        return None

    fields = _ann_map(annotations)
    name = _first_field(fields, NAME_COLUMNS)
    address = _first_field(fields, ADDRESS_COLUMNS)
    hours = _first_field(fields, HOURS_COLUMNS)
    fee = _first_field(fields, FEE_COLUMNS)
    phone = fields.get("대표번호", {})
    closed = fields.get("휴무일", {})
    parking = fields.get("주차시설 유무", fields.get("주차", {}))
    menu = fields.get("취급메뉴", {})
    homepage = fields.get("홈페이지 주소", {})
    checkin = fields.get("입실 시간", fields.get("입실시간", {}))
    checkout = fields.get("퇴실 시간", fields.get("퇴실시간", {}))

    travel_type = clean_value(data.get("travelType")) or fields.get("관광타입", {}).get("ko")
    region = region_from_filename(path)
    city = city_from_address(address.get("ko"), region)
    poi_id = clean_value(data.get("POI_id"))
    if not poi_id or not name.get("ko"):
        return None

    name_ko = name.get("ko") or ""
    if any(k in name_ko for k in SKIP_NAME_KEYWORDS):
        return None

    ko_lines = []
    en_lines = []
    for key, pair in fields.items():
        if pair.get("ko"):
            ko_lines.append(f"{key}: {pair['ko']}")
        if pair.get("en"):
            en_label = key
            en_lines.append(f"{en_label}: {pair['en']}")

    page_content = "\n".join(
        [
            f"[POI] {name.get('ko')} / {name.get('en') or ''}".strip(),
            f"유형: {travel_type}",
            f"지역: {city or region or ''}",
            "한국어 정보:",
            *ko_lines,
            "English:",
            *en_lines,
        ]
    )

    return {
        "poi_id": poi_id,
        "travel_type": travel_type,
        "name_ko": name.get("ko"),
        "name_en": name.get("en"),
        "name_norm": normalize_name(name.get("ko")),
        "region": region,
        "city": city,
        "address_ko": address.get("ko"),
        "address_en": address.get("en"),
        "phone": phone.get("ko") if isinstance(phone, dict) else None,
        "hours_ko": hours.get("ko"),
        "hours_en": hours.get("en"),
        "closed_ko": closed.get("ko") if isinstance(closed, dict) else None,
        "fee_ko": fee.get("ko"),
        "fee_en": fee.get("en"),
        "parking_ko": parking.get("ko") if isinstance(parking, dict) else None,
        "menu_ko": menu.get("ko") if isinstance(menu, dict) else None,
        "homepage": homepage.get("ko") if isinstance(homepage, dict) else None,
        "checkin": checkin.get("ko") if isinstance(checkin, dict) else None,
        "checkout": checkout.get("ko") if isinstance(checkout, dict) else None,
        "source": "poi",
        "source_path": str(path),
        "page_content": page_content,
    }


def _label_type_dirs(
    root: Path | None = None, *, include_validation: bool = False
) -> list[Path]:
    """Return only label folders for curated travel types (avoids scanning 350k files)."""
    root = root or poi_data_root()
    wanted_types = set(TRAVEL_POI_FULL_TYPES) | set(TRAVEL_POI_SAMPLED_TYPES)
    splits = ["Training"] + (["Validation"] if include_validation else [])
    dirs: list[Path] = []

    # Expected layout: .../01-1.*/{Training|Validation}/02.라벨링데이터/TL_영어_*
    for split in splits:
        split_dirs = [
            p
            for p in root.glob(f"*/{split}")
            if p.is_dir() and nfc(p.name) == split
        ]
        # Also tolerate deeper nesting
        if not split_dirs:
            split_dirs = [
                p for p in root.rglob(split) if p.is_dir() and nfc(p.name) == split
            ]
        for split_dir in split_dirs:
            label_dirs = [
                p
                for p in split_dir.iterdir()
                if p.is_dir() and "라벨링" in nfc(p.name)
            ]
            if not label_dirs:
                label_dirs = [
                    p
                    for p in split_dir.rglob("*")
                    if p.is_dir() and "라벨링" in nfc(p.name)
                ]
            for label_dir in label_dirs:
                for path in label_dir.iterdir():
                    if not path.is_dir():
                        continue
                    name = nfc(path.name)
                    if not (name.startswith("TL_영어_") or name.startswith("VL_영어_")):
                        continue
                    travel_type = name.split("_")[-1]
                    if travel_type in wanted_types:
                        dirs.append(path)
    return sorted(set(dirs))


def iter_poi_label_json_files(
    root: Path | None = None, *, include_validation: bool = False
) -> Iterator[Path]:
    for folder in _label_type_dirs(root, include_validation=include_validation):
        yield from folder.glob("*.json")


def collect_poi_records(
    *,
    include_validation: bool = False,
    sample_per_region: int = TRAVEL_POI_SAMPLE_PER_REGION,
    seed: int = 42,
    max_full: int | None = None,
    full_per_region: int | None = None,
) -> list[dict[str, Any]]:
    """Collect curated POI records for SQLite / FAISS.

    Full types (관광지/문화시설) are sampled per region so major destinations
    stay represented even when max_full is small.
    """
    rng = random.Random(seed)
    # (travel_type, region) -> rows
    full_buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    sampled_buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    sample_buffer = max(sample_per_region * 3, sample_per_region)
    # Default: spread full-type quota across regions
    per_region_full = full_per_region
    if per_region_full is None:
        if max_full is not None:
            per_region_full = max(20, max_full // 10)
        else:
            per_region_full = 10_000
    full_buffer = max(per_region_full * 3, per_region_full)

    for folder in _label_type_dirs(include_validation=include_validation):
        folder_type = nfc(folder.name).split("_")[-1]
        if folder_type not in TRAVEL_POI_FULL_TYPES and folder_type not in TRAVEL_POI_SAMPLED_TYPES:
            continue

        for path in folder.glob("*.json"):
            record = parse_poi_json(path)
            if not record:
                continue
            poi_id = record["poi_id"]
            if poi_id in seen:
                continue
            seen.add(poi_id)

            travel_type = record.get("travel_type") or folder_type
            region = record.get("region") or "unknown"
            key = (travel_type, region)

            if travel_type in TRAVEL_POI_FULL_TYPES:
                if len(full_buckets[key]) < full_buffer:
                    full_buckets[key].append(record)
            elif travel_type in TRAVEL_POI_SAMPLED_TYPES:
                if len(sampled_buckets[key]) < sample_buffer:
                    sampled_buckets[key].append(record)

    selected: list[dict[str, Any]] = []
    for key, rows in full_buckets.items():
        if len(rows) <= per_region_full:
            selected.extend(rows)
        else:
            selected.extend(rng.sample(rows, per_region_full))

    if max_full is not None:
        # Keep regional diversity while respecting global cap per full type
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            if row.get("travel_type") in TRAVEL_POI_FULL_TYPES:
                by_type[row["travel_type"]].append(row)
        trimmed: list[dict[str, Any]] = [
            r for r in selected if r.get("travel_type") not in TRAVEL_POI_FULL_TYPES
        ]
        for travel_type, rows in by_type.items():
            if len(rows) <= max_full:
                trimmed.extend(rows)
            else:
                trimmed.extend(rng.sample(rows, max_full))
        selected = trimmed

    for _key, rows in sampled_buckets.items():
        if len(rows) <= sample_per_region:
            selected.extend(rows)
        else:
            selected.extend(rng.sample(rows, sample_per_region))

    selected.sort(key=lambda r: (r.get("travel_type") or "", r.get("poi_id") or ""))
    return selected


def load_course_records() -> list[dict[str, Any]]:
    path = course_csv_path()
    records: list[dict[str, Any]] = []
    seen_keys: set[tuple[str | None, str | None, str | None]] = set()

    with path.open(encoding="cp949", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            package_id = clean_value(row.get("Package_Id"))
            title = clean_value(row.get("Package Title"))
            city = clean_value(row.get("City"))
            duration_raw = clean_value(row.get("Duration"))
            places = clean_value(row.get("Places to Visit"))
            price_raw = clean_value(row.get("Price"))
            departure_city = clean_value(row.get("Departure City"))
            departure_date = clean_value(row.get("Departure Date"))

            dedupe_key = (package_id, title, duration_raw)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)

            price = parse_price(price_raw)
            days = parse_duration_days(duration_raw)
            nights = parse_duration_nights(duration_raw)
            page_content = "\n".join(
                [
                    f"[COURSE] {title or ''}",
                    f"도시: {city or ''}",
                    f"기간: {duration_raw or ''}",
                    f"출발지: {departure_city or ''}",
                    f"방문지: {places or ''}",
                    f"참고가격: {price_raw or '없음'}",
                    "주의: 가격과 출발일은 과거 상품 참고값이며 실시간 예약 정보가 아닙니다.",
                ]
            )
            records.append(
                {
                    "package_id": package_id,
                    "title": title,
                    "city": city,
                    "duration_raw": duration_raw,
                    "duration_days": days,
                    "duration_nights": nights,
                    "departure_city": departure_city,
                    "departure_date": departure_date,
                    "places": places,
                    "price_ref": price,
                    "price_raw": price_raw,
                    "source": "course",
                    "page_content": page_content,
                }
            )
    return records


def records_to_documents(records: list[dict[str, Any]]) -> list[Document]:
    docs: list[Document] = []
    for row in records:
        metadata = {k: v for k, v in row.items() if k != "page_content" and v is not None}
        # FAISS metadata values should be simple types
        clean_meta = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)
        docs.append(Document(page_content=row["page_content"], metadata=clean_meta))
    return docs


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_travel_data(
    *,
    include_validation: bool = False,
    sample_per_region: int = TRAVEL_POI_SAMPLE_PER_REGION,
    max_full: int | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    out = output_dir or TRAVEL_NORMALIZED_DIR
    out.mkdir(parents=True, exist_ok=True)

    pois = collect_poi_records(
        include_validation=include_validation,
        sample_per_region=sample_per_region,
        max_full=max_full,
        full_per_region=None,
    )
    courses = load_course_records()

    poi_path = out / "pois.jsonl"
    course_path = out / "courses.jsonl"
    write_jsonl(pois, poi_path)
    write_jsonl(courses, course_path)

    print(f"POI 정규화 완료: {len(pois)}건 -> {poi_path}")
    print(f"코스 정규화 완료: {len(courses)}건 -> {course_path}")
    return {"pois": poi_path, "courses": course_path}
