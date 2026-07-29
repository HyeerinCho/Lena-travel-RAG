"""한국관광공사_국문 관광정보 서비스 (KorService2, TourAPI 4.0).

https://www.data.go.kr/data/15101578/openapi.do
"""

from __future__ import annotations

from typing import Any

from src.config import KOR_SERVICE_BASE_URL

from ._client import normalize_items, paged, request

_BASE = KOR_SERVICE_BASE_URL


def _call(operation: str, params: dict[str, Any]) -> dict[str, Any]:
    return request(_BASE, operation, params)


def area_code(area_code: str | None = None, *, num_of_rows: int = 50) -> list[dict[str, Any]]:
    """지역코드 조회 (areaCode2). area_code 미지정 시 광역시도 목록."""
    body = _call("areaCode2", {"areaCode": area_code, "numOfRows": num_of_rows})
    return normalize_items(body)


def category_code(
    *,
    content_type_id: str | None = None,
    cat1: str | None = None,
    cat2: str | None = None,
    cat3: str | None = None,
    num_of_rows: int = 100,
) -> list[dict[str, Any]]:
    """서비스 분류코드 조회 (categoryCode2)."""
    body = _call(
        "categoryCode2",
        {
            "contentTypeId": content_type_id,
            "cat1": cat1,
            "cat2": cat2,
            "cat3": cat3,
            "numOfRows": num_of_rows,
        },
    )
    return normalize_items(body)


def area_based_list(
    *,
    area_code: str | None = None,
    sigungu_code: str | None = None,
    content_type_id: str | None = None,
    cat1: str | None = None,
    cat2: str | None = None,
    cat3: str | None = None,
    arrange: str = "C",
    page_no: int = 1,
    num_of_rows: int = 10,
) -> dict[str, Any]:
    """지역기반 관광정보 조회 (areaBasedList2)."""
    body = _call(
        "areaBasedList2",
        {
            "areaCode": area_code,
            "sigunguCode": sigungu_code,
            "contentTypeId": content_type_id,
            "cat1": cat1,
            "cat2": cat2,
            "cat3": cat3,
            "arrange": arrange,
            "pageNo": page_no,
            "numOfRows": num_of_rows,
        },
    )
    return paged(body)


def location_based_list(
    *,
    map_x: float,
    map_y: float,
    radius: int = 2000,
    content_type_id: str | None = None,
    arrange: str = "E",
    page_no: int = 1,
    num_of_rows: int = 10,
) -> dict[str, Any]:
    """위치기반 관광정보 조회 (locationBasedList2). map_x=경도, map_y=위도."""
    body = _call(
        "locationBasedList2",
        {
            "mapX": map_x,
            "mapY": map_y,
            "radius": radius,
            "contentTypeId": content_type_id,
            "arrange": arrange,
            "pageNo": page_no,
            "numOfRows": num_of_rows,
        },
    )
    return paged(body)


def search_keyword(
    keyword: str,
    *,
    area_code: str | None = None,
    sigungu_code: str | None = None,
    content_type_id: str | None = None,
    arrange: str = "C",
    page_no: int = 1,
    num_of_rows: int = 10,
) -> dict[str, Any]:
    """키워드 검색 (searchKeyword2)."""
    body = _call(
        "searchKeyword2",
        {
            "keyword": keyword,
            "areaCode": area_code,
            "sigunguCode": sigungu_code,
            "contentTypeId": content_type_id,
            "arrange": arrange,
            "pageNo": page_no,
            "numOfRows": num_of_rows,
        },
    )
    return paged(body)


def search_festival(
    *,
    event_start_date: str,
    event_end_date: str | None = None,
    area_code: str | None = None,
    sigungu_code: str | None = None,
    arrange: str = "C",
    page_no: int = 1,
    num_of_rows: int = 10,
) -> dict[str, Any]:
    """행사정보 조회 (searchFestival2). event_start_date 형식 YYYYMMDD."""
    body = _call(
        "searchFestival2",
        {
            "eventStartDate": event_start_date,
            "eventEndDate": event_end_date,
            "areaCode": area_code,
            "sigunguCode": sigungu_code,
            "arrange": arrange,
            "pageNo": page_no,
            "numOfRows": num_of_rows,
        },
    )
    return paged(body)


def search_stay(
    *,
    area_code: str | None = None,
    sigungu_code: str | None = None,
    arrange: str = "C",
    page_no: int = 1,
    num_of_rows: int = 10,
) -> dict[str, Any]:
    """숙박정보 조회 (searchStay2)."""
    body = _call(
        "searchStay2",
        {
            "areaCode": area_code,
            "sigunguCode": sigungu_code,
            "arrange": arrange,
            "pageNo": page_no,
            "numOfRows": num_of_rows,
        },
    )
    return paged(body)


def detail_common(content_id: str) -> dict[str, Any] | None:
    """공통정보 조회 (detailCommon2)."""
    body = _call("detailCommon2", {"contentId": content_id})
    items = normalize_items(body)
    return items[0] if items else None


def detail_intro(content_id: str, content_type_id: str) -> dict[str, Any] | None:
    """소개정보 조회 (detailIntro2). content_type_id 필수."""
    body = _call(
        "detailIntro2",
        {"contentId": content_id, "contentTypeId": content_type_id},
    )
    items = normalize_items(body)
    return items[0] if items else None


def detail_info(content_id: str, content_type_id: str) -> list[dict[str, Any]]:
    """반복정보 조회 (detailInfo2). content_type_id 필수."""
    body = _call(
        "detailInfo2",
        {"contentId": content_id, "contentTypeId": content_type_id},
    )
    return normalize_items(body)


def detail_image(content_id: str, *, num_of_rows: int = 20) -> list[dict[str, Any]]:
    """이미지정보 조회 (detailImage2)."""
    body = _call(
        "detailImage2",
        {"contentId": content_id, "imageYN": "Y", "numOfRows": num_of_rows},
    )
    return normalize_items(body)
