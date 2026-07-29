"""한국관광공사_반려동물 동반여행 서비스 (KorPetTourService2).

반려동물 동반이 가능한 관광지/숙소/음식점 등의 정보와 동반 조건/유의사항.
https://www.data.go.kr/data/15135102/openapi.do
"""

from __future__ import annotations

from typing import Any

from src.config import KOR_PET_TOUR_BASE_URL

from ._client import normalize_items, paged, request

_BASE = KOR_PET_TOUR_BASE_URL


def _call(operation: str, params: dict[str, Any]) -> dict[str, Any]:
    return request(_BASE, operation, params)


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
    """지역기반 반려동물 동반여행 정보 조회 (areaBasedList2)."""
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
    """위치기반 반려동물 동반여행 정보 조회 (locationBasedList2)."""
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
    """키워드 기반 반려동물 동반여행 정보 조회 (searchKeyword2)."""
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


def detail_common(content_id: str) -> dict[str, Any] | None:
    """공통정보 조회 (detailCommon2)."""
    body = _call("detailCommon2", {"contentId": content_id})
    items = normalize_items(body)
    return items[0] if items else None


def detail_pet_tour(
    *,
    content_id: str | None = None,
    page_no: int = 1,
    num_of_rows: int = 10,
) -> dict[str, Any]:
    """반려동물 동반여행 상세정보 조회 (detailPetTour2)."""
    body = _call(
        "detailPetTour2",
        {
            "contentId": content_id,
            "pageNo": page_no,
            "numOfRows": num_of_rows,
        },
    )
    return paged(body)
