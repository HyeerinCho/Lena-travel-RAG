"""한국관광공사_무장애 여행 정보 서비스 (KorWithService2).

장애인/고령자/영유아가족 등 모든 관광객이 이용 가능한 무장애 관광정보.
https://www.data.go.kr/data/15101897/openapi.do
"""

from __future__ import annotations

from typing import Any

from src.config import KOR_WITH_SERVICE_BASE_URL

from ._client import normalize_items, paged, request

_BASE = KOR_WITH_SERVICE_BASE_URL


def _call(operation: str, params: dict[str, Any]) -> dict[str, Any]:
    return request(_BASE, operation, params)


def area_based_list(
    *,
    area_code: str | None = None,
    sigungu_code: str | None = None,
    content_type_id: str | None = None,
    arrange: str = "C",
    page_no: int = 1,
    num_of_rows: int = 10,
) -> dict[str, Any]:
    """지역기반 무장애 관광정보 조회 (areaBasedList2)."""
    body = _call(
        "areaBasedList2",
        {
            "areaCode": area_code,
            "sigunguCode": sigungu_code,
            "contentTypeId": content_type_id,
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
    """위치기반 무장애 관광정보 조회 (locationBasedList2)."""
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
    """키워드 기반 무장애 관광정보 조회 (searchKeyword2)."""
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


def detail_intro(content_id: str, content_type_id: str) -> dict[str, Any] | None:
    """소개정보 조회 (detailIntro2)."""
    body = _call(
        "detailIntro2",
        {"contentId": content_id, "contentTypeId": content_type_id},
    )
    items = normalize_items(body)
    return items[0] if items else None


def detail_info(content_id: str, content_type_id: str) -> list[dict[str, Any]]:
    """반복정보 조회 (detailInfo2)."""
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


def detail_with_tour(content_id: str) -> list[dict[str, Any]]:
    """무장애 상세정보 조회 (detailWithTour2).

    장애유형별(지체/시각/청각 등) 이용 가능 시설·편의 정보를 반환합니다.
    """
    body = _call("detailWithTour2", {"contentId": content_id})
    return normalize_items(body)
