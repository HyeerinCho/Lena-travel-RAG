"""한국관광공사_두루누비 정보 서비스 (Durunubi).

코리아둘레길(해파랑길/남파랑길/서해랑길/DMZ 평화의 길 등) 걷기여행길 코스와
경로 GPX 정보. 오퍼레이션은 courseList / routeList 2종.
https://www.data.go.kr/data/15101974/openapi.do
"""

from __future__ import annotations

from typing import Any

from src.config import DURUNUBI_BASE_URL

from ._client import paged, request

_BASE = DURUNUBI_BASE_URL


def course_list(
    *,
    course_name: str | None = None,
    brd_div: str | None = None,
    route_idx: str | None = None,
    page_no: int = 1,
    num_of_rows: int = 10,
) -> dict[str, Any]:
    """코스 목록 정보 조회 (courseList).

    course_name(crsKorNm): 코스명 검색어
    brd_div(brdDiv): DNWW=걷기여행길, DNWA=자전거길 등 구분
    route_idx(routeIdx): 노선(테마) 식별자
    """
    body = request(
        _BASE,
        "courseList",
        {
            "crsKorNm": course_name,
            "brdDiv": brd_div,
            "routeIdx": route_idx,
            "pageNo": page_no,
            "numOfRows": num_of_rows,
        },
    )
    return paged(body)


def route_list(
    *,
    route_idx: str | None = None,
    crs_idx: str | None = None,
    brd_div: str | None = None,
    page_no: int = 1,
    num_of_rows: int = 10,
) -> dict[str, Any]:
    """길(경로) 목록 정보 조회 (routeList).

    crs_idx(crsIdx): 코스 식별자. route_idx(routeIdx): 노선 식별자.
    각 코스의 GPX 경로 좌표/구간 정보를 반환합니다.
    """
    body = request(
        _BASE,
        "routeList",
        {
            "routeIdx": route_idx,
            "crsIdx": crs_idx,
            "brdDiv": brd_div,
            "pageNo": page_no,
            "numOfRows": num_of_rows,
        },
    )
    return paged(body)
