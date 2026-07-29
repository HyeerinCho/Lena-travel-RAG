"""한국관광공사_관광지별 연관 관광지 정보 (TarRlteTarService1).

티맵 모빌리티 내비게이션 데이터 기반으로 산출된 관광지별 연관 관광지 통계.
오퍼레이션은 areaBasedList1 / searchKeyword1 2종. 응답에 "2" 접미사가 없습니다.
https://www.data.go.kr/data/15128560/openapi.do
"""

from __future__ import annotations

from typing import Any

from src.config import TAR_RLTE_TAR_BASE_URL

from ._client import paged, request

_BASE = TAR_RLTE_TAR_BASE_URL


def area_based_list(
    *,
    base_ym: str,
    area_cd: str,
    signgu_cd: str,
    page_no: int = 1,
    num_of_rows: int = 10,
) -> dict[str, Any]:
    """지역기반 관광지별 연관 관광지 목록 조회 (areaBasedList1).

    base_ym(baseYm): 조회 연월 YYYYMM (제공 범위 2024-05 ~ 2025-04)
    area_cd(areaCd): 지역 코드, signgu_cd(signguCd): 시군구 코드 (필수)
    """
    body = request(
        _BASE,
        "areaBasedList1",
        {
            "baseYm": base_ym,
            "areaCd": area_cd,
            "signguCd": signgu_cd,
            "pageNo": page_no,
            "numOfRows": num_of_rows,
        },
    )
    return paged(body)


def search_keyword(
    keyword: str,
    *,
    base_ym: str,
    area_cd: str | None = None,
    signgu_cd: str | None = None,
    page_no: int = 1,
    num_of_rows: int = 10,
) -> dict[str, Any]:
    """키워드 검색 관광지별 연관 관광지 목록 조회 (searchKeyword1)."""
    body = request(
        _BASE,
        "searchKeyword1",
        {
            "keyword": keyword,
            "baseYm": base_ym,
            "areaCd": area_cd,
            "signguCd": signgu_cd,
            "pageNo": page_no,
            "numOfRows": num_of_rows,
        },
    )
    return paged(body)
