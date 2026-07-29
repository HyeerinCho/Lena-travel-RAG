"""한국관광공사 data.go.kr(B551011) 오픈 API FastAPI 라우터.

5개 서비스를 prefix 로 구분해 노출합니다.
  /travel/kor/*        국문 관광정보 (KorService2)
  /travel/pet/*        반려동물 동반여행 (KorPetTourService2)
  /travel/with/*       무장애 여행 (KorWithService2)
  /travel/durunubi/*   두루누비 걷기여행길 (Durunubi)
  /travel/related/*    관광지별 연관 관광지 (TarRlteTarService1)
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Query

from src.travel.tour_api import (
    TourAPIError,
    durunubi,
    kor_pet_tour,
    kor_service,
    kor_with_service,
    tar_rlte_tar,
)

router = APIRouter(prefix="/travel", tags=["tourapi"])


def _call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """TourAPIError 를 502 Bad Gateway 로 변환."""
    try:
        return fn(*args, **kwargs)
    except TourAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# KorService2 — 국문 관광정보
# ---------------------------------------------------------------------------


@router.get("/kor/area-codes")
def kor_area_codes(
    area_code: str | None = Query(default=None, description="상위 지역코드(미지정 시 광역시도)"),
    num_of_rows: int = Query(default=50, ge=1, le=100),
) -> list[dict[str, Any]]:
    return _call(kor_service.area_code, area_code, num_of_rows=num_of_rows)


@router.get("/kor/category-codes")
def kor_category_codes(
    content_type_id: str | None = Query(default=None),
    cat1: str | None = Query(default=None),
    cat2: str | None = Query(default=None),
    cat3: str | None = Query(default=None),
) -> list[dict[str, Any]]:
    return _call(
        kor_service.category_code,
        content_type_id=content_type_id,
        cat1=cat1,
        cat2=cat2,
        cat3=cat3,
    )


@router.get("/kor/area")
def kor_area(
    area_code: str | None = Query(default=None),
    sigungu_code: str | None = Query(default=None),
    content_type_id: str | None = Query(default=None),
    arrange: str = Query(default="C"),
    page_no: int = Query(default=1, ge=1),
    num_of_rows: int = Query(default=10, ge=1, le=100),
) -> dict[str, Any]:
    return _call(
        kor_service.area_based_list,
        area_code=area_code,
        sigungu_code=sigungu_code,
        content_type_id=content_type_id,
        arrange=arrange,
        page_no=page_no,
        num_of_rows=num_of_rows,
    )


@router.get("/kor/nearby")
def kor_nearby(
    map_x: float = Query(..., description="경도(longitude)"),
    map_y: float = Query(..., description="위도(latitude)"),
    radius: int = Query(default=2000, ge=1, le=20000),
    content_type_id: str | None = Query(default=None),
    page_no: int = Query(default=1, ge=1),
    num_of_rows: int = Query(default=10, ge=1, le=100),
) -> dict[str, Any]:
    return _call(
        kor_service.location_based_list,
        map_x=map_x,
        map_y=map_y,
        radius=radius,
        content_type_id=content_type_id,
        page_no=page_no,
        num_of_rows=num_of_rows,
    )


@router.get("/kor/search")
def kor_search(
    keyword: str = Query(...),
    area_code: str | None = Query(default=None),
    content_type_id: str | None = Query(default=None),
    page_no: int = Query(default=1, ge=1),
    num_of_rows: int = Query(default=10, ge=1, le=100),
) -> dict[str, Any]:
    return _call(
        kor_service.search_keyword,
        keyword,
        area_code=area_code,
        content_type_id=content_type_id,
        page_no=page_no,
        num_of_rows=num_of_rows,
    )


@router.get("/kor/festivals")
def kor_festivals(
    event_start_date: str = Query(..., description="행사 시작일 YYYYMMDD"),
    event_end_date: str | None = Query(default=None, description="행사 종료일 YYYYMMDD"),
    area_code: str | None = Query(default=None),
    page_no: int = Query(default=1, ge=1),
    num_of_rows: int = Query(default=10, ge=1, le=100),
) -> dict[str, Any]:
    return _call(
        kor_service.search_festival,
        event_start_date=event_start_date,
        event_end_date=event_end_date,
        area_code=area_code,
        page_no=page_no,
        num_of_rows=num_of_rows,
    )


@router.get("/kor/stays")
def kor_stays(
    area_code: str | None = Query(default=None),
    sigungu_code: str | None = Query(default=None),
    page_no: int = Query(default=1, ge=1),
    num_of_rows: int = Query(default=10, ge=1, le=100),
) -> dict[str, Any]:
    return _call(
        kor_service.search_stay,
        area_code=area_code,
        sigungu_code=sigungu_code,
        page_no=page_no,
        num_of_rows=num_of_rows,
    )


@router.get("/kor/detail/{content_id}")
def kor_detail(
    content_id: str,
    content_type_id: str | None = Query(
        default=None, description="소개/반복 정보 조회에 필요"
    ),
) -> dict[str, Any]:
    """공통 + (선택)소개/반복 + 이미지 정보를 한 번에 조회."""
    common = _call(kor_service.detail_common, content_id)
    images = _call(kor_service.detail_image, content_id)
    intro = None
    info: list[dict[str, Any]] = []
    if content_type_id:
        intro = _call(kor_service.detail_intro, content_id, content_type_id)
        info = _call(kor_service.detail_info, content_id, content_type_id)
    if common is None and not images:
        raise HTTPException(status_code=404, detail="콘텐츠를 찾을 수 없습니다.")
    return {
        "content_id": content_id,
        "common": common,
        "intro": intro,
        "info": info,
        "images": images,
    }


# ---------------------------------------------------------------------------
# KorPetTourService2 — 반려동물 동반여행
# ---------------------------------------------------------------------------


@router.get("/pet/search")
def pet_search(
    keyword: str = Query(...),
    area_code: str | None = Query(default=None),
    content_type_id: str | None = Query(default=None),
    page_no: int = Query(default=1, ge=1),
    num_of_rows: int = Query(default=10, ge=1, le=100),
) -> dict[str, Any]:
    return _call(
        kor_pet_tour.search_keyword,
        keyword,
        area_code=area_code,
        content_type_id=content_type_id,
        page_no=page_no,
        num_of_rows=num_of_rows,
    )


@router.get("/pet/area")
def pet_area(
    area_code: str | None = Query(default=None),
    sigungu_code: str | None = Query(default=None),
    content_type_id: str | None = Query(default=None),
    arrange: str = Query(default="C"),
    page_no: int = Query(default=1, ge=1),
    num_of_rows: int = Query(default=10, ge=1, le=100),
) -> dict[str, Any]:
    return _call(
        kor_pet_tour.area_based_list,
        area_code=area_code,
        sigungu_code=sigungu_code,
        content_type_id=content_type_id,
        arrange=arrange,
        page_no=page_no,
        num_of_rows=num_of_rows,
    )


@router.get("/pet/nearby")
def pet_nearby(
    map_x: float = Query(..., description="경도(longitude)"),
    map_y: float = Query(..., description="위도(latitude)"),
    radius: int = Query(default=2000, ge=1, le=20000),
    content_type_id: str | None = Query(default=None),
    page_no: int = Query(default=1, ge=1),
    num_of_rows: int = Query(default=10, ge=1, le=100),
) -> dict[str, Any]:
    return _call(
        kor_pet_tour.location_based_list,
        map_x=map_x,
        map_y=map_y,
        radius=radius,
        content_type_id=content_type_id,
        page_no=page_no,
        num_of_rows=num_of_rows,
    )


@router.get("/pet/{content_id}")
def pet_detail(
    content_id: str,
) -> dict[str, Any]:
    """콘텐츠의 공통 정보 + 반려동물 동반 상세정보 조회."""
    common = _call(kor_pet_tour.detail_common, content_id)
    pet_info = _call(kor_pet_tour.detail_pet_tour, content_id=content_id)
    if common is None and not pet_info["items"]:
        raise HTTPException(status_code=404, detail="콘텐츠를 찾을 수 없습니다.")
    return {
        "content_id": content_id,
        "common": common,
        "pet_tour": pet_info["items"],
    }


# ---------------------------------------------------------------------------
# KorWithService2 — 무장애 여행
# ---------------------------------------------------------------------------


@router.get("/with/area")
def with_area(
    area_code: str | None = Query(default=None),
    sigungu_code: str | None = Query(default=None),
    content_type_id: str | None = Query(default=None),
    arrange: str = Query(default="C"),
    page_no: int = Query(default=1, ge=1),
    num_of_rows: int = Query(default=10, ge=1, le=100),
) -> dict[str, Any]:
    return _call(
        kor_with_service.area_based_list,
        area_code=area_code,
        sigungu_code=sigungu_code,
        content_type_id=content_type_id,
        arrange=arrange,
        page_no=page_no,
        num_of_rows=num_of_rows,
    )


@router.get("/with/nearby")
def with_nearby(
    map_x: float = Query(..., description="경도(longitude)"),
    map_y: float = Query(..., description="위도(latitude)"),
    radius: int = Query(default=2000, ge=1, le=20000),
    content_type_id: str | None = Query(default=None),
    page_no: int = Query(default=1, ge=1),
    num_of_rows: int = Query(default=10, ge=1, le=100),
) -> dict[str, Any]:
    return _call(
        kor_with_service.location_based_list,
        map_x=map_x,
        map_y=map_y,
        radius=radius,
        content_type_id=content_type_id,
        page_no=page_no,
        num_of_rows=num_of_rows,
    )


@router.get("/with/search")
def with_search(
    keyword: str = Query(...),
    area_code: str | None = Query(default=None),
    content_type_id: str | None = Query(default=None),
    page_no: int = Query(default=1, ge=1),
    num_of_rows: int = Query(default=10, ge=1, le=100),
) -> dict[str, Any]:
    return _call(
        kor_with_service.search_keyword,
        keyword,
        area_code=area_code,
        content_type_id=content_type_id,
        page_no=page_no,
        num_of_rows=num_of_rows,
    )


@router.get("/with/detail/{content_id}")
def with_detail(
    content_id: str,
    content_type_id: str | None = Query(default=None),
) -> dict[str, Any]:
    """공통 정보 + 무장애 상세정보(detailWithTour2) 조회."""
    common = _call(kor_with_service.detail_common, content_id)
    with_tour = _call(kor_with_service.detail_with_tour, content_id)
    intro = None
    if content_type_id:
        intro = _call(kor_with_service.detail_intro, content_id, content_type_id)
    if common is None and not with_tour:
        raise HTTPException(status_code=404, detail="콘텐츠를 찾을 수 없습니다.")
    return {
        "content_id": content_id,
        "common": common,
        "intro": intro,
        "barrier_free": with_tour,
    }


# ---------------------------------------------------------------------------
# Durunubi — 두루누비 걷기여행길
# ---------------------------------------------------------------------------


@router.get("/durunubi/courses")
def durunubi_courses(
    course_name: str | None = Query(default=None, description="코스명 검색어"),
    brd_div: str | None = Query(default=None, description="걷기/자전거 구분(DNWW 등)"),
    route_idx: str | None = Query(default=None, description="노선 식별자"),
    page_no: int = Query(default=1, ge=1),
    num_of_rows: int = Query(default=10, ge=1, le=100),
) -> dict[str, Any]:
    return _call(
        durunubi.course_list,
        course_name=course_name,
        brd_div=brd_div,
        route_idx=route_idx,
        page_no=page_no,
        num_of_rows=num_of_rows,
    )


@router.get("/durunubi/routes")
def durunubi_routes(
    route_idx: str | None = Query(default=None, description="노선 식별자"),
    crs_idx: str | None = Query(default=None, description="코스 식별자"),
    brd_div: str | None = Query(default=None),
    page_no: int = Query(default=1, ge=1),
    num_of_rows: int = Query(default=10, ge=1, le=100),
) -> dict[str, Any]:
    return _call(
        durunubi.route_list,
        route_idx=route_idx,
        crs_idx=crs_idx,
        brd_div=brd_div,
        page_no=page_no,
        num_of_rows=num_of_rows,
    )


# ---------------------------------------------------------------------------
# TarRlteTarService1 — 관광지별 연관 관광지
# ---------------------------------------------------------------------------


@router.get("/related/area")
def related_area(
    base_ym: str = Query(..., description="조회 연월 YYYYMM (2024-05~2025-04)"),
    area_cd: str = Query(..., description="지역 코드"),
    signgu_cd: str = Query(..., description="시군구 코드"),
    page_no: int = Query(default=1, ge=1),
    num_of_rows: int = Query(default=10, ge=1, le=100),
) -> dict[str, Any]:
    return _call(
        tar_rlte_tar.area_based_list,
        base_ym=base_ym,
        area_cd=area_cd,
        signgu_cd=signgu_cd,
        page_no=page_no,
        num_of_rows=num_of_rows,
    )


@router.get("/related/search")
def related_search(
    keyword: str = Query(...),
    base_ym: str = Query(..., description="조회 연월 YYYYMM"),
    area_cd: str | None = Query(default=None),
    signgu_cd: str | None = Query(default=None),
    page_no: int = Query(default=1, ge=1),
    num_of_rows: int = Query(default=10, ge=1, le=100),
) -> dict[str, Any]:
    return _call(
        tar_rlte_tar.search_keyword,
        keyword,
        base_ym=base_ym,
        area_cd=area_cd,
        signgu_cd=signgu_cd,
        page_no=page_no,
        num_of_rows=num_of_rows,
    )
