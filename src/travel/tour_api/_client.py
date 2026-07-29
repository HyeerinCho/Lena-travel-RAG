"""한국관광공사 data.go.kr (B551011) 공통 HTTP 클라이언트.

모든 서비스(KorService2, KorPetTourService2, KorWithService2, Durunubi,
TarRlteTarService1)가 동일한 인증/응답 규격을 공유하므로 여기에 모읍니다.
기존 프로젝트 규칙에 맞춰 외부 SDK 없이 stdlib urllib 로 호출합니다.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from src.config import (
    DATA_GO_KR_API_KEY,
    DATA_GO_KR_MOBILE_APP,
    DATA_GO_KR_MOBILE_OS,
    DATA_GO_KR_REQUEST_TIMEOUT_SEC,
)


class TourAPIError(RuntimeError):
    """data.go.kr(B551011) 호출 실패 시 발생."""


def _service_key() -> str:
    key = (DATA_GO_KR_API_KEY or "").strip()
    if not key or key == "[APIkey]":
        raise TourAPIError(
            "DATA_GO_KR_API_KEY 가 설정되지 않았습니다. "
            ".env 에 data.go.kr 인증키(Decoding)를 넣어주세요."
        )
    return key


def _drop_none(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if v is not None}


def _extract_service_error(payload: dict[str, Any]) -> str | None:
    """data.go.kr 게이트웨이 레벨 에러(cmmMsgHeader)를 사람이 읽을 문자열로."""
    for key in ("OpenAPI_ServiceResponse", "response"):
        node = payload.get(key)
        if isinstance(node, dict):
            header = node.get("cmmMsgHeader")
            if isinstance(header, dict):
                code = header.get("returnReasonCode") or header.get("errMsg")
                msg = header.get("returnAuthMsg") or header.get("errMsg")
                return f"[{code}] {msg}"
    return None


def request(
    base_url: str,
    operation: str,
    params: dict[str, Any],
    *,
    include_type: bool = True,
) -> dict[str, Any]:
    """오퍼레이션을 호출하고 response.body(dict) 를 반환.

    include_type=False 이면 _type 파라미터를 붙이지 않습니다(일부 레거시 대응).
    """
    query: dict[str, Any] = {
        "serviceKey": _service_key(),
        "MobileOS": DATA_GO_KR_MOBILE_OS,
        "MobileApp": DATA_GO_KR_MOBILE_APP,
    }
    if include_type:
        query["_type"] = "json"
    query.update(_drop_none(params))

    url = f"{base_url}/{operation}?{urllib.parse.urlencode(query)}"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(
            req, timeout=DATA_GO_KR_REQUEST_TIMEOUT_SEC
        ) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # pragma: no cover - 네트워크 의존
        raise TourAPIError(f"HTTP {exc.code} 오류: {exc.reason}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - 네트워크 의존
        raise TourAPIError(f"네트워크 오류: {exc.reason}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        # 인증키 오류/트래픽 초과 시 XML 에러를 반환하기도 함
        snippet = raw.strip()[:300]
        raise TourAPIError(f"JSON 파싱 실패 (응답: {snippet})") from exc

    service_error = _extract_service_error(payload)
    if service_error:
        raise TourAPIError(f"API 게이트웨이 오류 {service_error}")

    response = payload.get("response") or {}
    header = response.get("header") or {}
    result_code = str(header.get("resultCode", ""))
    if result_code and result_code != "0000":
        raise TourAPIError(
            f"API 오류 [{result_code}] {header.get('resultMsg', '알 수 없는 오류')}"
        )
    return response.get("body") or {}


def normalize_items(body: dict[str, Any]) -> list[dict[str, Any]]:
    """response.body.items.item 을 항상 list 로 정규화."""
    items = body.get("items")
    if not items or not isinstance(items, dict):
        return []
    item = items.get("item")
    if item is None:
        return []
    if isinstance(item, dict):
        return [item]
    if isinstance(item, list):
        return item
    return []


def paged(body: dict[str, Any]) -> dict[str, Any]:
    """목록형 응답을 표준 형태로 변환."""
    return {
        "items": normalize_items(body),
        "page_no": body.get("pageNo"),
        "num_of_rows": body.get("numOfRows"),
        "total_count": body.get("totalCount"),
    }
