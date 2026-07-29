# 관광공사 오픈 API → LLM 모델 통합 흐름

> 한국관광공사 data.go.kr(B551011) 오픈 API가 **가장 낮은 HTTP 엔드포인트 호출**부터
> 최종적으로 **LLM(Gemini) 모델의 답변**에 반영되기까지 어떤 순서로 통합되는지 정리한 문서입니다.

---

## 0. 한눈에 보는 전체 그림

```
[외부] data.go.kr B551011 오픈 API
        │  (HTTP GET)
        ▼
① config.py            ← 인증키 · Base URL · 타임아웃 등 설정값
        │
        ▼
② tour_api/_client.py  ← 공통 HTTP 클라이언트 (request/정규화/에러처리)
        │
        ▼
③ tour_api/*.py        ← 서비스별 래퍼 (kor_service, kor_pet_tour, durunubi ...)
        │
        ├───────────────────────────┐
        ▼                           ▼
④-A tour_routes.py            ④-B tour_enrich.py
   (원본 데이터를 REST로 그대로 노출)   (질문 의도 감지 → 보강 텍스트 생성)
                                    │
                                    ▼
                            ⑤ travel_graph.py
                               search_candidates 노드에서 build_tour_context() 호출
                               → state["external_text"] 에 저장
                                    │
                                    ▼
                            ⑥ build_itinerary 노드
                               external_text 를 realtime_text 에 합쳐 프롬프트에 주입
                                    │
                                    ▼
                            ⑦ LLM(Gemini) 호출 → 일정 답변 생성
                                    │
                                    ▼
                            ⑧ travel_agent.py (facade) → api.py (FastAPI 응답)
```

핵심 요약: **관광공사 API는 두 갈래로 쓰입니다.**
- **④-A 경로**: 데이터를 가공 없이 그대로 REST 엔드포인트(`/travel/kor/*` 등)로 노출 → 프론트/외부에서 직접 사용.
- **④-B → ⑦ 경로**: 데이터를 "보강 컨텍스트(텍스트)"로 만들어 **LLM 프롬프트에 끼워 넣어** 답변에 반영.

이 문서의 주제인 "엔드포인트부터 모델에 통합"은 주로 **④-B → ⑦ 경로**입니다.

---

## 1단계 — 설정값 정의 (`src/config.py`)

모든 호출의 출발점. 어디로 어떤 키로 요청할지 정의합니다.

- `DATA_GO_KR_API_KEY` : data.go.kr 인증키(Decoding 키). `.env` 에서 주입, 없으면 `"[APIkey]"` 플레이스홀더.
- Base URL 5종을 한 곳에서 관리:

```50:55:src/config.py
_TOURAPI_BASE = "https://apis.data.go.kr/B551011"
KOR_SERVICE_BASE_URL = f"{_TOURAPI_BASE}/KorService2"
KOR_PET_TOUR_BASE_URL = f"{_TOURAPI_BASE}/KorPetTourService2"
KOR_WITH_SERVICE_BASE_URL = f"{_TOURAPI_BASE}/KorWithService2"
DURUNUBI_BASE_URL = f"{_TOURAPI_BASE}/Durunubi"
TAR_RLTE_TAR_BASE_URL = f"{_TOURAPI_BASE}/TarRlteTarService1"
```

> 5개 서비스(국문 관광정보 / 반려동물 / 무장애 / 두루누비 / 연관관광지)가 **같은 인증키**를 공유합니다.

---

## 2단계 — 공통 HTTP 클라이언트 (`src/travel/tour_api/_client.py`)

실제 네트워크 요청을 담당하는 **가장 낮은 레벨의 엔드포인트 호출부**. 외부 SDK 없이 `urllib`만 사용합니다.

`request()` 함수가 하는 일 (순서대로):
1. 공통 쿼리 파라미터 조립 — `serviceKey`, `MobileOS`, `MobileApp`, `_type=json`.
2. `{base_url}/{operation}?{params}` 형태로 URL 생성 후 `urllib`로 GET 호출.
3. 응답을 JSON 파싱. (인증 실패 시 XML이 오면 `TourAPIError` 발생)
4. 게이트웨이 에러(`cmmMsgHeader`)와 서비스 에러(`resultCode != "0000"`) 검사.
5. 성공 시 `response.body`(dict) 반환.

부가 유틸:
- `normalize_items(body)` : `items.item` 을 항상 **list** 로 정규화 (단건이면 `[dict]`).
- `paged(body)` : 목록형 응답을 `{items, page_no, num_of_rows, total_count}` 표준 형태로 변환.
- `TourAPIError` : 이 레이어의 모든 실패를 하나의 예외로 통일 → 상위에서 일관되게 처리.

---

## 3단계 — 서비스별 래퍼 (`src/travel/tour_api/*.py`)

각 오픈 API 서비스를 파이썬 함수로 감싼 얇은 레이어. 오퍼레이션명·파라미터명(카멜케이스)을 숨기고, 읽기 쉬운 함수 시그니처를 제공합니다.

| 모듈 | 서비스 | 대표 함수 |
|------|--------|-----------|
| `kor_service.py` | 국문 관광정보 KorService2 | `search_keyword`, `area_based_list`, `detail_common`, `detail_image` ... |
| `kor_pet_tour.py` | 반려동물 동반여행 | `search_keyword`, `area_based_list`, `detail_pet_tour` |
| `kor_with_service.py` | 무장애 여행 | `search_keyword`, `area_based_list`, `detail_with_tour` |
| `durunubi.py` | 두루누비 걷기여행길 | `course_list`, `route_list` |
| `tar_rlte_tar.py` | 관광지별 연관 관광지 | `area_based_list`, `search_keyword` |

예시 — 키워드 검색은 결국 `_client.request()` 로 위임됩니다.

```105:128:src/travel/tour_api/kor_service.py
def search_keyword(
    keyword: str,
    *,
    ...
) -> dict[str, Any]:
    """키워드 검색 (searchKeyword2)."""
    body = _call(
        "searchKeyword2",
        {
            "keyword": keyword,
            ...
        },
    )
    return paged(body)
```

> 여기까지가 "**엔드포인트**" 계층입니다. (설정 → HTTP 클라이언트 → 서비스 함수)

---

## 4단계 — 두 갈래로 나뉘는 소비 지점

### ④-A. REST 엔드포인트로 그대로 노출 (`src/tour_routes.py`)

관광공사 데이터를 **가공 없이** FastAPI 라우터(`/travel/...`)로 외부에 열어줍니다. LLM과는 무관한 "직통" 경로입니다.

- 라우터 prefix: `/travel`, 서비스별로 `/kor/*`, `/pet/*`, `/with/*`, `/durunubi/*`, `/related/*`.
- `_call()` 헬퍼가 `TourAPIError` → **HTTP 502**로 변환.

```29:34:src/tour_routes.py
def _call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """TourAPIError 를 502 Bad Gateway 로 변환."""
    try:
        return fn(*args, **kwargs)
    except TourAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
```

### ④-B. 보강 컨텍스트 생성 (`src/travel/tour_enrich.py`) ← 모델 통합의 핵심 진입점

사용자 질문을 분석해 **관광공사 데이터를 LLM이 읽을 수 있는 텍스트**로 만듭니다.

1. **의도 감지** `detect_kinds(query)` — 정규식으로 질문에서 의도 추출:
   - `pet`(반려동물), `barrier`(무장애), `trail`(걷기여행길)
2. **지역코드 변환** `resolve_area_code(destination)` — "제주" → `39` 처럼 도시명을 areaCode로 매핑.
3. **데이터 수집** — 감지된 의도에 맞는 서비스 함수 호출 (`kor_pet_tour`, `kor_with_service`, `durunubi`).
   - **실패해도 조용히 빈 결과 반환** → 챗봇 기본 동작을 절대 깨지 않음.
4. **텍스트 조립** `build_tour_context()` — 사람이 읽는 형태의 섹션 텍스트로 변환해 반환:

```227:231:src/travel/tour_enrich.py
    return {
        "kinds": [k for k in kinds if k in data],
        "data": data,
        "text": "\n".join(sections),
    }
```

> `text` 필드가 곧 프롬프트에 주입될 문자열, `data` 는 응답 JSON의 `external` 필드로 나갑니다.

---

## 5단계 — 그래프에서 컨텍스트 수집 (`src/travel/travel_graph.py` · `search_candidates` 노드)

LangGraph 여행 그래프는 3개 노드로 구성됩니다:

```
extract_requirements → search_candidates → build_itinerary
```

`search_candidates` 노드는 내부 DB(SQLite+FAISS)에서 후보 장소를 찾은 뒤,
**관광공사 보강 컨텍스트도 함께 수집**해 그래프 상태(state)에 저장합니다.

```694:702:src/travel/travel_graph.py
        tour = build_tour_context(
            state.get("query") or "",
            destination,
            language=state.get("language") or "ko",
        )
        if tour:
            result["external"] = tour.get("data") or {}
            result["external_text"] = tour.get("text") or ""
        return result
```

- `external_text` : ⑥에서 프롬프트에 주입될 텍스트.
- `external` : 원본 데이터. 최종 응답의 `external` 필드로 그대로 전달.

---

## 6단계 — 프롬프트에 주입 (`build_itinerary` 노드)

보강 텍스트를 **실시간 컨텍스트(날씨/휴무)와 합쳐** 하나의 문자열로 만든 뒤, 일정 생성 프롬프트의 `{realtime}` 자리에 끼워 넣습니다.

```735:743:src/travel/travel_graph.py
        realtime = build_realtime_context(
            state.get("destination"),
            state.get("places") or [],
            state.get("days"),
        )
        realtime_text = realtime.get("text") or "(제공 가능한 실시간 정보 없음)"
        external_text = state.get("external_text") or ""
        if external_text:
            realtime_text = f"{realtime_text}\n{external_text}"
```

그리고 이 `realtime_text` 가 프롬프트의 `실시간 참고 정보:` 섹션으로 들어갑니다.

```791:804:src/travel/travel_graph.py
            raw = (TRAVEL_ITINERARY_PROMPT | llm).invoke(
                {
                    "question": state.get("query"),
                    ...
                    "places": json.dumps(places, ensure_ascii=False),
                    "courses": json.dumps(courses, ensure_ascii=False),
                    "realtime": realtime_text,
                }
            )
```

프롬프트(`src/prompts.py`)는 LLM에게 **"실시간 참고 정보에 있으면 반드시 반영하라"** 고 지시합니다.

```93:94:src/prompts.py
실시간 참고 정보:
{realtime}
```

> 이 지점이 바로 **관광공사 API 데이터가 "모델에 통합"되는 순간**입니다.
> API 원본 → 텍스트 → 프롬프트 컨텍스트 → LLM 입력.

---

## 7단계 — LLM(Gemini) 호출 및 답변 생성

- 모델: `ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)` (`LLM_MODEL = "gemini-2.5-flash"`).
- 프롬프트 체인 `(TRAVEL_ITINERARY_PROMPT | llm).invoke(...)` 실행.
- 응답(JSON 문자열)을 `_extract_json()` 으로 파싱해 `itinerary` / `answer` / `warnings` 추출.
- 스트리밍 API의 경우 `stream_build_itinerary_tokens()` 가 동일 로직을 토큰 단위(SSE)로 흘려보냅니다.

즉, 관광공사 데이터가 반영된 프롬프트를 받은 Gemini가 최종 여행 일정 답변을 만들어냅니다.

---

## 8단계 — Facade & FastAPI 응답

### 8-1. Facade (`src/travel/travel_agent.py`)

그래프를 감싸 외부에서 쓰기 쉬운 함수를 제공합니다.

- `ask_travel(...)` : 그래프를 1회 실행(`get_travel_agent().invoke(state)`)하고 결과 dict 반환. 이때 `external` 필드도 포함.
- `ask_travel_in_session(...)` : 세션 히스토리/이전 일정을 붙여 실행 후 결과를 세션에 저장.
- `stream_travel_in_session(...)` : SSE 스트리밍용.

```70:89:src/travel/travel_agent.py
    result = get_travel_agent().invoke(state)
    return {
        "question": question,
        "answer": result.get("answer") or "",
        ...
        "external": result.get("external") or {},
    }
```

### 8-2. FastAPI 엔드포인트 (`src/api.py`)

사용자 HTTP 요청의 최종 진입/반환 지점.

- `POST /travel/query` → `ask_travel()` 호출 → `TravelQueryResponse` 반환.
- `POST /travel/sessions/{id}/query` → `ask_travel_in_session()`.
- `POST /travel/sessions/{id}/query/stream` → `stream_travel_in_session()` (SSE).

응답 모델에 `external` 필드가 있어, 프롬프트에 주입된 관광공사 원본 데이터도 함께 내려갑니다.

```148:159:src/api.py
@app.post("/travel/query", response_model=TravelQueryResponse)
def travel_query(request: TravelQueryRequest) -> TravelQueryResponse:
    result = ask_travel(
        request.question,
        ...
    )
    return TravelQueryResponse(**result)
```

---

## 정리 — 요청 한 건의 전체 여정

사용자가 `"제주 반려동물이랑 갈만한 2박3일 여행 추천해줘"` 라고 물으면:

1. **api.py** `POST /travel/query` 가 요청을 받음.
2. **travel_agent.py** `ask_travel()` 가 그래프 실행.
3. **travel_graph** `extract_requirements` → 목적지=제주, 일수=3 추출.
4. **travel_graph** `search_candidates` → 내부 DB 후보 검색 + `build_tour_context()` 호출.
5. **tour_enrich** → 질문에서 `pet` 의도 감지 → 지역코드(제주=39) 변환 →
6. **tour_api → _client** → data.go.kr `KorPetTourService2` 실제 HTTP 호출 → 반려동물 동반 장소 목록 수신.
7. 수신 데이터를 보강 텍스트(`external_text`)로 만들어 state에 저장.
8. **travel_graph** `build_itinerary` → 보강 텍스트를 `realtime` 컨텍스트에 합쳐 프롬프트 주입.
9. **Gemini** 가 반려동물 장소가 반영된 일정 답변 생성.
10. **api.py** 가 `answer` + `external`(원본 데이터)을 JSON으로 응답.

> **핵심 통찰**: 관광공사 API는 LLM이 "함수 호출(tool calling)"로 직접 부르는 방식이 아니라,
> **코드가 미리 호출해 텍스트로 정리한 뒤 프롬프트 컨텍스트로 밀어 넣는 (RAG 방식의) 통합**입니다.
> 그래서 API/네트워크가 실패해도 챗봇은 빈 컨텍스트로 정상 동작합니다.

---

## 참고: 관련 파일 지도

| 계층 | 파일 | 역할 |
|------|------|------|
| 설정 | `src/config.py` | 인증키·Base URL·타임아웃 |
| HTTP | `src/travel/tour_api/_client.py` | 공통 요청/정규화/에러 |
| 서비스 | `src/travel/tour_api/{kor_service,kor_pet_tour,kor_with_service,durunubi,tar_rlte_tar}.py` | 서비스별 래퍼 |
| REST 노출 | `src/tour_routes.py` | 원본 데이터를 `/travel/*` 로 직통 노출 |
| 보강 | `src/travel/tour_enrich.py` | 의도 감지 → 보강 텍스트 생성 |
| 그래프 | `src/travel/travel_graph.py` | 컨텍스트 수집 + 프롬프트 주입 + LLM 호출 |
| 실시간 | `src/travel/travel_realtime.py` | 날씨/휴무 컨텍스트 (보강 텍스트와 병합) |
| 프롬프트 | `src/prompts.py` | `{realtime}` 자리로 컨텍스트 삽입 |
| Facade | `src/travel/travel_agent.py` | 그래프 실행 래퍼 + 세션 |
| API | `src/api.py` | FastAPI 엔드포인트/응답 모델 |
