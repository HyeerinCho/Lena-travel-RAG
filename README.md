# LENA-PJ

LangGraph 기반 **RAG(Retrieval-Augmented Generation) 챗봇**입니다.

| 에이전트 | 역할 | 검색 백엔드 |
|----------|------|-------------|
| **Travel Planner** | 한국 여행 POI·코스 검색 및 일정 추천 | SQLite + FAISS 하이브리드 |

**기술 스택**: `Python 3.14+` · `FastAPI` · `LangGraph` · `LangChain` · `Google Gemini(LLM+임베딩)` · `FAISS` · `SQLite` · `Open-Meteo` · `한국관광공사 TourAPI(data.go.kr)`

> 아키텍처 다이어그램과 상세 구조 설명은 [`docs/아키텍처_구조도.md`](docs/아키텍처_구조도.md)를, TourAPI → LLM 통합 흐름은 [`docs/API_통합_흐름.md`](docs/API_통합_흐름.md)를 참고하세요.

---

## 아키텍처 개요

![LENA-PJ RAG Architecture](docs/assets/lena_rag_architecture.png)

LENA-PJ는 단순 "검색 → 생성"의 Naive RAG가 아니라, 아래 모듈이 추가된 **Advanced / Modular RAG**입니다.

| RAG 단계 | LENA-PJ 구현 |
|----------|--------------|
| **Pre-Retrieval** | 정규화 → SQLite/FAISS 이중 인덱싱, 의도(intent) 추출 |
| **Retrieval** | SQL 필터 + FAISS 시맨틱 검색의 **하이브리드 검색** |
| **Post-Retrieval** | 중복 제거·의미 우선 병합, 지역 포커스/제외 필터 |
| **Generation** | 의도별 프롬프트 라우팅 + 실시간(날씨)·외부(TourAPI) 컨텍스트 주입 후 Gemini 생성 |
| **Modular** | TourAPI 보강, Open-Meteo 실시간, 세션 메모리, SSE 스트리밍 |

---

## 빠른 시작

### 1. 요구사항

- Python **3.14+**
- [`uv`](https://github.com/astral-sh/uv) (의존성/실행 관리)

### 2. 설치

```bash
uv sync
```

### 3. 환경 변수

프로젝트 루트에 `.env` 파일을 만듭니다 (`.env.example` 참고).

| 변수 | 필수 | 용도 |
|------|------|------|
| `GOOGLE_API_KEY` | 예 | Gemini LLM + 임베딩 |
| `DATA_GO_KR_API_KEY` | 선택 | TourAPI 보강/노출 (없으면 해당 기능만 비활성) |
| `LANGSMITH_API_KEY` / `LANGCHAIN_TRACING_V2` / `LANGCHAIN_PROJECT` | 선택 | LangSmith 트레이싱 |

> `DATA_GO_KR_API_KEY`는 data.go.kr의 **"일반 인증키(Decoding)"** 값을 사용하세요.

### 4. 여행 인덱스 빌드 (여행 에이전트 사용 시 최초 1회)

원본 데이터를 검색 가능한 `travel.db`(SQLite) + FAISS 인덱스로 변환합니다.

```bash
# 전체 빌드 (정규화 + SQLite + FAISS)
uv run python scripts/build_travel_index.py

# SQLite만 빠르게 (FAISS 생략)
uv run python scripts/build_travel_index.py --skip-faiss

# FAISS만 다시 생성
uv run python scripts/build_travel_index.py --faiss-only --force-faiss
```

### 5. 실행

```bash
uv run python scripts/CLI/lena-travel   # 여행 에이전트 CLI
uv run python scripts/CLI/lena-api      # 웹/API 서버 → http://localhost:8000
```

---

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/` | 여행 챗 웹 UI |
| `POST` | `/travel/query` | 여행 에이전트 (무상태) |
| `POST` | `/travel/sessions/{id}/query` | 세션 기반 여행 질의 |
| `POST` | `/travel/sessions/{id}/query/stream` | SSE 토큰 스트리밍 |
| `GET/POST/PATCH/DELETE` | `/travel/sessions/*` | 세션 CRUD |
| `GET` | `/travel/places/{poi_id}` | POI 상세 조회 |
| `/travel/kor/*` · `/pet/*` · `/with/*` · `/durunubi/*` · `/related/*` | TourAPI 원본 직통 노출 |

### 예시

여행 Q&A (`question`만 필수):

```bash
curl -X POST http://localhost:8000/travel/query \
  -H "Content-Type: application/json" \
  -d '{"question": "제주 반려동물이랑 갈만한 2박3일 여행 추천해줘", "destination": "제주", "days": 3}'
```

---

## 프로젝트 구조

```
LENA-PJ/
├── pyproject.toml              # 패키지/의존성/CLI 스크립트 정의 (uv)
├── docs/                       # 아키텍처/통합 흐름 문서
│   ├── 아키텍처_구조도.md
│   ├── API_통합_흐름.md
│   └── assets/                 # 다이어그램 이미지
├── eval/dataset.py             # LangSmith 평가 데이터셋 시딩
├── scripts/
│   ├── CLI/{lena-travel,lena-api}   # CLI/서버 진입점
│   ├── build_travel_index.py   # 여행 인덱스 빌드
│   └── smoke_travel.py         # 여행 스모크 테스트
├── data/                       # (gitignore) 원본 + 생성 데이터(DB/FAISS/세션)
└── src/
    ├── config.py               # 전역 설정 (키/모델/경로/TourAPI/날씨)
    ├── prompts.py              # 여행 프롬프트 템플릿 (6종)
    ├── main.py                 # 여행 CLI 진입점
    ├── api.py                  # FastAPI 앱
    ├── tour_routes.py          # TourAPI 원본 REST 노출 라우터
    ├── front/index.html        # 여행 챗 웹 UI (세션 · SSE)
    │
    └── travel/                 # [여행] 에이전트 패키지
        ├── travel_agent.py         # 파사드 (세션/스트리밍 래퍼)
        ├── travel_graph.py         # 3노드 그래프 + 스트리밍 변형
        ├── travel_tools.py         # 하이브리드 검색 (SQL + FAISS 병합)
        ├── travel_repository.py    # SQLite places/courses 검색
        ├── travel_vectorstore.py   # 여행 FAISS 빌드(배치/재시도/체크포인트)
        ├── travel_ingestion.py     # 원본 → JSONL 정규화 + Document 변환
        ├── session_store.py        # 세션/메시지 영속화 (sessions.db)
        ├── travel_realtime.py      # Open-Meteo 날씨 + 휴무 휴리스틱
        ├── tour_enrich.py          # TourAPI 의도 감지 → 보강 텍스트
        └── tour_api/               # TourAPI 저수준 HTTP 클라이언트
```

---

## Travel Planner 에이전트 (핵심)

**3노드 LangGraph + 하이브리드 검색 + 다중 의도 라우팅 + 실시간/외부 보강 + 세션 + SSE 스트리밍**을 갖춘 Modular RAG입니다.

### 런타임 3노드

```
① extract_requirements   휴리스틱 정규식 + LLM JSON 추출 → 의도/목적지/일수/예산 결정
        ↓
② search_candidates      SQL 필터 + FAISS 시맨틱 → 병합/중복제거 (+ TourAPI 보강 텍스트 수집)
        ↓
③ build_itinerary        실시간(날씨)·외부(TourAPI) 컨텍스트 주입 → 의도별 프롬프트 → Gemini 생성
```

### 의도(intent) 라우팅

| intent | 트리거 | 프롬프트 |
|--------|--------|----------|
| `city_list` | "어디가 좋아?" 도시 나열형 | `TRAVEL_CITY_LIST_PROMPT` |
| `multi_itinerary` | 여러 개 일정 요청 | `TRAVEL_MULTI_ITINERARY_PROMPT` |
| `qa` | 단순 질의응답 | `TRAVEL_QA_PROMPT` |
| `rewrite_day` | "2일차만 바꿔줘" | `TRAVEL_REWRITE_DAY_PROMPT` |
| `itinerary` (기본) | 일반 일정 생성 | `TRAVEL_ITINERARY_PROMPT` |

### 주요 파일

| 파일 | 역할 |
|------|------|
| `travel/travel_agent.py` | 파사드: `ask_travel`, `ask_travel_in_session`, `stream_travel_in_session` |
| `travel/travel_graph.py` | `TravelState` + 3노드 정의, 토큰 스트리밍 변형 |
| `travel/travel_tools.py` | `TravelSearchService`: SQL+FAISS 하이브리드 병합 |
| `travel/travel_repository.py` | SQLite `places`/`courses` 검색, 도시 별칭 |
| `travel/travel_vectorstore.py` | 여행 FAISS 빌드/로드 (배치 임베딩·429 재시도·체크포인트) |
| `travel/travel_ingestion.py` | 원본 POI JSON·코스 CSV → JSONL 정규화 → Document |
| `travel/session_store.py` | `sessions.db`에 세션/메시지 영속화, 최근 히스토리 제공 |
| `travel/travel_realtime.py` | Open-Meteo 날씨 예보 + 휴무 휴리스틱 |
| `travel/tour_enrich.py` | `pet`/`barrier`/`trail` 의도 감지 → TourAPI 호출 → 보강 텍스트 |
| `travel/tour_api/*` | 5개 관광공사 서비스별 저수준 HTTP 클라이언트 |

> TourAPI는 LLM이 직접 tool-calling으로 부르지 않고, **코드가 미리 호출해 텍스트로 정리한 뒤 프롬프트 컨텍스트로 주입**하는 RAG 방식입니다. 외부 API가 실패해도 빈 컨텍스트로 챗봇은 정상 동작합니다.

---

## 평가 (선택)

LangSmith 평가용 데이터셋(`lena-travel`)을 시딩합니다.

```bash
uv run python eval/dataset.py
```

---

## 라이선스

프로젝트 내부용. (별도 라이선스 미지정)
