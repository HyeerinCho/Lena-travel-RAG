# LENA-PJ 정리

여행/노트 RAG 프로젝트의 실행 스크립트, 인덱스 빌드, 웹 UI에 대한 요약입니다.

---

## 1. 프로젝트 개요

LENA-PJ는 두 가지 에이전트를 제공합니다.

| 에이전트 | 역할 |
|----------|------|
| Notes RAG | 개인 마크다운 노트 기반 Q&A |
| Travel planner | 한국 여행(제주, 서울 등) POI·코스 검색 및 일정 추천 |

기술 스택: **FastAPI + LangGraph + Gemini + SQLite + FAISS**

---

## 2. 실행 스크립트 (`scripts/`)

프로젝트 루트에서 아래처럼 실행합니다.

```bash
uv run python scripts/lena
uv run python scripts/lena-travel
uv run python scripts/lena-api
```

### 2.1 `scripts/lena`

- **역할:** 노트 RAG CLI
- **동작:** `src.main`을 실행 (기본 agent: notes)
- **용도:** 터미널에서 노트 기반 질문/답변

### 2.2 `scripts/lena-travel`

- **역할:** 여행 에이전트 CLI
- **동작:** 같은 `src.main`을 실행하되, `--agent travel`을 기본으로 넣음
- **용도:** 터미널에서 여행 관련 질문/답변

### 2.3 `scripts/lena-api`

- **역할:** 웹 API + 웹 UI 서버
- **동작:** FastAPI 앱(`src.api:app`)을 `http://0.0.0.0:8000`에서 실행
- **용도:** 브라우저/클라이언트로 Q&A 사용

```bash
uv run python scripts/lena-api
# → http://localhost:8000
```

---

## 3. API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/` | 웹 UI (질문·답변 페이지) |
| `GET` | `/api` | API 상태/엔드포인트 안내 |
| `POST` | `/query` | 노트 RAG Q&A |
| `POST` | `/travel/query` | 여행 에이전트 Q&A |

### 노트 Q&A (`POST /query`)

요청:

```json
{ "question": "RAG가 뭐야?" }
```

응답:

```json
{ "question": "...", "answer": "..." }
```

### 여행 Q&A (`POST /travel/query`)

요청:

```json
{
  "question": "제주도에 유명한게 뭐야?",
  "destination": "제주",
  "days": 3,
  "budget": 500000,
  "language": "ko",
  "preferences": ["자연", "경관"]
}
```

- `question`만 필수
- `destination`, `days`, `budget`, `language`, `preferences`는 선택

응답에는 `answer`와 함께 `itinerary`, `places`, `courses`, `sources` 등이 포함될 수 있습니다.

---

## 4. 웹 UI

### 목적

질문과 답변만 보이는 간단한 채팅 페이지.

예시:

- **질문:** 제주도에 유명한게 뭐야?
- **에이전트 답변:** (어디어디)가 유명하고 어디 가면 좋습니다.

### 구성

| 파일 | 역할 |
|------|------|
| `src/static/index.html` | 질문/답변 채팅 UI |
| `src/api.py` | `/`에서 HTML 제공, 질문은 `POST /travel/query`로 전달 |

### 사용 방법

1. `uv run python scripts/lena-api` 실행
2. 브라우저에서 http://localhost:8000 접속
3. 질문 입력 후 보내기 → 에이전트 답변 표시

---

## 5. `scripts/build_travel_index.py`

여행 에이전트가 검색할 수 있도록 **데이터를 전처리하고 DB·벡터 인덱스를 만드는 스크립트**입니다.

### 5.1 전체 흐름

```
원본 여행 데이터
    ↓ normalize_travel_data()
pois.jsonl / courses.jsonl (정규화)
    ↓ build_database()
travel.db (SQLite)
    ↓ prioritize + build_travel_vectorstore()
FAISS 벡터 인덱스 (의미 검색용)
```

### 5.2 코드별 역할

#### 프로젝트 루트 설정 (1–12행)

스크립트 위치 기준으로 프로젝트 루트를 찾아 `sys.path`에 넣습니다. `src.*` 모듈을 import할 수 있게 합니다.

#### import (14–27행)

경로/설정값과 실제 작업 함수를 가져옵니다.

| 함수 | 역할 |
|------|------|
| `normalize_travel_data` | 원본 → JSONL 정규화 |
| `build_database` | JSONL → SQLite |
| `prioritize_documents_for_faiss` | FAISS에 넣을 문서 선별 |
| `build_travel_vectorstore` | FAISS 인덱스 구축 |

#### `_read_jsonl` (30–39행)

정규화된 `.jsonl` 파일을 한 줄씩 읽어 딕셔너리 리스트로 만듭니다.

#### CLI 옵션 (42–90행)

| 옵션 | 의미 |
|------|------|
| `--skip-faiss` | 정규화 + SQLite만, FAISS는 생략 |
| `--faiss-only` | 이미 있는 JSONL로 FAISS만 재생성 |
| `--force-faiss` | FAISS가 있어도 다시 만듦 |
| `--include-validation` | Validation용 POI 라벨도 포함 |
| `--sample-per-region` | 지역별 음식점/숙박 샘플 수 |
| `--max-full` | 관광지/문화시설 개수 상한 (테스트용) |
| `--faiss-poi-limit` / `--faiss-course-limit` | FAISS에 넣을 POI·코스 개수 |
| `--batch-size` / `--batch-delay` | 임베딩 API 배치 크기·대기 시간 |

#### 정규화 + DB (92–112행)

- `--faiss-only`면 기존 JSONL만 사용
- 아니면 `normalize_travel_data` → `build_database` 실행
- `--skip-faiss`면 여기서 종료

#### FAISS 빌드 (114–131행)

1. POI·코스 JSONL 로드
2. 우선순위 도시 위주로 문서 선별
3. 임베딩해서 FAISS 인덱스 저장
4. “여행 인덱스 빌드 완료” 출력

### 5.3 실행 예

```bash
# 전체 빌드 (정규화 + SQLite + FAISS)
uv run python scripts/build_travel_index.py

# SQLite만 빠르게
uv run python scripts/build_travel_index.py --skip-faiss

# FAISS만 다시
uv run python scripts/build_travel_index.py --faiss-only --force-faiss
```

---

## 6. 환경 변수

| 변수 | 필수 | 용도 |
|------|------|------|
| `GOOGLE_API_KEY` | 예 | Gemini LLM + 임베딩 |
| `LANGSMITH_API_KEY` / `LANGCHAIN_API_KEY` | 아니오 | LangSmith 트레이싱 |
| `LANGCHAIN_TRACING_V2` | 아니오 | 예: `true` |
| `LANGCHAIN_PROJECT` | 아니오 | 예: `LENA-PJ` |

---

## 7. 주요 디렉터리

```
scripts/
  lena                 # 노트 CLI
  lena-travel          # 여행 CLI
  lena-api             # 웹/API 서버
  build_travel_index.py  # 여행 인덱스 빌드

src/
  api.py               # FastAPI 앱
  main.py              # CLI 진입점
  static/index.html    # 웹 UI
  travel/              # 여행 에이전트 패키지 (agent, graph, tools, …)
  rag.py, graph.py     # 노트 에이전트

data/
  alex-notes/          # 노트 코퍼스
  travel/              # travel.db + FAISS 인덱스
```
