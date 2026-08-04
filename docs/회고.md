# LENA-PJ — 과제 회고

LangChain + LangGraph 기반 RAG 파이프라인을 직접 구축하며 정리한 과제 회고 문서입니다.

---

## 1. 과제 목표

- LangChain + LangGraph 기반 **RAG 파이프라인 구축**
- **FastAPI**로 REST API 래핑
- **LangSmith**로 체인 실행 Tracing 및 Dataset 기반 평가 (진행 중)

---

## 2. 버전별 변경 사항 요약

| 버전 | 핵심 변경 | 결과 | 날짜 |
|------|-----------|------|------|
| **v0** | `plain_rag.py` — LangChain 없이 RAG 직접 구현 (로딩 → 청킹 → 임베딩 → 코사인 유사도 검색 → 생성) | RAG 원리 이해, 단일 파일 프로토타입 | 6/30 |
| **v1** | `alex-rag` 데모 참고, LangChain 스택 도입 (ingestion → vectorstore → chain / graph 분리), `data/alex-notes` 연결하여 DB로 사용 | 기본 구조 완성, CLI(graph)와 API(chain) 이원화 | 7/6 |
| **v2** | Phase 1 리팩터링 — `prompts.py` 프롬프트 분리, `rag.py` + `ask()` 단일 진입점, LangGraph를 RAG 코어로 통일, `chain.py` 제거 · 그래프 마이그레이션 | CLI·API가 동일한 LangGraph 파이프라인 공유 | 7/9 |
| **v3** | Phase 2 부분 완료 — `config.py` API 키 검증, `VECTORSTORE_PATH` 일원화, `CHUNK_SIZE`/`CHUNK_OVERLAP` config 이동 | 설정 중복 제거, 시작 시 에러 명확화 | 7/9 |

---

## 3. 핵심 개념 정리

### 3-1. RAG 흐름

```
사용자 입력
    ↓
[ingestion] data/alex-notes/*.md 로딩 → RecursiveCharacterTextSplitter 청킹 (500자, overlap 100)
    ↓
[vectorstore] Gemini embedding → FAISS 인덱스 (data/faiss_index/ 디스크 캐싱)
    ↓
[LangGraph — retrieve 노드] 질문과 유사한 문서 청크 검색
    ↓
[LangGraph — generate 노드] context + prompt → Gemini 답변 생성
    ↓
답변 반환
```

### 3-2. 모델 구성 (현재)

| 역할 | 모델 |
|------|------|
| 임베딩 | Google `gemini-embedding-001` |
| 답변 생성 | Google `gemini-2.5-flash` |
| 벡터 DB | FAISS (로컬 파일) |
| 평가 (예정) | LangSmith Dataset `lena-rag` |

### 3-3. 파일 구조

| 파일 | 역할 |
|------|------|
| `data/alex-notes/*.md` | RAG 검색 대상 — 개인 학습·일기·프로젝트 노트 |
| `src/config.py` | 환경변수, 모델명, 경로, 청킹 설정 |
| `src/ingestion.py` | 마크다운 로딩 + 텍스트 청킹 |
| `src/vectorstore.py` | FAISS 생성/로드 (디스크 캐싱) |
| `src/prompts.py` | RAG 프롬프트 템플릿 |
| `src/graph.py` | LangGraph RAG — retrieve → generate 노드 |
| `src/rag.py` | lazy init + `ask(query)` 단일 API |
| `src/main.py` | CLI — `lena` 명령 진입점 |
| `src/api.py` | FastAPI — `POST /query` 엔드포인트 |
| `eval/dataset.py` | LangSmith Dataset 생성 및 예제 시드 |
| `scripts/lena` | CLI 래퍼 |
| `scripts/lena-api` | Uvicorn 서버 래퍼 |

### 3-4. 실행 흐름 (CLI 기준)

```
사용자가 lena "LangChain이 뭐야?" 실행
    ↓
scripts/lena → src/main.py
    ↓
src/rag.py — ask(query)
    ↓
src/rag.py — get_rag_graph()  [최초 1회만 실행]
    ├── ingestion.load_and_split(DATA_PATH)  → 45개 문서, 387개 청크
    ├── vectorstore.build_vectorstore(chunks)  → FAISS 로드 또는 생성
    └── graph.build_rag_graph(vectorstore)  → LangGraph 컴파일
    ↓
graph — retrieve 노드
    ├── retriever.invoke(query)  → 유사 문서 k개 검색
    └── state["documents"] 업데이트
    ↓
graph — generate 노드
    ├── context = 문서 청크 join
    ├── RAG_PROMPT + gemini-2.5-flash → 답변 생성
    └── state["answer"] 업데이트
    ↓
ask() — result["answer"] 반환 → 터미널 출력
```

### 3-5. 실행 흐름 (FastAPI 기준)

```
사용자가 POST /query {"question": "RAG가 뭐야?"} 전송
    ↓
src/api.py — query()
    ↓
src/rag.py — ask(question)  [CLI와 동일한 파이프라인]
    ↓
QueryResponse(question, answer) JSON 응답
```

---

## 4. 겪었던 문제와 해결

| 문제 | 원인 | 해결 | 파일 |
|------|------|------|------|
| `python` 명령 not found | `.python-version`이 3.14인데 pyenv에는 3.11만 설치 | `source .venv/bin/activate` 또는 `uv run` 사용 | `.python-version`, `.venv` |
| `build_rag_graph() takes 0 positional arguments` | `graph.py`에 새/옛 `build_rag_graph` 함수가 중복 정의, 나중 정의가 덮어씀 | 41행 이후 예전 코드 삭제 | `src/graph.py` |
| CLI와 API가 다른 RAG 사용 | `main.py`는 graph, `api.py`는 chain 사용 | `ask()` 단일 진입점으로 통일, `chain.py` 제거 | `src/rag.py`, `src/api.py` |
| import 시마다 문서 로딩·인덱싱 | `graph.py` 모듈 레벨에서 `load_and_split` 실행 | `@lru_cache` + `get_rag_graph()`로 lazy init | `src/rag.py` |
| 매번 인덱싱 (초기) | 벡터를 RAM에만 저장 | FAISS `save_local` / `load_local`로 디스크 캐싱 | `src/vectorstore.py` |
| API 키 없을 때 불명확한 에러 | `GOOGLE_API_KEY` None 검증 없음 | 시작 시 `ValueError` 발생 | `src/config.py` |
| Gemini API 토큰 부족 | free tier embed 100 req/min | 1분당 할당되기 때문에 토큰 초기화까지 대기 걸어둠 | `src/config.py` |

---

## 5. 실행 방법

### 가상환경 실행

```bash
uv sync
source .venv/bin/activate   # 또는 매번 uv run 사용
```

### CLI

```bash
lena                        # 기본 질문: "RAG가 뭐야?"
lena "LangChain이 뭐야?"    # 질문 지정
```

### API 서버

```bash
lena-api
# 다른 터미널
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "RAG가 뭐야?"}'
```

### LangSmith Dataset 시드

```bash
uv run python eval/dataset.py
```

---

## 6. 회고

### 6-1. 과정 서술

수업시간에 그렇게 `.env` 파일을 항상 신경써야 된다고 했는데 왜 push가 안되지? 하고 보니까 `.env`파일을 push 하려고 하고 있었다. 깜짝 놀랬다...

중간에 청크가 안되고 멈춰있는 상황이 있어서 무슨일인지 보아하니 gemini api 분당 최대 토큰을 다쓴 상황이여서 토큰 없으면 60초 기다려 새롭게 갱신되면 돌아가도록 했다. 좀 오래 걸리긴했지만 한 번만 하고 벡터 DB에 저장해둬서 불러와서 사용할 수 있어서 편했다.

중간중간 실습이나 수업시간에 했던 내용과 프로젝트의 내용이 연결되지 않고 따로 논다는 생각이 들었다. 이걸 어떻게 해결하면 좋을까... → 우선은 교재를 다시 봐야겠다.

생각보다 LLM에 의지하는 일이 많다는 걸 새삼다시 느꼈다.

**〈〈마음가짐〉〉**

- LLM을 과외쌤처럼 사용할 것 (알려주세요 X, 길잡이 용도)
- 인공지능에 자아의탁을 하지 않는다.
- 항상 어떤 의미로 이 코드를 왜 추가했는지 정리해달라고 한다.
- ask 모드로 사용, agent 모드는 지양

### 6-2. 앞으로 할 일

- [ ] `pyproject.toml` 미사용 의존성 정리 (`google-generativeai`, `numpy`, `pypdf`, `ragas`)
- [ ] LangSmith Dataset 기반 평가
- [ ] retriever `top_k` 튜닝, 빈 검색 fallback
