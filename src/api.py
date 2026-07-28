import json
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.rag import ask
from src.travel.travel_agent import (
    ask_travel,
    ask_travel_in_session,
    get_search_service,
    get_session_store,
    stream_travel_in_session,
)

FRONT_DIR = Path(__file__).resolve().parent / "front"

app = FastAPI(title="LENA Travel + Notes RAG API")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str


class TravelQueryRequest(BaseModel):
    question: str
    destination: str | None = None
    days: int | None = Field(default=None, ge=1, le=30)
    budget: int | None = Field(default=None, ge=0)
    language: str | None = Field(default=None, pattern="^(ko|en)$")
    preferences: list[str] | None = None
    rewrite_day: int | None = Field(default=None, ge=1, le=30)


class TravelQueryResponse(BaseModel):
    question: str
    answer: str
    destination: str | None = None
    days: int | None = None
    budget: int | None = None
    preferences: list[str] = []
    language: str = "ko"
    itinerary: list[dict[str, Any]] = []
    cities: list[dict[str, Any]] = []
    intent: str = "itinerary"
    places: list[dict[str, Any]] = []
    courses: list[dict[str, Any]] = []
    warnings: list[str] = []
    sources: list[dict[str, Any]] = []
    session_id: str | None = None
    session: dict[str, Any] | None = None
    rewrite_day: int | None = None


class CreateSessionRequest(BaseModel):
    title: str | None = None


class UpdateSessionRequest(BaseModel):
    title: str | None = None
    destination: str | None = None
    days: int | None = Field(default=None, ge=1, le=30)
    budget: int | None = Field(default=None, ge=0)
    language: str | None = Field(default=None, pattern="^(ko|en)$")
    preferences: list[str] | None = None


class SessionSummary(BaseModel):
    id: str
    title: str
    destination: str | None = None
    days: int | None = None
    budget: int | None = None
    language: str = "ko"
    preferences: list[str] = []
    created_at: str
    updated_at: str
    message_count: int = 0


class SessionMessage(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    payload: dict[str, Any] | None = None
    created_at: str


class SessionDetail(SessionSummary):
    messages: list[SessionMessage] = []


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/")
def root():
    return FileResponse(FRONT_DIR / "index.html")


@app.get("/api")
def api_info():
    return {
        "message": "LENA API 실행 중",
        "endpoints": [
            "/query",
            "/travel/query",
            "/travel/sessions",
            "/travel/sessions/{session_id}",
            "/travel/sessions/{session_id}/query",
            "/travel/sessions/{session_id}/query/stream",
            "/travel/places/{poi_id}",
        ],
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    answer = ask(request.question)
    return QueryResponse(question=request.question, answer=answer)


@app.post("/travel/query", response_model=TravelQueryResponse)
def travel_query(request: TravelQueryRequest) -> TravelQueryResponse:
    result = ask_travel(
        request.question,
        destination=request.destination,
        days=request.days,
        budget=request.budget,
        language=request.language,
        preferences=request.preferences,
        rewrite_day=request.rewrite_day,
    )
    return TravelQueryResponse(**result)


@app.get("/travel/places/{poi_id}")
def get_place(poi_id: str) -> dict[str, Any]:
    place = get_search_service().get_place_details(poi_id)
    if not place:
        raise HTTPException(status_code=404, detail="장소를 찾을 수 없습니다.")
    return place


@app.get("/travel/sessions", response_model=list[SessionSummary])
def list_travel_sessions() -> list[SessionSummary]:
    store = get_session_store()
    return [SessionSummary(**item) for item in store.list_sessions()]


@app.post("/travel/sessions", response_model=SessionSummary)
def create_travel_session(
    request: CreateSessionRequest = CreateSessionRequest(),
) -> SessionSummary:
    store = get_session_store()
    session = store.create_session(title=request.title)
    session["message_count"] = 0
    return SessionSummary(**session)


@app.get("/travel/sessions/{session_id}", response_model=SessionDetail)
def get_travel_session(session_id: str) -> SessionDetail:
    store = get_session_store()
    detail = store.get_session_detail(session_id)
    if not detail:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    detail["message_count"] = len(detail.get("messages") or [])
    return SessionDetail(**detail)


@app.patch("/travel/sessions/{session_id}", response_model=SessionSummary)
def update_travel_session(
    session_id: str, request: UpdateSessionRequest
) -> SessionSummary:
    store = get_session_store()
    updated = store.update_session_meta(
        session_id,
        title=request.title,
        destination=request.destination,
        days=request.days,
        budget=request.budget,
        language=request.language,
        preferences=request.preferences,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    detail = store.get_session_detail(session_id) or {}
    updated["message_count"] = len(detail.get("messages") or [])
    return SessionSummary(**updated)


@app.delete("/travel/sessions/{session_id}")
def delete_travel_session(session_id: str) -> dict[str, Any]:
    store = get_session_store()
    deleted = store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    return {"ok": True, "session_id": session_id}


@app.post("/travel/sessions/{session_id}/query", response_model=TravelQueryResponse)
def travel_session_query(
    session_id: str, request: TravelQueryRequest
) -> TravelQueryResponse:
    try:
        result = ask_travel_in_session(
            session_id,
            request.question,
            destination=request.destination,
            days=request.days,
            budget=request.budget,
            language=request.language,
            preferences=request.preferences,
            rewrite_day=request.rewrite_day,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.") from None
    return TravelQueryResponse(**result)


@app.post("/travel/sessions/{session_id}/query/stream")
def travel_session_query_stream(
    session_id: str, request: TravelQueryRequest
) -> StreamingResponse:
    store = get_session_store()
    if not store.get_session(session_id):
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    def event_gen() -> Iterator[str]:
        try:
            for event_type, payload in stream_travel_in_session(
                session_id,
                request.question,
                destination=request.destination,
                days=request.days,
                budget=request.budget,
                language=request.language,
                preferences=request.preferences,
                rewrite_day=request.rewrite_day,
            ):
                yield _sse(event_type, payload)
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(event_gen(), media_type="text/event-stream")
