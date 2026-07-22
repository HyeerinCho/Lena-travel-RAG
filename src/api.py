from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.rag import ask
from src.travel.travel_agent import (
    ask_travel,
    ask_travel_in_session,
    get_session_store,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

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


class TravelQueryResponse(BaseModel):
    question: str
    answer: str
    destination: str | None = None
    days: int | None = None
    budget: int | None = None
    preferences: list[str] = []
    language: str = "ko"
    itinerary: list[dict[str, Any]] = []
    places: list[dict[str, Any]] = []
    courses: list[dict[str, Any]] = []
    warnings: list[str] = []
    sources: list[dict[str, Any]] = []
    session_id: str | None = None
    session: dict[str, Any] | None = None


class CreateSessionRequest(BaseModel):
    title: str | None = None


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


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


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
    )
    return TravelQueryResponse(**result)


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
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.") from None
    return TravelQueryResponse(**result)
