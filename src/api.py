from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.rag import ask
from src.travel.travel_agent import ask_travel

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


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api")
def api_info():
    return {
        "message": "LENA API 실행 중",
        "endpoints": ["/query", "/travel/query"],
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
