"""Travel-specific FAISS helpers with batched embedding and 429 retries."""

from __future__ import annotations

import json
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import (
    EMBED_BATCH_DELAY_SEC,
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL,
    TRAVEL_VECTORSTORE_PATH,
)


def _index_exists(path: Path | str) -> bool:
    root = Path(path)
    return (root / "index.faiss").is_file() and (
        (root / "index.pkl").is_file() or (root / "index.faiss").is_file()
    )


def load_vectorstore(path: Path | str) -> FAISS:
    store_path = Path(path)
    if not _index_exists(store_path):
        raise FileNotFoundError(f"FAISS 인덱스가 없습니다: {store_path}")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(
        str(store_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def _checkpoint_path(store_path: Path) -> Path:
    return store_path / "embed_checkpoint.json"


def _load_checkpoint(store_path: Path) -> int:
    path = _checkpoint_path(store_path)
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data.get("embedded_count", 0))
    except (OSError, ValueError, json.JSONDecodeError):
        return 0


def _save_checkpoint(store_path: Path, embedded_count: int, total: int) -> None:
    path = _checkpoint_path(store_path)
    path.write_text(
        json.dumps(
            {"embedded_count": embedded_count, "total": total},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _embed_with_retry(embeddings: GoogleGenerativeAIEmbeddings, texts: list[str], *, retries: int = 6):
    delay = 15.0
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            return embeddings.embed_documents(texts)
        except Exception as exc:  # noqa: BLE001 - retry quota/network errors
            last_err = exc
            msg = str(exc)
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                wait = delay * (attempt + 1)
                print(f"  rate limited, retry in {wait:.0f}s (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
                continue
            raise
    assert last_err is not None
    raise last_err


def build_travel_vectorstore(
    documents: list[Document],
    *,
    path: Path | str | None = None,
    batch_size: int = EMBED_BATCH_SIZE,
    batch_delay_sec: float = EMBED_BATCH_DELAY_SEC,
    force: bool = False,
    # Free-tier embed is request/min limited; keep micro-batches small
    embed_micro_batch: int = 5,
) -> FAISS:
    store_path = Path(path) if path else TRAVEL_VECTORSTORE_PATH
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    if _index_exists(store_path) and not force:
        checkpoint = _load_checkpoint(store_path)
        if checkpoint >= len(documents):
            print(f"여행 벡터DB 로딩... ({store_path})")
            return load_vectorstore(store_path)

    if not documents:
        raise ValueError("여행 벡터DB를 생성할 문서가 없습니다.")

    store_path.mkdir(parents=True, exist_ok=True)
    # Effective batch: how many docs before pause. Cap for free tier.
    effective_batch = max(1, min(batch_size, 20))
    print(
        f"여행 벡터DB 생성: 문서 {len(documents)}건, "
        f"batch={effective_batch}, micro={embed_micro_batch}"
    )

    start_at = 0
    vectorstore: FAISS | None = None
    if _index_exists(store_path) and not force:
        start_at = _load_checkpoint(store_path)
        if start_at > 0:
            print(f"체크포인트에서 재개: {start_at}/{len(documents)}")
            vectorstore = load_vectorstore(store_path)
    elif force and store_path.exists():
        for child in store_path.iterdir():
            if child.is_file():
                child.unlink()
        start_at = 0

    total = len(documents)
    i = start_at
    while i < total:
        chunk_end = min(i + effective_batch, total)
        print(f"  embedding {i + 1}-{chunk_end}/{total}...")
        batch_docs = documents[i:chunk_end]

        for micro_start in range(0, len(batch_docs), embed_micro_batch):
            micro = batch_docs[micro_start : micro_start + embed_micro_batch]
            texts = [d.page_content for d in micro]
            vectors = _embed_with_retry(embeddings, texts)
            text_embeddings = list(zip(texts, vectors))
            metadatas = [d.metadata for d in micro]
            if vectorstore is None:
                vectorstore = FAISS.from_embeddings(
                    text_embeddings,
                    embeddings,
                    metadatas=metadatas,
                )
            else:
                vectorstore.add_embeddings(text_embeddings, metadatas=metadatas)

        assert vectorstore is not None
        vectorstore.save_local(str(store_path))
        _save_checkpoint(store_path, chunk_end, total)
        i = chunk_end
        if i < total and batch_delay_sec > 0:
            print(f"  rate-limit wait {batch_delay_sec}s...")
            time.sleep(batch_delay_sec)

    assert vectorstore is not None
    print(f"여행 벡터DB 저장 완료: {store_path}")
    return vectorstore


def get_travel_vectorstore(path: Path | str | None = None) -> FAISS:
    store_path = Path(path) if path else TRAVEL_VECTORSTORE_PATH
    return load_vectorstore(store_path)


def prioritize_documents_for_faiss(
    pois: list[dict],
    courses: list[dict],
    *,
    poi_limit: int,
    course_limit: int,
    priority_cities: tuple[str, ...] = ("제주", "서울", "부산", "강릉", "경주"),
) -> list[Document]:
    """Prefer major destinations when selecting FAISS subset."""
    from src.travel.travel_ingestion import records_to_documents

    def score_poi(row: dict) -> tuple[int, str]:
        blob = " ".join(
            str(row.get(k) or "")
            for k in ("city", "region", "address_ko", "name_ko")
        )
        for idx, city in enumerate(priority_cities):
            if city in blob:
                return (idx, row.get("poi_id") or "")
        return (len(priority_cities) + 1, row.get("poi_id") or "")

    def score_course(row: dict) -> tuple[int, str]:
        blob = " ".join(str(row.get(k) or "") for k in ("city", "title", "places"))
        for idx, city in enumerate(priority_cities):
            if city in blob:
                return (idx, row.get("package_id") or row.get("title") or "")
        return (len(priority_cities) + 1, row.get("package_id") or "")

    selected_pois = sorted(pois, key=score_poi)[:poi_limit]
    selected_courses = sorted(courses, key=score_course)[:course_limit]
    return records_to_documents(selected_pois + selected_courses)
