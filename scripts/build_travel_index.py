#!/usr/bin/env python3
"""Normalize travel data, build SQLite, and optionally build FAISS index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    TRAVEL_DB_PATH,
    TRAVEL_FAISS_COURSE_LIMIT,
    TRAVEL_FAISS_POI_LIMIT,
    TRAVEL_NORMALIZED_DIR,
    TRAVEL_POI_SAMPLE_PER_REGION,
    TRAVEL_VECTORSTORE_PATH,
)
from src.travel.travel_ingestion import normalize_travel_data  # noqa: E402
from src.travel.travel_repository import build_database  # noqa: E402
from src.travel.travel_vectorstore import (  # noqa: E402
    build_travel_vectorstore,
    prioritize_documents_for_faiss,
)


def _read_jsonl(path: Path) -> list[dict]:
    import json

    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build travel SQLite + FAISS indexes")
    parser.add_argument("--skip-faiss", action="store_true", help="Only normalize + SQLite")
    parser.add_argument(
        "--faiss-only",
        action="store_true",
        help="Skip normalize/SQLite and build FAISS from existing JSONL",
    )
    parser.add_argument("--force-faiss", action="store_true", help="Rebuild FAISS even if present")
    parser.add_argument(
        "--include-validation",
        action="store_true",
        help="Include Validation POI labels",
    )
    parser.add_argument(
        "--sample-per-region",
        type=int,
        default=TRAVEL_POI_SAMPLE_PER_REGION,
        help="Sample size for 음식점/숙박 per region",
    )
    parser.add_argument(
        "--max-full",
        type=int,
        default=None,
        help="Cap for 관광지/문화시설 (useful for smoke tests)",
    )
    parser.add_argument(
        "--faiss-poi-limit",
        type=int,
        default=TRAVEL_FAISS_POI_LIMIT,
    )
    parser.add_argument(
        "--faiss-course-limit",
        type=int,
        default=TRAVEL_FAISS_COURSE_LIMIT,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Embedding batch size (defaults to config)",
    )
    parser.add_argument(
        "--batch-delay",
        type=float,
        default=None,
        help="Seconds between embedding batches",
    )
    args = parser.parse_args()

    if args.faiss_only:
        paths = {
            "pois": TRAVEL_NORMALIZED_DIR / "pois.jsonl",
            "courses": TRAVEL_NORMALIZED_DIR / "courses.jsonl",
        }
        if not paths["pois"].exists() or not paths["courses"].exists():
            raise SystemExit(
                "normalized JSONL이 없습니다. 먼저 전체 빌드를 실행하세요."
            )
    else:
        paths = normalize_travel_data(
            include_validation=args.include_validation,
            sample_per_region=args.sample_per_region,
            max_full=args.max_full,
            output_dir=TRAVEL_NORMALIZED_DIR,
        )
        build_database(db_path=TRAVEL_DB_PATH, normalized_dir=TRAVEL_NORMALIZED_DIR)

        if args.skip_faiss:
            print("FAISS 인덱싱을 건너뜁니다.")
            return

    pois = _read_jsonl(paths["pois"])
    courses = _read_jsonl(paths["courses"])
    documents = prioritize_documents_for_faiss(
        pois,
        courses,
        poi_limit=args.faiss_poi_limit,
        course_limit=args.faiss_course_limit,
    )
    print(f"FAISS 대상: {len(documents)} documents (priority cities preferred)")

    kwargs = {"force": args.force_faiss, "path": TRAVEL_VECTORSTORE_PATH}
    if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
    if args.batch_delay is not None:
        kwargs["batch_delay_sec"] = args.batch_delay

    build_travel_vectorstore(documents, **kwargs)
    print("여행 인덱스 빌드 완료.")


if __name__ == "__main__":
    main()
