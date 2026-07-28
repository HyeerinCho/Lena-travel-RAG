import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
elif not os.getenv("GOOGLE_API_KEY"):
    raise ValueError(
        "GOOGLE_API_KEY가 설정되지 않았습니다. "
        ".env 파일을 확인하세요."
    )

EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"

# Real-time weather via Open-Meteo (무료, API 키 불필요). https://open-meteo.com
# 위/경도만으로 조회하며, 네트워크 실패 시 날씨 없이 동작합니다.
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
# Open-Meteo는 최대 16일까지 일별 예보 제공
WEATHER_FORECAST_MAX_DAYS = 16
WEATHER_REQUEST_TIMEOUT_SEC = 6

DATA_PATH = PROJECT_ROOT / "data" / "alex-notes"
VECTORSTORE_PATH = PROJECT_ROOT / "data" / "faiss_index"

# Travel data (NFC names; resolve_data_dir handles macOS NFD paths)
POI_DATA_DIRNAME = "221.관광지 소개 다국어 번역 데이터"
TRAVEL_INFO_DIRNAME = "여행 정보 데이터셋"
TRAVEL_DIR = PROJECT_ROOT / "data" / "travel"
TRAVEL_NORMALIZED_DIR = TRAVEL_DIR / "normalized"
TRAVEL_DB_PATH = TRAVEL_DIR / "travel.db"
TRAVEL_VECTORSTORE_PATH = TRAVEL_DIR / "faiss_index"

# Gemini free tier: embed_content 100 requests/min
EMBED_BATCH_SIZE = 80
EMBED_BATCH_DELAY_SEC = 65

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# MVP indexing defaults
TRAVEL_POI_FULL_TYPES = ("관광지", "문화시설")
TRAVEL_POI_SAMPLED_TYPES = ("음식점", "숙박")
TRAVEL_POI_SAMPLE_PER_REGION = 80
TRAVEL_FAISS_POI_LIMIT = 1500
TRAVEL_FAISS_COURSE_LIMIT = 2000


def resolve_data_dir(dirname: str) -> Path:
    """Resolve a data subdirectory even when the filesystem uses NFD Hangul."""
    import unicodedata

    data_root = PROJECT_ROOT / "data"
    target = unicodedata.normalize("NFC", dirname)
    direct = data_root / target
    if direct.exists():
        return direct

    for child in data_root.iterdir():
        if unicodedata.normalize("NFC", child.name) == target:
            return child

    raise FileNotFoundError(f"data 하위 폴더를 찾을 수 없습니다: {dirname}")


def poi_data_root() -> Path:
    return resolve_data_dir(POI_DATA_DIRNAME)


def travel_info_root() -> Path:
    return resolve_data_dir(TRAVEL_INFO_DIRNAME)


def course_csv_path() -> Path:
    root = travel_info_root()
    # Nested folder: 여행 정보 데이터셋/여행 정보 데이터셋/여행코스데이터.csv
    candidates = list(root.rglob("여행코스데이터.csv"))
    if not candidates:
        raise FileNotFoundError("여행코스데이터.csv 를 찾을 수 없습니다.")
    return candidates[0]
