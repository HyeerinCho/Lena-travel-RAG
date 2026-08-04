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
# 여행 기간이 지정되지 않았을 때 보여줄 기본 예보 일수(오늘부터 1주일)
WEATHER_DEFAULT_DAYS = 7
WEATHER_REQUEST_TIMEOUT_SEC = 6

# 한국관광공사 공공데이터(data.go.kr B551011) 통합 설정.
# 아래 5개 서비스가 동일 계정 인증키를 공유합니다.
#  - KorService2         : 국문 관광정보          https://www.data.go.kr/data/15101578/openapi.do
#  - KorPetTourService2  : 반려동물 동반여행       https://www.data.go.kr/data/15135102/openapi.do
#  - KorWithService2     : 무장애 여행            https://www.data.go.kr/data/15101897/openapi.do
#  - Durunubi            : 두루누비(코리아둘레길)  https://www.data.go.kr/data/15101974/openapi.do
#  - TarRlteTarService1  : 관광지별 연관 관광지    https://www.data.go.kr/data/15128560/openapi.do
#
# 인증키는 .env 의 DATA_GO_KR_API_KEY 로 주입하세요. 값이 없으면 "[APIkey]"
# 플레이스홀더가 사용되며, 실제 호출 전에 반드시 채워야 합니다.
# 주의: data.go.kr 의 "일반 인증키(Decoding)" 값을 사용하세요.
# (urlencode 가 자동 인코딩하므로 Encoding 키를 쓰면 이중 인코딩됩니다.)
DATA_GO_KR_API_KEY = (
    os.getenv("DATA_GO_KR_API_KEY")
    or os.getenv("KOR_PET_TOUR_API_KEY")  # 하위 호환
    or "[APIkey]"
)
DATA_GO_KR_MOBILE_OS = os.getenv("DATA_GO_KR_MOBILE_OS", "ETC")
DATA_GO_KR_MOBILE_APP = os.getenv("DATA_GO_KR_MOBILE_APP", "LENA")
DATA_GO_KR_REQUEST_TIMEOUT_SEC = 8

_TOURAPI_BASE = "https://apis.data.go.kr/B551011"
KOR_SERVICE_BASE_URL = f"{_TOURAPI_BASE}/KorService2"
KOR_PET_TOUR_BASE_URL = f"{_TOURAPI_BASE}/KorPetTourService2"
KOR_WITH_SERVICE_BASE_URL = f"{_TOURAPI_BASE}/KorWithService2"
DURUNUBI_BASE_URL = f"{_TOURAPI_BASE}/Durunubi"
TAR_RLTE_TAR_BASE_URL = f"{_TOURAPI_BASE}/TarRlteTarService1"

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
