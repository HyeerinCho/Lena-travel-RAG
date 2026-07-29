"""한국관광공사 data.go.kr(B551011) 오픈 API 클라이언트 모음.

서비스별 모듈:
  - kor_service      : 국문 관광정보 (KorService2)
  - kor_pet_tour     : 반려동물 동반여행 (KorPetTourService2)
  - kor_with_service : 무장애 여행 (KorWithService2)
  - durunubi         : 두루누비 걷기여행길 (Durunubi)
  - tar_rlte_tar     : 관광지별 연관 관광지 (TarRlteTarService1)
"""

from ._client import TourAPIError, normalize_items, paged, request
from . import (
    durunubi,
    kor_pet_tour,
    kor_service,
    kor_with_service,
    tar_rlte_tar,
)

__all__ = [
    "TourAPIError",
    "request",
    "normalize_items",
    "paged",
    "kor_service",
    "kor_pet_tour",
    "kor_with_service",
    "durunubi",
    "tar_rlte_tar",
]
