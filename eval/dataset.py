import os

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

# LangSmith client uses LANGSMITH_API_KEY / LANGSMITH_ENDPOINT from .env
client = Client(
    api_key=os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"),
)

DATASET_NAME = "lena-travel"

examples = [
    {
        "input": {"query": "제주도에 유명한게 뭐야?"},
        "output": {
            "answer": (
                "제주도에서는 한라산, 성산일출봉, 우도, 협재·함덕 해수욕장, "
                "카페거리와 올레길 같은 자연·관광지가 유명합니다. "
                "자연 경관을 보고 싶다면 성산일출봉이나 한라산 둘레, "
                "바다를 즐기려면 협재나 함덕 해수욕장에 가면 좋습니다."
            )
        },
    },
    {
        "input": {"query": "제주 2박3일, 자연 경관 위주, 예산 50만원"},
        "output": {
            "answer": (
                "제주 2박3일 자연 경관 위주라면 1일차에 성산일출봉·우도, "
                "2일차에 한라산/오름·중문 해안, 3일차에 협재·애월 카페거리를 "
                "추천합니다. 예산 50만원 기준으로는 대중교통·렌터카와 "
                "현지 식사 위주로 일정과 동선을 짜는 것이 좋습니다."
            )
        },
    },
    {
        "input": {"query": "서울에서 비 오는 날 실내 문화시설 추천해줘"},
        "output": {
            "answer": (
                "비 오는 날 서울에서는 국립중앙박물관, 국립현대미술관, "
                "서울시립미술관, 디큐브시티·코엑스 같은 실내 문화·전시 공간을 "
                "추천합니다. 한곳에 오래 머물기 좋은 박물관·미술관 위주로 "
                "동선을 잡는 것이 좋습니다."
            )
        },
    },
]


def ensure_dataset(name: str):
    try:
        return client.read_dataset(dataset_name=name)
    except Exception:
        return client.create_dataset(dataset_name=name)


ensure_dataset(DATASET_NAME)

for example in examples:
    client.create_example(
        dataset_name=DATASET_NAME,
        inputs=example["input"],
        outputs=example["output"],
    )

print(f"Dataset '{DATASET_NAME}'에 {len(examples)}개 예제 추가 완료!")
