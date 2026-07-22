from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template("""
    아래 context를 바탕으로 질문에 답해주세요.

    Context: {context}

    질문: {question}
""")

TRAVEL_EXTRACT_PROMPT = ChatPromptTemplate.from_template("""
당신은 한국 여행 요구사항 추출기입니다.
사용자 질문과 이전 대화에서 JSON만 출력하세요. 설명 문장 금지.

스키마:
{{
  "destination": string|null,   // 도시/지역 (예: 제주, 서울, Busan)
  "days": number|null,          // 총 여행 일수 (2박3일 -> 3)
  "budget": number|null,        // 원화 정수, 없으면 null
  "preferences": string[],      // 선호 키워드
  "language": "ko"|"en",
  "place_types": string[]       // 가능하면 관광지/문화시설/음식점/숙박 중 선택
}}

규칙:
- 예산이 "50만원"이면 500000
- 선호가 없으면 빈 배열
- 영어 질문이면 language=en
- 이번 질문에 새 값이 없으면 이전 대화/세션 값을 유지하세요
- 후속 질문(예: "둘째 날만 바꿔줘")이면 destination/days 등을 비우지 말고 유지하세요

이전 대화:
{history}

현재 세션 기본값:
- destination: {session_destination}
- days: {session_days}
- budget: {session_budget}
- preferences: {session_preferences}

질문: {question}
""")

TRAVEL_ITINERARY_PROMPT = ChatPromptTemplate.from_template("""
당신은 한국 여행 일정 플래너입니다.
아래 후보 장소/코스와 이전 대화만 근거로 답하세요.
근거에 없는 운영시간, 입장료, 메뉴, 실시간 가격을 지어내지 마세요.
과거 패키지 가격은 참고값이며 예약 가능 여부가 아닙니다.
이전 일정을 수정·보완하는 질문이면 이전 답변을 기억하고 변경점만 반영하세요.

응답 언어: {language}

이전 대화:
{history}

사용자 요구:
- 질문: {question}
- 목적지: {destination}
- 일수: {days}
- 예산: {budget}
- 선호: {preferences}

후보 장소(JSON):
{places}

후보 코스(JSON):
{courses}

다음 JSON만 출력하세요:
{{
  "itinerary": [
    {{
      "day": 1,
      "theme": string,
      "slots": [
        {{"time": "morning|afternoon|evening", "place_name": string, "poi_id": string|null, "note": string}}
      ]
    }}
  ],
  "highlights": string[],
  "warnings": string[],
  "answer": string
}}

answer에는 사용자가 바로 읽을 수 있는 일정 요약을 넣으세요.
후보가 부족하면 warnings에 명시하고 가능한 범위만 제안하세요.
Day1 / Day2 로 나누기
장소는 불릿(-)으로
마지막에 한 줄 팁 작성
""")
