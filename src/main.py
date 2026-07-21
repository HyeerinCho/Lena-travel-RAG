import argparse

from src.rag import ask
from src.travel.travel_agent import ask_travel


def main():
    parser = argparse.ArgumentParser(description="LENA CLI")
    parser.add_argument("question", nargs="?", default=None, help="질문")
    parser.add_argument(
        "--agent",
        choices=["notes", "travel"],
        default="notes",
        help="사용할 에이전트",
    )
    parser.add_argument("--destination", default=None)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--language", choices=["ko", "en"], default=None)
    args = parser.parse_args()

    question = args.question or (
        "제주 2박3일 자연 경관 위주 여행 추천해줘"
        if args.agent == "travel"
        else "RAG가 뭐야?"
    )

    if args.agent == "travel":
        result = ask_travel(
            question,
            destination=args.destination,
            days=args.days,
            budget=args.budget,
            language=args.language,
        )
        print(f"질문: {result['question']}")
        print(f"답변:\n{result['answer']}")
        if result.get("warnings"):
            print("경고:")
            for w in result["warnings"]:
                print(f"- {w}")
        return

    print(f"질문: {question}")
    print(f"답변: {ask(question)}")


if __name__ == "__main__":
    main()
