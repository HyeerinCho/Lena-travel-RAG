from functools import lru_cache

from src.config import DATA_PATH
from src.graph import build_rag_graph
from src.ingestion import load_and_split
from src.vectorstore import build_vectorstore


@lru_cache
def get_agent():
    chunks = load_and_split(str(DATA_PATH))
    vectorstore = build_vectorstore(chunks)
    return build_rag_graph(vectorstore)


# 하위 호환
get_rag_graph = get_agent


def ask(query: str) -> str:
    result = get_agent().invoke({"query": query})
    return result["answer"]
