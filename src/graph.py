from typing import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from src.prompts import RAG_PROMPT
from src.config import LLM_MODEL

# 1. State 정의 (그래프에서 노드 간에 공유되는 데이터)
class RAGState(TypedDict):
    query: str
    documents: List[Document]
    answer: str 


def build_rag_graph(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)

    def retrieve(state: RAGState) -> RAGState:
        print(f"[검색 노드] 쿼리: {state['query']}")
        documents = retriever.invoke(state["query"])
        return {"documents": documents}

    def generate(state: RAGState) -> RAGState:
        print(f"[생성 노드] 문서 {len(state['documents'])}개로 답변 생성 중...")
        context = "\n\n".join(doc.page_content for doc in state["documents"])
        chain = RAG_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": state["query"]})
        return {"answer": answer}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    
    return graph.compile()

