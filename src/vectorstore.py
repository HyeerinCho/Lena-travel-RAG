import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import EMBEDDING_MODEL, VECTORSTORE_PATH


def _index_exists(path: Path | str) -> bool:
    root = Path(path)
    return (root / "index.faiss").is_file() and (
        (root / "index.pkl").is_file() or (root / "index.faiss").is_file()
    )


def build_vectorstore(chunks, path: Path | str | None = None):
    store_path = Path(path) if path is not None else Path(VECTORSTORE_PATH)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    if _index_exists(store_path):
        print(f"저장된 벡터DB 로딩... ({store_path})")
        return FAISS.load_local(
            str(store_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    if store_path.exists() and not any(store_path.iterdir()):
        print(f"빈 인덱스 디렉터리 감지, 새로 생성합니다: {store_path}")
    elif store_path.exists() and not _index_exists(store_path):
        print(f"불완전한 인덱스 디렉터리 감지, 새로 생성합니다: {store_path}")

    if not chunks:
        raise ValueError("벡터DB를 생성할 문서 청크가 없습니다.")

    print("벡터DB 새로 생성 중...")
    store_path.mkdir(parents=True, exist_ok=True)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(store_path))
    print("벡터DB 저장 완료!")
    return vectorstore


def load_vectorstore(path: Path | str):
    store_path = Path(path)
    if not _index_exists(store_path):
        raise FileNotFoundError(f"FAISS 인덱스가 없습니다: {store_path}")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(
        str(store_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
