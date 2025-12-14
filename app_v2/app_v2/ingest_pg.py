import json
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_postgres import PGVector

from .settings import DATABASE_URL, COLLECTION_NAME, require_env
from .vectorstore import get_embeddings

PROCESSED_DIR = Path("data/processed")

def load_processed_chunks() -> List[Document]:
    if not PROCESSED_DIR.exists():
        raise RuntimeError(f"{PROCESSED_DIR} 폴더가 없습니다. (docker-compose의 data 볼륨 마운트 확인)")

    json_files = sorted(PROCESSED_DIR.glob("*.json"))
    if not json_files:
        raise RuntimeError("data/processed/*.json 파일이 없습니다. v1 전처리 결과가 필요합니다.")

    docs: List[Document] = []
    for jf in json_files:
        with jf.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        # 흔한 형태를 방어적으로 처리
        if isinstance(payload, list):
            items = payload
            doc_id = jf.stem
        else:
            items = payload.get("chunks") or payload.get("items")
            if items is None:
                raise RuntimeError(f"{jf.name}에서 chunks/items를 찾지 못했습니다. processed 포맷 확인 필요")
            doc_id = payload.get("doc_id", jf.stem)

        for it in items:
            if not isinstance(it, dict):
                continue
            text = (it.get("text") or "").strip()
            if not text:
                continue
            chunk_id = it.get("chunk_id", None)

            md: Dict[str, Any] = {"doc_id": doc_id}
            if chunk_id is not None:
                md["chunk_id"] = chunk_id

            docs.append(Document(page_content=text, metadata=md))

    if not docs:
        raise RuntimeError("인덱싱할 Document가 0개입니다. processed json의 text가 비어있는지 확인하세요.")
    return docs

def main():
    require_env("DATABASE_URL", DATABASE_URL)
    embeddings = get_embeddings()
    documents = load_processed_chunks()

    PGVector.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        pre_delete_collection=True,
    )
    print(f"OK: indexed {len(documents)} documents into collection='{COLLECTION_NAME}'")

if __name__ == "__main__":
    main()
