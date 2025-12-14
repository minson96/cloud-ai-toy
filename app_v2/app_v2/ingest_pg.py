# app_v2/ingest_pg.py
import json
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_postgres import PGVector

from .classifier import get_classifier
from .settings import CLASSIFIER_MODEL_PATH, COLLECTION_NAME, DATABASE_URL, require_env
from .vectorstore import get_embeddings

PROCESSED_DIR = Path("data/processed")


def load_processed_chunks() -> List[Document]:
    if not PROCESSED_DIR.exists():
        raise RuntimeError(f"{PROCESSED_DIR} 폴더가 없습니다. (docker-compose의 data 볼륨 마운트 확인)")

    clf = get_classifier(CLASSIFIER_MODEL_PATH)

    docs: List[Document] = []
    for jf in PROCESSED_DIR.glob("*.json"):
        with jf.open("r", encoding="utf-8") as f:
            doc = json.load(f)

        doc_id = doc.get("doc_id", jf.stem)
        chunks = doc.get("chunks") or doc.get("items") or []
        for ch in chunks:
            text = (ch.get("text") or "").strip()
            if not text:
                continue

            chunk_id = ch.get("chunk_id")
            # 분류(문서/청크 단위)
            pred = clf.predict(text)

            metadata: Dict[str, Any] = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "source": ch.get("source"),
                "category": pred.category,
                "category_confidence": pred.confidence,
                "category_method": pred.method,
            }
            docs.append(Document(page_content=text, metadata=metadata))

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
