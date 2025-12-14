from langchain_postgres import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings

from .settings import (
    DATABASE_URL,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    require_env,
)

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def get_vectorstore():
    require_env("DATABASE_URL", DATABASE_URL)
    embeddings = get_embeddings()
    return PGVector(
        connection=DATABASE_URL,
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
    )
