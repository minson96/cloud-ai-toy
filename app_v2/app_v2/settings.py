import os

DATABASE_URL = os.getenv("DATABASE_URL", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "")

COLLECTION_NAME = "nds_food_safety_docs"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

# 분류 모델 아티팩트(컨테이너 내부 경로 기준)
CLASSIFIER_MODEL_PATH = os.getenv("CLASSIFIER_MODEL_PATH", "models/tfidf_classifier.joblib")

def require_env(name: str, value: str) -> None:
    if not value:
        raise RuntimeError(f"환경변수 {name} 가(이) 비어 있습니다. .env를 확인하세요.")