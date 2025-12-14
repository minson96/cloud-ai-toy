# app_v2/train_classifier.py
import json
from pathlib import Path
from typing import List, Tuple

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from .classifier import heuristic_classify, CATEGORY_KEYWORDS
from .settings import CLASSIFIER_MODEL_PATH

PROCESSED_DIR = Path("data/processed")


def load_texts() -> List[str]:
    if not PROCESSED_DIR.exists():
        raise RuntimeError(f"{PROCESSED_DIR} 폴더가 없습니다.")

    texts: List[str] = []
    for jf in PROCESSED_DIR.glob("*.json"):
        with jf.open("r", encoding="utf-8") as f:
            doc = json.load(f)
        # v1/v2 processed 포맷이 다양할 수 있어 방어적으로 처리
        chunks = doc.get("chunks") or doc.get("items") or []
        for ch in chunks:
            t = (ch.get("text") or "").strip()
            if t:
                texts.append(t)
    if not texts:
        raise RuntimeError("학습에 사용할 텍스트가 없습니다. data/processed 포맷/내용을 확인하세요.")
    return texts


def build_dataset(texts: List[str]) -> Tuple[List[str], List[str]]:
    X: List[str] = []
    y: List[str] = []
    for t in texts:
        cat, conf = heuristic_classify(t)
        # 너무 약한 매칭(확신 낮음)은 학습에서 제외해 노이즈를 줄임
        if conf < 0.4:
            continue
        X.append(t)
        y.append(cat)
    if len(set(y)) < 2:
        raise RuntimeError(
            "라벨 다양성이 부족합니다(카테고리가 2개 이상 필요). "
            "CATEGORY_KEYWORDS를 늘리거나 processed 텍스트를 확인하세요."
        )
    return X, y


def main():
    texts = load_texts()
    X, y = build_dataset(texts)

    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=300)),
        ]
    )

    pipe.fit(X, y)

    out_path = Path(CLASSIFIER_MODEL_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)

    print(f"OK: trained classifier on n={len(X)} samples, saved to {out_path}")
    print(f"Classes: {list(pipe.classes_)}")
    print("Keyword categories:", list(CATEGORY_KEYWORDS.keys()))


if __name__ == "__main__":
    main()
