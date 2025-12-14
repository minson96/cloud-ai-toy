# app_v2/classifier.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os

import joblib

# 카테고리/키워드(초기 baseline)
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "제조": ["제조", "가공", "제조가공", "공정", "작업장", "자가품질검사", "원료", "포장", "창고"],
    "수입": ["수입", "해외제조업소", "해외작업장", "현지실사", "통관", "수입식품", "주문자상표부착"],
    "축산": ["축산", "도축", "식육", "축산물", "식용란", "가공업", "식육포장"],
    "행정": ["행정처분", "고발", "과태료", "공표", "점검결과", "재점검", "특별관리업체", "조치"],
    "검사": ["검사", "수거", "검체", "부적합", "검사항목", "검사주기", "시험", "실험실", "시약"],
    "표시광고": ["표시", "광고", "허위", "과대", "부당", "표시기준", "원산지"],
}

DEFAULT_CATEGORY = "제조"


def heuristic_classify(text: str) -> Tuple[str, float]:
    """
    키워드 기반 분류(모델이 없을 때 fallback).
    반환: (category, confidence[0~1])
    """
    t = (text or "").strip()
    if not t:
        return DEFAULT_CATEGORY, 0.0

    best_cat = DEFAULT_CATEGORY
    best_score = 0

    for cat, kws in CATEGORY_KEYWORDS.items():
        score = 0
        for kw in kws:
            if kw in t:
                score += 1
        if score > best_score:
            best_cat = cat
            best_score = score

    # confidence는 단순 스코어 기반(정확한 확률 아님)
    conf = min(1.0, best_score / 5.0) if best_score > 0 else 0.2
    return best_cat, conf


@dataclass
class ClassificationResult:
    category: str
    confidence: float
    method: str  # "model" | "heuristic"


class TfidfClassifier:
    """
    joblib로 저장한 sklearn 파이프라인(예: TfidfVectorizer + LogisticRegression)을 로드하여 예측한다.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.pipeline = None

    def load(self) -> bool:
        if not self.model_path:
            return False
        if not os.path.exists(self.model_path):
            return False
        self.pipeline = joblib.load(self.model_path)
        return True

    def predict(self, text: str) -> ClassificationResult:
        if self.pipeline is None:
            cat, conf = heuristic_classify(text)
            return ClassificationResult(category=cat, confidence=conf, method="heuristic")

        proba = self.pipeline.predict_proba([text])[0]
        classes = list(self.pipeline.classes_)
        idx = int(proba.argmax())
        return ClassificationResult(category=str(classes[idx]), confidence=float(proba[idx]), method="model")


def get_classifier(model_path: str) -> TfidfClassifier:
    clf = TfidfClassifier(model_path=model_path)
    clf.load()  # 실패해도 OK(heuristic으로 fallback)
    return clf
