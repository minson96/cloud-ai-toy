import json
import re
from typing import Any, Dict, List, Tuple

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from .rag_chain import generate_answer, retrieve_contexts
from .settings import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, require_env

REWRITE_SYSTEM = """너는 검색 질의 재작성기다.
규칙:
- 원 질문의 의미를 유지하면서, 문서 검색에 유리한 핵심 키워드 형태로 바꾼다.
- 사실을 추가하지 않는다.
- 출력은 JSON 한 줄로만 한다: {"rewritten_query": "..."}"""

def _rewrite_query_for_search(query: str) -> str:
    require_env("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)
    require_env("ANTHROPIC_MODEL", ANTHROPIC_MODEL)

    llm = ChatAnthropic(
        model=ANTHROPIC_MODEL,
        temperature=0,
        max_tokens=200,
        api_key=ANTHROPIC_API_KEY,
    )

    msg = f"원 질문: {query}\n검색용으로 재작성해."
    res = llm.invoke([SystemMessage(content=REWRITE_SYSTEM), HumanMessage(content=msg)])
    raw = str(res.content).strip()

    try:
        data = json.loads(raw)
        rq = str(data.get("rewritten_query", "")).strip()
        return rq or query
    except Exception:
        return query


def _decide_style(query: str, top_k: int) -> Dict[str, Any]:
    q = query.strip()
    wants_summary = any(k in q for k in ["요약", "정리", "핵심", "한눈에", "간단히"])
    if wants_summary:
        return {"summary_level": "high", "max_bullets": 8, "include_sections": True}
    return {"summary_level": "normal", "max_bullets": 12, "include_sections": True}


def _clean_answer(text: str) -> str:
    """
    LLM이 가끔 남기는 마크다운 찌꺼기(마지막 '**', '---' 등) 제거.
    """
    if not text:
        return text

    t = text.strip()

    lines = t.splitlines()
    while lines and lines[-1].strip() in {"**", "*", "```", "---"}:
        lines.pop()
    t = "\n".join(lines).rstrip()

    while "\n\n\n" in t:
        t = t.replace("\n\n\n", "\n\n")

    return t.strip()


_BULLET_RE = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s+", re.UNICODE)
_HEADING_RE = re.compile(r"^\s*#{1,6}\s+", re.UNICODE)
_CITE_RE = re.compile(r"\(doc_id=.*?,\s*chunk_id=.*?\)\s*$", re.UNICODE)


def _validate_output(answer: str, summary_level: str, max_bullets: int) -> Tuple[bool, List[str]]:
    """
    summary_level == 'high'이면:
    - heading(#...) / 빈줄 / bullet만 허용
    - bullet은 반드시 (doc_id=..., chunk_id=...)로 끝나야 함
    - bullet 개수는 max_bullets 이하여야 함
    """
    bad: List[str] = []
    bullet_count = 0

    for line in answer.splitlines():
        s = line.strip()
        if not s:
            continue

        if _HEADING_RE.match(s):
            continue

        if _BULLET_RE.match(s):
            bullet_count += 1
            if not _CITE_RE.search(s):
                bad.append(f"[근거없음] {s}")
            continue

        if summary_level == "high":
            bad.append(f"[본문금지] {s}")

    if bullet_count > max_bullets:
        bad.append(f"[불릿초과] bullets={bullet_count} > max_bullets={max_bullets}")

    return (len(bad) == 0, bad)


def agent_answer(query: str, top_k: int = 5) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    style = _decide_style(query, top_k)

    # 1차 검색
    contexts = retrieve_contexts(query, top_k=top_k)
    decision: Dict[str, Any] = {"search_rounds": 1, "rewritten": False, "style": style}

    if not contexts:
        # 2차: 질의 재작성 후 재검색
        rewritten = _rewrite_query_for_search(query)
        contexts = retrieve_contexts(rewritten, top_k=min(20, top_k + 5))
        decision["search_rounds"] = 2
        decision["rewritten"] = (rewritten != query)

        if not contexts:
            return "문서 근거가 부족하다", [], decision

    # 1차 답변 생성 + 후처리
    answer = _clean_answer(generate_answer(query, contexts, style=style))

    # 구조+근거 검증 + 재생성(최대 2회)
    for _ in range(2):
        ok, _issues = _validate_output(
            answer,
            summary_level=str(style.get("summary_level", "normal")),
            max_bullets=int(style.get("max_bullets", 10)),
        )
        if ok:
            return answer, contexts, decision

        # 재생성(요약이면 더 짧게 유지)
        stronger_style = dict(style)
        if stronger_style.get("summary_level") == "high":
            stronger_style["max_bullets"] = min(int(stronger_style.get("max_bullets", 8)), 8)

        answer = _clean_answer(generate_answer(query, contexts, style=stronger_style))

    # 끝까지 실패하면 안전 종료
    return "문서 근거가 부족하다", contexts, decision
