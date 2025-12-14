from typing import Any, Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from .settings import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, require_env
from .vectorstore import get_vectorstore

SYSTEM_PROMPT = """너는 식품 안전/위생/영양 문서에 기반해 한국어로 답변한다.

규칙:
- 아래 컨텍스트에 '명시적으로' 있는 내용만 사용한다. 추측/일반상식으로 채우지 않는다.
- 답변의 각 핵심 bullet 끝에 근거를 (doc_id=..., chunk_id=...)로 표기한다.
- 컨텍스트에 근거가 없으면 '문서 근거가 부족하다'고 말한다.
"""

def retrieve_contexts(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """PGVector에서 유사 문서 조각을 top_k개 가져와 API 응답용 dict로 변환."""
    vs = get_vectorstore()
    docs: List[Document] = vs.similarity_search(query, k=top_k)

    contexts: List[Dict[str, Any]] = []
    for d in docs:
        md = d.metadata or {}
        contexts.append(
            {
                "text": d.page_content,
                "doc_id": md.get("doc_id"),
                "chunk_id": md.get("chunk_id"),
                "source": md.get("source"),
            }
        )
    return contexts


def generate_answer(
    query: str,
    contexts: List[Dict[str, Any]],
    style: Optional[Dict[str, Any]] = None,
) -> str:
    require_env("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)
    require_env("ANTHROPIC_MODEL", ANTHROPIC_MODEL)

    if not contexts:
        return "문서 근거가 부족하다"

    style = style or {}
    summary_level = style.get("summary_level", "normal")  # "high" | "normal"
    max_bullets = int(style.get("max_bullets", 10))
    include_sections = bool(style.get("include_sections", True))

    joined = "\n\n".join(
        f"[doc_id={c.get('doc_id')}, chunk_id={c.get('chunk_id')}]\n{c.get('text','')}"
        for c in contexts
    )

    if summary_level == "high":
        format_hint = f"""출력 형식:
- 가장 중요한 항목만 {max_bullets}개 이하 bullet로 작성
- 불필요한 세부 공정/절차/예시는 제외
- bullet 외의 본문 문장은 작성하지 말 것(heading과 bullet만)
- 각 bullet 끝에 반드시 (doc_id=..., chunk_id=...)를 붙일 것
"""
    else:
        format_hint = f"""출력 형식:
- 핵심을 구조적으로 정리하되 bullet 총량은 {max_bullets}개 내로 제한
- 각 bullet 끝에 반드시 (doc_id=..., chunk_id=...)를 붙일 것
"""

    if include_sections:
        format_hint += "- 섹션(제목)은 가능하면 2~4개로만 나눌 것\n"
    else:
        format_hint += "- 섹션 제목 없이 bullet만 출력할 것\n"

    user_prompt = f"""질문:
{query}

{format_hint}

컨텍스트(문서 발췌):
{joined}
"""

    llm = ChatAnthropic(
        model=ANTHROPIC_MODEL,
        temperature=0,
        max_tokens=900,
        api_key=ANTHROPIC_API_KEY,
    )
    res = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)])
    return str(res.content)
