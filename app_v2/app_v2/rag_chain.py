from typing import Any, Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from .settings import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, require_env
from .vectorstore import get_vectorstore

SYSTEM_PROMPT = """너는 식품 안전/위생/영양 문서에 기반해 한국어로 답변한다.

규칙:
- 아래 컨텍스트에 '명시적으로' 있는 내용만 사용한다. 추측/일반상식으로 채우지 않는다.
- 답변의 각 핵심 bullet 끝에 근거를 (doc_id, chunk_id)로 표기한다.
- 컨텍스트에 근거가 없으면 '문서 근거가 부족하다'고 말한다.
"""

def retrieve_contexts(query: str, top_k: int) -> List[Dict[str, Any]]:
    vs = get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(query)

    contexts: List[Dict[str, Any]] = []
    for d in docs:
        md = dict(d.metadata or {})
        contexts.append(
            {
                "doc_id": md.get("doc_id"),
                "chunk_id": md.get("chunk_id"),
                "text": d.page_content,
            }
        )
    return contexts

def generate_answer(query: str, contexts: List[Dict[str, Any]]) -> str:
    require_env("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)

    joined = "\n\n".join(
        [f"[doc_id={c.get('doc_id')}, chunk_id={c.get('chunk_id')}]\n{c.get('text')}" for c in contexts]
    )

    user_prompt = f"""질문: {query}

컨텍스트(문서 발췌):
{joined}
"""

    llm = ChatAnthropic(
        model=ANTHROPIC_MODEL,
        temperature=0,
        max_tokens=800,
        api_key=ANTHROPIC_API_KEY,
    )

    res = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)])
    return str(res.content)
