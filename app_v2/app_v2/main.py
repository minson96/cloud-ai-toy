from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent_chain import agent_answer
from .rag_chain import generate_answer, retrieve_contexts

app = FastAPI(title="cloud-ai-toy v3 (agent)")

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)

@app.get("/health")
def health():
    return {"status": "ok", "service": "app_v3"}

@app.post("/query")
def query(req: QueryRequest):
    contexts = retrieve_contexts(req.query, req.top_k)
    answer = generate_answer(req.query, contexts)
    return {"query": req.query, "answer": answer, "contexts": contexts}

@app.post("/agent_query")
def agent_query(req: QueryRequest):
    answer, contexts, decision = agent_answer(req.query, req.top_k)
    return {
        "query": req.query,
        "decision": decision,
        "answer": answer,
        "contexts": contexts,
    }
