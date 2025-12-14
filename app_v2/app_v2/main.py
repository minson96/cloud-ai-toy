from fastapi import FastAPI
from pydantic import BaseModel, Field

from .rag_chain import retrieve_contexts, generate_answer

app = FastAPI(title="cloud-ai-toy v2")

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)

@app.get("/health")
def health():
    return {"status": "ok", "service": "app_v2"}

@app.post("/query")
def query(req: QueryRequest):
    contexts = retrieve_contexts(req.query, req.top_k)
    answer = generate_answer(req.query, contexts)
    return {"query": req.query, "answer": answer, "contexts": contexts}
