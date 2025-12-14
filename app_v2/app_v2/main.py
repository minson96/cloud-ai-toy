from fastapi import FastAPI

app = FastAPI(title="cloud-ai-toy v2")

@app.get("/health")
def health():
    return {"status": "ok", "service": "app_v2"}
