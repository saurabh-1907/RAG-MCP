# rag_api.py
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import os

app = FastAPI(title="RAG API (safe)")

# take key from either env name
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

genai = None
if API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=API_KEY)
    except Exception:
        genai = None  # don't crash if import fails


class RAGQuery(BaseModel):
    query: str


DOCS = [
    "This is a RAG MCP project. MCP server runs locally and forwards queries.",
    "The RAG HTTP endpoint is deployed on Render and protected with Bearer token.",
    "You can plug Gemini to generate final answers from retrieved context.",
]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/rag")
async def rag_endpoint(body: RAGQuery, authorization: str = Header(None)):
    # auth
    expected = os.getenv("RAG_API_TOKEN", "test")
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid token")

    # simple retrieval
    qwords = body.query.lower().split()
    relevant = [d for d in DOCS if any(w in d.lower() for w in qwords)] or DOCS[:1]

    # default answer in case LLM fails
    answer = f"(no LLM) best match: {relevant[0]}"

    if genai:
        prompt = (
            "You are a RAG-style assistant. Use ONLY the context below.\n\n"
            f"Context:\n{chr(10).join(relevant)}\n\n"
            f"Question: {body.query}\n"
            "Answer clearly and mention which context you used."
        )
        # 1) try your preferred model
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            resp = model.generate_content(prompt)
            answer = resp.text
        except Exception as e1:
            # 2) fall back to older, more available one
            try:
                model = genai.GenerativeModel("gemini-1.0-pro")
                resp = model.generate_content(prompt)
                answer = resp.text
            except Exception as e2:
                # 3) still return JSON, don't 500
                answer = (
                    f"(LLM error: {e1}; fallback error: {e2}) "
                    f"Context: {relevant[0]}"
                )

    return {
        "answer": answer,
        "sources": relevant,
    }
