from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import os

app = FastAPI(title="RAG API (Render + Gemini)")

# read API key from either name
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

genai = None
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        genai = None  # don't kill the app


class RAGQuery(BaseModel):
    query: str


DOCS = [
    "This is a RAG MCP project. MCP server runs locally and forwards queries.",
    "The RAG HTTP endpoint is deployed on Render and protected with Bearer token.",
    "You can plug Gemini (2.5 flash) to generate final answers from retrieved context.",
]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/rag")
async def rag_endpoint(body: RAGQuery, authorization: str = Header(None)):
    # 1. auth
    expected = os.getenv("RAG_API_TOKEN", "test")
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid token")

    # 2. simple retrieval
    qwords = body.query.lower().split()
    relevant = [d for d in DOCS if any(w in d.lower() for w in qwords)] or DOCS[:1]

    # 3. LLM call with fallback
    answer = None
    if genai:
        prompt = (
            "You are a RAG answerer. Use ONLY the context below.\n\n"
            f"Context:\n{chr(10).join(relevant)}\n\n"
            f"Question: {body.query}\n"
            "Answer clearly and mention which context line you used."
        )
        try:
            # your preferred model
            model = genai.GenerativeModel("gemini-2.5-flash")
            resp = model.generate_content(prompt)
            answer = resp.text
        except Exception as e1:
            # fallback to a more common model name
            try:
                model = genai.GenerativeModel("gemini-1.0-pro")
                resp = model.generate_content(prompt)
                answer = resp.text
            except Exception as e2:
                answer = f"(LLM error: {e1} ; fallback error: {e2}) Context: {relevant[0]}"
    else:
        answer = f"(Gemini not configured) Best match: {relevant[0]}"

    return {
        "answer": answer,
        "sources": relevant,
    }
