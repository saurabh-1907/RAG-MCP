# rag_api.py
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import os
import google.generativeai as genai

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class RAGQuery(BaseModel):
    query: str

# fake docs for now – replace with DB/files later
DOCS = [
    "Job refer platform for students and recruiters.",
    "MCP server forwards queries to RAG backend with bearer token.",
    "Admin dashboard supports CRUD for users, recruiters and job posts.",
]

@app.post("/rag")
async def rag_endpoint(body: RAGQuery, authorization: str = Header(None)):
    expected = os.getenv("RAG_API_TOKEN", "test")
    if authorization != f"Bearer {expected}":
        raise HTTPException(401, "Invalid token")

    # simple retrieval
    q = body.query.lower()
    relevant = [d for d in DOCS if any(w in d.lower() for w in q.split())] or DOCS[:1]

    # call gemini if configured
    if GEMINI_API_KEY:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "Use the context to answer.\n"
            f"Context:\n{chr(10).join(relevant)}\n\n"
            f"Question: {body.query}\n"
            "Answer with sources."
        )
        resp = model.generate_content(prompt)
        answer = resp.text
    else:
        answer = f"(no gemini key) best match: {relevant[0]}"

    return {
        "answer": answer,
        "sources": relevant
    }
