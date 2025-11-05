from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI(title="RAG API (multi)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
genai = None
if API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=API_KEY)
    except Exception:
        genai = None

RAG_TOKEN = os.getenv("RAG_API_TOKEN", "test")

# start with some docs
DOCS = [
    "This is a RAG MCP project. MCP server runs locally and forwards queries.",
    "The RAG HTTP endpoint is deployed on Render and protected with Bearer token.",
    "You can plug Gemini to generate final answers from retrieved context.",
]


class RAGQuery(BaseModel):
    query: str
    extra_context: str | None = None


class IngestBody(BaseModel):
    text: str


def simple_retrieve(query: str, top_k: int = 3):
    qwords = query.lower().split()
    scored = []
    for d in DOCS:
        score = sum(1 for w in qwords if w in d.lower())
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    docs = [d for s, d in scored[:top_k] if s > 0]
    return docs or DOCS[:1]


def call_gemini(prompt: str):
    if not genai:
        return "(Gemini not configured) " + prompt
    # try new model → fallback
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        return resp.text
    except Exception:
        model = genai.GenerativeModel("gemini-1.0-pro")
        resp = model.generate_content(prompt)
        return resp.text


@app.get("/health")
async def health():
    return {"status": "ok"}


def check_auth(auth: str | None):
    if auth != f"Bearer {RAG_TOKEN}":
        raise HTTPException(401, "Invalid token")


@app.post("/rag")
async def rag_endpoint(body: RAGQuery, authorization: str = Header(None)):
    # auth
    expected = os.getenv("RAG_API_TOKEN", "test")
    if authorization != f"Bearer {expected}":
        raise HTTPException(401, "Invalid token")

    # your existing retrieval
    docs = simple_retrieve(body.query)

    # 👇 put user-provided context first so the model sees it
    if body.extra_context:
        docs = [body.extra_context] + docs

    prompt = (
        "You are a RAG assistant. Use ONLY this context.\n\n"
        f"Context:\n{chr(10).join(docs)}\n\n"
        f"Question: {body.query}\n"
        "Answer clearly and mention which context you used."
    )

    answer = call_gemini(prompt)  # whatever you already had
    return {"answer": answer, "sources": docs}


@app.post("/retrieve")
async def retrieve_endpoint(body: QueryBody, authorization: str = Header(None)):
    check_auth(authorization)
    docs = simple_retrieve(body.query)
    return {"query": body.query, "results": docs}


@app.post("/summarize")
async def summarize_endpoint(body: QueryBody, authorization: str = Header(None)):
    check_auth(authorization)
    prompt = (
        "Summarize the following text in 3-5 bullet points:\n\n"
        f"{body.query}"
    )
    summary = call_gemini(prompt)
    return {"summary": summary}


@app.post("/ingest")
async def ingest_endpoint(body: IngestBody, authorization: str = Header(None)):
    check_auth(authorization)
    DOCS.append(body.text)
    return {"status": "ok", "count": len(DOCS)}
