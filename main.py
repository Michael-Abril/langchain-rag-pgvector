import os
import json
import logging
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "tinyllama")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "tinyllama")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "2048"))

DATABASE_URL = os.environ.get("DATABASE_URL", "")


def get_db():
    return psycopg2.connect(DATABASE_URL)


def wait_for_ollama(max_wait: int = 120) -> bool:
    """Wait until Ollama is reachable."""
    import time
    for i in range(max_wait // 5):
        try:
            resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            if resp.status_code == 200:
                logger.info("Ollama is reachable")
                return True
        except Exception:
            pass
        logger.info(f"Waiting for Ollama... ({i*5}s)")
        time.sleep(5)
    logger.error("Ollama not reachable after waiting")
    return False


def pull_model(model: str) -> bool:
    """Pull an Ollama model if not already available."""
    try:
        # Check if model is already available
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        models = [m["name"] for m in resp.json().get("models", [])]
        if any(model in m for m in models):
            logger.info(f"Model {model} already available")
            return True
        logger.info(f"Pulling model {model}...")
        resp = requests.post(
            f"{OLLAMA_URL}/api/pull",
            json={"name": model, "stream": False},
            timeout=600,
        )
        resp.raise_for_status()
        logger.info(f"Model {model} pulled successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to pull model {model}: {e}")
        return False


def init_db():
    db = get_db()
    cur = db.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            chunk TEXT NOT NULL,
            embedding vector(2048),
            source TEXT
        )
    """)
    db.commit()
    cur.close()
    db.close()
    logger.info("Database initialized with pgvector")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — waiting for Ollama and initializing DB")
    ollama_up = wait_for_ollama(max_wait=120)
    if ollama_up:
        pull_model(CHAT_MODEL)
    else:
        logger.warning("Ollama not available at startup; models will be pulled on first request")
    init_db()
    logger.info("Startup complete")
    yield


app = FastAPI(title="RAG App", description="LangChain-style RAG using Ollama + pgvector", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


def ensure_model_ready(model: str):
    """Pull model if not available (lazy, called before inference)."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(model in m for m in models):
            logger.info(f"Lazy pull: {model}")
            pull_model(model)
    except Exception as e:
        logger.warning(f"Could not check/pull {model}: {e}")


def embed_text(text: str) -> list[float]:
    ensure_model_ready(EMBED_MODEL)
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def generate_answer(prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": CHAT_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


@app.get("/")
def root():
    return {
        "service": "RAG App",
        "version": "1.0.0",
        "embed_model": EMBED_MODEL,
        "chat_model": CHAT_MODEL,
        "embed_dim": EMBED_DIM,
        "usage": {
            "upload": "POST /upload  (multipart file: text or PDF)",
            "query": "POST /query  {\"query\": \"your question\"}",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return {"ollama_url": OLLAMA_URL, "models": resp.json()}
    except Exception as e:
        return {"ollama_url": OLLAMA_URL, "error": str(e)}


@app.get("/ready")
def ready():
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT 1")
        cur.close()
        db.close()
        db_ok = True
    except Exception:
        db_ok = False

    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        embed_ready = any(EMBED_MODEL in m for m in models)
        chat_ready = any(CHAT_MODEL in m for m in models)
    except Exception:
        embed_ready = chat_ready = False

    all_ready = db_ok and embed_ready and chat_ready
    return {
        "ready": all_ready,
        "db": db_ok,
        "embed_model_ready": embed_ready,
        "chat_model_ready": chat_ready,
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8", errors="replace")

    if not text.strip():
        raise HTTPException(status_code=400, detail="File appears empty")

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text content found")

    db = get_db()
    cur = db.cursor()
    stored = 0
    for chunk in chunks:
        try:
            embedding = embed_text(chunk)
            cur.execute(
                "INSERT INTO documents (chunk, embedding, source) VALUES (%s, %s::vector, %s)",
                (chunk, json.dumps(embedding), file.filename),
            )
            stored += 1
        except Exception as e:
            logger.warning(f"Failed to embed chunk: {e}")
    db.commit()
    cur.close()
    db.close()

    return {
        "uploaded": True,
        "filename": file.filename,
        "total_chunks": len(chunks),
        "stored_chunks": stored,
    }


@app.post("/query")
async def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query_embedding = embed_text(req.query)

    db = get_db()
    cur = db.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        "SELECT chunk, source FROM documents ORDER BY embedding <-> %s::vector LIMIT %s",
        (json.dumps(query_embedding), req.top_k),
    )
    rows = cur.fetchall()
    cur.close()
    db.close()

    if not rows:
        raise HTTPException(status_code=404, detail="No documents found. Upload documents first via POST /upload")

    chunks = [row["chunk"] for row in rows]
    sources = list({row["source"] for row in rows if row["source"]})
    context = "\n\n---\n\n".join(chunks)

    prompt = (
        f"You are a helpful assistant. Answer the question using only the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {req.query}\n\n"
        f"Answer:"
    )

    answer = generate_answer(prompt)

    return {
        "query": req.query,
        "answer": answer,
        "sources": sources,
        "chunks_retrieved": len(chunks),
        "model": CHAT_MODEL,
    }
