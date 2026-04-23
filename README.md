# RAG App — Retrieval-Augmented Generation on Varity

A FastAPI application demonstrating AI document retrieval and question answering.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Liveness check |
| GET | `/ready` | Readiness (DB + models) |
| POST | `/upload` | Upload a text document for indexing |
| POST | `/query` | Ask a question against uploaded documents |

## How it works

1. **Upload**: Text files are chunked, embedded using a local embedding model, and stored in a vector database
2. **Query**: Your question is embedded, similar chunks are retrieved, and a language model generates an answer from the context

## Example

```bash
# Upload a document
curl -X POST https://your-app.varity.app/upload \
  -F "file=@document.txt"

# Ask a question
curl -X POST https://your-app.varity.app/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic of this document?"}'
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | (auto-injected) | PostgreSQL connection string |
| `OLLAMA_URL` | `http://ollama:11434` | Ollama service URL |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model name |
| `CHAT_MODEL` | `tinyllama` | Generation model name |
| `EMBED_DIM` | `768` | Embedding vector dimensions |
