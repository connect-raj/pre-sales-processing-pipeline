from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from models import fileProcess
from pydantic import BaseModel
from chunking import chunking
from embeddings import embed_and_store, embed_query
import logging

logger = logging.getLogger("python-service")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Chunking Storage")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get('/api/health')
def health():
    return JSONResponse({"status": "ok", "message": "Service is up and Running"})

@app.post('/api/store')
def store(request: fileProcess):
    try:
        if not request.fileURL:
            return JSONResponse({"error": "fileURL is required"}, status_code=400)

        chunks = chunking(request.fileURL)
        embeddings = embed_and_store(chunks=chunks)
        return JSONResponse({"stored": embeddings})
    except Exception as e:
        logger.exception("/api/store failed")
        return JSONResponse({"error": str(e)}, status_code=500)

class EmbedRequest(BaseModel):
    # Accept a single text or a list of texts for batch embedding
    text: str | None = None
    texts: list[str] | None = None


@app.post('/api/embed')
def embed(request: EmbedRequest):
    try:
        # Validate input
        if not request.text and not request.texts:
            return JSONResponse({"error": "'text' or 'texts' is required"}, status_code=400)

        # Single text
        if request.text:
            vector = embed_query(request.text)
            return JSONResponse({"vector": vector})

        # Batch texts
        vectors = []
        for t in request.texts:
            vectors.append(embed_query(t))
        return JSONResponse({"vectors": vectors})
    except Exception as e:
        logger.exception("/api/embed failed")
        # Return a consistent JSON error payload
        return JSONResponse({"error": str(e)}, status_code=500)