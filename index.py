from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import fileProcess
from chunking import chunking
from embeddings import embed_and_store

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
    return "Service is up and Running..........🚀"

@app.post('/api/store')
def store(request: fileProcess):
    chunks = chunking(request.fileURL)
    embeddings = embed_and_store(chunks=chunks)


    return embeddings