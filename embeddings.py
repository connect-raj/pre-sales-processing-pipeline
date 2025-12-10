import os
import math
import uuid
from typing import List, Optional

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# Import chunking function from local module
from chunking import chunking
from docling.chunking import HybridChunker


def embed_and_store(
    chunks: List,
    collection_name: str = "doc_chunks",
    model_name: str = "all-MiniLM-L6-v2",
    qdrant_host: str | None = "localhost",
    qdrant_port: int | None = 6333,
    batch_size: int = 64,
):
    """Compute embeddings for `chunks` and store them in a Qdrant collection.

    Each point payload will include the chunk text and basic metadata when available.
    """

    # Resolve Qdrant location
    host = qdrant_host or os.environ.get("QDRANT_HOST", "localhost")
    port = qdrant_port or int(os.environ.get("QDRANT_PORT", 6333))
    url = f"http://{host}:{port}"

    print(f"Connecting to Qdrant at {url}...")
    client = QdrantClient(url=url)

    # Load embedding model
    if model_name.startswith("all-") or model_name.startswith("sentence-transformers"):
        model_id = (
            f"sentence-transformers/{model_name}"
            if not model_name.startswith("sentence-transformers/")
            and not model_name.startswith("all-MiniLM")
            and not model_name.startswith("all-")
            else model_name
        )
    else:
        model_id = model_name

    print(f"Loading embedding model '{model_id}'...")
    model = SentenceTransformer(model_id)

    vector_size = model.get_sentence_embedding_dimension()

    # Create collection if it doesn't exist
    try:
        client.get_collection(collection_name)
        print(f"Collection '{collection_name}' exists.")
    except Exception:
        print(f"Creating collection '{collection_name}' with vector size {vector_size}...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    # Prepare texts to embed
    chunker = HybridChunker()

    texts = []
    payloads = []
    ids = []

    # Prepare texts, payloads and valid ids in a single pass
    for i, chunk in enumerate(chunks):
        try:
            enriched = chunker.contextualize(chunk=chunk)
        except Exception:
            enriched = None

        text_for_embedding = enriched or getattr(chunk, "text", "")
        texts.append(text_for_embedding)

        payload = {"text": getattr(chunk, "text", "")}
        if hasattr(chunk, "meta"):
            payload["meta"] = chunk.meta
        if hasattr(chunk, "source"):
            payload["source"] = getattr(chunk, "source")

        payloads.append(payload)

        # Build a valid Qdrant point id for this chunk now (avoid zeros)
        raw_id = getattr(chunk, "id", None)
        if raw_id is None:
            pid = i + 1
        else:
            try:
                pid_int = int(raw_id)
                pid = pid_int if pid_int > 0 else i + 1
            except Exception:
                raw_str = str(raw_id)
                pid = raw_str if raw_str.strip() != "" else str(uuid.uuid4())

        ids.append(pid)

    # Compute embeddings in batches using the prepared lists
    total = len(texts)
    print(f"Embedding {total} chunks with model (batch_size={batch_size})...")

    for start in range(0, total, batch_size):
        end = start + batch_size
        batch_texts = texts[start:end]
        batch_ids = ids[start:end]
        batch_payloads = payloads[start:end]

        vectors = model.encode(batch_texts, show_progress_bar=False)

        points = []
        for pid, vec, payload in zip(batch_ids, vectors, batch_payloads):
            vector_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)
            points.append({
                "id": pid,
                "vector": vector_list,
                "payload": payload,
            })

        print(f"Upserting points {start}:{min(end, total)}...")
        client.upsert(collection_name=collection_name, points=points)

    return f"{total} chunks stored successfuly"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk a document, embed chunks and store to Qdrant")
    parser.add_argument("url", help="Document URL to chunk and embed")
    parser.add_argument("--collection", default="doc_chunks", help="Qdrant collection name")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--qhost", help="Qdrant host (default: localhost)")
    parser.add_argument("--qport", type=int, help="Qdrant port (default: 6333)")
    args = parser.parse_args()

    print("Chunking document...")
    chunks = chunking(args.url)
    embed_and_store(
        chunks,
        collection_name=args.collection,
        model_name=args.model,
        qdrant_host=args.qhost,
        qdrant_port=args.qport,
    )

def embed_query(query: str, model_name: str = "all-MiniLM-L6-v2") -> List[float]:
    """Generate embedding for a single query string."""
    from sentence_transformers import SentenceTransformer
    
    if model_name.startswith("all-") or model_name.startswith("sentence-transformers"):
        model_id = (
            f"sentence-transformers/{model_name}"
            if not model_name.startswith("sentence-transformers/")
            and not model_name.startswith("all-MiniLM")
            and not model_name.startswith("all-")
            else model_name
        )
    else:
        model_id = model_name

    model = SentenceTransformer(model_id)
    vector = model.encode(query, show_progress_bar=False)
    
    return vector.tolist() if hasattr(vector, "tolist") else list(vector)
