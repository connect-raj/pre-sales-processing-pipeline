# python-service

This folder contains small utilities for document chunking and embedding storage used by the pre-sales estimator project. It includes:

- `chunking.py` — downloads/converts a document (via `docling`) and returns text chunks.
- `embeddings.py` — computes embeddings (via `sentence-transformers`) and persists them into a local Qdrant instance.
- `docker-compose.yml` — launches a local Qdrant container with a mounted `qdrant_storage/` directory.
- `requirements.txt` — pinned Python dependencies for the service.

This README explains how to set up a local environment, run Qdrant, chunk a document, and persist embeddings.

**Prerequisites**

- Python 3.10+ (3.12 is used in this workspace)
- Docker & Docker Compose (for Qdrant)
- Git (to clone the repo)

**Quick setup**

1. Create and activate a virtual environment:

```bash
cd python-service
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Qdrant (local vector DB):

```bash
docker compose up -d
# Qdrant HTTP: http://localhost:6333
```

If you prefer not to use Docker, you can run a managed Qdrant instance and set `QDRANT_HOST`/`QDRANT_PORT` when running `embeddings.py`.

**Chunking (convert a document to chunks)**

`chunking.py` exposes a `chunking(url: str) -> list` function and also works as a CLI script.

Example (run directly):

```bash
python chunking.py
# or
python -c "from chunking import chunking; print(len(chunking('<document_url>')) )"
```

**Embedding and storing in Qdrant**

`embeddings.py` will:
- import `chunking.chunking(url)` to get chunks
- compute embeddings using `sentence-transformers`
- create a Qdrant collection (if missing)
- upsert points with `id`, `vector`, and `payload` (payload includes original `text` and metadata)

Usage example:

```bash
python embeddings.py \
  "<DOCUMENT_URL>" \
  --collection sow_kore_phase1 \
  --model all-MiniLM-L6-v2
```

Options:
- `--collection`: Qdrant collection name (default: `sow_chunks`)
- `--model`: SentenceTransformer model name (default: `all-MiniLM-L6-v2`)
- `--qhost` / `--qport`: override Qdrant host and port (default `localhost:6333`)

Environment variables:
- `QDRANT_HOST` — host for Qdrant (default `localhost`)
- `QDRANT_PORT` — port for Qdrant (default `6333`)

**Notes on IDs and payloads**

- The embedding script generates valid Qdrant point IDs: it prefers an existing positive integer `chunk.id`, otherwise uses a positive integer starting at 1, and falls back to a UUID string when necessary. This avoids Qdrant rejecting `0` as an ID.
- Payloads include the raw chunk text and any available metadata fields (e.g., `meta`, `source`).

**Inspecting Qdrant**

- HTTP API: `http://localhost:6333/collections` to list collections.
- If you need to clear local storage, stop the container and remove `qdrant_storage/` (note: files may be owned by the container; use Docker to remove safely).

**Troubleshooting**

- Import errors (e.g. `ModuleNotFoundError: sentence_transformers`): ensure your venv is active and run `pip install -r requirements.txt`.
- Torch/torchvision mismatch: if you see errors like `operator torchvision::nms does not exist`, reinstall matching `torch` and `torchvision` wheels that match your Python/CUDA environment. Use the PyTorch website for the correct wheel selection.
- Permission errors when removing `qdrant_storage/`: stop the container, then remove files using Docker (or `sudo` on the host) to avoid permission issues.

**Development notes**

- `chunking.py` now returns chunks programmatically (useful for tests or downstream scripts).
- `embeddings.py` batches embeddings and upserts them to Qdrant. Consider adding retry/backoff when connecting to Qdrant in case the service is still starting.

**License & Contributing**

This directory inherits the repository license. Contributions welcome — open issues or PRs against the repo.

If you'd like, I can:
- run the end-to-end flow (start Qdrant, chunk the sample document, compute embeddings, and verify collections/points), or
- add a tiny test script that runs the pipeline with a mocked embedding model for CI.

----
Generated README for the `python-service` folder.