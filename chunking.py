from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from typing import List


def chunking(url: str) -> List:
    """Convert a document at `url` to a list of docling chunk objects.

    Returns the list of chunks so the function can be imported and used
    by other modules (for example, embedding upload code).
    """

    # Initialize the DocumentConverter
    converter = DocumentConverter()

    # Convert documents
    data = converter.convert(url).document

    # convert the docs to chunks
    chunker = HybridChunker()
    chunks = chunker.chunk(dl_doc=data)

    return chunks


if __name__ == "__main__":
    url = (
        "https://res.cloudinary.com/ddrsjswyc/raw/upload/"
        "v1765263412/pre-sales-estimator/sow/10272025%20SOW%20for%20Kore%20Phase%201%20by%20Simform"
    )

    chunks = chunking(url)

    for i, chunk in enumerate(chunks):
        print(f"=== {i} ===")
        print(f"chunk.text:\n{chunk.text[:300]}…")
        enriched_text = HybridChunker().contextualize(chunk=chunk)
        print(f"chunker.contextualize(chunk):\n{enriched_text[:300]}…")
        print()