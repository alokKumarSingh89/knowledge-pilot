from pathlib import Path


DOCUMENT_FILE = Path(
    "data/employee_handbook.txt"
)


content = DOCUMENT_FILE.read_text(
    encoding="utf-8",
)

def chunk_words(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative"
        )
    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for start in range(0,len(words),step):
        end = start + chunk_size

        chunk = words[start:end]

        if not chunk:
            break
        chunks.append(
            " ".join(chunk)
        )
        if end >= len(words):
            break
    return chunks
    

chunks = chunk_words(
    text=content,
    chunk_size=60,
    overlap=10,
)


print(f"Document words: {len(content.split())}")

print(f"Generated chunks: {len(chunks)}")


for index, chunk in enumerate(
    chunks,
    start=1,
):
    print(f"\n=== CHUNK {index} ===")

    print(f"Words: {len(chunk.split())}")

    print(chunk)