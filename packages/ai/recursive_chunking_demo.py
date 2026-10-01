from pathlib import Path


DOCUMENT_FILE = Path(
    "data/employee_handbook.txt"
)

CHUNK_SIZE = 500

SEPARATORS = [
    "\n\n",
    ". ",
    " ",
]


content = DOCUMENT_FILE.read_text(
    encoding="utf-8",
)

def split_text(
    text: str,
    separator: str,
) -> list[str]:
    return [
        part.strip()
        for part in text.split(separator)
        if part.strip()
    ]
    
def merge_pieces(
    pieces: list[str],
    separator: str,
    chunk_size: int,
) -> list[str]:
    chunks = []
    current = ""

    for piece in pieces:
        candidate = (
            f"{current}{separator}{piece}"
            if current
            else piece
        )
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        
        current = piece
    if current:
        chunks.append(current)
    
    return chunks
    
def recursive_split(
    text: str,
    chunk_size: int,
    separators: list[str],
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    
    if len(text) <= chunk_size:
        return [text]
    
    if not separators:
        return [
            text[start:start + chunk_size]
            for start in range(
                0,
                len(text),
                chunk_size,
            )
        ]
    separator = separators[0]
    pieces = split_text(
        text=text,
        separator=separator,
    )
    if len(pieces) == 1:
        return recursive_split(
            text=text,
            chunk_size=chunk_size,
            separators=separators[1:],
        )
    chunks = []

    merged_pieces = merge_pieces(
        pieces=pieces,
        separator=separator,
        chunk_size=chunk_size,
    )
    for piece in merged_pieces:
        if len(piece) <= chunk_size:
            chunks.append(piece)
        else:
            chunks.extend(
                recursive_split(
                    text=piece,
                    chunk_size=chunk_size,
                    separators=separators[1:],
                )
            )
    return chunks

chunks = recursive_split(
    text=content,
    chunk_size=CHUNK_SIZE,
    separators=SEPARATORS,
)

print(
    f"Document characters: {len(content)}"
)

print(
    f"Generated chunks: {len(chunks)}"
)


for index, chunk in enumerate(
    chunks,
    start=1,
):
    print(
        f"\n=== CHUNK {index} ==="
    )

    print(
        f"Characters: {len(chunk)}"
    )

    print(chunk)