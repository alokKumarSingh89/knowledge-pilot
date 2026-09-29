import json
from pathlib import Path
from openai import OpenAI

POLICY_FILE = Path("data/company_policies.txt")
INDEX_FILE = Path("data/indexes/policy_index.json")

client = OpenAI()

def create_embedding(text: str) -> list[float]:
    res = client.embeddings.create(model="text-embedding-3-small", input=text)
    return res.data[0].embedding

def load_chunks() -> list[str]:
    content = POLICY_FILE.read_text(
        encoding="utf-8",
    )
    return [
        chunk.strip()
        for chunk in content.split("\n\n")
        if chunk.strip()
    ]

def build_index() -> list[dict]:
    chunks = load_chunks()

    index = []
    for position, chunk in enumerate(chunks):
        print(f"Embedding chunk {position + 1}/{len(chunks)}")

        embedding = create_embedding(chunk)

        index.append(
            {
                "id": position,
                "text": chunk,
                "embedding": embedding,
            }
        )

    return index

def save_index(index: list[dict]) -> None:
    INDEX_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    INDEX_FILE.write_text(
        json.dumps(index),
        encoding="utf-8",
    )

index = build_index()

save_index(index)

print()
print(f"Indexed {len(index)} chunks.")
print(f"Saved index to: {INDEX_FILE}")