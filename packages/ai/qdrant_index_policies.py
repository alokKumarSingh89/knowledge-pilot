from pathlib import Path
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams
)

POLICY_FILE = Path("data/company_policies.txt")

COLLECTION_NAME = "company_policies"

ORGANIZATION_ID = "knowledge-pilot-demo"

EMBEDDING_MODEL = "text-embedding-3-small"

o_client = OpenAI()

q_client = QdrantClient(
    url="http://localhost:6333"
)

def load_chunks() -> list[str]:
    content = POLICY_FILE.read_text(encoding="utf-8")
    return [
        chunk.strip()
        for chunk in content.split("\n\n")
        if chunk.strip()
    ]
    
def create_embeddings(texts: list[str]) -> list[list[float]]:
    res = o_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    
    return [
        item.embedding
        for item in res.data
    ]
    
def recreate_collection(
    vector_size: int,
) -> None:
    if q_client.collection_exists(
        COLLECTION_NAME
    ):
        q_client.delete_collection(collection_name=COLLECTION_NAME)
    
    q_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

def build_points(
    chunks: list[str],
    embeddings: list[list[float]]
    ) -> list[PointStruct]:
    points = []
    
    for index, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):
        points.append(
            PointStruct(
                id=index+1,
                vector=embedding,
                payload={
                    "text":chunk,
                    "source": POLICY_FILE.name,
                    "chunk_index": index,
                    "organization_id": ORGANIZATION_ID,
                }
            )
        )
    return points

def main() -> None:
    chunks = load_chunks()
    print(
        f"Loaded {len(chunks)} policy chunks."
    )
    
    embeddings = create_embeddings(chunks)
    print(
        f"Created {len(embeddings)} embeddings."
    )
    
    vector_size = len(embeddings[0])
    
    recreate_collection(
        vector_size=vector_size,
    )
    
    points = build_points(
        chunks=chunks,
        embeddings=embeddings,
    )
    
    q_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    
    print(
        f"Indexed {len(points)} points "
        f"into '{COLLECTION_NAME}'."
    )

if __name__ == "__main__":
    main()