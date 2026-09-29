from pathlib import Path

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

DATA_FILE = Path("data/tenant_policies.txt")

COLLECTION_NAME = "tenant_policies"

EMBEDDING_MODEL = "text-embedding-3-small"

openai_client = OpenAI()

qdrant_client = QdrantClient(
    url="http://localhost:6333",
)

def load_policies() -> list[dict]:
    content = DATA_FILE.read_text(
        encoding="utf-8",
    )

    blocks = [
        block.strip()
        for block in content.split("---")
        if block.strip()
    ]
    
    policies = []
    
    for block in blocks:
        lines = block.splitlines()
        organization_line = lines[0]
        organization_id = (
            organization_line
            .split("=", maxsplit=1)[1]
            .strip()
        )
        
        text = "\n".join(
            lines[1:]
        ).strip()
        
        policies.append(
            {
                "organization_id": organization_id,
                "text": text,
            }
        )
    return policies

def create_embeddings(
    texts: list[str],
) -> list[list[float]]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    return [
        item.embedding
        for item in response.data
    ]
    
def index_policies() -> None:
    policies = load_policies()
    texts = [
        policy["text"]
        for policy in policies
    ]
    
    embeddings = create_embeddings(
        texts
    )
    vector_size = len(
        embeddings[0]
    )
    
    if qdrant_client.collection_exists(
        COLLECTION_NAME
    ):
        qdrant_client.delete_collection(
            collection_name=COLLECTION_NAME,
        )
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )
    
    points = []
    
    for index, (
        policy,
        embedding,
    ) in enumerate(
        zip(policies, embeddings),
        start=1,
    ):
        points.append(
            PointStruct(
                id=index,
                vector=embedding,
                payload={
                    "organization_id": (
                        policy["organization_id"]
                    ),
                    "text": policy["text"],
                    "source": DATA_FILE.name,
                }
            )
        )
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )
    
    print(
        f"Indexed {len(points)} "
        "multi-tenant policy chunks."
    )
    
def unsafe_retrieve(
    question: str,
    top_k: int = 3,
):
    question_embedding = create_embeddings(
        [question]
    )[0]
    
    result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=question_embedding,
        limit=top_k
    )
    return result.points

def retrieve_for_organization(
    question: str,
    organization_id: str,
    top_k: int = 3,
):
    question_embedding = create_embeddings(
        [question]
    )[0]
    tenant_filter = Filter(
        must=[
            FieldCondition(
                key="organization_id",
                match=MatchValue(value=organization_id)
            )
        ]
    )
    result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=question_embedding,
        query_filter=tenant_filter,
        limit=top_k
    )
    
    return result.points

def print_results(
    title: str,
    points) -> None:
    print(f"\n=== {title} ===")
    for point in points:
        payload = point.payload or {}

        print()
        print("Organization:",payload.get("organization_id"),)

        print("Score:",f"{point.score:.4f}",)

        print(payload.get("text", ""))

def main() -> None:
    index_policies()
    
    question = (
        "What is the enterprise "
        "refund period?"
    )

    unsafe_results = unsafe_retrieve(
        question=question,
    )

    print_results(
        title="UNSAFE GLOBAL SEARCH",
        points=unsafe_results,
    )

    acme_results = retrieve_for_organization(
        question=question,
        organization_id="globex",
    )

    print_results(
        title="SAFE ACME SEARCH",
        points=acme_results,
    )


if __name__ == "__main__":
    main()