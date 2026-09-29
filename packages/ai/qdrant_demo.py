from qdrant_client import QdrantClient
from openai import OpenAI
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams
)


COLLECTION_NAME = "company_policies"

o_client = OpenAI()

q_client = QdrantClient(
    url="http://localhost:6333"
)

text = (
    "Enterprise customers can request "
    "a refund within 45 days."
)

response = o_client.embeddings.create(
    model="text-embedding-3-small",
    input=text
)

embedding = response.data[0].embedding

if not q_client.collection_exists(
    COLLECTION_NAME
):
    q_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=len(embedding),
            distance=Distance.COSINE
        )
    )

q_client.upsert(
    collection_name=COLLECTION_NAME,
    points=[
        PointStruct(
            id=1,
            vector=embedding,
            payload={
                "text":text,
                "source": "company_policies.txt",
                "policy_type": "refund",
            }
        )
    ]
)

question = (
    "How long can a corporate client wait "
    "before asking for their money back?"
)
question_response = o_client.embeddings.create(
    model="text-embedding-3-small",
    input=question,
)

question_embedding = (
    question_response.data[0].embedding
)

search_result = q_client.query_points(
    collection_name=COLLECTION_NAME,
    query=question_embedding,
    limit=1
)

print("=== QUESTION ===")
print(question)

print("\n=== SEARCH RESULT ===")

for point in search_result.points:
    print(f"Score: {point.score:.4f}")
    print(f"Text: {point.payload['text']}")
    print(f"Source: {point.payload['source']}")