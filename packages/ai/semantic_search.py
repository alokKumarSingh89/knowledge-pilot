from pathlib import Path
import math
from openai import OpenAI

POLICY_FILE = Path("data/company_policies.txt")

client = OpenAI()

def create_embedding(text:str)->list[float]:
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    
    mag_a = math.sqrt(
        sum(a * a for a in vec_a)
    )
    mag_b = math.sqrt(
            sum(b * b for b in vec_b)
    )
    
    return dot_product/(mag_a * mag_b)


content = POLICY_FILE.read_text(encoding="utf-8")

chunks = [
    chunk.strip()
    for chunk in content.split("\n\n")
    if chunk.strip()
]

chunk_embeddings = []

for chunk in chunks:
    embedding = create_embedding(chunk)
    
    chunk_embeddings.append(
        {
            "text": chunk,
            "embedding": embedding,
        }
    )

query = (
    "How long does a corporate client have "
    "to ask for their money back?"
)

query_embedding = create_embedding(query)

results = []

for item in chunk_embeddings:
    similarity = cosine_similarity(
        query_embedding,
        item["embedding"],
    )
    
    results.append(
        {
            "text": item["text"],
            "similarity": similarity,
        }
    )

results.sort(key=lambda result: result["similarity"], reverse=True)

print("Query:")
print(query)

print("\n=== SEARCH RESULTS ===")

for index, result in enumerate(results, start=1):
    print(f"\n#{index}")
    print(f"Similarity: {result['similarity']:.4f}")
    print(result["text"])