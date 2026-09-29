import json
import math
from pathlib import Path

from openai import OpenAI


INDEX_FILE = Path("data/indexes/policy_index.json")

client = OpenAI()

def create_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )

    return response.data[0].embedding


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    return dot_product / (
        magnitude_a * magnitude_b
    )

def load_index() -> list[dict]:
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            "Policy index does not exist. "
            "Run packages.ai.index_policies first."
        )
    return json.loads(
        INDEX_FILE.read_text(encoding='utf-8')
    )
    
def search(
    question: str,
    index: list[dict],
    top_k: int = 1,
) -> list[dict]:
    question_embedding = create_embedding(
        question
    )

    results = []
    
    for item in index:
        similarity = cosine_similarity(
            question_embedding,
            item["embedding"],
        )
        results.append(
            {
                "id": item["id"],
                "text": item["text"],
                "similarity": similarity,
            }
        )
    results.sort(
        key=lambda result: result["similarity"],
        reverse=True,
    )
    return results[:top_k]

index = load_index()


question = (
    "How long does a corporate client have "
    "to ask for their money back?"
)

results = search(
    question=question,
    index=index,
    top_k=1,
)

retrieved_chunk = results[0]


print("=== QUESTION ===")
print(question)

print("\n=== RETRIEVED CONTEXT ===")
print(retrieved_chunk["text"])

print("\n=== SIMILARITY ===")
print(f"{retrieved_chunk['similarity']:.4f}")

rag_input = f"""
Use the following company policy to answer the question.

Company policy:
{retrieved_chunk["text"]}

Question:
{question}
"""

response = client.responses.create(
    model="gpt-5.6-luna",
    instructions=(
        "You are KnowledgePilot, an enterprise knowledge assistant. "
        "Answer using only the supplied company policy. "
        "If the policy does not contain enough information, "
        "say that you do not have enough information. "
        "Do not invent company policies."
    ),
    input=rag_input,
)


print("\n=== ANSWER ===")
print(response.output_text)