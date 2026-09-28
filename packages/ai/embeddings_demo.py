from openai import OpenAI
import math

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
    '''
                         A · B
        cos(A, B) = ───────────────
                    ||A|| × ||B||
    '''
    
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
    
    return dot_product / (magnitude_a * magnitude_b)


refund_policy = (
    "Enterprise customers can request "
    "refunds within 45 days."
)

refund_question = (
    "How long can a corporate client wait "
    "before asking for their money back?"
)

password_policy = (
    "Employees must change their password "
    "every 90 days."
)

refund_policy_embedding = create_embedding(refund_policy)

refund_question_embedding = create_embedding(refund_question)

password_policy_embedding = create_embedding(password_policy)


refund_similarity = cosine_similarity(
    refund_question_embedding,
    refund_policy_embedding,
)

password_similarity = cosine_similarity(
    refund_question_embedding,
    password_policy_embedding,
)

print("Question:")
print(refund_question)

print("\nSimilarity with refund policy:")
print(refund_similarity)

print("\nSimilarity with password policy:")
print(password_similarity)