from openai import OpenAI
from qdrant_client import QdrantClient

COLLECTION_NAME = "company_policies"

EMBEDDING_MODEL = "text-embedding-3-small"

TOP_K = 3

o_client = OpenAI()

q_client = QdrantClient(
    url="http://localhost:6333"
)

def create_embedding(
    text: str,
) -> list[float]:
    response = o_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return response.data[0].embedding

def retrieve(
    question: str,
    top_k: int = TOP_K,
):
    question_embedding = create_embedding(
        question
    )
    result = q_client.query_points(
        collection_name=COLLECTION_NAME,
        query=question_embedding,
        limit=top_k
    )
    
    return result.points

def build_context(points) -> str:
    context_parts = []
    
    for index, point in enumerate(
        points,
        start=1
    ):
        payload = point.payload or {}
        
        text = payload.get(
            "text",
            "",
        )
        
        source = payload.get(
            "source",
            "unknown",
        )
        
        chunk_index = payload.get(
            "chunk_index",
            "unknown",
        )
        
        context_parts.append(
            f"""
                Context {index}
                Source: {source}
                Chunk: {chunk_index}
                
                {text}
            """.strip())
    return "\n\n".join(context_parts)

def generate_answer(
    question: str,
    context: str,
) -> str:
    rag_input = f"""
        Use the retrieved company information below
        to answer the user's question.
        
        Retrieved context:
        {context}
        
        Question:
        
        {question}
    """
    
    res = o_client.responses.create(
        model="gpt-5.6-luna",
        instructions=(
            "You are KnowledgePilot, an enterprise "
            "knowledge assistant. "
            "Answer using only the retrieved context. "
            "If the retrieved context does not contain "
            "enough information, say that you do not "
            "have enough information. "
            "Do not invent company policies."
        ),
        input=rag_input
    )
    
    return res.output_text

def main()->None:
    question = (
        "What are the refund periods for "
        "enterprise, premium, and standard customers?"
    )
    points = retrieve(
        question=question,
    )
    
    print("=== QUESTION ===")
    print(question)

    print("\n=== RETRIEVAL RESULTS ===")
    
    for index, point in enumerate(
        points,
        start=1,
    ):
        payload = point.payload or {}
        print(
            f"\n#{index} "
            f"score={point.score:.4f}"
        )
        
        print(
            payload.get(
                "text",
                "",
            )
        )
        
    context = build_context(points)
    
    answer = generate_answer(
        question=question,
        context=context,
    )
    print("\n=== ANSWER ===")
    print(answer)

if __name__ == "__main__":
    main()