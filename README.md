# KnowledgePilot

KnowledgePilot is a learning project for building production-oriented
AI, RAG, and agentic AI systems from first principles.

## Learning Goals

- LLM fundamentals
- Prompting
- Embeddings
- RAG
- Document ingestion
- Chunking
- Vector databases
- Hybrid search
- Reranking
- LangChain
- LangGraph
- AI agents
- Agentic RAG
- Multi-agent systems
- Guardrails
- Evaluation
- Graph RAG
- Production AI engineering

## Current Stage

AI and LLM fundamentals.

```bash
    # Run qdrant
    docker run -d \
  --name knowledge-pilot-qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v knowledge-pilot-qdrant-data:/qdrant/storage \
  qdrant/qdrant
```
