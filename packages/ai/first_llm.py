from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.6-luna",
    instructions=(
    "You are KnowledgePilot, an enterprise knowledge assistant. "
    "Answer questions using only information provided by the application. "
    "If the required information has not been provided, say that you "
    "do not have enough information. Do not invent company policies."
    ),
    input=("""
    Company policy:

    Standard customers: 15 days.
    Premium customers: 30 days.
    Enterprise customers: 45 days.

    Question:
    What is the refund period for enterprise customers?
    """)
)

print("=== ANSWER ===")
print(response.output_text)

print("\n=== RESPONSE INFO ===")
print(f"Response ID: {response.id}")
print(f"Model: {response.model}")
print(f"Status: {response.status}")

print("\n=== TOKEN USAGE ===")

if response.usage:
    print(f"Input tokens:  {response.usage.input_tokens}")
    print(f"Input tokens:  {response.usage.output_tokens}")
    print(f"Input tokens:  {response.usage.total_tokens}")