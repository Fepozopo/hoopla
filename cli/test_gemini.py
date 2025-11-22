import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
print(f"Using key {api_key[:6]}...")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
)
print(response.text)

usage = response.usage_metadata
prompt_tokens = getattr(usage, "prompt_token_count", None)
response_tokens = getattr(usage, "candidates_token_count", None)
if prompt_tokens is not None and response_tokens is not None:
    print(f"Prompt Tokens: {prompt_tokens}")
    print(f"Response Tokens: {response_tokens}")
else:
    print("Prompt Tokens: N/A")
    print("Response Tokens: N/A")
