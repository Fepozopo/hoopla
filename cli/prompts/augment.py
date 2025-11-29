import os

from dotenv import load_dotenv
from google import genai


def ai_augment(cmd: str, query: str, docs: list):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    match cmd:
        case "rag":
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                Query: {query}

                Documents:
                {docs}

                Provide a comprehensive answer that addresses the query:""",
            )
            return response.text
