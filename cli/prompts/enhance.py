import os

from dotenv import load_dotenv
from google import genai


def get_response(method: str, query: str):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)

    match method:
        case "spell":
            response = client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=f"""Fix any spelling errors in this movie search query.

                Only correct obvious typos. Don't change correctly spelled words.

                Query: "{query}"

                If no errors, return the original query.
                Corrected:""",
            )
            return response.text
