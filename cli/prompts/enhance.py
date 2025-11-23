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
        case "rewrite":
            response = client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=f"""Rewrite this movie search query to be more specific and searchable.

                Original: "{query}"

                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep it concise (under 10 words)
                - It should be a google style search query that's very specific
                - Don't use boolean logic

                Examples:

                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                Rewritten query:""",
            )
            return response.text
        case "expand":
            response = client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=f"""Expand this movie search query with related terms.

                Add synonyms and related concepts that might appear in movie descriptions.
                Keep expansions relevant and focused.
                This will be appended to the original query.

                Examples:

                - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                - "action movie with bear" -> "action thriller bear chase fight adventure"
                - "comedy with bear" -> "comedy funny bear humor lighthearted"

                Query: "{query}"
                """,
            )
            return response.text
