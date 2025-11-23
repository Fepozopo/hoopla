import os
from time import sleep

from dotenv import load_dotenv
from google import genai


def ai_rerank_method(method: str, query: str, results: list):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)

    match method:
        case "individual":
            for item in results:
                doc = item.get("doc") or {}
                title = doc.get("title", "")
                document = doc.get("document", "")
                response = client.models.generate_content(
                    model="gemini-2.0-flash-001",
                    contents=f"""Rate how well this movie matches the search query.

                    Query: "{query}"
                    Movie: {title} - {document}

                    Consider:
                    - Direct relevance to query
                    - User intent (what they're looking for)
                    - Content appropriateness

                    Rate 0-10 (10 = perfect match).
                    Give me ONLY the number in your response, no other text or explanation.

                    Score:""",
                )
                if response.text is not None:
                    # Assign the last score to the item's rerank_score
                    item["rerank_score"] = float(response.text.strip())

                sleep(3)  # To avoid rate limiting

            results.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            return results
