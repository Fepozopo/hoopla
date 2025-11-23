import json
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
        case "batch":
            response = client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=f"""Rank these movies by relevance to the search query.

                Query: "{query}"

                Movies:
                {results}

                Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

                [75, 12, 34, 2, 1]

                Do not include any text other than the list.
                """,
            )
            if response.text is None:
                print("Error: No response from AI model.")
                return None
            try:
                ranked_ids = json.loads(response.text)
                # Build a rank lookup (id -> rank)
                rank_map = {}
                for idx, rid in enumerate(ranked_ids, start=1):
                    if not isinstance(rid, int):
                        raise TypeError(
                            f"Expected integer IDs in the ranked list, got: {rid!r}"
                        )
                    # keep the first occurrence if duplicates exist
                    rank_map.setdefault(rid, idx)

                # One pass over results to assign rerank_rank where applicable
                for item in results:
                    doc = item.get("doc") or {}
                    doc_id = doc.get("id")
                    if doc_id in rank_map:
                        item["rerank_rank"] = rank_map[doc_id]
            except json.JSONDecodeError:
                print("Error: Failed to parse AI response as JSON.")
                print(f"Response was: {response.text}")
                return None

            results.sort(key=lambda x: x.get("rerank_rank", float("inf")))
            return results
