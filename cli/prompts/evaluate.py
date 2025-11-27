import json
import os

from dotenv import load_dotenv
from google import genai


def ai_evaluate_results(query: str, formatted_results: list):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    debug = os.environ.get("DEBUG", "0") == "1"
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=f"""Rate how relevant each result is to this query on a 0-3 scale:

        Query: "{query}"

        Results:
        {formatted_results}

        Scale:
        - 3: Highly relevant
        - 2: Relevant
        - 1: Marginally relevant
        - 0: Not relevant

        Do NOT give any numbers out than 0, 1, 2, or 3.

        Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

        [2, 0, 3, 2, 0, 1]

        DO NOT include any text other than the list, including explanations, clarifications, or 'json'""",
    )

    # Parse the JSON response, and match up each score to the corresponding result.
    scores = []
    try:
        if response.text is not None:
            if debug:
                print("AI Evaluation Response:", response.text)
            scores = json.loads(response.text)
    except json.JSONDecodeError:
        print("Error decoding JSON response from AI:")

    return scores
