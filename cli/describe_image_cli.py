#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure the project root (one level above the `cli` package) is on sys.path
# so `from cli.helpers import ...` works when running the script directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import mimetypes
import os

from dotenv import load_dotenv
from google import genai


def main():
    parser = argparse.ArgumentParser(description="Multi-modal query rewriting CLI")
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the image file"
    )
    parser.add_argument(
        "--query", type=str, required=True, help="Text query to combine with image"
    )

    args = parser.parse_args()
    image = args.image
    query = args.query

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(image, "rb") as f:
        image_bytes = f.read()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)
    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary"""
    parts = [
        system_prompt,
        genai.types.Part.from_bytes(data=image_bytes, mime_type=mime),
        query.strip(),
    ]

    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=parts,
    )

    if response.text is not None:
        print(f"Rewritten query: {response.text.strip()}")
        if response.usage_metadata is not None:
            print(f"Total tokens:    {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()
