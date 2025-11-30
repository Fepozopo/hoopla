import os

from dotenv import load_dotenv
from google import genai


def ai_augment_rag(query: str, docs: list):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

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


def ai_augment_summarize(query: str, results: list):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=f"""
        Provide information useful to this query by synthesizing information from multiple search results in detail.
        The goal is to provide comprehensive information so that users know what their options are.
        Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
        This should be tailored to Hoopla users. Hoopla is a movie streaming service.
        Query: {query}
        Search Results:
        {results}
        Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
        """,
    )
    return response.text


def ai_augment_citations(query: str, documents: list):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=f"""Answer the question or provide information based on the provided documents.

        This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

        Query: {query}

        Documents:
        {documents}

        Instructions:
        - Provide a comprehensive answer that addresses the query
        - Cite sources using [1], [2], etc. format when referencing information
        - If sources disagree, mention the different viewpoints
        - If the answer isn't in the documents, say "I don't have enough information"
        - Be direct and informative

        Answer:""",
    )
    return response.text


def ai_augment_question(question: str, documents: list):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=f"""Answer the user's question based on the provided movies that are available on Hoopla.

        This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        Question: {question}

        Documents:
        {documents}

        Instructions:
        - Answer questions directly and concisely
        - Be casual and conversational
        - Don't be cringe or hype-y
        - Talk like a normal person would in a chat conversation

        Answer:""",
    )
    return response.text
