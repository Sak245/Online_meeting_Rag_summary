"""
test_rag.py

Run:
    uv run python test_rag.py
"""

import os

from dotenv import load_dotenv

from core.vector_store import (
    MeetingVectorStore,
)

from core.rag_chain import (
    MeetingRAG,
)


load_dotenv()


def main():

    # =====================================================
    # API KEY
    # =====================================================

    mistral_api_key = os.getenv(
        "MISTRAL_API_KEY"
    )

    if not mistral_api_key:

        raise ValueError(
            "MISTRAL_API_KEY missing in .env"
        )

    # =====================================================
    # TRANSCRIPT PATH
    # =====================================================

    transcript_path = (
        "data/transcripts/"
        "test_transcript.txt"
    )

    # =====================================================
    # VECTOR STORE INGESTION
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 1 — VECTOR STORE INGESTION")
    print("=" * 60)

    vector_manager = (
        MeetingVectorStore()
    )

    vector_manager.ingest_transcript(
        transcript_path
    )

    # =====================================================
    # INITIALIZE RAG
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 2 — INITIALIZE RAG")
    print("=" * 60)

    rag = MeetingRAG(
        api_key=mistral_api_key
    )

    # =====================================================
    # TEST QUESTIONS
    # =====================================================

    test_questions = [

        "What is the main topic of the meeting?",

        "What decisions were made?",

        "Were there any action items?",

        "What song was mentioned?",

        "Was there any technical discussion?",
    ]

    # =====================================================
    # RUN QUESTIONS
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 3 — ASKING QUESTIONS")
    print("=" * 60)

    for question in test_questions:

        print("\n" + "-" * 60)

        print(
            f"\nQUESTION:\n{question}"
        )

        result = rag.ask(question)

        print(
            f"\nANSWER:\n"
        )

        print(result["answer"])

    # =====================================================
    # OPTIONAL INTERACTIVE CHAT
    # =====================================================

    start_chat = input(
        "\nStart interactive chat? (y/n): "
    )

    if start_chat.lower() == "y":

        rag.chat()


if __name__ == "__main__":
    main()