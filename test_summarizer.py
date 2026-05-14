"""
test_summarizer.py

Run:
    uv run python test_summarizer.py
"""

import os

from dotenv import load_dotenv

from core.summarizer import (
    MeetingSummarizer,
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
            "MISTRAL_API_KEY not found in .env"
        )

    # =====================================================
    # INITIALIZE SUMMARIZER
    # =====================================================

    summarizer = MeetingSummarizer(
        api_key=mistral_api_key,
        model="mistral-medium-latest",
    )

    # =====================================================
    # TRANSCRIPT PATH
    # =====================================================

    transcript_path = (
        "data/transcripts/"
        "test_transcript.txt"
    )

    # =====================================================
    # RUN PIPELINE
    # =====================================================

    print("\n" + "=" * 60)
    print("RUNNING MEETING SUMMARIZER")
    print("=" * 60)

    results = summarizer.process_meeting(
        transcript_path
    )

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(results["summary"])

    print("\n" + "=" * 60)
    print("KEY DECISIONS")
    print("=" * 60)

    print(results["decisions"])

    print("\n" + "=" * 60)
    print("ACTION ITEMS")
    print("=" * 60)

    print(results["actions"])

    print("\n" + "=" * 60)
    print("OPEN QUESTIONS")
    print("=" * 60)

    print(results["questions"])


if __name__ == "__main__":
    main()