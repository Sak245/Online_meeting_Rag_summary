"""
test_transcriber.py

Run:
    uv run python test_transcriber.py
"""

from pathlib import Path
import os

from dotenv import load_dotenv

from core.audio_processor import AudioProcessor
from core.transcriber import Transcriber


load_dotenv()


def main():

    # =====================================================
    # INITIALIZE
    # =====================================================

    processor = AudioProcessor()

    transcriber = Transcriber(
        sarvam_api_key=os.getenv(
            "SARVAM_API_KEY"
        ),
        model_size="base",
    )

    # =====================================================
    # TEST INPUT
    # =====================================================

    input_source = (
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )

    # Example local file:
    # input_source = "sample.mp3"

    # =====================================================
    # AUDIO PROCESSING
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 1 — PROCESS INPUT")
    print("=" * 60)

    processed_wav = (
        processor.process_input(
            input_source
        )
    )

    print(
        f"\nProcessed WAV:\n{processed_wav}"
    )

    # =====================================================
    # LANGUAGE CONFIG
    # =====================================================

    language = "en"

    # For Hindi/Hinglish:
    # language = "hi"

    # =====================================================
    # CHUNKING STRATEGY
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 2 — CHUNKING")
    print("=" * 60)

    if language == "en":

        chunk_paths = (
            processor.split_audio(
                processed_wav,
                chunk_duration_minutes=1,
            )
        )

    else:

        chunk_paths = (
            processor.split_audio_seconds(
                processed_wav,
                chunk_duration_seconds=25,
            )
        )

    print("\nGenerated Chunks:")

    for chunk in chunk_paths:
        print(chunk)

    # =====================================================
    # TRANSCRIPTION
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 3 — TRANSCRIPTION")
    print("=" * 60)

    transcript_path = (
        transcriber.process_transcription(
            chunk_paths=chunk_paths,
            language=language,
            output_name="test_transcript.txt",
        )
    )

    # =====================================================
    # SHOW OUTPUT
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 4 — FINAL OUTPUT")
    print("=" * 60)

    print(
        f"\nTranscript Saved At:\n"
        f"{transcript_path}"
    )

    print("\nTranscript Preview:\n")

    transcript_text = Path(
        transcript_path
    ).read_text(
        encoding="utf-8"
    )

    print(
        transcript_text[:2000]
    )


if __name__ == "__main__":
    main()