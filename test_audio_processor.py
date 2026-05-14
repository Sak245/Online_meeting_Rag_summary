"""
test_audio_processor.py

Run:
    python test_audio_processor.py
"""

from core.audio_processor import AudioProcessor


def main():

    processor = AudioProcessor()

    # =====================================================
    # TEST 1 — LOCAL FILE
    # =====================================================

    print("\n" + "=" * 60)
    print("TEST 1 — LOCAL FILE")
    print("=" * 60)

    local_path = "sample.mp3"
    # Replace with your local file

    try:

        wav_path = processor.process_input(
            local_path
        )

        print(f"\nProcessed WAV File:\n{wav_path}")

        chunks = processor.split_audio(
            wav_path,
            chunk_duration_minutes=1,
        )

        print("\nGenerated Chunks:")

        for chunk in chunks:
            print(chunk)

    except Exception as e:
        print(f"\nERROR:\n{e}")

    # =====================================================
    # TEST 2 — YOUTUBE URL
    # =====================================================

    print("\n" + "=" * 60)
    print("TEST 2 — YOUTUBE URL")
    print("=" * 60)

    youtube_url = (
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )

    try:

        wav_path = processor.process_input(
            youtube_url
        )

        print(f"\nProcessed WAV File:\n{wav_path}")

        chunks = processor.split_audio(
            wav_path,
            chunk_duration_minutes=1,
        )

        print("\nGenerated Chunks:")

        for chunk in chunks:
            print(chunk)

    except Exception as e:
        print(f"\nERROR:\n{e}")


if __name__ == "__main__":
    main()