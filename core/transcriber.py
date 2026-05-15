"""
core/transcriber.py

Handles:
- Whisper transcription (English)
- Sarvam transcription (Hindi/Hinglish)
- Dynamic transcription routing
- Transcript merging
- Transcript saving
"""

from __future__ import annotations

import requests

from pathlib import Path
from typing import List

import whisper


class Transcriber:

    def __init__(
        self,
        sarvam_api_key: str | None = None,
        model_size: str = "base",
        transcripts_dir: str = "data/transcripts",
    ) -> None:

        self.sarvam_api_key = sarvam_api_key
        self.model_size = model_size

        self.transcripts_dir = Path(
            transcripts_dir
        )

        self.transcripts_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Lazy load Whisper model
        self.whisper_model = None

    def _ensure_model(self):
        """Lazy load Whisper model when needed."""
        if self.whisper_model is None:
            import whisper
            print(f"\nLoading Whisper Model: {self.model_size}")
            self.whisper_model = whisper.load_model(self.model_size)
            print("\nWhisper Model Loaded Successfully")

    # =====================================================
    # LANGUAGE ROUTER
    # =====================================================

    def should_use_sarvam(
        self,
        language: str,
    ) -> bool:
        """
        Decide whether Sarvam
        should be used.
        """

        supported_languages = [
            "hi",
            "hinglish",
            "hindi",
        ]

        return language.lower() in (
            supported_languages
        )

    # =====================================================
    # WHISPER TRANSCRIPTION
    # =====================================================

    def transcribe_with_whisper(
        self,
        chunk_path: str | Path,
        language: str = "en",
    ) -> str:

        chunk_path = Path(chunk_path)

        print(
            f"\nWhisper Transcribing:\n"
            f"{chunk_path.name}"
        )

        # Ensure model is loaded
        self._ensure_model()

        result = (
            self.whisper_model.transcribe(
                str(chunk_path),
                language=language,
                fp16=False,
            )
        )

        transcript = result["text"].strip()

        return transcript

    # =====================================================
    # SARVAM TRANSCRIPTION
    # =====================================================

    def transcribe_with_sarvam(
        self,
        chunk_path: str | Path,
    ) -> str:
        """
        Transcribe Hindi/Hinglish audio
        using Sarvam AI.
        """

        if not self.sarvam_api_key:
            raise ValueError(
                "SARVAM_API_KEY missing"
            )

        chunk_path = Path(chunk_path)

        print(
            f"\nSarvam Transcribing:\n"
            f"{chunk_path.name}"
        )

        url = (
            "https://api.sarvam.ai/"
            "speech-to-text"
        )

        headers = {
            "api-subscription-key":
            self.sarvam_api_key
        }

        # Determine content type
        ext = chunk_path.suffix.lower()
        content_types = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".mp4": "audio/mp4",
            ".m4a": "audio/mp4",
            ".aac": "audio/aac",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac",
            ".webm": "audio/webm",
        }
        content_type = content_types.get(ext, "audio/wav")

        with open(chunk_path, "rb") as audio:

            files = {
                "file": (chunk_path.name, audio, content_type)
            }

            data = {
                "model": "saarika:v2.5",
                "language_code": "hi-IN",
            }

            response = requests.post(
                url=url,
                headers=headers,
                files=files,
                data=data,
            )

        if response.status_code != 200:

            raise Exception(
                f"Sarvam API Error:\n"
                f"{response.text}"
            )

        result = response.json()

        transcript = (
            result
            .get("transcript", "")
            .strip()
        )

        return transcript

    # =====================================================
    # DYNAMIC TRANSCRIPTION
    # =====================================================

    def transcribe_chunk(
        self,
        chunk_path: str | Path,
        language: str = "en",
    ) -> str:
        """
        Dynamic transcription router.
        """

        if self.should_use_sarvam(
            language
        ):

            return self.transcribe_with_sarvam(
                chunk_path
            )

        return self.transcribe_with_whisper(
            chunk_path,
            language,
        )

    # =====================================================
    # MULTI-CHUNK TRANSCRIPTION
    # =====================================================

    def transcribe_chunks(
        self,
        chunk_paths: List[Path],
        language: str = "en",
    ) -> str:

        full_transcript = []

        total_chunks = len(chunk_paths)

        print(
            f"\nTotal Chunks: {total_chunks}"
        )

        for idx, chunk_path in enumerate(
            chunk_paths,
            start=1,
        ):

            print(
                f"\nProcessing "
                f"{idx}/{total_chunks}"
            )

            transcript = (
                self.transcribe_chunk(
                    chunk_path=chunk_path,
                    language=language,
                )
            )

            full_transcript.append(
                transcript
            )

        combined_transcript = (
            "\n\n".join(full_transcript)
        )

        return combined_transcript

    # =====================================================
    # SAVE TRANSCRIPT
    # =====================================================

    def save_transcript(
        self,
        transcript: str,
        output_name: str = "transcript.txt",
    ) -> Path:

        output_path = (
            self.transcripts_dir
            / output_name
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            f.write(transcript)

        print(
            f"\nTranscript Saved:\n"
            f"{output_path}"
        )

        return output_path

    # =====================================================
    # COMPLETE PIPELINE
    # =====================================================

    def process_transcription(
        self,
        chunk_paths: List[Path],
        language: str = "en",
        output_name: str = (
            "meeting_transcript.txt"
        ),
    ) -> Path:

        transcript = (
            self.transcribe_chunks(
                chunk_paths=chunk_paths,
                language=language,
            )
        )

        transcript_path = (
            self.save_transcript(
                transcript=transcript,
                output_name=output_name,
            )
        )

        return transcript_path