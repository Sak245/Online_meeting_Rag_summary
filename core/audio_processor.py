"""
core/audio_processor.py

Handles:
- Local audio/video processing
- YouTube URL downloading
- WAV conversion
- Mono conversion
- 16kHz resampling
- Audio chunking
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List
from urllib.parse import urlparse

from pydub import AudioSegment
import yt_dlp


class AudioProcessor:

    def __init__(
        self,
        temp_dir: str = "data/temp",
        processed_dir: str = "data/processed",
        chunks_dir: str = "data/chunks",
    ) -> None:

        self.temp_dir = Path(temp_dir)
        self.processed_dir = Path(processed_dir)
        self.chunks_dir = Path(chunks_dir)

        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # INPUT DETECTION
    # =========================================================

    def is_url(self, input_source: str) -> bool:
        """
        Detect whether input is URL or local path.
        """

        parsed = urlparse(input_source)

        return parsed.scheme in ("http", "https")

    # =========================================================
    # MAIN ENTRY PIPELINE
    # =========================================================

    def process_input(self, input_source: str) -> Path:
        """
        Dynamically process:
        - YouTube URL
        - Local audio/video file

        Returns:
            Standardized WAV file path.
        """

        if self.is_url(input_source):

            print("\nDetected YouTube URL")

            downloaded_file = self.download_youtube_audio(
                input_source
            )

            wav_path = self.convert_to_wav(
                downloaded_file
            )

            return wav_path

        else:

            print("\nDetected Local File")

            local_file = Path(input_source)

            if not local_file.exists():
                raise FileNotFoundError(
                    f"File not found: {local_file}"
                )

            wav_path = self.convert_to_wav(
                local_file
            )

            return wav_path

    # =========================================================
    # YOUTUBE DOWNLOAD
    # =========================================================

    def download_youtube_audio(self, url: str) -> Path:
        """
        Download audio from YouTube.
        """

        output_template = str(
            self.temp_dir / "%(title)s.%(ext)s"
        )

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "quiet": False,
            "noplaylist": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:

            info = ydl.extract_info(
                url,
                download=True,
            )

            downloaded_file = Path(
                ydl.prepare_filename(info)
            )

        return downloaded_file

    # =========================================================
    # AUDIO STANDARDIZATION
    # =========================================================

    def convert_to_wav(
        self,
        input_path: str | Path,
        output_name: str | None = None,
    ) -> Path:
        """
        Convert into:
        - WAV
        - Mono
        - 16kHz
        """

        input_path = Path(input_path)

        if output_name is None:
            output_name = f"{input_path.stem}.wav"

        output_path = (
            self.processed_dir / output_name
        )

        audio = AudioSegment.from_file(
            input_path
        )

        # Mono
        audio = audio.set_channels(1)

        # 16kHz
        audio = audio.set_frame_rate(16000)

        audio.export(
            output_path,
            format="wav",
        )

        return output_path

    # =========================================================
    # LONG CHUNKING
    # =========================================================

    def split_audio(
        self,
        audio_path: str | Path,
        chunk_duration_minutes: int = 10,
    ) -> List[Path]:

        audio_path = Path(audio_path)

        audio = AudioSegment.from_wav(
            audio_path
        )

        chunk_length_ms = (
            chunk_duration_minutes
            * 60
            * 1000
        )

        total_length_ms = len(audio)

        total_chunks = math.ceil(
            total_length_ms / chunk_length_ms
        )

        chunk_paths = []

        for i in range(total_chunks):

            start_ms = i * chunk_length_ms
            end_ms = start_ms + chunk_length_ms

            chunk = audio[start_ms:end_ms]

            chunk_filename = (
                f"{audio_path.stem}_chunk_{i + 1}.wav"
            )

            chunk_path = (
                self.chunks_dir
                / chunk_filename
            )

            chunk.export(
                chunk_path,
                format="wav",
            )

            chunk_paths.append(chunk_path)

        return chunk_paths

    # =========================================================
    # SHORT CHUNKING (SARVAM)
    # =========================================================

    def split_audio_seconds(
        self,
        audio_path: str | Path,
        chunk_duration_seconds: int = 25,
    ) -> List[Path]:

        audio_path = Path(audio_path)

        audio = AudioSegment.from_wav(
            audio_path
        )

        chunk_length_ms = (
            chunk_duration_seconds * 1000
        )

        total_length_ms = len(audio)

        total_chunks = math.ceil(
            total_length_ms / chunk_length_ms
        )

        chunk_paths = []

        for i in range(total_chunks):

            start_ms = i * chunk_length_ms
            end_ms = start_ms + chunk_length_ms

            chunk = audio[start_ms:end_ms]

            chunk_filename = (
                f"{audio_path.stem}_short_chunk_{i + 1}.wav"
            )

            chunk_path = (
                self.chunks_dir
                / chunk_filename
            )

            chunk.export(
                chunk_path,
                format="wav",
            )

            chunk_paths.append(chunk_path)

        return chunk_paths