"""
Test HuggingFace Whisper inference
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load env
load_dotenv()

# Set your HF token
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("Error: HF_TOKEN not set in environment")
    exit(1)

client = InferenceClient(
    api_key=HF_TOKEN,
)

# Test with a sample audio file
test_audio = "data/processed/KHAAB ｜ AKHIL ｜ PARMISH VERMA ｜  PUNJABI SONG 2025 ｜ CROWN RECORDS ｜ LATEST PUNJABI 2025 ｜.wav"

if not Path(test_audio).exists():
    print(f"File not found: {test_audio}")
    exit(1)

print(f"Testing HF Whisper on: {test_audio}")

# Use large model with fal-ai provider
model = "openai/whisper-large-v3"

client = InferenceClient(
    provider="fal-ai",
    api_key=HF_TOKEN,
)

print(f"Testing HF Whisper on: {Path(test_audio).name}")

try:
    with open(test_audio, "rb") as f:
        output = client.automatic_speech_recognition(
            f,
            model=model
        )
    print(f"Transcription: {output}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")