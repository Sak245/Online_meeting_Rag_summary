"""
Test HuggingFace Whisper via direct API - user's approach
"""
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://router.huggingface.co/fal-ai/whisper"
headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
}

test_audio = "data/processed/KHAAB ｜ AKHIL ｜ PARMISH VERMA ｜  PUNJABI SONG 2025 ｜ CROWN RECORDS ｜ LATEST PUNJABI 2025 ｜.wav"

print(f"Testing HF Whisper on: {Path(test_audio).name}")

try:
    with open(test_audio, "rb") as f:
        data = f.read()

    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "audio/wav"},
        data=data,
        timeout=300
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Transcription: {result}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")