# AI Meeting Assistant

An end-to-end AI-powered meeting intelligence system built using Streamlit, OpenAI Whisper, LangChain, ChromaDB, and Mistral AI APIs.

This application can:

- Transcribe meetings from audio/video
- Process YouTube meeting recordings
- Generate intelligent summaries
- Extract key decisions and action items
- Build semantic memory using vector embeddings
- Perform conversational Q&A with RAG
- Export reports as TXT and PDF

---

# Features

## Audio / Video Ingestion

- Upload:
  - MP3
  - MP4
  - WAV
  - M4A
  - WEBM

- Or provide:
  - YouTube URL

---

# AI Processing Pipeline

```text
Input
↓
Audio Processing
↓
WAV Normalization
↓
Chunking
↓
Whisper Transcription
↓
Meeting Summarization
↓
Decision Extraction
↓
Action Item Extraction
↓
Open Questions Extraction
↓
Embedding Generation
↓
ChromaDB Vector Storage
↓
RAG-based Conversational Q&A
```

---

# Tech Stack

## Frontend

- Streamlit

## AI / LLM

- OpenAI Whisper
- Mistral AI
- LangChain

## Vector Database

- ChromaDB

## Embeddings

- HuggingFace Sentence Transformers
- all-MiniLM-L6-v2

## Audio Processing

- yt-dlp
- ffmpeg
- pydub

---

# Project Structure

```text
AI_Meeting_Assistant/
│
├── core/
│   ├── audio_processor.py
│   ├── transcriber.py
│   ├── summarizer.py
│   ├── vector_store.py
│   └── rag_chain.py
│
├── data/
│   ├── uploads/
│   ├── temp/
│   ├── processed/
│   ├── chunks/
│   ├── transcripts/
│   └── vectorstore/
│
├── outputs/
│   └── summaries/
│
├── exports/
│
├── main.py
├── requirements.txt
├── packages.txt
├── .env
└── README.md
```

---

# Installation

## 1. Clone Repository

```bash
git clone <your-repo-url>
cd AI_Meeting_Assistant
```

---

# 2. Create Virtual Environment

Using uv:

```bash
uv venv
```

Activate:

### Mac/Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\\Scripts\\activate
```

---

# 3. Install Dependencies

```bash
uv pip install -r requirements.txt
```

---

# 4. Install FFmpeg

## Mac

```bash
brew install ffmpeg
```

## Ubuntu

```bash
sudo apt update
sudo apt install ffmpeg
```

## Windows

Install FFmpeg manually and add it to PATH.

---

# 5. Environment Variables

Create `.env`

```env
MISTRAL_API_KEY=your_mistral_api_key
SARVAM_API_KEY=your_sarvam_api_key
```

---

# Run Application

```bash
streamlit run main.py
```

---

# Usage

## Option 1 — YouTube URL

1. Paste meeting YouTube URL
2. Click:
   - `Process Meeting`
3. Wait for:
   - download
   - transcription
   - summarization
   - vectorization

---

## Option 2 — Upload File

1. Upload:
   - MP3
   - MP4
   - WAV
2. Click:
   - `Process Meeting`

---

# Output Features

After processing you get:

## Transcript

Full meeting transcript.

---

## AI Summary

Concise intelligent meeting summary.

---

## Key Decisions

Automatically extracted decisions.

---

## Action Items

Tasks and assignments extracted from meeting.

---

## Open Questions

Pending or unresolved discussions.

---

## RAG Chat

Ask questions like:

```text
What decisions were made?
What deadlines were discussed?
Who was assigned the task?
What was the main topic?
```

---

## Export

Download:
- TXT report
- PDF report

---

# Example Questions

```text
What was discussed about deployment?
What were the blockers?
What action items were assigned?
Summarize the meeting in short.
```

---

# Current Limitations

- No speaker diarization
- No timestamps
- Local Whisper inference may be slow
- Large files require higher RAM
- Streamlit is single-process for heavy workloads

---

# Future Improvements

- Speaker diarization
- Live meeting support
- Real-time transcription
- Multi-user authentication
- Cloud deployment
- Hybrid search retrieval
- Better multilingual support
- Async background processing

---

# Deployment

Recommended platforms:

- Render
- Streamlit Community Cloud
- Railway

---

# Required Deployment Files

## packages.txt

```text
ffmpeg
```

## .streamlit/config.toml

```toml
[server]
maxUploadSize = 500
```

---

# Built With

- OpenAI Whisper
- LangChain
- ChromaDB
- Streamlit
- Mistral AI
- HuggingFace Transformers

---

# License

MIT License

---

# Author

Aryal Katkar