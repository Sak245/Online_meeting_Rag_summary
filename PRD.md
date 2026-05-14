# PRD — AI Video Meeting Assistant

## Project Name

AI Video Meeting Assistant

---

# 1. Problem Statement

In modern remote work environments, people attend numerous online meetings daily. These meetings are often long, repetitive, and information-dense.

Common problems include:

* Users forget important discussions
* Key decisions are lost
* Action items are missed
* Watching full recordings is time-consuming
* Missed meetings require reviewing hours of content
* Finding specific information inside recordings is difficult

The goal is to build an AI-powered assistant that transforms long meeting recordings into structured, searchable, and actionable knowledge.

---

# 2. Product Vision

Build a local-first AI system capable of:

* Processing meeting audio/video
* Generating transcripts
* Creating concise summaries
* Extracting decisions and action items
* Identifying open questions
* Enabling conversational Q&A over meetings using RAG

The system should significantly reduce the time required to review meetings and improve productivity.

---

# 3. Core Features

## 3.1 Input Sources

The system should support:

### Local File Upload

Supported formats:

* MP3
* MP4
* WAV
* M4A
* AAC

### YouTube URL Input

* Download audio from YouTube videos
* Convert downloaded video into audio pipeline

---

# 4. Functional Requirements

## 4.1 Audio Ingestion Pipeline

### Local File Flow

1. User uploads file
2. System validates format
3. System converts audio into standardized WAV format

### YouTube Flow

1. User provides YouTube URL
2. System downloads video/audio using yt-dlp
3. Extract audio
4. Convert into standardized WAV format

---

## 4.2 Audio Standardization

All audio must be converted into:

* WAV format
* Mono channel
* 16kHz sample rate

### Tools

* FFmpeg
* pydub

---

## 4.3 Language Detection

The system must determine:

* English
* Hindi
* Hinglish

This determines which transcription pipeline to use.

---

## 4.4 Audio Chunking

Large audio files must be split into chunks.

### English Pipeline

* 5–10 minute chunks

### Sarvam Pipeline

* ~25 second chunks due to free-tier limitation

---

# 5. Transcription System

## 5.1 English Transcription

### Model

OpenAI Whisper

### Requirements

* Process chunk-wise audio
* Combine transcripts sequentially

---

## 5.2 Hindi / Hinglish Transcription

### Model

Sarvam AI Speech Model

### Requirements

* Better Hindi/Hinglish handling
* Handle API constraints
* Chunk-wise transcription

---

## 5.3 Transcript Assembly

After all chunks are processed:

* Combine transcripts in sequence
* Add spacing/newlines between chunks
* Preserve readability

---

# 6. Summarization Pipeline

## 6.1 Chunk-Level Summaries

1. Split transcript into manageable chunks
2. Send chunk to LLM
3. Generate chunk summary

---

## 6.2 Final Summary Generation

1. Combine chunk summaries
2. Generate final consolidated summary

---

# 7. Meeting Intelligence Extraction

The system should extract:

* Key Decisions
* Action Items
* Open Questions

---

# 8. RAG-Based Q&A System

## 8.1 Text Splitting

Use:

* RecursiveCharacterTextSplitter

---

## 8.2 Embedding Generation

Generate embeddings using HuggingFace embedding models.

---

## 8.3 Vector Database

Use:

* ChromaDB (local)

---

## 8.4 Retrieval Pipeline

1. Receive user query
2. Retrieve relevant transcript chunks
3. Send context to LLM
4. Generate grounded response

---

## 8.5 Example Queries

* "What decisions were made?"
* "Who was assigned backend work?"
* "What was discussed about deployment?"
* "What are the pending tasks?"

---

# 9. Technical Stack

| Component             | Technology                     |
| --------------------- | ------------------------------ |
| Programming Language  | Python                         |
| Package Manager       | UV                             |
| Video Download        | yt-dlp                         |
| Audio Processing      | FFmpeg                         |
| Audio Manipulation    | pydub                          |
| English Transcription | Whisper                        |
| Hindi/Hinglish STT    | Sarvam AI                      |
| LLM Framework         | LangChain LCEL                 |
| LLM Provider          | Mistral AI                     |
| Embeddings            | HuggingFace                    |
| Vector Database       | ChromaDB                       |
| Text Splitting        | RecursiveCharacterTextSplitter |
| Q&A Pipeline          | LangChain RAG                  |

---

# 10. Folder Structure

project/
│
├── main.py
├── config.py
├── pyproject.toml
│
├── audio_processor.py
├── transcriber.py
├── summarizer.py
├── insights_extractor.py
├── vectorstore.py
├── rag_pipeline.py
├── youtube_handler.py
├── utils.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── chunks/
│   ├── transcripts/
│   └── vectordb/
│
├── outputs/
│   ├── summaries/
│   ├── decisions/
│   ├── qa_logs/
│   └── transcripts/
│
└── temp/

---

# 11. Future Enhancements

* Speaker diarization
* Timestamp support
* Streamlit frontend
* React frontend
* Real-time meeting transcription
* Zoom/Meet integrations
* Authentication system
* Cloud deployment

---

# 12. Risks and Constraints

## API Constraints

* Sarvam free-tier limitations
* LLM context window limits

## Hallucination Risk

Mitigation:

* Retrieval grounding
* transcript-first prompting

---

# 13. MVP Scope

Initial MVP should include:

* Local file upload
* YouTube URL ingestion
* WAV standardization
* Whisper transcription
* Sarvam transcription
* Transcript generation
* Summary generation
* Key decision extraction
* Action item extraction
* ChromaDB integration
* RAG Q&A

---

# 14. Conclusion

The AI Video Meeting Assistant aims to transform long-form meeting content into searchable and actionable intelligence.

By combining:

* transcription,
* summarization,
* structured extraction,
* and retrieval-augmented generation,

the system will help users save time, improve productivity, and reduce information loss from meetings.
