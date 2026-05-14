"""
main.py

AI Meeting Assistant

Features:
- YouTube URL ingestion
- Local file upload
- Whisper transcription
- Sarvam-ready routing
- Meeting summarization
- Key decisions extraction
- Action items extraction
- Open questions extraction
- RAG-based Q&A
- TXT export
- PDF export
"""
from __future__ import annotations
import warnings
import logging
import transformers

# Suppress warnings
warnings.filterwarnings("ignore")

# Reduce transformers logging
transformers.logging.set_verbosity_error()

# Reduce sentence-transformers logging
logging.getLogger(
    "sentence_transformers"
).setLevel(logging.ERROR)

logging.getLogger(
    "transformers"
).setLevel(logging.ERROR)


import os
from pathlib import Path

import streamlit as st

from dotenv import load_dotenv

from fpdf import FPDF

from core.audio_processor import AudioProcessor
from core.transcriber import Transcriber
from core.summarizer import MeetingSummarizer
from core.vector_store import MeetingVectorStore
from core.rag_chain import MeetingRAG



# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Meeting Assistant",
    page_icon="🎙️",
    layout="wide",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

.main {
    padding-top: 1rem;
}

.stTextArea textarea {
    min-height: 250px;
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# DIRECTORIES
# =========================================================

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# SESSION STATE
# =========================================================

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "decisions" not in st.session_state:
    st.session_state.decisions = ""

if "actions" not in st.session_state:
    st.session_state.actions = ""

if "questions" not in st.session_state:
    st.session_state.questions = ""

if "rag" not in st.session_state:
    st.session_state.rag = None


# =========================================================
# SAFE PDF EXPORT
# =========================================================

def create_pdf(
    transcript: str,
    summary: str,
    decisions: str,
    actions: str,
    questions: str,
) -> str:

    def clean_text(text: str) -> str:

        text = text.replace("\t", " ")
        text = text.replace("\r", " ")

        # normalize spaces
        text = " ".join(text.split())

        # safe encoding
        text = (
            text.encode(
                "latin-1",
                "replace",
            )
            .decode("latin-1")
        )

        return text

    pdf = FPDF()

    pdf.set_auto_page_break(
        auto=True,
        margin=15,
    )

    pdf.add_page()

    # =====================================================
    # TITLE
    # =====================================================

    pdf.set_font(
        "Arial",
        "B",
        18,
    )

    pdf.cell(
        0,
        10,
        "AI Meeting Assistant Report",
        ln=True,
        align="C",
    )

    sections = [
        ("Final Summary", summary),
        ("Key Decisions", decisions),
        ("Action Items", actions),
        ("Open Questions", questions),
        ("Transcript", transcript),
    ]

    # =====================================================
    # CONTENT
    # =====================================================

    for title, content in sections:

        pdf.ln(8)

        # section title
        pdf.set_font(
            "Arial",
            "B",
            14,
        )

        pdf.multi_cell(
            0,
            10,
            clean_text(title),
        )

        pdf.ln(2)

        # section body
        pdf.set_font(
            "Arial",
            size=11,
        )

        cleaned_content = clean_text(
            content
        )

        paragraphs = cleaned_content.split(
            ". "
        )

        for para in paragraphs:

            para = para.strip()

            if not para:
                continue

            try:

                pdf.multi_cell(
                    0,
                    8,
                    para,
                )

                pdf.ln(1)

            except Exception:
                continue

    output_path = (
        EXPORT_DIR
        / "meeting_report.pdf"
    )

    pdf.output(str(output_path))

    return str(output_path)


# =========================================================
# TITLE
# =========================================================

st.title("🎙️ AI Meeting Assistant")

st.markdown(
    """
Upload a meeting recording or provide a YouTube URL.

The system will:
- Transcribe audio
- Generate summaries
- Extract decisions/actions/questions
- Create searchable RAG memory
- Allow conversational Q&A
"""
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    language = st.selectbox(
        "Language",
        ["en", "hi"],
    )

    whisper_model = st.selectbox(
        "Whisper Model",
        [
            "base",
            "small",
            "medium",
        ],
        index=1,
    )

    st.markdown("---")

    st.caption(
        "Whisper + Mistral + ChromaDB + Streamlit"
    )


# =========================================================
# INPUT
# =========================================================

st.header("📥 Input Source")

input_type = st.radio(
    "Choose Input Type",
    [
        "YouTube URL",
        "Upload File",
    ],
)

input_source = None


# =========================================================
# YOUTUBE
# =========================================================

if input_type == "YouTube URL":

    youtube_url = st.text_input(
        "Enter YouTube URL"
    )

    if youtube_url:
        input_source = youtube_url


# =========================================================
# FILE UPLOAD
# =========================================================

else:

    uploaded_file = st.file_uploader(
        "Upload Audio/Video File",
        type=[
            "mp3",
            "mp4",
            "wav",
            "m4a",
            "webm",
        ],
    )

    if uploaded_file:

        save_path = (
            UPLOAD_DIR
            / uploaded_file.name
        )

        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())

        input_source = str(save_path)


# =========================================================
# PROCESS BUTTON
# =========================================================

if st.button(
    "🚀 Process Meeting",
    use_container_width=True,
):

    if not input_source:

        st.error(
            "Please provide input source"
        )

    else:

        try:

            with st.spinner(
                "Initializing AI Pipeline..."
            ):

                # =====================================
                # INITIALIZE COMPONENTS
                # =====================================

                processor = AudioProcessor()

                transcriber = Transcriber(
                    sarvam_api_key=os.getenv(
                        "SARVAM_API_KEY"
                    ),
                    model_size=whisper_model,
                )

                summarizer = (
                    MeetingSummarizer(
                        api_key=os.getenv(
                            "MISTRAL_API_KEY"
                        )
                    )
                )

                vector_store = (
                    MeetingVectorStore()
                )

                # =====================================
                # AUDIO PROCESSING
                # =====================================

                st.info(
                    "Processing Audio..."
                )

                processed_wav = (
                    processor.process_input(
                        input_source
                    )
                )

                # =====================================
                # CHUNKING
                # =====================================

                st.info(
                    "Creating Audio Chunks..."
                )

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

                # =====================================
                # TRANSCRIPTION
                # =====================================

                st.info(
                    "Transcribing Meeting..."
                )

                transcript_path = (
                    transcriber.process_transcription(
                        chunk_paths=chunk_paths,
                        language=language,
                        output_name="meeting_transcript.txt",
                    )
                )

                transcript = Path(
                    transcript_path
                ).read_text(
                    encoding="utf-8"
                )

                st.session_state.transcript = transcript

                # =====================================
                # SUMMARIZATION
                # =====================================

                st.info(
                    "Generating Meeting Intelligence..."
                )

                results = (
                    summarizer.process_meeting(
                        transcript_path
                    )
                )

                st.session_state.summary = (
                    results["summary"]
                )

                st.session_state.decisions = (
                    results["decisions"]
                )

                st.session_state.actions = (
                    results["actions"]
                )

                st.session_state.questions = (
                    results["questions"]
                )

                # =====================================
                # VECTOR STORE
                # =====================================

                st.info(
                    "Creating Vector Database..."
                )

                vector_store.ingest_transcript(
                    transcript_path
                )

                # =====================================
                # RAG SYSTEM
                # =====================================

                st.session_state.rag = (
                    MeetingRAG(
                        api_key=os.getenv(
                            "MISTRAL_API_KEY"
                        )
                    )
                )

            st.success(
                "Meeting Processed Successfully"
            )

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )


# =========================================================
# TRANSCRIPT
# =========================================================

if st.session_state.transcript:

    st.header("📝 Transcript")

    st.text_area(
        "Transcript",
        st.session_state.transcript,
        height=300,
    )


# =========================================================
# SUMMARY
# =========================================================

if st.session_state.summary:

    st.header("📌 Final Summary")

    st.markdown(
        st.session_state.summary
    )


# =========================================================
# DECISIONS + ACTIONS
# =========================================================

col1, col2 = st.columns(2)

with col1:

    if st.session_state.decisions:

        st.subheader(
            "✅ Key Decisions"
        )

        st.markdown(
            st.session_state.decisions
        )

with col2:

    if st.session_state.actions:

        st.subheader(
            "📋 Action Items"
        )

        st.markdown(
            st.session_state.actions
        )


# =========================================================
# QUESTIONS
# =========================================================

if st.session_state.questions:

    st.subheader(
        "❓ Open Questions"
    )

    st.markdown(
        st.session_state.questions
    )


# =========================================================
# RAG CHAT
# =========================================================

if st.session_state.rag:

    st.header("💬 Ask Questions")

    user_query = st.text_input(
        "Ask about the meeting"
    )

    if st.button(
        "Ask",
        use_container_width=True,
    ):

        if user_query:

            with st.spinner(
                "Generating Answer..."
            ):

                result = (
                    st.session_state.rag.ask(
                        user_query
                    )
                )

                st.subheader(
                    "Answer"
                )

                st.write(
                    result["answer"]
                )

                with st.expander(
                    "Retrieved Context"
                ):

                    st.write(
                        result["context"]
                    )


# =========================================================
# EXPORTS
# =========================================================

if st.session_state.summary:

    st.header("📤 Export")

    txt_export = f"""
FINAL SUMMARY
====================

{st.session_state.summary}


KEY DECISIONS
====================

{st.session_state.decisions}


ACTION ITEMS
====================

{st.session_state.actions}


OPEN QUESTIONS
====================

{st.session_state.questions}


TRANSCRIPT
====================

{st.session_state.transcript}
"""

    # TXT DOWNLOAD

    st.download_button(
        label="📄 Download TXT Report",
        data=txt_export,
        file_name="meeting_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

    # PDF DOWNLOAD

    pdf_path = create_pdf(
        transcript=st.session_state.transcript,
        summary=st.session_state.summary,
        decisions=st.session_state.decisions,
        actions=st.session_state.actions,
        questions=st.session_state.questions,
    )

    with open(pdf_path, "rb") as pdf_file:

        st.download_button(
            label="📕 Download PDF Report",
            data=pdf_file,
            file_name="meeting_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "AI Meeting Assistant • Local Whisper + Mistral + ChromaDB"
)