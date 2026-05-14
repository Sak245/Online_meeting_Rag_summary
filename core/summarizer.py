"""
core/summarizer.py

Handles:
- Transcript chunking
- Chunk-wise summarization
- Final summary generation
- Key decisions extraction
- Action items extraction
- Open questions extraction
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from langchain_core.prompts import (
    ChatPromptTemplate,
)

from langchain_core.output_parsers import (
    StrOutputParser,
)

from langchain_openai import ChatOpenAI


class MeetingSummarizer:

    def __init__(
        self,
        api_key: str,
        model: str = "mistral-medium-latest",
        base_url: str = (
            "https://api.mistral.ai/v1"
        ),
        temperature: float = 0.3,
        chunk_size: int = 3000,
        chunk_overlap: int = 300,
        summaries_dir: str = "outputs/summaries",
    ) -> None:

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.summaries_dir = Path(
            summaries_dir
        )

        self.summaries_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # =================================================
        # LLM
        # =================================================

        self.llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
        )

        # =================================================
        # TEXT SPLITTER
        # =================================================

        self.text_splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

        # =================================================
        # OUTPUT PARSER
        # =================================================

        self.output_parser = (
            StrOutputParser()
        )

    # =====================================================
    # LOAD TRANSCRIPT
    # =====================================================

    def load_transcript(
        self,
        transcript_path: str | Path,
    ) -> str:

        transcript_path = Path(
            transcript_path
        )

        transcript = transcript_path.read_text(
            encoding="utf-8"
        )

        return transcript

    # =====================================================
    # SPLIT TRANSCRIPT
    # =====================================================

    def split_transcript(
        self,
        transcript: str,
    ) -> List[str]:

        chunks = (
            self.text_splitter.split_text(
                transcript
            )
        )

        print(
            f"\nTranscript Split Into "
            f"{len(chunks)} Chunks"
        )

        return chunks

    # =====================================================
    # CHUNK SUMMARY
    # =====================================================

    def summarize_chunk(
        self,
        chunk: str,
    ) -> str:

        prompt = ChatPromptTemplate.from_template(
            """
You are an AI meeting assistant.

Summarize the following meeting transcript chunk.

Focus on:
- Important discussions
- Key ideas
- Technical points
- Main conclusions

Transcript:
{chunk}
"""
        )

        chain = (
            prompt
            | self.llm
            | self.output_parser
        )

        summary = chain.invoke(
            {
                "chunk": chunk
            }
        )

        return summary

    # =====================================================
    # MULTI-CHUNK SUMMARIZATION
    # =====================================================

    def summarize_transcript(
        self,
        transcript: str,
    ) -> str:

        chunks = self.split_transcript(
            transcript
        )

        chunk_summaries = []

        total_chunks = len(chunks)

        for idx, chunk in enumerate(
            chunks,
            start=1,
        ):

            print(
                f"\nSummarizing "
                f"Chunk {idx}/{total_chunks}"
            )

            summary = self.summarize_chunk(
                chunk
            )

            chunk_summaries.append(
                summary
            )

        combined_summary = "\n\n".join(
            chunk_summaries
        )

        return combined_summary

    # =====================================================
    # FINAL GLOBAL SUMMARY
    # =====================================================

    def generate_final_summary(
        self,
        combined_summary: str,
    ) -> str:

        prompt = ChatPromptTemplate.from_template(
            """
You are an AI meeting assistant.

Create a FINAL CONSOLIDATED summary
from the provided chunk summaries.

The final summary should:
- Be concise
- Preserve important details
- Preserve technical context
- Preserve conclusions

Chunk Summaries:
{summary}
"""
        )

        chain = (
            prompt
            | self.llm
            | self.output_parser
        )

        final_summary = chain.invoke(
            {
                "summary":
                combined_summary
            }
        )

        return final_summary

    # =====================================================
    # KEY DECISIONS
    # =====================================================

    def extract_key_decisions(
        self,
        transcript: str,
    ) -> str:

        prompt = ChatPromptTemplate.from_template(
            """
Extract KEY DECISIONS
from the meeting transcript.

Return concise bullet points.

Transcript:
{transcript}
"""
        )

        chain = (
            prompt
            | self.llm
            | self.output_parser
        )

        decisions = chain.invoke(
            {
                "transcript":
                transcript
            }
        )

        return decisions

    # =====================================================
    # ACTION ITEMS
    # =====================================================

    def extract_action_items(
        self,
        transcript: str,
    ) -> str:

        prompt = ChatPromptTemplate.from_template(
            """
Extract ACTION ITEMS
from the meeting transcript.

Include:
- Tasks
- Responsibilities
- Deadlines if present

Return concise bullet points.

Transcript:
{transcript}
"""
        )

        chain = (
            prompt
            | self.llm
            | self.output_parser
        )

        actions = chain.invoke(
            {
                "transcript":
                transcript
            }
        )

        return actions

    # =====================================================
    # OPEN QUESTIONS
    # =====================================================

    def extract_open_questions(
        self,
        transcript: str,
    ) -> str:

        prompt = ChatPromptTemplate.from_template(
            """
Extract OPEN QUESTIONS
or unresolved discussions
from the meeting transcript.

Return concise bullet points.

Transcript:
{transcript}
"""
        )

        chain = (
            prompt
            | self.llm
            | self.output_parser
        )

        questions = chain.invoke(
            {
                "transcript":
                transcript
            }
        )

        return questions

    # =====================================================
    # SAVE OUTPUT
    # =====================================================

    def save_output(
        self,
        content: str,
        filename: str,
    ) -> Path:

        output_path = (
            self.summaries_dir
            / filename
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(content)

        print(
            f"\nSaved:\n{output_path}"
        )

        return output_path

    # =====================================================
    # COMPLETE PIPELINE
    # =====================================================

    def process_meeting(
        self,
        transcript_path: str | Path,
    ) -> dict:

        # ================================================
        # LOAD TRANSCRIPT
        # ================================================

        transcript = self.load_transcript(
            transcript_path
        )

        # ================================================
        # CHUNK SUMMARIES
        # ================================================

        combined_summary = (
            self.summarize_transcript(
                transcript
            )
        )

        # ================================================
        # FINAL SUMMARY
        # ================================================

        final_summary = (
            self.generate_final_summary(
                combined_summary
            )
        )

        # ================================================
        # EXTRACTIONS
        # ================================================

        decisions = (
            self.extract_key_decisions(
                transcript
            )
        )

        actions = (
            self.extract_action_items(
                transcript
            )
        )

        questions = (
            self.extract_open_questions(
                transcript
            )
        )

        # ================================================
        # SAVE OUTPUTS
        # ================================================

        self.save_output(
            final_summary,
            "final_summary.txt",
        )

        self.save_output(
            decisions,
            "key_decisions.txt",
        )

        self.save_output(
            actions,
            "action_items.txt",
        )

        self.save_output(
            questions,
            "open_questions.txt",
        )

        return {
            "summary": final_summary,
            "decisions": decisions,
            "actions": actions,
            "questions": questions,
        }