"""
core/vector_store.py

Handles:
- Transcript loading
- Text chunking
- Embedding generation
- ChromaDB storage
- Vector database persistence
- Similarity search
"""

from __future__ import annotations

import os
import shutil

from pathlib import Path
from typing import List

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from langchain_core.documents import (
    Document,
)

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma


def _get_base_dir():
    """Get the project base directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MeetingVectorStore:

    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = (
            "meeting_transcripts"
        ),
        embedding_model: str = (
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:

        # Use absolute path if not provided
        if persist_directory is None:
            base_dir = _get_base_dir()
            persist_directory = os.path.join(base_dir, "data", "vectorstore")

        self.persist_directory = (
            persist_directory
        )

        self.collection_name = (
            collection_name
        )

        self.chunk_size = chunk_size

        self.chunk_overlap = chunk_overlap

        self._embedding_model_name = embedding_model
        self._embedding_model = None

        # =================================================
        # TEXT SPLITTER
        # =================================================

        self.text_splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    def _ensure_embeddings(self):
        """Lazy load embeddings."""
        if self._embedding_model is None:
            print("\nLoading Embedding Model...")
            self._embedding_model = HuggingFaceEmbeddings(
                model_name=self._embedding_model_name
            )
            print("\nEmbedding Model Loaded")

    @property
    def embedding_model(self):
        self._ensure_embeddings()
        return self._embedding_model

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

        transcript = (
            transcript_path.read_text(
                encoding="utf-8"
            )
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
            f"\nGenerated "
            f"{len(chunks)} Chunks"
        )

        return chunks

    # =====================================================
    # CREATE DOCUMENTS
    # =====================================================

    def create_documents(
        self,
        chunks: List[str],
    ) -> List[Document]:

        documents = []

        for idx, chunk in enumerate(
            chunks,
            start=1,
        ):

            document = Document(
                page_content=chunk,
                metadata={
                    "chunk_id": idx,
                },
            )

            documents.append(
                document
            )

        print(
            f"\nCreated "
            f"{len(documents)} Documents"
        )

        return documents

    # =====================================================
    # CREATE VECTOR STORE
    # =====================================================

    def create_vector_store(
        self,
        documents: List[Document],
    ) -> Chroma:

        print(
            "\nCreating Chroma Vector Store..."
        )

        # Delete and recreate the directory fresh
        persist_path = Path(self.persist_directory)

        # Remove entire directory if exists
        if persist_path.exists():
            try:
                shutil.rmtree(persist_path)
            except Exception as e:
                print(f"Warning deleting old vector store: {e}")

        # Create fresh directory
        persist_path.mkdir(parents=True, exist_ok=True)

        # ================================================
        # CREATE NEW VECTOR STORE
        # =============================================

        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=(
                self.persist_directory
            ),
            collection_name=(
                self.collection_name
            ),
        )

        print(
            "\nVector Store Created Successfully"
        )

        return vector_store

    # =====================================================
    # LOAD EXISTING VECTOR STORE
    # =====================================================

    def load_vector_store(
        self,
    ) -> Chroma:

        print(
            "\nLoading Existing Vector Store..."
        )

        # Check if directory exists and has content
        persist_path = Path(self.persist_directory)
        if not persist_path.exists() or not any(persist_path.iterdir()):
            raise ValueError(f"Vector store at {self.persist_directory} is empty or missing")

        vector_store = Chroma(
            persist_directory=(
                self.persist_directory
            ),
            embedding_function=(
                self.embedding_model
            ),
            collection_name=(
                self.collection_name
            ),
        )

        print(
            "\nVector Store Loaded"
        )

        return vector_store

    # =====================================================
    # SIMILARITY SEARCH
    # =====================================================

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> List[Document]:

        vector_store = (
            self.load_vector_store()
        )

        results = (
            vector_store.similarity_search(
                query=query,
                k=k,
            )
        )

        return results

    # =====================================================
    # COMPLETE INGESTION PIPELINE
    # =====================================================

    def ingest_transcript(
        self,
        transcript_path: str | Path,
    ) -> Chroma:

        # ================================================
        # LOAD TRANSCRIPT
        # ================================================

        transcript = self.load_transcript(
            transcript_path
        )

        # ================================================
        # SPLIT TEXT
        # ================================================

        chunks = self.split_transcript(
            transcript
        )

        # ================================================
        # CREATE DOCUMENTS
        # ================================================

        documents = self.create_documents(
            chunks
        )

        # ================================================
        # VECTOR STORE
        # =============================================

        vector_store = (
            self.create_vector_store(
                documents
            )
        )

        return vector_store