"""
core/rag_chain.py

Handles:
- Retriever creation
- RAG pipeline
- Context retrieval
- Conversational Q&A
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain_core.documents import (
    Document,
)

from langchain_core.prompts import (
    ChatPromptTemplate,
)

from langchain_core.output_parsers import (
    StrOutputParser,
)

from langchain_openai import ChatOpenAI

from core.vector_store import (
    MeetingVectorStore,
)


def _get_base_dir():
    """Get the project base directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MeetingRAG:

    def __init__(
        self,
        api_key: str,
        model: str = (
            "mistral-medium-latest"
        ),
        base_url: str = (
            "https://api.mistral.ai/v1"
        ),
        temperature: float = 0.2,
    ) -> None:

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
        # OUTPUT PARSER
        # =================================================

        self.output_parser = (
            StrOutputParser()
        )

        # =================================================
        # VECTOR STORE (Lazy load)
        # =================================================

        self.vector_store_manager = None
        self.vector_store = None
        self.retriever = None
        # Use absolute paths
        base_dir = _get_base_dir()
        self._transcript_path = os.path.join(base_dir, "data", "transcripts", "meeting_transcript.txt")
        self._vectorstore_path = os.path.join(base_dir, "data", "vectorstore")

    def _ensure_vector_store(self):
        """Lazy load vector store when needed. Auto-recover from corruption."""
        from pathlib import Path

        # Check if we have a transcript
        if not Path(self._transcript_path).exists():
            raise ValueError("No transcript processed yet. Process a meeting first.")

        # Check if vector store exists
        vectorstore_path = Path(self._vectorstore_path)
        has_existing = vectorstore_path.exists() and any(vectorstore_path.iterdir())

        # If vector store not loaded, create it fresh from transcript
        if self.vector_store is None:
            if not has_existing:
                # No existing vector store - create one
                print("Creating vector store from transcript...")
                self.vector_store_manager = MeetingVectorStore()
                self.vector_store_manager.ingest_transcript(self._transcript_path)
                self.vector_store = self.vector_store_manager.load_vector_store()
                self.retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                )
                print("\nRAG Pipeline Created")
            else:
                try:
                    # Try to load existing
                    self.vector_store_manager = MeetingVectorStore()
                    self.vector_store = self.vector_store_manager.load_vector_store()
                    self.retriever = self.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 4}
                    )
                    print("\nRAG Pipeline Initialized")
                except Exception as e:
                    print(f"Vector store corrupted, recreating: {e}")
                    # Recreate from scratch
                    self.vector_store = None
                    self.retriever = None
                    # Delete corrupted db
                    import shutil
                    shutil.rmtree(vectorstore_path, ignore_errors=True)
                    vectorstore_path.mkdir(exist_ok=True)
                    # Create fresh
                    self.vector_store_manager = MeetingVectorStore()
                    self.vector_store_manager.ingest_transcript(self._transcript_path)
                    self.vector_store = self.vector_store_manager.load_vector_store()
                    self.retriever = self.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 4}
                    )
                    print("\nRAG Pipeline Recreated")

    # =====================================================
    # RETRIEVE DOCUMENTS
    # =====================================================

    def retrieve_documents(
        self,
        query: str,
    ) -> List[Document]:
        """
        Retrieve relevant transcript chunks.
        """
        self._ensure_vector_store()

        documents = self.retriever.invoke(query)

        return documents

    # =====================================================
    # FORMAT CONTEXT
    # =====================================================

    def format_context(
        self,
        documents: List[Document],
    ) -> str:
        """
        Convert retrieved documents
        into context string.
        """

        context_parts = []

        for doc in documents:

            chunk_id = (
                doc.metadata.get(
                    "chunk_id",
                    "unknown",
                )
            )

            content = (
                doc.page_content
            )

            formatted_chunk = (
                f"[Chunk {chunk_id}]\n"
                f"{content}"
            )

            context_parts.append(
                formatted_chunk
            )

        context = "\n\n".join(
            context_parts
        )

        return context

    # =====================================================
    # GENERATE ANSWER
    # =====================================================

    def generate_answer(
        self,
        query: str,
        context: str,
    ) -> str:
        """
        Generate grounded answer
        using retrieved context.
        """

        prompt = (
            ChatPromptTemplate
            .from_template(
                """
You are an AI Meeting Assistant.

Answer the question using ONLY
the provided meeting transcript context.

Be intelligent while interpreting
the transcript even if:
- transcription errors exist
- text is noisy
- wording is imperfect

If partial information exists,
provide the best possible answer.

Only say:
"I could not find this information
in the meeting transcript."

if absolutely no relevant context exists.

========================
CONTEXT:
{context}
========================

QUESTION:
{question}
"""
            )
        )

        chain = (
            prompt
            | self.llm
            | self.output_parser
        )

        answer = chain.invoke(
            {
                "context": context,
                "question": query,
            }
        )

        return answer

    # =====================================================
    # ASK QUESTION
    # =====================================================

    def ask(
        self,
        query: str,
    ) -> dict:
        """
        Complete RAG pipeline.

        Steps:
        - Retrieve documents
        - Build context
        - Generate answer
        """

        print(
            f"\nUser Question:\n{query}"
        )

        # ================================================
        # RETRIEVAL
        # ================================================

        documents = (
            self.retrieve_documents(
                query
            )
        )

        print(
            f"\nRetrieved "
            f"{len(documents)} Documents"
        )

        # ================================================
        # CONTEXT
        # ================================================

        context = (
            self.format_context(
                documents
            )
        )

        # ================================================
        # DEBUG CONTEXT
        # ================================================

        print("\n" + "=" * 60)
        print("RETRIEVED CONTEXT")
        print("=" * 60)

        print(context[:2000])

        # ================================================
        # GENERATION
        # ================================================

        answer = (
            self.generate_answer(
                query=query,
                context=context,
            )
        )

        return {
            "question": query,
            "answer": answer,
            "documents": documents,
            "context": context,
        }

    # =====================================================
    # INTERACTIVE CHAT LOOP
    # =====================================================

    def chat(
        self,
    ) -> None:
        """
        Interactive terminal chat.
        """

        print("\n" + "=" * 60)
        print("MEETING RAG CHAT")
        print("=" * 60)

        print(
            "\nType 'exit' to quit.\n"
        )

        while True:

            query = input(
                "\nAsk Question:\n> "
            )

            if (
                query.lower().strip()
                == "exit"
            ):

                print(
                    "\nExiting RAG Chat..."
                )

                break

            try:

                result = self.ask(
                    query
                )

                print(
                    "\n" + "=" * 60
                )

                print("ANSWER")

                print(
                    "=" * 60
                )

                print(
                    f"\n{result['answer']}"
                )

            except Exception as e:

                print(
                    f"\nERROR:\n{e}"
                )