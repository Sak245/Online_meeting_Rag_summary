"""
core/rag_chain.py

Handles:
- Retriever creation
- RAG pipeline
- Context retrieval
- Conversational Q&A
"""

from __future__ import annotations

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
        # VECTOR STORE
        # =================================================

        self.vector_store_manager = (
            MeetingVectorStore()
        )

        self.vector_store = (
            self.vector_store_manager
            .load_vector_store()
        )

        # =================================================
        # RETRIEVER
        # =================================================

        self.retriever = (
            self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 4
                },
            )
        )

        print(
            "\nRAG Pipeline Initialized"
        )

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

        documents = (
            self.retriever.invoke(query)
        )

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