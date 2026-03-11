import logging
from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from app.config import get_settings
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)
settings = get_settings()


class RAGPipeline:
    """RAG pipeline using ChromaDB for vector storage and Ollama for generation."""

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self._embedding_model: Optional[SentenceTransformer] = None
        self._chroma_client: Optional[chromadb.Client] = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self._embedding_model = SentenceTransformer(settings.embedding_model)
        return self._embedding_model

    @property
    def chroma_client(self) -> chromadb.Client:
        if self._chroma_client is None:
            self._chroma_client = chromadb.Client(
                ChromaSettings(
                    persist_directory=settings.chroma_persist_dir,
                    anonymized_telemetry=False,
                )
            )
        return self._chroma_client

    def _get_or_create_collection(self, doc_id: str):
        """Get or create a ChromaDB collection for a document."""
        collection_name = f"doc_{doc_id}"
        return self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_document(self, doc_id: str, chunks: list[str]) -> int:
        """Index document chunks into ChromaDB with embeddings."""
        if not chunks:
            logger.warning(f"No chunks to index for document {doc_id}")
            return 0

        collection = self._get_or_create_collection(doc_id)

        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks).tolist()

        # Add to ChromaDB
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{"chunk_index": i, "doc_id": doc_id} for i in range(len(chunks))],
        )

        logger.info(f"Indexed {len(chunks)} chunks for document {doc_id}")
        return len(chunks)

    def retrieve_relevant_chunks(
        self, doc_id: str, query: str, top_k: int = 4
    ) -> list[str]:
        """Retrieve the most relevant chunks for a query."""
        collection = self._get_or_create_collection(doc_id)

        query_embedding = self.embedding_model.encode([query]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, collection.count()),
        )

        chunks = results.get("documents", [[]])[0]
        logger.info(f"Retrieved {len(chunks)} relevant chunks for query")
        return chunks

    async def answer_question(self, doc_id: str, question: str) -> tuple[str, list[str]]:
        """Answer a question about a document using RAG."""
        # Retrieve relevant context
        relevant_chunks = self.retrieve_relevant_chunks(doc_id, question)

        if not relevant_chunks:
            return "No relevant information found in the document.", []

        # Build context
        context = "\n\n---\n\n".join(relevant_chunks)

        system_prompt = (
            "You are a precise document analysis assistant. Answer the user's question "
            "based ONLY on the provided context. If the answer cannot be found in the "
            "context, say so clearly. Do not make up information."
        )

        prompt = (
            f"Context from the document:\n\n{context}\n\n"
            f"---\n\n"
            f"Question: {question}\n\n"
            f"Answer based on the context above:"
        )

        answer = await self.llm_service.generate(prompt, system_prompt)
        return answer, relevant_chunks
