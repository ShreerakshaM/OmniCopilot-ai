"""Vector store interface for RAG knowledge base."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RetrievedDocument:
    """A document retrieved from the knowledge base."""

    content: str
    source: str
    score: float
    metadata: dict[str, str]


class KnowledgeStore:
    """Qdrant-backed vector store for RAG.

    Stores embedded documents (papers, standards, documentation)
    and retrieves relevant context for the reasoning agents.
    """

    def __init__(self, collection_name: str = "omnicopilot_knowledge") -> None:
        """Initialize knowledge store.

        Args:
            collection_name: Qdrant collection name.
        """
        self._collection_name = collection_name

    def index_documents(self, documents_dir: Path) -> int:
        """Index all documents in a directory.

        Args:
            documents_dir: Path containing documents to index.

        Returns:
            Number of documents indexed.
        """
        # TODO: Implement — chunk, embed, upsert to Qdrant.
        raise NotImplementedError

    def search(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        """Search for relevant documents.

        Args:
            query: Natural language query.
            top_k: Number of results to return.

        Returns:
            Top-K most relevant documents.
        """
        # TODO: Implement — embed query, search Qdrant, rerank.
        raise NotImplementedError
