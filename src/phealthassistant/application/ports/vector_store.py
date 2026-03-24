from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class VectorChunk:
    """A unit of text stored in the vector database, with metadata and an optional embedding."""

    id: str
    text: str
    metadata: dict[str, str]
    embedding: list[float] = field(default_factory=list)


class VectorStorePort(ABC):
    """
    Abstract interface for any vector database.

    To add a new provider (e.g. Pinecone, Weaviate):
      1. Create infrastructure/vector_store/<provider>_store.py
      2. Implement all methods below
      3. Wire it in main.py — nothing else changes
    """

    @abstractmethod
    async def upsert(self, chunks: list[VectorChunk]) -> None:
        """Insert or update chunks. IDs are used for deduplication."""
        ...

    @abstractmethod
    async def get_by_metadata(
        self,
        filters: dict[str, str],
        limit: int = 20,
    ) -> list[VectorChunk]:
        """
        Retrieve chunks matching all supplied metadata key-value pairs.
        No semantic ranking — returns all matches up to `limit`.
        """
        ...

    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: list[float],
        filters: dict[str, str],
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[VectorChunk]:
        """
        Return the `top_k` chunks whose embeddings are most similar to
        `query_embedding`, filtered to rows matching `filters`.
        Only chunks with similarity >= `min_similarity` are returned.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the store is reachable and healthy."""
        ...
