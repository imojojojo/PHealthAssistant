"""
ChromaDB adapter — implements VectorStorePort using the async ChromaDB HTTP client.

To add a different vector database:
  1. Create infrastructure/vector_store/<provider>_store.py
  2. Implement VectorStorePort
  3. Swap the wiring in main.py — nothing else in the codebase changes
"""

import logging

import chromadb

from phealthassistant.application.ports.vector_store import VectorChunk, VectorStorePort
from phealthassistant.config import Settings

logger = logging.getLogger(__name__)

# ChromaDB uses cosine distance (0 = identical, 1 = orthogonal).
# similarity = 1 - distance, so we store with cosine space for intuitive thresholds.
_HNSW_SPACE = "cosine"


class ChromaVectorStore(VectorStorePort):
    """
    Async ChromaDB adapter.
    The collection is created/retrieved lazily on first use to keep __init__ sync.
    Call await store.initialise() in the app lifespan before serving requests.
    """

    def __init__(self, settings: Settings) -> None:
        self._host = settings.chroma_host
        self._port = settings.chroma_port
        self._collection_name = settings.chroma_collection
        self._client: chromadb.AsyncHttpClient | None = None
        self._collection = None

    async def initialise(self) -> None:
        """Connect to ChromaDB and ensure the collection exists."""
        self._client = await chromadb.AsyncHttpClient(
            host=self._host, port=self._port
        )
        self._collection = await self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": _HNSW_SPACE},
        )
        logger.info(
            "ChromaDB connected — collection '%s' ready", self._collection_name
        )

    # ── VectorStorePort ───────────────────────────────────────────────────────

    async def upsert(self, chunks: list[VectorChunk]) -> None:
        await self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=[c.embedding for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        logger.info("Upserted %d chunks into ChromaDB", len(chunks))

    async def get_by_metadata(
        self,
        filters: dict[str, str],
        limit: int = 20,
    ) -> list[VectorChunk]:
        where = self._build_where(filters)
        result = await self._collection.get(
            where=where,
            limit=limit,
            include=["documents", "metadatas"],
        )
        return self._to_chunks(result.ids, result.documents, result.metadatas)

    async def similarity_search(
        self,
        query_embedding: list[float],
        filters: dict[str, str],
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[VectorChunk]:
        where = self._build_where(filters)
        result = await self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.ids[0]
        documents = result.documents[0]
        metadatas = result.metadatas[0]
        distances = result.distances[0]

        # cosine distance → similarity: similarity = 1 - distance
        chunks = []
        for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            similarity = 1.0 - dist
            if similarity >= min_similarity:
                chunks.append(VectorChunk(id=chunk_id, text=doc, metadata=meta))

        return chunks

    async def health_check(self) -> bool:
        try:
            await self._client.heartbeat()
            return True
        except Exception:
            return False

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_where(filters: dict[str, str]) -> dict:
        """
        Convert a flat {key: value} dict to ChromaDB's explicit $eq filter format.
        Handles single and multiple filters via $and.
        """
        conditions = [{k: {"$eq": v}} for k, v in filters.items()]
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @staticmethod
    def _to_chunks(
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
    ) -> list[VectorChunk]:
        return [
            VectorChunk(id=chunk_id, text=doc, metadata=meta)
            for chunk_id, doc, meta in zip(ids, documents, metadatas)
        ]
