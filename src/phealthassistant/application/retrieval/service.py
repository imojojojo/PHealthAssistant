import logging

from phealthassistant.application.ports.llm import EmbeddingPort
from phealthassistant.application.ports.vector_store import VectorChunk, VectorStorePort
from phealthassistant.domain.patient.exceptions import PatientNotFoundError

logger = logging.getLogger(__name__)


class PatientContextService:
    """
    Retrieves patient context from the vector store.

    Two retrieval modes:
      A. get_patient_history   — metadata-only filter, returns all chunks for a patient
      B. find_relevant_history — combines metadata filter + semantic similarity search
    """

    def __init__(self, vector_store: VectorStorePort, embedding: EmbeddingPort) -> None:
        self._vector_store = vector_store
        self._embedding = embedding

    async def get_patient_history(self, patient_id: str) -> list[VectorChunk]:
        """
        Return ALL stored chunks for the given patient (no semantic ranking).
        Raises PatientNotFoundError when no chunks exist for that patient.
        """
        chunks = await self._vector_store.get_by_metadata(
            filters={"patientId": patient_id},
            limit=20,
        )
        if not chunks:
            raise PatientNotFoundError(patient_id)

        logger.info("Retrieved %d history chunks for patient %s", len(chunks), patient_id)
        return chunks

    async def find_relevant_history(
        self, patient_id: str, symptoms: str, top_k: int = 5
    ) -> list[VectorChunk]:
        """
        Return the top-K chunks most semantically similar to `symptoms`,
        scoped to the given patient via a metadata filter.
        Raises PatientNotFoundError when no results meet the similarity threshold.
        """
        query_embedding = await self._embedding.embed(symptoms)
        chunks = await self._vector_store.similarity_search(
            query_embedding=query_embedding,
            filters={"patientId": patient_id},
            top_k=top_k,
            min_similarity=0.5,
        )
        if not chunks:
            raise PatientNotFoundError(patient_id)

        logger.info(
            "Found %d relevant chunks for patient %s (symptoms: %s)",
            len(chunks),
            patient_id,
            symptoms,
        )
        return chunks

    @staticmethod
    def assemble_context(chunks: list[VectorChunk]) -> str:
        """Concatenate chunk texts into a single context string for the LLM."""
        if not chunks:
            return "No relevant patient history found."
        return "\n---\n".join(chunk.text for chunk in chunks)
