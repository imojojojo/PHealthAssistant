import logging

from phealthassistant.application.ports.llm import EmbeddingPort
from phealthassistant.application.ports.vector_store import VectorChunk, VectorStorePort
from phealthassistant.domain.patient.exceptions import IngestionError
from phealthassistant.domain.patient.models import Patient
from phealthassistant.infrastructure.data.patient_loader import PatientDataLoader

logger = logging.getLogger(__name__)


class PatientIngestionService:
    """
    Loads patient JSON files, splits them into focused chunks,
    embeds each chunk, and stores everything in the vector store.

    Chunking strategy (same as the original Java project):
      - 1 profile chunk per patient  (demographics + conditions + medications)
      - 1 visit chunk per clinical visit
    Smaller, focused chunks → better retrieval precision.
    """

    def __init__(
        self,
        loader: PatientDataLoader,
        vector_store: VectorStorePort,
        embedding: EmbeddingPort,
    ) -> None:
        self._loader = loader
        self._vector_store = vector_store
        self._embedding = embedding

    async def ingest_all(self) -> int:
        """
        Ingest every patient JSON file found in the configured data directory.
        Returns the total number of chunks stored.
        """
        try:
            patients = self._loader.load_all()
        except Exception as exc:
            raise IngestionError(f"Failed to load patient data: {exc}") from exc

        all_chunks: list[VectorChunk] = []
        for patient in patients:
            chunks = self._build_chunks(patient)
            for chunk in chunks:
                chunk.embedding = await self._embedding.embed(chunk.text)
            all_chunks.extend(chunks)
            logger.info("Prepared %d chunks for patient %s", len(chunks), patient.patient_id)

        await self._vector_store.upsert(all_chunks)
        logger.info("Ingested %d total chunks into vector store", len(all_chunks))
        return len(all_chunks)

    def _build_chunks(self, patient: Patient) -> list[VectorChunk]:
        chunks: list[VectorChunk] = []

        # Chunk 1 — patient profile (demographics + conditions + medications)
        profile_text = (
            f"Patient: {patient.name} (ID: {patient.patient_id}), Age: {patient.age}\n"
            f"Chronic Conditions: {', '.join(patient.conditions)}\n"
            f"Current Medications: {', '.join(patient.medications)}"
        )
        chunks.append(
            VectorChunk(
                id=f"{patient.patient_id}_profile",
                text=profile_text,
                metadata={"patientId": patient.patient_id, "chunkType": "profile"},
            )
        )

        # Chunks 2+ — one per clinical visit
        for idx, visit in enumerate(patient.visits):
            visit_text = (
                f"Patient: {patient.name} (ID: {patient.patient_id})\n"
                f"Visit Date: {visit.date} | Reason: {visit.reason}\n"
                f"Clinical Notes: {visit.notes}"
            )
            chunks.append(
                VectorChunk(
                    id=f"{patient.patient_id}_visit_{idx}",
                    text=visit_text,
                    metadata={
                        "patientId": patient.patient_id,
                        "chunkType": "visit",
                        "visitDate": visit.date,
                        "visitReason": visit.reason,
                    },
                )
            )

        return chunks
