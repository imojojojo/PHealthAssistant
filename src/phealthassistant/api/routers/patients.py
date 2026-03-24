from fastapi import APIRouter, Query

from phealthassistant.api.deps import RetrievalDep
from phealthassistant.application.retrieval.service import PatientContextService

router = APIRouter(prefix="/patients", tags=["Patients"])


@router.get("/{patient_id}/history")
async def get_history(patient_id: str, retrieval: RetrievalDep) -> dict:
    """Return all stored chunks for a patient (no semantic ranking)."""
    chunks = await retrieval.get_patient_history(patient_id)
    return {
        "patientId": patient_id,
        "chunksFound": len(chunks),
        "chunks": [chunk.text for chunk in chunks],
    }


@router.get("/{patient_id}/context")
async def get_context(
    patient_id: str,
    retrieval: RetrievalDep,
    symptoms: str = Query(..., description="Symptoms or clinical concerns to search for"),
) -> dict:
    """Return the top-5 chunks most relevant to the given symptoms."""
    chunks = await retrieval.find_relevant_history(patient_id, symptoms)
    return {
        "patientId": patient_id,
        "symptoms": symptoms,
        "chunksFound": len(chunks),
        "assembledContext": PatientContextService.assemble_context(chunks),
    }
