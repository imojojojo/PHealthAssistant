from fastapi import APIRouter

from phealthassistant.api.deps import IngestionDep

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.post("/ingest")
async def ingest(ingestion: IngestionDep) -> dict:
    """
    Load all patient JSON files, chunk them, embed each chunk,
    and store everything in the vector database.
    """
    count = await ingestion.ingest_all()
    return {"status": "success", "chunksIngested": count}
