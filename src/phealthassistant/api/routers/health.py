from fastapi import APIRouter, Query

from phealthassistant.api.deps import EmbeddingDep, LLMDep

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/ping")
async def ping(llm: LLMDep) -> dict:
    """Verify LLM connectivity by asking a simple question."""
    response = await llm.health_check()
    return {"response": response}


@router.get("/embed")
async def embed(
    embedding: EmbeddingDep,
    text: str = Query(default="fever"),
) -> dict:
    """Verify embedding model connectivity and return vector metadata."""
    result = await embedding.health_check(text)
    return {
        "text": result.text,
        "dimensions": result.dimensions,
        "first5Values": result.first_5_values,
    }
