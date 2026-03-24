import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from phealthassistant.domain.patient.exceptions import IngestionError, PatientNotFoundError

logger = logging.getLogger(__name__)


async def patient_not_found_handler(request: Request, exc: PatientNotFoundError) -> JSONResponse:
    logger.warning("Patient not found: %s", exc.patient_id)
    return JSONResponse(
        status_code=404,
        content={"error": "PATIENT_NOT_FOUND", "detail": str(exc)},
    )


async def ingestion_error_handler(request: Request, exc: IngestionError) -> JSONResponse:
    logger.error("Ingestion error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "INGESTION_ERROR", "detail": str(exc)},
    )


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    logger.warning("Validation error: %s", exc)
    return JSONResponse(
        status_code=400,
        content={"error": "BAD_REQUEST", "detail": str(exc)},
    )


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unexpected error on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"error": "INTERNAL_ERROR", "detail": "An unexpected error occurred."},
    )
