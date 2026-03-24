"""
Application entry point.

This is the ONLY file that imports from all layers.
It wires together infrastructure adapters, application services, and API routers.

To swap a provider (e.g. ChromaDB → Pinecone, Gemini → OpenAI):
  - Change the adapter instantiation below
  - Nothing else in the codebase needs to change
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from phealthassistant.api.exception_handlers import (
    generic_error_handler,
    ingestion_error_handler,
    patient_not_found_handler,
    value_error_handler,
)
from phealthassistant.api.routers import admin, consultation, health, patients
from phealthassistant.application.agent.service import ClinicalAgentService
from phealthassistant.application.ingestion.service import PatientIngestionService
from phealthassistant.application.retrieval.service import PatientContextService
from phealthassistant.config import Settings
from phealthassistant.domain.patient.exceptions import IngestionError, PatientNotFoundError

# ── Infrastructure adapters (swap these to change providers) ─────────────────
from phealthassistant.infrastructure.data.patient_loader import PatientDataLoader
from phealthassistant.infrastructure.llm.gemini_client import GeminiClient
from phealthassistant.infrastructure.vector_store.chroma_store import ChromaVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: build and wire all singletons, store on app.state.
    Shutdown: nothing to clean up for now (connections are stateless HTTP).
    """
    settings = Settings()

    # ── Infrastructure ────────────────────────────────────────────────────────
    llm_client = GeminiClient(settings)           # swap → OpenAIClient(settings)
    embedding_client = GeminiClient(settings)     # swap → OllamaEmbeddingClient(settings)
    vector_store = ChromaVectorStore(settings)    # swap → PineconeVectorStore(settings)
    await vector_store.initialise()

    loader = PatientDataLoader(settings.data_dir)

    # ── Application services ──────────────────────────────────────────────────
    ingestion_service = PatientIngestionService(loader, vector_store, embedding_client)
    retrieval_service = PatientContextService(vector_store, embedding_client)
    agent_service = ClinicalAgentService(llm_client, retrieval_service)

    # ── Store on app.state for dependency injection ───────────────────────────
    app.state.llm_client = llm_client
    app.state.embedding_client = embedding_client
    app.state.ingestion_service = ingestion_service
    app.state.retrieval_service = retrieval_service
    app.state.agent_service = agent_service

    logger.info("PHealthAssistant started — all services ready")
    yield
    logger.info("PHealthAssistant shutting down")


app = FastAPI(
    title="PHealthAssistant",
    description="Healthcare RAG AI Agent",
    version="0.1.0",
    lifespan=lifespan,
)

# ── Exception handlers ────────────────────────────────────────────────────────
app.add_exception_handler(PatientNotFoundError, patient_not_found_handler)
app.add_exception_handler(IngestionError, ingestion_error_handler)
app.add_exception_handler(ValueError, value_error_handler)
app.add_exception_handler(Exception, generic_error_handler)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(admin.router)
app.include_router(patients.router)
app.include_router(consultation.router)
