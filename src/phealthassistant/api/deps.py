"""
FastAPI dependency injection.

Services are created once during app startup (lifespan) and stored in app.state.
These Annotated dependencies pull them out of state for use in route functions.
"""

from typing import Annotated

from fastapi import Depends, Request

from phealthassistant.application.agent.langgraph_service import LangGraphClinicalAgentService
from phealthassistant.application.ingestion.service import PatientIngestionService
from phealthassistant.application.retrieval.service import PatientContextService
from phealthassistant.application.ports.llm import EmbeddingPort, LLMPort


def _get_ingestion(request: Request) -> PatientIngestionService:
    return request.app.state.ingestion_service


def _get_retrieval(request: Request) -> PatientContextService:
    return request.app.state.retrieval_service


def _get_langgraph_agent(request: Request) -> LangGraphClinicalAgentService:
    return request.app.state.langgraph_agent_service


def _get_llm(request: Request) -> LLMPort:
    return request.app.state.llm_client


def _get_embedding(request: Request) -> EmbeddingPort:
    return request.app.state.embedding_client


IngestionDep = Annotated[PatientIngestionService, Depends(_get_ingestion)]
RetrievalDep = Annotated[PatientContextService, Depends(_get_retrieval)]
LanggraphAgentDep = Annotated[LangGraphClinicalAgentService, Depends(_get_langgraph_agent)]
LLMDep = Annotated[LLMPort, Depends(_get_llm)]
EmbeddingDep = Annotated[EmbeddingPort, Depends(_get_embedding)]
