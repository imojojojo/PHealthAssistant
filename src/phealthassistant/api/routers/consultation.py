from pydantic import BaseModel, Field, field_validator
from fastapi import APIRouter
from typing import Optional

from phealthassistant.api.deps import LanggraphAgentDep
from phealthassistant.domain.consultation.models import ConsultationResponse

router = APIRouter(prefix="/consultation", tags=["Consultation"])


class ConsultationRequest(BaseModel):
    # Accept both camelCase (patientId) and snake_case (patient_id) from callers
    patient_id: str = Field(alias="patientId")
    question: str
    thread_id: Optional[str] = Field(default=None, alias="threadId")

    @field_validator("patient_id", "question")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field must not be blank")
        return value

    model_config = {"populate_by_name": True}


class ResumeRequest(BaseModel):
    thread_id: str = Field(alias="threadId")
    decision: str

    @field_validator("thread_id", "decision")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field must not be blank")
        return value

    model_config = {"populate_by_name": True}


@router.post("", response_model=ConsultationResponse)
async def consult(request: ConsultationRequest, agent: LanggraphAgentDep) -> ConsultationResponse:
    """
    Run a clinical consultation for the given patient.
    The LLM decides which tools to call and loops until it has enough to answer.
    Returns status 'pending_review' if risk is high, 'completed' otherwise.
    Supports multi-turn conversations via threadId.
    """
    return await agent.consult(request.patient_id, request.question, request.thread_id)


@router.post("/resume", response_model=ConsultationResponse)
async def resume(request: ResumeRequest, agent: LanggraphAgentDep) -> ConsultationResponse:
    """
    Resume a pending consultation that was paused for senior doctor review.
    Pass the threadId from the pending_review response and a decision (approved/rejected).
    """
    return await agent.resume_review(request.thread_id, request.decision)
