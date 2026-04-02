from pydantic import BaseModel, Field, field_validator
from fastapi import APIRouter

from phealthassistant.api.deps import AgentDep, LanggraphAgentDep
from phealthassistant.domain.consultation.models import ConsultationResult

router = APIRouter(prefix="/consultation", tags=["Consultation"])


class ConsultationRequest(BaseModel):
    # Accept both camelCase (patientId) and snake_case (patient_id) from callers
    patient_id: str = Field(alias="patientId")
    question: str

    @field_validator("patient_id", "question")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field must not be blank")
        return value

    model_config = {"populate_by_name": True}


@router.post("", response_model=ConsultationResult)
async def consult(request: ConsultationRequest, agent: LanggraphAgentDep) -> ConsultationResult:
    """
    Run a clinical consultation for the given patient.
    The AI agent will autonomously retrieve relevant patient history
    and return a structured clinical assessment.
    """
    return await agent.consult(request.patient_id, request.question)

@router.post("/react", response_model=ConsultationResult)
async def consult_react(request: ConsultationRequest, agent: LanggraphAgentDep) -> ConsultationResult:
    """
    Run a clinical consultation using the ReAct agent.
    The LLM decides which tools to call and loops until it has enough to answer.
    """
    return await agent.consult_react(request.patient_id, request.question)
