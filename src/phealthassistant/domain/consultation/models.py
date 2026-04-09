from pydantic import BaseModel
from typing import Optional


class StructuredConsultation(BaseModel):
    """Structured clinical output produced by the AI agent."""

    patient_summary: str
    active_conditions: list[str]
    current_medications: list[str]
    relevant_findings: list[str]
    clinical_recommendations: list[str]
    risk_level: str  # "low" | "moderate" | "high"


class ConsultationResult(BaseModel):
    """Full API response for a clinical consultation."""

    patient_id: str
    question: str
    consultation: StructuredConsultation

class ConsultationResponse(BaseModel):
    """ Unified response model for all consultation endpoints. """

    status: str  # "completed" | "pending_review"
    thread_id: str
    patient_id: str
    question: str
    consultation: Optional[StructuredConsultation] = None
    review_decision: Optional[str] = None  # "approved" | "rejected
