from pydantic import BaseModel


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
