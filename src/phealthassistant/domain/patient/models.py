from pydantic import BaseModel, Field


class Visit(BaseModel, frozen=True):
    date: str
    reason: str
    notes: str


class Patient(BaseModel, frozen=True):
    """Core patient aggregate. Immutable after construction."""

    patient_id: str = Field(alias="patientId")
    name: str
    age: int
    conditions: list[str]
    medications: list[str]
    visits: list[Visit]

    model_config = {"populate_by_name": True}
