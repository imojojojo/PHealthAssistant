class PatientNotFoundError(Exception):
    """Raised when a patient cannot be found in the vector store."""

    def __init__(self, patient_id: str) -> None:
        self.patient_id = patient_id
        super().__init__(f"No records found for patient '{patient_id}'")


class IngestionError(Exception):
    """Raised when patient data ingestion fails."""
