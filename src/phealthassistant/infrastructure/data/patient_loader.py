import json
import logging
from pathlib import Path

from phealthassistant.domain.patient.models import Patient

logger = logging.getLogger(__name__)


class PatientDataLoader:
    """Reads patient JSON files from the configured data directory."""

    def __init__(self, data_dir: str) -> None:
        self._data_dir = Path(data_dir)

    def load_all(self) -> list[Patient]:
        if not self._data_dir.exists():
            raise FileNotFoundError(f"Patient data directory not found: {self._data_dir}")

        files = sorted(self._data_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No JSON files found in {self._data_dir}")

        patients: list[Patient] = []
        for path in files:
            raw = json.loads(path.read_text(encoding="utf-8"))
            patient = Patient.model_validate(raw)
            patients.append(patient)
            logger.info("Loaded patient %s from %s", patient.patient_id, path.name)

        return patients
