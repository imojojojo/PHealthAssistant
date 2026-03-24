import asyncio
import json
import logging
import re

from phealthassistant.application.ports.llm import LLMPort, ToolDefinition
from phealthassistant.application.retrieval.service import PatientContextService
from phealthassistant.domain.consultation.models import ConsultationResult, StructuredConsultation

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a clinical decision support assistant.
You have access to tools that retrieve patient medical history from a vector database.
Always use the appropriate tool to fetch patient data before answering.
Be concise, medically precise, and cite the patient data you retrieved.

After retrieving patient data, respond ONLY with a valid JSON object \
matching this exact structure (no markdown, no extra text):
{
  "patient_summary": "Brief demographic and clinical summary",
  "active_conditions": ["condition1", "condition2"],
  "current_medications": ["med1", "med2"],
  "relevant_findings": ["finding1", "finding2"],
  "clinical_recommendations": ["recommendation1", "recommendation2"],
  "risk_level": "low"
}
risk_level must be one of: "low", "moderate", "high".\
"""

_TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="get_patient_history",
        description=(
            "Retrieves the complete medical history for a patient. "
            "Use this when you need a full overview of a patient's profile, "
            "conditions, medications, and all past visits."
        ),
        parameters={
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "The unique patient identifier (e.g. P001)",
                }
            },
            "required": ["patient_id"],
        },
    ),
    ToolDefinition(
        name="find_relevant_history",
        description=(
            "Finds patient history chunks semantically relevant to specific symptoms "
            "or clinical concerns. Use this when you need focused context for a specific "
            "complaint rather than the full history."
        ),
        parameters={
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "The unique patient identifier",
                },
                "symptoms": {
                    "type": "string",
                    "description": "Description of symptoms or clinical concerns",
                },
            },
            "required": ["patient_id", "symptoms"],
        },
    ),
]


class ClinicalAgentService:
    """
    Orchestrates the LLM agent for clinical consultations.

    Tool definitions live here (application layer) because they describe
    business capabilities, not infrastructure. The LLMPort implementation
    handles the provider-specific tool-calling loop.
    """

    def __init__(self, llm: LLMPort, retrieval: PatientContextService) -> None:
        self._llm = llm
        self._retrieval = retrieval

    async def consult(self, patient_id: str, question: str) -> ConsultationResult:
        logger.info("Starting consultation for patient %s — %s", patient_id, question)

        # Capture the running event loop so the sync tool executor can dispatch
        # async retrieval calls back to it from within the LLM provider's thread.
        loop = asyncio.get_running_loop()

        def tool_executor(name: str, args: dict) -> str:
            if name == "get_patient_history":
                coro = self._get_history_text(args.get("patient_id", patient_id))
            elif name == "find_relevant_history":
                coro = self._find_relevant_text(
                    args.get("patient_id", patient_id),
                    args.get("symptoms", ""),
                )
            else:
                return f"Unknown tool: {name}"

            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result(timeout=30)

        response_text = await self._llm.chat_with_tools(
            system_prompt=_SYSTEM_PROMPT,
            user_message=f"Patient ID: {patient_id}\n\nQuestion: {question}",
            tools=_TOOLS,
            tool_executor=tool_executor,
        )

        consultation = self._parse_consultation(response_text)
        logger.info("Consultation complete for patient %s", patient_id)
        return ConsultationResult(
            patient_id=patient_id,
            question=question,
            consultation=consultation,
        )

    async def _get_history_text(self, patient_id: str) -> str:
        chunks = await self._retrieval.get_patient_history(patient_id)
        return PatientContextService.assemble_context(chunks)

    async def _find_relevant_text(self, patient_id: str, symptoms: str) -> str:
        chunks = await self._retrieval.find_relevant_history(patient_id, symptoms)
        return PatientContextService.assemble_context(chunks)

    @staticmethod
    def _parse_consultation(response_text: str) -> StructuredConsultation:
        """
        Extract a JSON object from the LLM response and parse it into
        a StructuredConsultation. Strips markdown code fences if present.
        """
        text = response_text.strip()

        # Remove markdown code fences: ```json ... ``` or ``` ... ```
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        # If there are multiple JSON objects, take the last one (the final answer)
        json_matches = re.findall(r"\{[\s\S]*\}", text)
        if json_matches:
            text = json_matches[-1]

        data = json.loads(text)
        return StructuredConsultation.model_validate(data)
