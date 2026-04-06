import json
import pytest

from langchain.messages import AIMessage
from unittest.mock import AsyncMock, MagicMock, patch

from phealthassistant.application.agent.langgraph_service import LangGraphClinicalAgentService
from phealthassistant.application.ports.vector_store import VectorChunk
from phealthassistant.domain.patient.exceptions import PatientNotFoundError


VALID_LLM_RESPONSE = json.dumps({
    "patient_summary": "52-year-old female with Type 2 Diabetes",
    "active_conditions": ["Type 2 Diabetes", "Hypertension"],
    "current_medications": ["Metformin 500mg"],
    "relevant_findings": ["HbA1c 7.2"],
    "clinical_recommendations": ["Monitor blood pressure"],
    "risk_level": "moderate",
})

SAMPLE_CHUNKS = [
    VectorChunk(
        id="1",
        text="Patient Alice Morgan, 52F, Type 2 Diabetes, Hypertension",
        metadata={"patientId": "P001"},
        embedding=[],
    )
]

@pytest.fixture
def mock_retrieval():
    retrieval = MagicMock()
    retrieval.get_patient_history = AsyncMock(return_value=SAMPLE_CHUNKS)
    retrieval.find_relevant_history = AsyncMock(return_value=SAMPLE_CHUNKS)
    retrieval.assemble_context = MagicMock(return_value="Patient history text")
    return retrieval

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat_with_tools = AsyncMock(return_value=VALID_LLM_RESPONSE)
    return llm

@pytest.fixture
def mock_chat_llm():
    chat_llm = MagicMock()
    chat_llm.ainvoke = AsyncMock(return_value=AIMessage(content=VALID_LLM_RESPONSE))
    chat_llm.bind_tools = MagicMock(return_value=chat_llm)
    return chat_llm

@pytest.fixture
def service(mock_llm, mock_retrieval, mock_chat_llm):
    return LangGraphClinicalAgentService(mock_llm, mock_retrieval, mock_chat_llm)

async def test_consult(service):
    result = await service.consult("P001", "What are the main health concerns?")

    assert result.patient_id == "P001"
    assert result.question == "What are the main health concerns?"
    assert result.consultation.risk_level == "moderate"
    assert "Type 2 Diabetes" in result.consultation.active_conditions

async def test_consult_patient_not_found(service, mock_retrieval):
    mock_retrieval.get_patient_history = AsyncMock(
        side_effect = PatientNotFoundError("P999")
    )

    with pytest.raises(PatientNotFoundError):
        await service.consult("P999", "What are the main health concerns?")

async def test_consult_invalid_llm_response(service, mock_llm):
    mock_llm.chat_with_tools = AsyncMock(return_value="Not a valid JSON")

    with pytest.raises(ValueError, match="LLM returned invalid JSON"):
        await service.consult("P001", "What are the main health concerns?")

async def test_consult_react(service, mock_chat_llm):
    tool_call_response = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "1",
                "name": "get_patient_history",
                "args": {"patient_id": "P001"},
                "type": "tool_call",
            }
        ],
    )

    final_response = AIMessage(content = VALID_LLM_RESPONSE)

    mock_chat_llm.ainvoke = AsyncMock(side_effect=[tool_call_response, final_response])

    result = await service.consult_react("P001", "What are the main health concerns?")

    assert result.patient_id == "P001"
    assert result.consultation.risk_level == "moderate"

async def test_consult_react_patient_not_found(service, mock_retrieval, mock_chat_llm):
    mock_retrieval.get_patient_history = AsyncMock(
        side_effect = PatientNotFoundError("P999")
    )

    mock_chat_llm.ainvoke = AsyncMock(
        return_value=AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "1",
                    "name": "get_patient_history",
                    "args": {"patient_id": "P999"},
                    "type": "tool_call",
                }
            ],
        )
    )

    with pytest.raises(PatientNotFoundError):
        await service.consult_react("P999", "What are the main health concerns?")
