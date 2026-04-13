import json
import pytest

from langchain_core.messages import AIMessage
from unittest.mock import AsyncMock, MagicMock

from phealthassistant.application.agent.multi_agent_service import MultiAgentClinicalService
from phealthassistant.application.ports.vector_store import VectorChunk


SAMPLE_CHUNKS = [
    VectorChunk(
        id="1",
        text="Patient Alice Morgan, 52F, Type 2 Diabetes, Hypertension, Metformin 500mg",
        metadata={"patientId": "P001"},
        embedding=[],
    )
]

SUPERVISOR_RESPONSE_BOTH = json.dumps({
    "agents_to_call": ["medication_analysis", "risk_assessment"],
    "reasoning": "Question covers both medications and risk",
})

SUPERVISOR_RESPONSE_MEDICATION = json.dumps({
    "agents_to_call": ["medication_analysis"],
    "reasoning": "Question is about drug interactions",
})

MEDICATION_RESULT = json.dumps({
    "medication_summary": "Patient on Metformin 500mg for Type 2 Diabetes",
    "interactions": ["No significant interactions detected"],
    "contraindications": [],
    "recommendations": ["Continue current regimen"],
})

RISK_RESULT = json.dumps({
    "risk_summary": "Moderate cardiovascular risk due to diabetes and hypertension",
    "risk_factors": ["Type 2 Diabetes", "Hypertension"],
    "urgency": "moderate",
    "immediate_actions": ["Monitor blood pressure"],
})

FINAL_SYNTHESIS = json.dumps({
    "patient_summary": "52-year-old female with Type 2 Diabetes and Hypertension",
    "active_conditions": ["Type 2 Diabetes", "Hypertension"],
    "current_medications": ["Metformin 500mg"],
    "relevant_findings": ["No drug interactions", "Moderate cardiovascular risk"],
    "clinical_recommendations": ["Continue Metformin", "Monitor blood pressure"],
    "risk_level": "moderate",
})

FINAL_SYNTHESIS_HIGH_RISK = json.dumps({
    "patient_summary": "65-year-old male with chest pain",
    "active_conditions": ["Coronary Artery Disease"],
    "current_medications": ["Aspirin 81mg"],
    "relevant_findings": ["Acute chest pain", "History of CAD"],
    "clinical_recommendations": ["Immediate cardiac evaluation"],
    "risk_level": "high",
})


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
    llm.bind_tools = MagicMock(return_value=llm)

    # Default: supervisor calls both agents, specialists use tools then respond, synthesizer produces final
    llm.ainvoke = AsyncMock(side_effect=[
        # 1. Supervisor decides
        AIMessage(content=SUPERVISOR_RESPONSE_BOTH),
        # 2. Medication specialist calls tool
        AIMessage(content="", tool_calls=[{
            "id": "call_med_1", "name": "get_patient_history",
            "args": {"patient_id": "P001"}, "type": "tool_call",
        }]),
        # 3. Medication specialist responds
        AIMessage(content=MEDICATION_RESULT),
        # 4. Risk specialist calls tool
        AIMessage(content="", tool_calls=[{
            "id": "call_risk_1", "name": "get_patient_history",
            "args": {"patient_id": "P001"}, "type": "tool_call",
        }]),
        # 5. Risk specialist responds
        AIMessage(content=RISK_RESULT),
        # 6. Synthesizer combines
        AIMessage(content=FINAL_SYNTHESIS),
    ])
    return llm


@pytest.fixture
def service(mock_llm, mock_retrieval):
    return MultiAgentClinicalService(mock_llm, mock_retrieval)


async def test_multi_agent_happy_path(service):
    response = await service.consult("P001", "What medications and risks?")

    assert response.status == "completed"
    assert response.patient_id == "P001"
    assert response.consultation.risk_level == "moderate"
    assert "Type 2 Diabetes" in response.consultation.active_conditions
    assert "Metformin 500mg" in response.consultation.current_medications


async def test_multi_agent_single_specialist(mock_retrieval):
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.ainvoke = AsyncMock(side_effect=[
        # 1. Supervisor calls medication only
        AIMessage(content=SUPERVISOR_RESPONSE_MEDICATION),
        # 2. Medication specialist calls tool
        AIMessage(content="", tool_calls=[{
            "id": "call_med_1", "name": "get_patient_history",
            "args": {"patient_id": "P001"}, "type": "tool_call",
        }]),
        # 3. Medication specialist responds
        AIMessage(content=MEDICATION_RESULT),
        # 4. Synthesizer combines
        AIMessage(content=FINAL_SYNTHESIS),
    ])

    service = MultiAgentClinicalService(llm, mock_retrieval)
    response = await service.consult("P001", "What about drug interactions?")

    assert response.status == "completed"
    assert response.consultation is not None


async def test_multi_agent_high_risk_triggers_review(mock_retrieval):
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.ainvoke = AsyncMock(side_effect=[
        # 1. Supervisor calls risk only
        AIMessage(content=json.dumps({
            "agents_to_call": ["risk_assessment"],
            "reasoning": "Chest pain requires risk assessment",
        })),
        # 2. Risk specialist calls tool
        AIMessage(content="", tool_calls=[{
            "id": "call_risk_1", "name": "get_patient_history",
            "args": {"patient_id": "P003"}, "type": "tool_call",
        }]),
        # 3. Risk specialist responds
        AIMessage(content=RISK_RESULT),
        # 4. Synthesizer produces high risk
        AIMessage(content=FINAL_SYNTHESIS_HIGH_RISK),
    ])

    service = MultiAgentClinicalService(llm, mock_retrieval)
    response = await service.consult("P003", "Patient has chest pain")

    assert response.status == "pending_review"
    assert response.consultation is None


async def test_multi_agent_invalid_supervisor_json(mock_retrieval):
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="I'm not sure what to do"))

    service = MultiAgentClinicalService(llm, mock_retrieval)

    with pytest.raises(ValueError, match="Supervisor returned invalid JSON"):
        await service.consult("P001", "What medications?")
