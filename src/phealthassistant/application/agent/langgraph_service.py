import json
import re
import logging

from typing import Optional
from typing_extensions import TypedDict

from functools import partial
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from phealthassistant.application.retrieval.service import PatientContextService
from phealthassistant.application.ports.llm import LLMPort
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

class AgentState(TypedDict):
    patient_id: str
    question: str
    history_text: Optional[str]
    relevant_text: Optional[str]
    llm_response: Optional[str]
    risk_level: Optional[str]

async def retrieve_context(state: AgentState, retrieval: PatientContextService) -> dict:
    history_chunks = await retrieval.get_patient_history(state["patient_id"])
    relevant_chunks = await retrieval.find_relevant_history(state["patient_id"], state["question"])
    return {
        "history_text": retrieval.assemble_context(history_chunks),
        "relevant_text": retrieval.assemble_context(relevant_chunks),
    }

async def call_llm(state: AgentState, llm: LLMPort) -> dict:
    user_message = (
        f"Patient ID: {state['patient_id']}\n\n"
        f"Question: {state['question']}\n\n"
        f"Full History:\n{state['history_text']}\n\n"
        f"Relevant Context:\n{state['relevant_text']}"
    )
    response = await llm.chat_with_tools(
        system_prompt=_SYSTEM_PROMPT,
        user_message=user_message,
        tools=[],
        tool_executor=lambda name, args: "",
    )

    return {"llm_response": response}

def parse_output(state: AgentState) -> dict:
    text = state["llm_response"].strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    json_matches = re.findall(r"\{[\s\S]*\}", text)
    if json_matches:
        text = json_matches[-1]
    data = json.loads(text)
    StructuredConsultation.model_validate(data)

    return {
        "llm_response": text,
        "risk_level": data.get("risk_level", "low"),
    }

def route_by_risk_level(state: AgentState) -> str:
    if state["risk_level"] == "high":
        return "flag_for_review"
    
    return END

def flag_for_review(state: AgentState) -> dict:
    logger.warning(
        "HIGH RISK patient flagged for senior review — patient_id=%s",
        state["patient_id"],
    )

    return {}
    

def build_graph(llm: LLMPort, retrieval: PatientContextService) -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("retrieve_context", partial(retrieve_context, retrieval=retrieval))
    graph.add_node("call_llm", partial(call_llm, llm=llm))
    graph.add_node("parse_output", parse_output)

    graph.add_edge(START, "retrieve_context")
    graph.add_edge("retrieve_context", "call_llm")
    graph.add_edge("call_llm", "parse_output")

    graph.add_conditional_edges("parse_output", route_by_risk_level)
    graph.add_edge("flag_for_review", END)
    graph.add_node("flag_for_review", flag_for_review)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)

class LangGraphClinicalAgentService:
    def __init__(self, llm: LLMPort, retrieval: PatientContextService) -> None:
        self._graph = build_graph(llm, retrieval)

    async def consult(self, patient_id: str, question: str) -> ConsultationResult:
        initial_state: AgentState = {
            "patient_id": patient_id,
            "question": question,
            "history_text": None,
            "relevant_text": None,
            "llm_response": None
        }

        config = {"configurable": {"thread_id": f"{patient_id}"}}
        final_state = await self._graph.ainvoke(initial_state, config=config)
        consultation = StructuredConsultation.model_validate(json.loads(final_state["llm_response"]))
        return ConsultationResult(patient_id=patient_id, question=question, consultation=consultation)
