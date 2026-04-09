import json
import re
import logging
import httpx

from typing import Annotated, Optional
from typing_extensions import TypedDict

from functools import partial
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from phealthassistant.application.retrieval.service import PatientContextService
from phealthassistant.domain.consultation.models import ConsultationResponse, StructuredConsultation

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

TRANSIENT_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    retry_on=(httpx.NetworkError, httpx.TimeoutException, ConnectionError),
)


class AgentState(TypedDict):
    patient_id: str
    question: str
    llm_response: Optional[str]
    risk_level: Optional[str]
    messages: Annotated[list[BaseMessage], add_messages]
    review_decision: Optional[str]


# ── Nodes ────────────────────────────────────────────────────────────────────

async def call_llm(state: AgentState, llm: BaseChatModel) -> dict:
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "execute_tools"
    return "parse_output"


async def execute_tools(state: AgentState, retrieval: PatientContextService) -> dict:
    last_message = state["messages"][-1]
    tool_messages = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "get_patient_history":
            chunks = await retrieval.get_patient_history(tool_args["patient_id"])
            result = PatientContextService.assemble_context(chunks)
        elif tool_name == "find_relevant_history":
            chunks = await retrieval.find_relevant_history(
                tool_args["patient_id"],
                tool_args["symptoms"],
            )
            result = PatientContextService.assemble_context(chunks)
        else:
            result = f"Unknown tool: {tool_name}"

        tool_messages.append(ToolMessage(
            content=result,
            tool_call_id=tool_call["id"],
        ))

    return {"messages": tool_messages}


def parse_output(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    content = last_message.content

    if isinstance(content, list):
        text = " ".join(
            part["text"] if isinstance(part, dict) else str(part)
            for part in content
        )
    else:
        text = content

    if not text:
        raise ValueError("LLM returned an empty response")

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    json_matches = re.findall(r"\{[\s\S]*\}", text)
    if json_matches:
        text = json_matches[-1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("LLM returned invalid JSON: %s | raw text: %r", e, text)
        raise ValueError(f"LLM returned invalid JSON: {e}") from e

    try:
        StructuredConsultation.model_validate(data)
    except Exception as e:
        logger.error("LLM JSON failed schema validation: %s | data: %r", e, data)
        raise ValueError(f"LLM response failed validation: {e}") from e

    return {
        "llm_response": text,
        "risk_level": data.get("risk_level", "low"),
    }


def route_by_risk_level(state: AgentState) -> str:
    if state["risk_level"] == "high":
        return "flag_for_review"
    return END


async def flag_for_review(state: AgentState) -> dict:
    logger.warning(
        "HIGH RISK patient flagged for senior review — patient_id=%s",
        state["patient_id"],
    )
    if state["review_decision"] == "rejected":
        raise ValueError(
            f"Consultation rejected by senior reviewer for patient {state['patient_id']}"
        )
    logger.info("Consultation approved by senior reviewer for patient %s", state["patient_id"])
    return {}


# ── Graph builder ────────────────────────────────────────────────────────────

def build_graph(llm: BaseChatModel, retrieval: PatientContextService) -> StateGraph:
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    async def get_patient_history(patient_id: str) -> str:
        """Retrieves the complete medical history for a patient."""
        chunks = await retrieval.get_patient_history(patient_id)
        return PatientContextService.assemble_context(chunks)

    @lc_tool
    async def find_relevant_history(patient_id: str, symptoms: str) -> str:
        """Finds patient history chunks semantically relevant to specific symptoms."""
        chunks = await retrieval.find_relevant_history(patient_id, symptoms)
        return PatientContextService.assemble_context(chunks)

    tools = [get_patient_history, find_relevant_history]
    llm_with_tools = llm.bind_tools(tools)

    graph = StateGraph(AgentState)

    graph.add_node("call_llm", partial(call_llm, llm=llm_with_tools), retry_policy=TRANSIENT_RETRY_POLICY)
    graph.add_node("execute_tools", partial(execute_tools, retrieval=retrieval), retry_policy=TRANSIENT_RETRY_POLICY)
    graph.add_node("parse_output", parse_output)
    graph.add_node("flag_for_review", flag_for_review)

    graph.add_edge(START, "call_llm")
    graph.add_conditional_edges("call_llm", should_continue)
    graph.add_edge("execute_tools", "call_llm")
    graph.add_conditional_edges("parse_output", route_by_risk_level)
    graph.add_edge("flag_for_review", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer, interrupt_before=["flag_for_review"])


# ── Service ──────────────────────────────────────────────────────────────────

class LangGraphClinicalAgentService:
    def __init__(self, llm: BaseChatModel, retrieval: PatientContextService) -> None:
        self._graph = build_graph(llm, retrieval)

    async def consult(self, patient_id: str, question: str, thread_id: str | None = None) -> ConsultationResponse:
        resolved_thread_id = thread_id or f"{patient_id}"
        config = {"configurable": {"thread_id": resolved_thread_id}}

        existing_state = await self._graph.aget_state(config)
        is_new_thread = not existing_state.values

        if is_new_thread:
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=f"Patient ID: {patient_id}\n\nQuestion: {question}"),
            ]
        else:
            messages = [
                HumanMessage(content=(
                    f"Follow-up question: {question}\n\n"
                    "Remember: respond ONLY with a valid JSON object matching the required schema."
                )),
            ]

        initial_state: AgentState = {
            "patient_id": patient_id,
            "question": question,
            "llm_response": None,
            "risk_level": None,
            "messages": messages,
            "review_decision": None,
        }

        try:
            final_state = await self._graph.ainvoke(initial_state, config=config)
        except Exception as e:
            logger.error("Agent failed for patient %s: %s", patient_id, e)
            raise

        state_snapshot = await self._graph.aget_state(config)
        if state_snapshot.next:
            return ConsultationResponse(
                status="pending_review",
                thread_id=resolved_thread_id,
                patient_id=patient_id,
                question=question,
            )

        consultation = StructuredConsultation.model_validate(json.loads(final_state["llm_response"]))
        return ConsultationResponse(
            status="completed",
            thread_id=resolved_thread_id,
            patient_id=patient_id,
            question=question,
            consultation=consultation,
        )

    async def resume_review(self, thread_id: str, decision: str) -> ConsultationResponse:
        config = {"configurable": {"thread_id": thread_id}}

        state_snapshot = await self._graph.aget_state(config)
        if not state_snapshot.next:
            raise ValueError(f"No pending review found for thread_id: {thread_id}")

        await self._graph.aupdate_state(config, {"review_decision": decision})
        final_state = await self._graph.ainvoke(None, config=config)

        patient_id = final_state["patient_id"]
        question = final_state["question"]
        consultation = StructuredConsultation.model_validate(json.loads(final_state["llm_response"]))
        return ConsultationResponse(
            status="completed",
            thread_id=thread_id,
            patient_id=patient_id,
            question=question,
            consultation=consultation,
            review_decision=final_state["review_decision"],
        )
