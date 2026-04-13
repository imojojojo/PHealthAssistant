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


# ── Retry policy (transient errors only) ────────────────────────────────────

TRANSIENT_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    retry_on=(httpx.NetworkError, httpx.TimeoutException, ConnectionError),
)


# ── System prompts ──────────────────────────────────────────────────────────

_SUPERVISOR_PROMPT = """\
You are a clinical supervisor coordinating specialist agents.
Given a patient question, decide which specialist(s) to consult.

Available specialists:
- medication_analysis: Expert in drug interactions, dosages, contraindications, and medication management.
- risk_assessment: Expert in clinical risk evaluation, urgency assessment, and complication likelihood.

Respond ONLY with a valid JSON object (no markdown, no extra text):
{
  "agents_to_call": ["agent_name1", "agent_name2"],
  "reasoning": "Brief explanation of why these specialists are needed"
}

Rules:
- You may call one or both specialists.
- If the question is general and doesn't clearly need a specialist, call both.
- Order matters: if one agent's result should inform the other, put it first.\
"""

_MEDICATION_PROMPT = """\
You are a clinical pharmacology specialist.
Analyze the patient data and answer the question focusing on:
- Current medications and dosages
- Potential drug interactions
- Contraindications based on patient conditions
- Medication adjustment recommendations

After retrieving patient data, respond ONLY with a valid JSON object \
(no markdown, no extra text):
{
  "medication_summary": "Overview of current medication regimen",
  "interactions": ["interaction1", "interaction2"],
  "contraindications": ["contraindication1"],
  "recommendations": ["recommendation1", "recommendation2"]
}\
"""

_RISK_ASSESSMENT_PROMPT = """\
You are a clinical risk assessment specialist.
Analyze the patient data and answer the question focusing on:
- Immediate clinical risks
- Risk factors from patient history
- Urgency level (low, moderate, high)
- Recommended immediate actions

After retrieving patient data, respond ONLY with a valid JSON object \
(no markdown, no extra text):
{
  "risk_summary": "Overview of clinical risk profile",
  "risk_factors": ["factor1", "factor2"],
  "urgency": "low|moderate|high",
  "immediate_actions": ["action1", "action2"]
}\
"""

_SYNTHESIZER_PROMPT = """\
You are a clinical decision support assistant producing a final consultation.
You have received specialist reports. Synthesize them into a single consultation.

Respond ONLY with a valid JSON object (no markdown, no extra text):
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


# ── State definitions ───────────────────────────────────────────────────────

class SpecialistState(TypedDict):
    patient_id: str
    question: str
    messages: Annotated[list[BaseMessage], add_messages]
    result: str


class CoordinatorState(TypedDict):
    patient_id: str
    question: str
    supervisor_messages: Annotated[list[BaseMessage], add_messages]
    agents_to_call: list[str]
    current_agent_index: int
    medication_result: Optional[str]
    risk_result: Optional[str]
    final_response: Optional[str]
    risk_level: Optional[str]
    review_decision: Optional[str]


# ── Specialist sub-graph nodes ──────────────────────────────────────────────

async def specialist_call_llm(state: SpecialistState, llm: BaseChatModel) -> dict:
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}


def _get_message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, list):
        return " ".join(
            part["text"] if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    return (content or "").strip()


def specialist_should_continue(state: SpecialistState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "specialist_execute_tools"
    if not _get_message_text(last_message):
        logger.warning("Specialist LLM returned empty content, nudging for JSON response")
        return "specialist_nudge"
    return "specialist_parse_output"


def specialist_nudge(state: SpecialistState) -> dict:
    return {"messages": [HumanMessage(content="You have the patient data. Now respond with the JSON object as instructed.")]}


async def specialist_execute_tools(state: SpecialistState, retrieval: PatientContextService) -> dict:
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


def specialist_parse_output(state: SpecialistState) -> dict:
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
        raise ValueError("Specialist LLM returned an empty response")

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    json_matches = re.findall(r"\{[\s\S]*\}", text)
    if json_matches:
        text = json_matches[-1]

    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Specialist returned invalid JSON: %s | raw: %r", e, text)
        raise ValueError(f"Specialist returned invalid JSON: {e}") from e

    return {"result": text}


# ── Build a specialist sub-graph ────────────────────────────────────────────

def build_specialist_graph(
    llm: BaseChatModel,
    retrieval: PatientContextService,
    system_prompt: str,
) -> StateGraph:
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

    graph = StateGraph(SpecialistState)

    graph.add_node(
        "specialist_call_llm",
        partial(specialist_call_llm, llm=llm_with_tools),
        retry_policy=TRANSIENT_RETRY_POLICY,
    )
    graph.add_node(
        "specialist_execute_tools",
        partial(specialist_execute_tools, retrieval=retrieval),
        retry_policy=TRANSIENT_RETRY_POLICY,
    )
    graph.add_node("specialist_parse_output", specialist_parse_output)
    graph.add_node("specialist_nudge", specialist_nudge)

    graph.add_edge(START, "specialist_call_llm")
    graph.add_conditional_edges("specialist_call_llm", specialist_should_continue)
    graph.add_edge("specialist_execute_tools", "specialist_call_llm")
    graph.add_edge("specialist_nudge", "specialist_call_llm")
    graph.add_edge("specialist_parse_output", END)

    return graph.compile()


# ── Coordinator nodes ───────────────────────────────────────────────────────

async def supervisor_decide(state: CoordinatorState, llm: BaseChatModel) -> dict:
    """Supervisor LLM decides which specialist(s) to call."""
    response = await llm.ainvoke(state["supervisor_messages"])

    content = response.content
    if isinstance(content, list):
        content = " ".join(
            part["text"] if isinstance(part, dict) else str(part)
            for part in content
        )

    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    json_matches = re.findall(r"\{[\s\S]*\}", content)
    if json_matches:
        content = json_matches[-1]

    try:
        decision = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error("Supervisor returned invalid JSON: %s | raw: %r", e, content)
        raise ValueError(f"Supervisor returned invalid JSON: {e}") from e

    agents = decision.get("agents_to_call", [])
    valid_agents = {"medication_analysis", "risk_assessment"}
    agents = [a for a in agents if a in valid_agents]

    if not agents:
        agents = ["medication_analysis", "risk_assessment"]
        logger.warning("Supervisor returned no valid agents, defaulting to both")

    logger.info("Supervisor decision: %s — %s", agents, decision.get("reasoning", ""))

    return {
        "agents_to_call": agents,
        "current_agent_index": 0,
        "supervisor_messages": [response],
    }


def route_next_agent(state: CoordinatorState) -> str:
    """Route to the next specialist or move to synthesis."""
    index = state["current_agent_index"]
    agents = state["agents_to_call"]

    if index >= len(agents):
        return "synthesize"

    return agents[index]


async def run_medication_agent(
    state: CoordinatorState,
    medication_graph,
) -> dict:
    """Run the medication sub-graph and store its result."""
    specialist_input: SpecialistState = {
        "patient_id": state["patient_id"],
        "question": state["question"],
        "messages": [
            SystemMessage(content=_MEDICATION_PROMPT),
            HumanMessage(content=f"Patient ID: {state['patient_id']}\n\nQuestion: {state['question']}"),
        ],
        "result": "",
    }

    specialist_output = await medication_graph.ainvoke(specialist_input)
    logger.info("Medication agent completed for patient %s", state["patient_id"])

    return {
        "medication_result": specialist_output["result"],
        "current_agent_index": state["current_agent_index"] + 1,
    }


async def run_risk_agent(
    state: CoordinatorState,
    risk_graph,
) -> dict:
    """Run the risk assessment sub-graph and store its result."""
    specialist_input: SpecialistState = {
        "patient_id": state["patient_id"],
        "question": state["question"],
        "messages": [
            SystemMessage(content=_RISK_ASSESSMENT_PROMPT),
            HumanMessage(content=f"Patient ID: {state['patient_id']}\n\nQuestion: {state['question']}"),
        ],
        "result": "",
    }

    specialist_output = await risk_graph.ainvoke(specialist_input)
    logger.info("Risk agent completed for patient %s", state["patient_id"])

    return {
        "risk_result": specialist_output["result"],
        "current_agent_index": state["current_agent_index"] + 1,
    }


async def synthesize(state: CoordinatorState, llm: BaseChatModel) -> dict:
    """Combine specialist results into the final StructuredConsultation."""
    specialist_reports = []
    if state.get("medication_result"):
        specialist_reports.append(f"=== Medication Analysis ===\n{state['medication_result']}")
    if state.get("risk_result"):
        specialist_reports.append(f"=== Risk Assessment ===\n{state['risk_result']}")

    combined = "\n\n".join(specialist_reports)

    synthesis_messages = [
        SystemMessage(content=_SYNTHESIZER_PROMPT),
        HumanMessage(content=(
            f"Patient ID: {state['patient_id']}\n"
            f"Original question: {state['question']}\n\n"
            f"Specialist reports:\n{combined}\n\n"
            "Synthesize these into a single consultation JSON."
        )),
    ]

    response = await llm.ainvoke(synthesis_messages)

    content = response.content
    if isinstance(content, list):
        content = " ".join(
            part["text"] if isinstance(part, dict) else str(part)
            for part in content
        )

    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    json_matches = re.findall(r"\{[\s\S]*\}", content)
    if json_matches:
        content = json_matches[-1]

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error("Synthesizer returned invalid JSON: %s | raw: %r", e, content)
        raise ValueError(f"Synthesizer returned invalid JSON: {e}") from e

    try:
        StructuredConsultation.model_validate(data)
    except Exception as e:
        logger.error("Synthesizer JSON failed validation: %s | data: %r", e, data)
        raise ValueError(f"Synthesizer response failed validation: {e}") from e

    risk_level = data.get("risk_level", "low")
    logger.info("Synthesis complete — risk_level=%s", risk_level)

    return {
        "final_response": content,
        "risk_level": risk_level,
        "supervisor_messages": [response],
    }


def route_by_risk_level(state: CoordinatorState) -> str:
    if state["risk_level"] == "high":
        return "flag_for_review"
    return END


async def flag_for_review(state: CoordinatorState) -> dict:
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


# ── Build coordinator graph ─────────────────────────────────────────────────

def build_coordinator_graph(
    llm: BaseChatModel,
    retrieval: PatientContextService,
) -> StateGraph:

    medication_graph = build_specialist_graph(llm, retrieval, _MEDICATION_PROMPT)
    risk_graph = build_specialist_graph(llm, retrieval, _RISK_ASSESSMENT_PROMPT)

    graph = StateGraph(CoordinatorState)

    graph.add_node(
        "supervisor_decide",
        partial(supervisor_decide, llm=llm),
        retry_policy=TRANSIENT_RETRY_POLICY,
    )
    graph.add_node(
        "medication_analysis",
        partial(run_medication_agent, medication_graph=medication_graph),
        retry_policy=TRANSIENT_RETRY_POLICY,
    )
    graph.add_node(
        "risk_assessment",
        partial(run_risk_agent, risk_graph=risk_graph),
        retry_policy=TRANSIENT_RETRY_POLICY,
    )
    graph.add_node(
        "synthesize",
        partial(synthesize, llm=llm),
        retry_policy=TRANSIENT_RETRY_POLICY,
    )
    graph.add_node("flag_for_review", flag_for_review)

    graph.add_edge(START, "supervisor_decide")
    graph.add_conditional_edges("supervisor_decide", route_next_agent)
    graph.add_conditional_edges("medication_analysis", route_next_agent)
    graph.add_conditional_edges("risk_assessment", route_next_agent)
    graph.add_conditional_edges("synthesize", route_by_risk_level)
    graph.add_edge("flag_for_review", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer, interrupt_before=["flag_for_review"])


# ── Service ─────────────────────────────────────────────────────────────────

class MultiAgentClinicalService:
    def __init__(self, llm: BaseChatModel, retrieval: PatientContextService) -> None:
        self._graph = build_coordinator_graph(llm, retrieval)

    async def consult(self, patient_id: str, question: str, thread_id: str | None = None) -> ConsultationResponse:
        resolved_thread_id = thread_id or f"{patient_id}"
        config = {"configurable": {"thread_id": resolved_thread_id}}

        existing_state = await self._graph.aget_state(config)
        is_new_thread = not existing_state.values

        if is_new_thread:
            supervisor_messages = [
                SystemMessage(content=_SUPERVISOR_PROMPT),
                HumanMessage(content=f"Patient ID: {patient_id}\n\nQuestion: {question}"),
            ]
        else:
            supervisor_messages = [
                HumanMessage(content=(
                    f"Follow-up question: {question}\n\n"
                    "Remember: respond ONLY with a valid JSON object with agents_to_call."
                )),
            ]

        initial_state: CoordinatorState = {
            "patient_id": patient_id,
            "question": question,
            "supervisor_messages": supervisor_messages,
            "agents_to_call": [],
            "current_agent_index": 0,
            "medication_result": None,
            "risk_result": None,
            "final_response": None,
            "risk_level": None,
            "review_decision": None,
        }

        try:
            final_state = await self._graph.ainvoke(initial_state, config=config)
        except Exception as e:
            logger.error("Multi-agent failed for patient %s: %s", patient_id, e)
            raise

        state_snapshot = await self._graph.aget_state(config)
        if state_snapshot.next:
            return ConsultationResponse(
                status="pending_review",
                thread_id=resolved_thread_id,
                patient_id=patient_id,
                question=question,
            )

        consultation = StructuredConsultation.model_validate(json.loads(final_state["final_response"]))
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
        consultation = StructuredConsultation.model_validate(json.loads(final_state["final_response"]))
        return ConsultationResponse(
            status="completed",
            thread_id=thread_id,
            patient_id=patient_id,
            question=question,
            consultation=consultation,
            review_decision=final_state["review_decision"],
        )
