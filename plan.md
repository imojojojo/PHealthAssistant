# PHealthAssistant — LangGraph Learning Plan

A hands-on project to learn LangGraph and build enterprise-grade AI agents.
Each step builds on the previous one. Complete them in order.

---

## Completed

- [x] Install dependencies (langgraph, langchain, langchain-google-genai)
- [x] Understand LangGraph concepts (StateGraph, nodes, edges, state)
- [x] Build pre-fetch agent (retrieve context → LLM → parse output)
- [x] Add checkpointer (MemorySaver) for conversation persistence
- [x] Add conditional edges (risk-level routing → flag_for_review)
- [x] Build ReAct agent (LLM-driven tool calling loop)
- [x] Provider-agnostic LLM with BaseChatModel

---

## Remaining Steps (in order)

### ~~Step A — Error Handling & Retries~~ ✓ DONE
*Makes the agent production-resilient*

- What happens when the LLM returns invalid JSON?
- What happens when ChromaDB is down?
- Add retry logic with exponential backoff
- Add fallback responses instead of 500 errors
- Learn: try/except in async code, custom exception types, graceful degradation

### ~~Step B — Test & Refine~~ ✓ DONE
*Verify correctness across all scenarios*

- Run end-to-end tests with all 3 patients (P001, P002, P003)
- Test edge cases: patient not found, empty question, LLM bad JSON
- Verify ReAct tool calling loop in logs
- Learn: how to test async Python code, FastAPI test client

### ~~Step C — Multi-turn Conversations~~ ✓ DONE
*Use the checkpointer properly for stateful dialogue*

- Let a doctor ask follow-up questions in the same thread
- Agent remembers previous questions and answers
- Design a proper thread_id strategy (doctor + patient + session)
- Learn: LangGraph state accumulation across multiple ainvoke calls

### ~~Step D — Human-in-the-Loop~~ ✓ DONE
*Pause graph execution and wait for human approval*

- Pause the graph after risk assessment
- Wait for doctor to approve/reject before continuing
- Resume the graph with the doctor's input
- Learn: LangGraph interrupt(), Command, graph resumption

### Step E — Sub-graphs & Multi-Agent Architecture ← START HERE
*Compose specialized agents into a larger system*

- Build a specialized MedicationAnalysisAgent sub-graph
- Build a RiskAssessmentAgent sub-graph
- Compose them into a coordinator graph
- Learn: nested graphs, agent handoffs, supervisor pattern

### Step F — Deployment Tier 1: Self-hosted FastAPI on Fly.io / Railway
*Deploy the agent so clients can access it over the internet — cheap/free, full control*

- Dockerize the app (Dockerfile, docker-compose with ChromaDB)
- Environment variable management for secrets (OpenAI API key, etc.)
- Deploy to Fly.io or Railway free tier (or Hetzner ~€4/mo for more headroom)
- Handle ChromaDB data persistence with mounted volumes
- Health checks and graceful shutdown
- Add `guardrails-ai` as a library for PII / output validation (guardrails story on resume)
- Wire up **LangSmith free tier for tracing only** — no paid deployment, just observability
- Learn: containerization, cloud deployment, secrets management, agent observability

**Why this tier first:** Near-zero cost while learning. Your hexagonal architecture already
makes the app portable. LangSmith tracing gives you the debugging UI without vendor lock-in
on the deployment layer. Good enough for a live client demo.

### Step G — LangGraph Visualization & Observability
*Understand and monitor what the agent is doing*

- Print graph structure as ASCII diagram
- Add structured logging at every node (node name, input, output, duration)
- Learn: graph.get_graph().draw_ascii(), LangSmith tracing (optional)

### Step H — Streaming Responses
*Return results progressively instead of waiting for full completion*

- Stream LLM tokens to the client as they arrive
- Stream node-by-node updates (retrieval done, LLM thinking, etc.)
- Learn: astream(), astream_events(), Server-Sent Events in FastAPI

### Step I — Production Checkpointer
*Replace MemorySaver with a real persistent store*

- Swap MemorySaver for SqliteSaver
- Understand the trade-offs vs PostgresSaver
- Learn: checkpointer interfaces, connection management

### Step J — Deployment Tier 2: AWS Bedrock AgentCore
*Port the same agent to AWS Bedrock AgentCore — the enterprise resume differentiator*

- Install AgentCore CLI and scaffold a LangGraph project (`agentcore create`)
- Wrap the existing StateGraph with `BedrockAgentCoreApp` — minimal code changes thanks to
  the hexagonal architecture
- Keep OpenAI as the LLM provider (AgentCore Runtime is model-agnostic — your OpenAI key works)
- Store the OpenAI key in AWS Secrets Manager, inject at runtime
- Swap the Postgres/Sqlite checkpointer for **AgentCoreMemorySaver** (native LangGraph
  checkpointer for AgentCore Memory)
- Add **AgentCore Memory Store** for long-term memory across sessions (actor_id + thread_id)
- Enable **Bedrock Guardrails** for PII detection, toxicity, and topic filtering — works even
  with OpenAI as the underlying model
- Wire up **AgentCore Observability** (CloudWatch traces for agent reasoning, tool calls, LLM interactions)
- Deploy with `agentcore deploy`; test with `agentcore invoke`
- Optional: AgentCore Gateway to expose tools via MCP, AgentCore Identity for user auth
- Learn: serverless microVM agent runtime, AWS IAM for agents, enterprise-grade guardrails
  and observability, session isolation

**Why this tier second:** Highest-leverage resume move. "Deployed the same LangGraph agent on
AgentCore with Bedrock Guardrails, AgentCore Memory, and microVM session isolation" is exactly
the enterprise agentic systems narrative. Consumption-based pricing (I/O wait and idle time are
free, per-second CPU/memory billing) means it costs pennies while experimenting, and new AWS
accounts get $200 in free tier credits.


---

## Enterprise Agent Patterns (after completing A–H)

These are advanced patterns used in real production systems:

| Pattern | What it is | When you need it |
|---|---|---|
| **Supervisor / Orchestrator** | One agent coordinates others | Multi-domain problems |
| **Parallel tool execution** | Run multiple tools simultaneously | Performance optimization |
| **Long-term memory** | Persist facts across sessions in a DB | Personalization, user preferences |
| **Structured output enforcement** | Force LLM to always return valid JSON | Reliability |
| **Rate limiting & cost control** | Track and limit LLM API usage | Production cost management |
| **Auth & multi-tenancy** | Isolate data per user/org | SaaS products |
| **Evaluation & evals** | Measure agent answer quality automatically | Quality assurance |

---

## Recommended Learning Order

```
A (Error Handling)          ← foundation for everything else
  ↓
B (Testing)                 ← verify what you built
  ↓
C (Multi-turn)              ← deepens checkpointer understanding
  ↓
D (Human-in-the-loop)       ← most unique LangGraph capability
  ↓
E (Sub-graphs)              ← true enterprise architecture
  ↓
F (Deploy Tier 1: Fly.io)   ← cheap live system for client demos
  ↓
G (Visualization)           ← understand complex graphs you build
  ↓
H (Streaming)               ← production UX requirement
  ↓
I (Production Checkpointer) ← final piece for production readiness
  ↓
J (Deploy Tier 2: AgentCore)← enterprise resume differentiator
```

**Why this order:**
- A and B first — a fragile agent that crashes on bad input is not learnable or testable
- C before D — you need to understand stateful conversations before pausing them
- E after D — sub-graphs compose the patterns from all previous steps
- F after E — deploy the multi-agent system cheaply for client demos and learning
- G, H, I — polish and production-readiness on the Tier 1 deployment
- J last — port the hardened agent to AgentCore once it's genuinely production-ready;
  this tier is about enterprise credentials (guardrails, microVM isolation, managed memory),
  not about learning LangGraph itself
