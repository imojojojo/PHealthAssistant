from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable


# Synchronous tool executor: (tool_name, args_dict) -> result_string
# The Gemini adapter runs its tool-calling loop in a thread, so the executor
# must be synchronous. ClinicalAgentService provides a sync wrapper that
# dispatches async retrieval calls back to the running event loop via
# asyncio.run_coroutine_threadsafe().
ToolExecutor = Callable[[str, dict], str]


@dataclass
class ToolDefinition:
    """
    Provider-agnostic description of an LLM tool (function declaration).

    `parameters` is a JSON Schema object describing the tool's arguments:
    {
        "type": "object",
        "properties": { "arg_name": { "type": "string", "description": "..." } },
        "required": ["arg_name"]
    }
    """

    name: str
    description: str
    parameters: dict


@dataclass
class EmbeddingHealthResult:
    text: str
    dimensions: int
    first_5_values: list[float]


class LLMPort(ABC):
    """
    Abstract interface for chat / tool-calling LLM providers.

    To add a new provider (e.g. OpenAI, Anthropic):
      1. Create infrastructure/llm/<provider>_client.py
      2. Implement the methods below
      3. Wire it in main.py — nothing else changes
    """

    @abstractmethod
    async def chat_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[ToolDefinition],
        tool_executor: ToolExecutor,
    ) -> str:
        """
        Run the full multi-turn tool-calling loop:
          1. Send system_prompt + user_message to the LLM with the given tools.
          2. While the LLM responds with tool calls, execute them via tool_executor.
          3. Feed results back until the LLM produces a final text response.
          4. Return that final text.
        """
        ...

    @abstractmethod
    async def health_check(self) -> str:
        """Return a short LLM-generated string to verify connectivity."""
        ...


class EmbeddingPort(ABC):
    """
    Abstract interface for text embedding providers.
    Kept separate from LLMPort so chat and embedding can use different providers.
    """

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for the given text."""
        ...

    @abstractmethod
    async def health_check(self, text: str = "fever") -> EmbeddingHealthResult:
        """Embed `text` and return metadata about the resulting vector."""
        ...
