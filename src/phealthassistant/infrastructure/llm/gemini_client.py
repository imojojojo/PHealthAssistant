"""
Gemini adapter — implements LLMPort and EmbeddingPort using google-generativeai.

To add a different LLM provider:
  1. Create infrastructure/llm/<provider>_client.py
  2. Implement LLMPort and/or EmbeddingPort
  3. Swap the wiring in main.py — nothing else in the codebase changes
"""

import asyncio
import logging

import google.generativeai as genai
from google.generativeai import protos

from phealthassistant.application.ports.llm import (
    EmbeddingHealthResult,
    EmbeddingPort,
    LLMPort,
    ToolDefinition,
    ToolExecutor,
)
from phealthassistant.config import Settings

logger = logging.getLogger(__name__)


class GeminiClient(LLMPort, EmbeddingPort):
    """
    Single adapter for both chat (with tool calling) and text embeddings
    via the Google Generative AI SDK.

    All SDK calls are synchronous; they are dispatched to a thread pool
    via asyncio.to_thread() to keep the FastAPI event loop unblocked.
    """

    def __init__(self, settings: Settings) -> None:
        genai.configure(api_key=settings.gemini_api_key)
        self._chat_model = settings.llm_chat_model
        self._embedding_model = settings.llm_embedding_model

    # ── LLMPort ──────────────────────────────────────────────────────────────

    async def chat_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[ToolDefinition],
        tool_executor: ToolExecutor,
    ) -> str:
        return await asyncio.to_thread(
            self._run_tool_calling_loop,
            system_prompt,
            user_message,
            tools,
            tool_executor,
        )

    def _run_tool_calling_loop(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[ToolDefinition],
        tool_executor: ToolExecutor,
    ) -> str:
        """
        Synchronous multi-turn tool-calling loop.
        Runs inside a thread (via asyncio.to_thread) so it never blocks the event loop.
        """
        genai_tool = self._build_genai_tool(tools)
        model = genai.GenerativeModel(
            model_name=self._chat_model,
            system_instruction=system_prompt,
            tools=[genai_tool],
        )
        chat = model.start_chat()
        response = chat.send_message(user_message)

        while True:
            function_calls = [
                part.function_call
                for part in response.parts
                if part.function_call.name  # empty name means no call
            ]

            if not function_calls:
                return response.text

            # Execute every tool call the LLM requested
            result_parts: list[protos.Part] = []
            for fc in function_calls:
                logger.info("Agent calling tool '%s' with args %s", fc.name, dict(fc.args))
                result = tool_executor(fc.name, dict(fc.args))
                result_parts.append(
                    protos.Part(
                        function_response=protos.FunctionResponse(
                            name=fc.name,
                            response={"result": result},
                        )
                    )
                )

            response = chat.send_message(result_parts)

    async def health_check(self) -> str:
        return await asyncio.to_thread(self._ping_chat)

    def _ping_chat(self) -> str:
        model = genai.GenerativeModel(self._chat_model)
        response = model.generate_content(
            "What is RAG in healthcare? Answer in one sentence."
        )
        return response.text

    # ── EmbeddingPort ─────────────────────────────────────────────────────────

    async def embed(self, text: str) -> list[float]:
        result = await asyncio.to_thread(
            genai.embed_content,
            model=self._embedding_model,
            content=text,
        )
        return result["embedding"]

    async def health_check(self, text: str = "fever") -> EmbeddingHealthResult:  # type: ignore[override]
        vector = await self.embed(text)
        return EmbeddingHealthResult(
            text=text,
            dimensions=len(vector),
            first_5_values=vector[:5],
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_genai_tool(tools: list[ToolDefinition]) -> protos.Tool:
        """Convert provider-agnostic ToolDefinition list to a Gemini Tool proto."""
        declarations: list[protos.FunctionDeclaration] = []
        for tool in tools:
            props = {}
            for prop_name, prop_schema in tool.parameters.get("properties", {}).items():
                type_name = prop_schema.get("type", "string").upper()
                props[prop_name] = protos.Schema(
                    type=getattr(protos.Type, type_name),
                    description=prop_schema.get("description", ""),
                )
            declarations.append(
                protos.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=protos.Schema(
                        type=protos.Type.OBJECT,
                        properties=props,
                        required=tool.parameters.get("required", []),
                    ),
                )
            )
        return protos.Tool(function_declarations=declarations)
