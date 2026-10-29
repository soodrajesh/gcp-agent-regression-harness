"""Model providers with tool calling behind one tiny interface, plus a scripted one for tests."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

PRICES = {  # USD per 1M tokens (input, output); estimates for budgeting only
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.0),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
}


@dataclass
class ToolUse:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ModelTurn:
    text: str = ""
    tool_calls: list[ToolUse] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0

    def cost(self, model: str) -> float:
        pin, pout = PRICES.get(model, (0.30, 2.50))
        return (self.tokens_in * pin + self.tokens_out * pout) / 1e6


class ProviderError(RuntimeError):
    pass


class Provider:
    name = "base"
    model = "base"

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
        temperature: float,
    ) -> ModelTurn:
        raise NotImplementedError


class ScriptedProvider(Provider):
    """fn(messages, tools) -> ModelTurn. Used by tests and the mock provider."""

    name = "mock"

    def __init__(
        self,
        fn: Callable[[list[dict[str, Any]], list[dict[str, Any]]], ModelTurn],
        model: str = "mock-1",
    ):
        self.fn, self.model = fn, model

    async def generate(self, messages, tools, system, temperature):
        return self.fn(messages, tools)


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, model: str):
        from google import genai

        self.model = model
        self._c = genai.Client(
            vertexai=True,
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=os.environ.get("AGENTREG_GCP_LOCATION", "us-central1"),
        )

    async def generate(self, messages, tools, system, temperature):
        from google.genai import types

        decls = [
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=t["parameters"] if t["parameters"].get("properties") else None,
            )
            for t in tools
        ]
        contents: list[types.Content] = []
        for m in messages:
            if m["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=m["content"])]))
            elif m["role"] == "assistant":
                parts = [types.Part(text=m["content"])] if m.get("content") else []
                parts += [
                    types.Part(function_call=types.FunctionCall(name=c["name"], args=c["args"]))
                    for c in m.get("tool_calls", [])
                ]
                contents.append(types.Content(role="model", parts=parts))
            else:  # tool results, one message per turn of calls
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=r["name"], response={"output": r["content"]}
                            )
                            for r in m["results"]
                        ],
                    )
                )
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=1500,
            tools=[types.Tool(function_declarations=decls)] if decls else None,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        try:
            r = await self._c.aio.models.generate_content(
                model=self.model, contents=contents, config=cfg
            )
        except Exception as e:
            raise ProviderError(f"gemini: {type(e).__name__}: {str(e)[:160]}") from e
        parts = (
            (r.candidates[0].content.parts or [])
            if r.candidates and r.candidates[0].content
            else []
        )
        calls = [
            ToolUse(
                id=f"{p.function_call.name}-{i}",
                name=p.function_call.name,
                args=dict(p.function_call.args or {}),
            )
            for i, p in enumerate(x for x in parts if x.function_call)
        ]
        text = "".join(p.text for p in parts if getattr(p, "text", None))
        u = r.usage_metadata
        return ModelTurn(
            text,
            calls,
            (u.prompt_token_count or 0) if u else 0,
            (u.candidates_token_count or 0) if u else 0,
        )


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, model: str):
        from openai import AsyncOpenAI

        self.model = model
        self._c = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    async def generate(self, messages, tools, system, temperature):
        import json

        msgs: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                msgs.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                am: dict[str, Any] = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    am["tool_calls"] = [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": json.dumps(c["args"])},
                        }
                        for c in m["tool_calls"]
                    ]
                msgs.append(am)
            else:
                msgs += [
                    {"role": "tool", "tool_call_id": r["id"], "content": r["content"]}
                    for r in m["results"]
                ]
        kw: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "max_completion_tokens": 2500,
            "reasoning_effort": "minimal",
        }
        if tools:
            kw["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["parameters"],
                    },
                }
                for t in tools
            ]
        try:
            r = await self._c.chat.completions.create(**kw)
        except Exception as e:
            raise ProviderError(f"openai: {type(e).__name__}: {str(e)[:160]}") from e
        m = r.choices[0].message
        calls = [
            ToolUse(id=c.id, name=c.function.name, args=json.loads(c.function.arguments or "{}"))
            for c in (m.tool_calls or [])
        ]
        return ModelTurn(
            m.content or "",
            calls,
            r.usage.prompt_tokens if r.usage else 0,
            r.usage.completion_tokens if r.usage else 0,
        )


def make_provider(provider: str, model: str) -> Provider:
    if provider == "gemini":
        return GeminiProvider(model)
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise ProviderError("set OPENAI_API_KEY for the openai provider")
        return OpenAIProvider(model)
    if provider == "mock":
        return ScriptedProvider(lambda m, t: ModelTurn("mock answer"), model)
    raise ProviderError(f"unknown provider {provider}")
