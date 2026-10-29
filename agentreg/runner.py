"""The agent loop that produces a Trace. The same loop records a baseline and replays a candidate,
so differences in behavior come from the model/prompt, not from the harness."""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any

from .models import Call, Case, Trace, Usage
from .providers import Provider, ProviderError

MAX_TURNS = 8
RETRIES = 4
BACKOFF_S = 2.0
TRANSIENT = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "timeout", "Timeout", "500")


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


async def generate_with_retry(provider, messages, decls, system, temperature):
    for attempt in range(RETRIES):
        try:
            return await provider.generate(messages, decls, system, temperature)
        except ProviderError as e:
            if attempt == RETRIES - 1 or not any(t in str(e) for t in TRANSIENT):
                raise
            await asyncio.sleep(BACKOFF_S * (2**attempt))


async def run_case(
    provider: Provider,
    system: str,
    toolbox: Any,
    case: Case,
    temperature: float = 1.0,
    max_turns: int = MAX_TURNS,
) -> Trace:
    t0 = time.perf_counter()
    tr = Trace(
        case_id=case.id,
        provider=provider.name,
        model=provider.model,
        prompt=case.prompt,
        system_sha=sha(system),
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": case.prompt}]
    decls = toolbox.decls()
    cost = 0.0
    try:
        for _ in range(max_turns):
            turn = await generate_with_retry(provider, messages, decls, system, temperature)
            tr.turns += 1
            tr.usage.tokens_in += turn.tokens_in
            tr.usage.tokens_out += turn.tokens_out
            cost += turn.cost(provider.model)
            messages.append(
                {
                    "role": "assistant",
                    "content": turn.text,
                    "tool_calls": [
                        {"id": c.id, "name": c.name, "args": c.args} for c in turn.tool_calls
                    ],
                }
            )
            if not turn.tool_calls:
                tr.final = turn.text.strip()
                break
            results = []
            for c in turn.tool_calls:
                call = Call(id=c.id, name=c.name, args=c.args)
                novel_before = len(getattr(toolbox, "novel", []))
                try:
                    call.result = toolbox.invoke(c.name, c.args)
                except Exception as e:  # a tool crash is data for the model, not a harness failure
                    call.error = f"{type(e).__name__}: {str(e)[:120]}"
                    call.result = f"error: {call.error}"
                call.novel = len(getattr(toolbox, "novel", [])) > novel_before
                tr.calls.append(call)
                results.append({"id": c.id, "name": c.name, "content": call.result})
            messages.append({"role": "tool", "results": results})
        else:
            tr.error = f"no final answer within {max_turns} turns"
    except ProviderError as e:
        tr.error, tr.infra_error = str(e), True
    tr.usage = Usage(
        tokens_in=tr.usage.tokens_in, tokens_out=tr.usage.tokens_out, cost_usd=round(cost, 6)
    )
    tr.messages = messages
    tr.latency_ms = int((time.perf_counter() - t0) * 1000)
    return tr
