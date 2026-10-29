"""Deterministic checks over a trace. These, not a model's opinion, decide pass/fail."""

from __future__ import annotations

import re

from .models import Assertion, AssertionResult, Trace


def _subset(args: dict, want: dict) -> bool:
    return all(str(args.get(k, "")).lower() == str(v).lower() for k, v in want.items())


def check(a: Assertion, tr: Trace) -> AssertionResult:
    ans = tr.final.lower()
    names = [c.name for c in tr.calls]
    ok, detail = True, ""
    if a.type == "tool_called":
        n = sum(1 for c in tr.calls if c.name == a.name and _subset(c.args, a.args_subset))
        ok = n >= max(1, int(a.n or 1))
        detail = f"{a.name}{a.args_subset or ''} called {n}x"
    elif a.type == "tool_not_called":
        n = sum(1 for c in tr.calls if c.name == a.name and _subset(c.args, a.args_subset))
        ok, detail = n == 0, f"{a.name} called {n}x"
    elif a.type == "answer_contains":
        ok = any(s.lower() in ans for s in a.any_of)
        detail = f"answer contains one of {a.any_of}: {ok}"
    elif a.type == "answer_not_contains":
        hit = [s for s in a.any_of if s.lower() in ans]
        ok, detail = not hit, f"forbidden phrases present: {hit}" if hit else "none present"
    elif a.type == "answer_regex":
        ok = re.search(a.pattern, tr.final, re.I | re.S) is not None
        detail = f"/{a.pattern}/"
    elif a.type == "max_tool_calls":
        ok, detail = len(names) <= a.n, f"{len(names)} calls (max {int(a.n)})"
    elif a.type == "max_cost_usd":
        ok, detail = tr.usage.cost_usd <= a.n, f"${tr.usage.cost_usd:.5f} (max ${a.n})"
    elif a.type == "max_latency_ms":
        ok, detail = tr.latency_ms <= a.n, f"{tr.latency_ms} ms (max {int(a.n)})"
    elif a.type == "calls_before":
        i = next((k for k, x in enumerate(names) if x == a.name), None)
        j = next((k for k, x in enumerate(names) if x == a.then), None)
        ok = i is not None and (j is None or i < j)
        detail = f"{a.name} at {i}, {a.then} at {j}"
    elif a.type == "no_error":
        ok, detail = (
            not tr.error and bool(tr.final),
            tr.error or ("empty answer" if not tr.final else "ok"),
        )
    return AssertionResult(
        type=a.type + (f":{a.name}" if a.name else ""), ok=ok, detail=detail, critical=a.critical
    )


def check_all(assertions: list[Assertion], tr: Trace) -> list[AssertionResult]:
    res = [check(Assertion(type="no_error"), tr)] + [check(a, tr) for a in assertions]
    return res
