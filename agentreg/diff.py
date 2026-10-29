"""Behavior diffs between two sets of runs of the same case."""

from __future__ import annotations

import difflib
import json
from collections import Counter

from .models import Run, SideStats, Trace


def sequence(tr: Trace) -> str:
    return ">".join(c.name for c in tr.calls) or "(no tools)"


def side_stats(runs: list[Run]) -> SideStats:
    n = len(runs) or 1
    seqs = Counter(sequence(r.trace) for r in runs)
    failed: Counter[str] = Counter()
    for r in runs:
        for a in r.results:
            if not a.ok:
                failed[a.type] += 1
    return SideStats(
        runs=len(runs),
        passed=sum(r.passed for r in runs),
        mean_calls=round(sum(len(r.trace.calls) for r in runs) / n, 2),
        mean_cost_usd=round(sum(r.trace.usage.cost_usd for r in runs) / n, 6),
        mean_latency_ms=round(sum(r.trace.latency_ms for r in runs) / n),
        mean_tokens=round(
            sum(r.trace.usage.tokens_in + r.trace.usage.tokens_out for r in runs) / n
        ),
        errors=sum(bool(r.trace.error) for r in runs),
        tool_sequences=dict(seqs.most_common(5)),
        failed_assertions=dict(failed),
    )


def modal(runs: list[Run]) -> Run | None:
    if not runs:
        return None
    top = Counter(sequence(r.trace) for r in runs).most_common(1)[0][0]
    return next(r for r in runs if sequence(r.trace) == top)


def arg_diffs(base: Trace, cand: Trace) -> list[str]:
    """Same tool at the same position with different arguments (e.g. a different order id)."""
    out = []
    for i, (b, c) in enumerate(zip(base.calls, cand.calls, strict=False)):
        if b.name == c.name and b.args != c.args:
            out.append(
                f"call {i + 1} {b.name}: {json.dumps(b.args, sort_keys=True)} -> {json.dumps(c.args, sort_keys=True)}"
            )
    return out


def similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return round(difflib.SequenceMatcher(None, a.lower().split(), b.lower().split()).ratio(), 3)
