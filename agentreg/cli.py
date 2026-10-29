"""CLI.  Exit codes: 0 ok, 1 changed, 2 regression, 3 error.

agentreg compare suite.json --baseline gemini:gemini-2.5-flash --candidate gemini:gemini-2.5-flash-lite --repeats 5
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from pydantic import ValidationError

from .compare import compare_suites
from .models import Config, Report, Suite
from .providers import ProviderError, make_provider

EXIT = {"ok": 0, "changed": 1, "regression": 2}
COLOR = {"regression": "1;31", "changed": "33", "improvement": "32", "same": "2"}


def parse_cfg(spec: str, system: str | None, label: str) -> Config:
    prov, _, model = spec.partition(":")
    return Config(label=label, provider=prov, model=model or "gemini-2.5-flash", system=system)  # type: ignore[arg-type]


def render(r: Report, color: bool) -> str:
    def c(code, s):
        return f"\033[{code}m{s}\033[0m" if color else s

    head = {"ok": "32", "changed": "33", "regression": "1;31"}[r.verdict]
    out = [
        c(head, f"VERDICT: {r.verdict.upper()}"),
        r.summary,
        f"baseline {r.baseline.model} vs candidate {r.candidate.model}{' (different system prompt)' if r.candidate.system else ''}"
        f" · ${r.cost_usd:.4f} · {r.duration_ms / 1000:.0f}s",
        "",
    ]
    for cs in r.cases:
        out.append(
            f"{c(COLOR[cs.verdict], cs.verdict.upper().ljust(11))} {cs.case_id.ljust(22)} "
            f"pass {cs.base.passed}/{cs.base.runs} -> {cs.cand.passed}/{cs.cand.runs}   calls {cs.base.mean_calls} -> {cs.cand.mean_calls}"
        )
        for reason in cs.reasons:
            out.append(f"{' ' * 12}{reason}")
        for k, v in cs.cand.failed_assertions.items():
            out.append(f"{' ' * 12}candidate failed {k} in {v}/{cs.cand.runs} runs")
    for d in r.degraded:
        out.append(c("33", f"! {d}"))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agentreg")
    sub = ap.add_subparsers(dest="cmd", required=True)
    cp = sub.add_parser("compare", help="run a suite against a baseline and a candidate")
    cp.add_argument("suite")
    cp.add_argument("--baseline", default="gemini:gemini-2.5-flash", help="provider:model")
    cp.add_argument("--candidate", default="gemini:gemini-2.5-flash-lite", help="provider:model")
    cp.add_argument(
        "--candidate-system-file", help="use a different system prompt for the candidate"
    )
    cp.add_argument("--repeats", type=int, default=5)
    cp.add_argument("--judge", help="provider:model for the optional answer-quality judge")
    cp.add_argument("--alpha", type=float, default=0.1)
    cp.add_argument("--format", default="text", choices=["text", "json"])
    a = ap.parse_args(argv)
    try:
        suite = Suite.model_validate_json(Path(a.suite).read_text())
        cand_sys = (
            Path(a.candidate_system_file).read_text().strip() if a.candidate_system_file else None
        )
        judge = make_provider(*a.judge.split(":", 1)) if a.judge else None
        rep = asyncio.run(
            compare_suites(
                suite,
                parse_cfg(a.baseline, None, "baseline"),
                parse_cfg(a.candidate, cand_sys, "candidate"),
                a.repeats,
                judge_provider=judge,
                alpha=a.alpha,
            )
        )
    except (OSError, ValueError, ValidationError, ProviderError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    print(rep.model_dump_json(indent=1) if a.format == "json" else render(rep, sys.stdout.isatty()))
    return EXIT[rep.verdict]


if __name__ == "__main__":
    sys.exit(main())
