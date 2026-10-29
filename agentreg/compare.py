"""Run a suite against a baseline and a candidate and decide, per case and overall."""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import UTC, datetime

from . import diff
from .assertions import check_all
from .judge import compare as judge_compare
from .models import CaseComparison, Config, Report, Run, Suite
from .providers import Provider, make_provider
from .runner import run_case
from .stats import bh_significant, fisher_one_sided_lower
from .tools import ToolBox

ALPHA = 0.1
CRITICAL_ALPHA = 0.25  # more sensitive than ALPHA for safety checks, still not "any single failure"


def verdict_for(base_runs: list[Run], cand_runs: list[Run], alpha: float = ALPHA):
    bp, cp = sum(r.passed for r in base_runs), sum(r.passed for r in cand_runs)
    bn, cn = len(base_runs), len(cand_runs)
    p_reg = fisher_one_sided_lower(bp, bn, cp, cn)
    p_imp = fisher_one_sided_lower(cp, cn, bp, bn)
    reasons: list[str] = []
    # Critical (safety) assertions get a more sensitive test, but not "any single failure": a calibration
    # run of the same model against itself showed one flaky failure in a handful of runs is common.
    crit_b = sum(all(a.ok for a in r.results if a.critical) for r in base_runs)
    crit_c = sum(all(a.ok for a in r.results if a.critical) for r in cand_runs)
    p_crit = fisher_one_sided_lower(crit_b, bn, crit_c, cn)
    crit_rate_drop = crit_c / max(cn, 1) < crit_b / max(bn, 1)
    if crit_rate_drop and p_crit < CRITICAL_ALPHA:
        reasons.append(
            f"critical assertions held in {crit_b}/{bn} baseline runs but {crit_c}/{cn} candidate runs (p={p_crit:.3f})"
        )
    if p_reg < alpha and cp / max(cn, 1) < bp / max(bn, 1):
        reasons.append(f"pass rate {bp}/{bn} -> {cp}/{cn} (p={p_reg:.3f})")
    if reasons:
        return "regression", reasons, p_reg, p_imp
    if (
        crit_rate_drop
    ):  # below the bar for a regression, but a safety check failed where it never did before
        return (
            "changed",
            [
                f"a critical assertion failed in {cn - crit_c}/{cn} candidate runs (baseline {bn - crit_b}/{bn}); not significant"
            ],
            p_reg,
            p_imp,
        )
    if p_imp < alpha and cp / max(cn, 1) > bp / max(bn, 1):
        return "improvement", [f"pass rate {bp}/{bn} -> {cp}/{cn} (p={p_imp:.3f})"], p_reg, p_imp
    return "same", [], p_reg, p_imp


async def run_side(
    cfg: Config,
    provider: Provider,
    suite: Suite,
    repeats: int,
    sem: asyncio.Semaphore,
    progress=None,
) -> dict[str, list[Run]]:
    system = cfg.system if cfg.system is not None else suite.system

    async def one(case):
        async with sem:
            tb = ToolBox(suite.tools)
            tr = await run_case(provider, system, tb, case, cfg.temperature)
            if progress:
                progress()
            return Run(trace=tr, results=check_all(case.assertions, tr))

    out: dict[str, list[Run]] = {}
    jobs = [(c, one(c)) for c in suite.cases for _ in range(repeats)]
    done = await asyncio.gather(*(j for _, j in jobs))
    for (c, _), r in zip(jobs, done, strict=True):
        out.setdefault(c.id, []).append(r)
    return out


async def compare_suites(
    suite: Suite,
    baseline: Config,
    candidate: Config,
    repeats: int = 5,
    *,
    base_provider: Provider | None = None,
    cand_provider: Provider | None = None,
    judge_provider: Provider | None = None,
    concurrency: int = 6,
    alpha: float = ALPHA,
    progress=None,
) -> Report:
    t0 = time.perf_counter()
    bp = base_provider or make_provider(baseline.provider, baseline.model)
    cp = cand_provider or make_provider(candidate.provider, candidate.model)
    sem = asyncio.Semaphore(concurrency)
    base, cand = await asyncio.gather(
        run_side(baseline, bp, suite, repeats, sem, progress),
        run_side(candidate, cp, suite, repeats, sem, progress),
    )
    cases: list[CaseComparison] = []
    degraded: list[str] = []
    for case in suite.cases:
        br_all, cr_all = base[case.id], cand[case.id]
        br = [r for r in br_all if not r.trace.infra_error]
        cr = [r for r in cr_all if not r.trace.infra_error]
        v, reasons, p_reg, p_imp = verdict_for(br, cr, alpha)
        bs, cs = diff.side_stats(br), diff.side_stats(cr)
        bs.excluded, cs.excluded = len(br_all) - len(br), len(cr_all) - len(cr)
        if bs.excluded or cs.excluded:
            degraded.append(
                f"{case.id}: {bs.excluded + cs.excluded} run(s) excluded from the statistics because of provider errors"
            )
        if len(br) < 2 or len(cr) < 2:
            v, reasons, p_reg, p_imp = (
                "changed",
                [
                    f"too few valid runs (baseline {len(br)}, candidate {len(cr)}) after provider errors; no conclusion"
                ],
                1.0,
                1.0,
            )
        bm, cm = diff.modal(br), diff.modal(cr)
        sim = diff.similarity(bm.trace.final, cm.trace.final) if bm and cm else 1.0
        adiffs = diff.arg_diffs(bm.trace, cm.trace) if bm and cm else []
        if v == "same":
            top_b = next(iter(bs.tool_sequences), "")
            top_c = next(iter(cs.tool_sequences), "")
            if top_b != top_c:
                v, reasons = "changed", [f"most common tool sequence changed: {top_b} -> {top_c}"]
            elif sim < 0.5:
                v, reasons = "changed", [f"final answer differs a lot (similarity {sim})"]
            elif adiffs:
                v, reasons = "changed", ["same tools, different arguments"]
            elif bs.mean_cost_usd and cs.mean_cost_usd > 1.5 * bs.mean_cost_usd:
                v, reasons = (
                    "changed",
                    [f"cost per run up {cs.mean_cost_usd / bs.mean_cost_usd:.1f}x"],
                )
        jd = None
        if judge_provider and bm and cm:
            jd = await judge_compare(
                judge_provider, case.prompt, bm.trace.final, cm.trace.final, seed=case.id
            )
            if jd is None:
                degraded.append(f"judge unavailable for {case.id}")
        cases.append(
            CaseComparison(
                case_id=case.id,
                tags=case.tags,
                base=bs,
                cand=cs,
                p_regress=round(p_reg, 4),
                p_improve=round(p_imp, 4),
                verdict=v,
                reasons=reasons,
                arg_diffs=adiffs,
                answer_similarity=sim,
                judge=jd,
                base_example=bm.trace if bm else None,
                cand_example=cm.trace if cm else None,
            )
        )
    # Many cases are tested at once: keep ordinary regression/improvement calls only if they survive a
    # Benjamini-Hochberg correction. Critical-assertion regressions bypass it (safety is not traded for fewer alarms).
    for direction, attr in (("regression", "p_regress"), ("improvement", "p_improve")):
        sig = bh_significant([getattr(c, attr) for c in cases], alpha)
        for c, ok in zip(cases, sig, strict=True):
            critical = any(x.startswith("critical assertions held") for x in c.reasons)
            if c.verdict == direction and not ok and not critical:
                c.verdict = "changed"
                c.reasons = [
                    *c.reasons,
                    f"not significant after correcting for {len(cases)} cases (FDR {alpha})",
                ]
    counts = {
        k: sum(c.verdict == k for c in cases)
        for k in ("same", "changed", "regression", "improvement")
    }
    overall = "regression" if counts["regression"] else "changed" if counts["changed"] else "ok"
    cost = sum(r.trace.usage.cost_usd for runs in (*base.values(), *cand.values()) for r in runs)
    errs = sum(c.base.errors + c.cand.errors for c in cases)
    if errs:
        degraded.append(f"{errs} run(s) ended with a provider or loop error (counted as failures)")
    summary = (
        f"{counts['regression']} regression(s), {counts['improvement']} improvement(s), {counts['changed']} changed, "
        f"{counts['same']} unchanged across {len(cases)} case(s) x {repeats} runs."
    )
    return Report(
        id=uuid.uuid4().hex[:12],
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        suite=suite.name,
        baseline=baseline,
        candidate=candidate,
        repeats=repeats,
        verdict=overall,
        summary=summary,
        cases=cases,
        counts=counts,
        cost_usd=round(cost, 5),
        duration_ms=int((time.perf_counter() - t0) * 1000),
        degraded=degraded,
        alpha=alpha,
    )
