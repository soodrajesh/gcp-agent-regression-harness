"""HTTP API and viewer. Auth is Cloud Run IAM. A run is a background job (a comparison takes tens of
seconds); results live in memory only."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import OrderedDict
from importlib import resources

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .compare import compare_suites
from .config import Settings
from .models import Config, Report, Suite
from .providers import PRICES, Provider, ProviderError, make_provider

SAMPLES = resources.files("agentreg") / "samples"


class RunRequest(BaseModel):
    suite: dict | None = None
    sample: str | None = None
    baseline: Config
    candidate: Config
    candidate_prompt: str | None = None  # name from /api/prompts, or None
    repeats: int = Field(default=5, ge=1)
    judge: bool = False


class Job(BaseModel):
    id: str
    status: str = "running"  # running | done | error
    done: int = 0
    total: int = 0
    error: str = ""
    report: Report | None = None


def create_app(settings: Settings | None = None, provider_factory=None) -> FastAPI:
    cfg = settings or Settings()
    make = provider_factory or make_provider
    app = FastAPI(title="agentreg", version=__version__)
    jobs: OrderedDict[str, Job] = OrderedDict()
    starts: list[float] = []
    tasks: set[asyncio.Task] = set()

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "version": __version__}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (resources.files("agentreg") / "ui.html").read_text()

    @app.get("/api/config")
    def config():
        return {
            "version": __version__,
            "providers": cfg.allowed_providers,
            "models": sorted(PRICES),
            "max_repeats": cfg.max_repeats,
            "runs_per_hour": cfg.max_runs_per_hour,
        }

    @app.get("/api/samples")
    def samples():
        suite = json.loads((SAMPLES / "order_support.json").read_text())
        return {"suites": ["order_support"], "cases": [c["id"] for c in suite["cases"]]}

    @app.get("/api/prompts")
    def prompts():
        return json.loads((SAMPLES / "prompts.json").read_text())

    def provider(c: Config) -> Provider:
        if c.provider not in cfg.allowed_providers:
            raise HTTPException(
                403,
                f"provider '{c.provider}' not allowed here ({', '.join(cfg.allowed_providers)})",
            )
        try:
            return make(c.provider, c.model)
        except ProviderError as e:
            raise HTTPException(503, str(e)) from e

    @app.post("/api/runs", status_code=202)
    async def start(req: Request):
        raw = await req.body()
        if len(raw) > cfg.max_body_bytes:
            raise HTTPException(413, "too large")
        try:
            body = RunRequest.model_validate_json(raw)
            doc = body.suite or json.loads(
                (SAMPLES / f"{body.sample or 'order_support'}.json").read_text()
            )
            suite = Suite.model_validate(doc)
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(422, f"invalid request: {str(e)[:300]}") from e
        if body.repeats > cfg.max_repeats or len(suite.cases) > cfg.max_cases:
            raise HTTPException(
                422, f"limits: repeats <= {cfg.max_repeats}, cases <= {cfg.max_cases}"
            )
        now = time.time()
        starts[:] = [t for t in starts if now - t < 3600]
        if len(starts) >= cfg.max_runs_per_hour:
            raise HTTPException(429, "hourly run limit reached")
        if body.candidate_prompt:
            named = json.loads((SAMPLES / "prompts.json").read_text())
            if body.candidate_prompt not in named:
                raise HTTPException(422, "unknown candidate_prompt")
            body.candidate.system = named[body.candidate_prompt]
        bp, cp = provider(body.baseline), provider(body.candidate)
        jp = (
            provider(Config(provider=body.baseline.provider, model=body.baseline.model))
            if body.judge
            else None
        )
        starts.append(now)
        job = Job(id=f"{int(now)}{len(jobs):03d}", total=len(suite.cases) * body.repeats * 2)
        jobs[job.id] = job
        while len(jobs) > 30:
            jobs.popitem(last=False)

        async def work():
            try:

                def tick():
                    job.done += 1

                rep = await compare_suites(
                    suite,
                    body.baseline,
                    body.candidate,
                    body.repeats,
                    base_provider=bp,
                    cand_provider=cp,
                    judge_provider=jp,
                    concurrency=cfg.concurrency,
                    progress=tick,
                )
                if rep.cost_usd > cfg.max_cost_usd:
                    rep.degraded.append(
                        f"run cost ${rep.cost_usd:.3f} exceeded the ${cfg.max_cost_usd} guard"
                    )
                job.report, job.status = rep, "done"
            except Exception as e:  # keep the job visible with an error instead of losing it
                logging.getLogger("agentreg").exception("run failed")
                job.status, job.error = "error", f"{type(e).__name__}: {str(e)[:200]}"

        t = asyncio.create_task(work())
        tasks.add(t)
        t.add_done_callback(tasks.discard)
        return {"id": job.id}

    @app.get("/api/runs")
    def list_runs():
        return [
            {
                "id": j.id,
                "status": j.status,
                "done": j.done,
                "total": j.total,
                "verdict": j.report.verdict if j.report else None,
                "suite": j.report.suite if j.report else "",
            }
            for j in reversed(jobs.values())
        ]

    @app.get("/api/runs/{jid}")
    def get_run(jid: str):
        j = jobs.get(jid)
        if j is None:
            raise HTTPException(404, "no such run")
        return j

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        logging.getLogger("agentreg").exception("unhandled error", exc_info=exc)
        return JSONResponse({"detail": "internal error"}, status_code=500)

    return app


def app_factory() -> FastAPI:
    return create_app()
