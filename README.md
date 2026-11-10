# gcp-agent-regression-harness

"Is the new model / prompt worse?" is hard to answer for an agent: runs are sampled, so one bad run proves nothing, and the interesting
differences are in *behavior* (which tools it called, with what arguments, whether it did something it must never do), not just the final text.

This harness runs the same suite against a **baseline** and a **candidate** (a different model, or the same model with a different system
prompt), checks deterministic assertions over every trace, and uses statistics to say **regression / improvement / changed / same** per case
and overall. It is built to be a CI gate (exit codes) and has a viewer.

![Comparison live on Cloud Run: a careless prompt refunds an order delivered 45 days ago](docs/img/comparison-live.png)

## How it works

```
suite.json (system prompt, tools, cases with assertions)
   -> run each case N times on each side            (retry 429/503; infrastructure errors excluded, not counted as failures)
   -> assertions over each trace                     (tool called / not called / ordering / answer text / cost / latency)
   -> per case: one-sided Fisher exact test on pass rates, Benjamini-Hochberg across cases
                critical (safety) assertions use a looser bar; one critical failure is still surfaced
   -> behavior diff: tool-sequence changes, argument changes, answer similarity, cost/latency/token deltas
   -> optional model judge of answer quality: informational only
```

Assertions, not a model's opinion, decide pass/fail. Mark checks that must never break as `"critical": true`
(for example "do not refund an order that is not delivered").

## Architecture

[![Architecture: private Cloud Run service, engine, read-only sources, optional Gemini, up.sh delivery](docs/img/architecture.png)](docs/img/architecture.svg)

Regenerate with `python3 docs/diagrams/architecture.py` then `python3 docs/diagrams/render.py docs/img/architecture.svg docs/img/architecture.png` (needs Playwright).

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
GOOGLE_CLOUD_PROJECT=<project> .venv/bin/agentreg compare agentreg/samples/order_support.json \
    --baseline gemini:gemini-2.5-flash --candidate gemini:gemini-2.5-flash --candidate-system-file new_prompt.txt --repeats 5
echo "exit=$?"       # 0 ok, 1 changed, 2 regression, 3 error

AGENTREG_ALLOWED_PROVIDERS=gemini,mock GOOGLE_CLOUD_PROJECT=<project> \
  .venv/bin/uvicorn agentreg.api:app_factory --factory --port 8080                  # viewer + API
```

Providers: `gemini` (Vertex AI, uses ADC), `openai` (needs `OPENAI_API_KEY`; for local tests and evals only), `mock`.
The bundled example is an order-support agent with five deterministic tools (lookups, calculator, policy search, refund) and a 10-case suite
that includes refund-safety cases. To test your own agent, implement `ToolBox` (`decls()` and `invoke()`), see [runbook 02](docs/runbooks/02-write-a-suite.md).

## Deploy to GCP

```bash
cp deploy.env.example deploy.env && ./scripts/up.sh
gcloud run services proxy agent-regression --region us-central1 --port 8081
./scripts/down.sh
```

Terraform creates APIs, Artifact Registry, a build identity, a runtime service account (`aiplatform.user`, `logging.logWriter`) and a
private Cloud Run service (one instance, CPU always allocated, one invoker, no `allUsers`). Guards: runs per hour, repeats and cases per run, a
cost guard, provider allow-list (`gemini`, `mock`; `openai` is refused). Results are in memory only.

Docs: [architecture](docs/architecture.md) · [threat model](docs/threat-model.md) · [runbooks](docs/runbooks/) · [ADRs](docs/adr/) · [eval results](docs/eval-results.md)

## Verification

| Check | Result |
|---|---|
| Offline tests | 34 pass: assertions, tools and replay, agent loop and retries, exact statistics, per-case verdicts, scripted careless/good agents, judge, API job lifecycle and limits, CLI |
| **A/A calibration** (same model against itself, 4 runs, 40 case-comparisons) | **0 false regressions, 0 false improvements**; 3-5 of 10 cases per run are reported as `changed` (behavioral noise shown, not alarmed on) |
| Careless system prompt vs careful prompt | **regression** caught: refunding an order delivered 45 days ago passed on the baseline 5/5 and on the candidate 1/5 (p=0.024), found with 3 and with 5 runs per case, locally and on Cloud Run |
| flash vs flash-lite, flash vs gpt-5-nano | each reports regressions (1 and 2 cases) with p-values and example traces; see [eval results](docs/eval-results.md) |
| Deployed | unauthenticated = 403; `openai` = 403; a full comparison runs as a background job (about $0.02 for 5 runs x 10 cases x 2 sides) |

### What running it for real taught me (all fixed, test-covered)

1. A single flaky safety failure was called a regression in a same-model comparison: critical assertions now use a looser *test*, not "any failure".
2. Ten cases at alpha 0.1 produced false "improvements": added Benjamini-Hochberg correction across cases.
3. Rate-limit errors from running two jobs at once made a healthy baseline look broken: provider errors are now retried, excluded from the
   statistics and listed; too few valid runs gives "no conclusion".

## Limits

- The bundled suite is small (10 cases) and written by the author. The A/A noise floor and the detection results are for this suite and these models.
- Statistics assume independent runs and a handful of repeats; with fewer than 5 runs a real regression often cannot reach significance.
- Assertions are only as good as you write them; the harness cannot notice a regression you did not assert on, beyond the reported behavior diffs.
- "changed" is deliberately noisy (answer similarity, tool-sequence differences); treat it as a prompt to look, not a failure.
- Replay mode serves recorded tool results for recorded calls only; a candidate that takes a different path hits live tools (and is flagged `novel`).
  Non-deterministic tools make comparisons noisier.
- Costs are estimates from a price table; the service is single-instance with in-memory results.
