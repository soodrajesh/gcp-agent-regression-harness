# Architecture

```
suite.json (system prompt, tools, cases + assertions)
        |
   for each side (baseline, candidate):  cases x repeats, concurrently, one shared semaphore
        |   [runner]  agent loop: model turn -> tool calls -> tool results -> ... -> final answer
        |             retry transient provider errors (429/503) with backoff; infra errors are marked, not counted as failures
        |   [assertions]  deterministic checks over the trace (tool called/not called, order, answer text, cost, latency)
        v
   per case: pass counts for both sides
        |   [stats]   one-sided Fisher exact test: is the candidate's pass rate lower than chance explains?
        |             critical (safety) assertions use a more sensitive bar (alpha 0.25 vs 0.1)
        |   [BH]      Benjamini-Hochberg across cases so ten cases do not produce a false alarm each
        |   [diff]    tool-sequence changes, argument changes, answer similarity, cost/latency/token deltas
        |   [judge]   optional model comparison of answers (randomised A/B order): informational only
        v
   verdict per case: regression | improvement | changed | same   ->   overall: regression | changed | ok
```

**Trace** (`models.Trace`): the prompt, every message, each tool call with arguments and result, the final answer, token usage, estimated cost,
latency and any error. Tools are deterministic in the bundled example (`tools.py`), and `ReplayToolBox` can serve recorded results for recorded
calls, flagging any call the recording never made as `novel` (so a candidate that takes a different path is visible rather than silently fed
fresh data).

**Assertions decide, models do not.** Pass/fail comes from `assertions.py`. The optional judge only annotates a case; it cannot change a verdict.

**Why statistics.** Sampled agents are flaky. A single failed run is not a regression. Each case is run N times per side; a regression needs a
pass-rate drop that chance would rarely produce (Fisher exact, one-sided), corrected across cases. Critical assertions (for example "never refund
an order that is not delivered") use a looser threshold, and a single critical failure is surfaced as `changed` with the counts rather than ignored.

**Infrastructure errors** (rate limits, outages) are retried, then excluded from the statistics and listed in the report. If fewer than two valid
runs remain on a side the case reports "no conclusion" instead of a verdict.

**Providers** (`providers.py`): Gemini on Vertex AI, OpenAI (test/eval only, never deployed) and a scripted provider for tests. All take the same
neutral message format; cost is estimated from a price table for budgeting.

**Service** (`api.py`): a comparison is a background job (`POST /api/runs` returns an id; `GET /api/runs/{id}` reports progress and the result).
Limits: runs per hour, repeats and cases per run, a cost guard, and an allow-list of providers. Results are in memory only, so the Cloud Run
service is pinned to one instance with CPU always allocated.
