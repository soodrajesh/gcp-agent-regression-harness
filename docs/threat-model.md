# Threat model

The harness runs models and tools on behalf of whoever can call it, and compares their behavior; the risks are cost, misleading verdicts and
data exposure.

| Threat | Mitigation | Test |
|---|---|---|
| Runaway cost from a large run | per-hour run cap, max repeats and cases, cost guard that flags an over-budget run, provider allow-list | `test_api_job_lifecycle_and_limits`, `test_api_hourly_limit` |
| Use of a vendor key that must not be deployed | the deployed allow-list is `gemini,mock`; `openai` returns 403 (live-verified) | live check |
| A flaky run being called a regression | exact test with a significance bar, correction across cases, repeated runs | `test_one_flaky_failure_is_not_a_regression`, `test_benjamini_hochberg_correction` |
| Rate limits or outages being read as agent failures | retries with backoff; infra errors excluded; "no conclusion" when too few valid runs | `test_infrastructure_errors_are_excluded_from_the_statistics` |
| A safety regression hidden by noise correction | critical assertions bypass the cross-case correction and use a looser bar; one critical failure is still surfaced | `test_repeated_critical_failures_are_a_regression...`, `test_one_critical_failure_is_flagged...` |
| The judge model deciding the outcome | judge is informational only, randomised order, schema-validated; failures degrade to no opinion | `test_judge_is_informational_and_position_randomised`, `test_judge_garbage_is_degraded_not_fatal` |
| Tool code abuse | the bundled calculator is an AST-restricted arithmetic evaluator; tools are deterministic and side effects (refunds) are in-memory per run | `test_calculator_is_arithmetic_only`, `test_toolbox_isolated_side_effects...` |
| Sensitive prompts/answers at rest | results live in memory only, 30 most recent runs | by design |
| Unauthorised use | Cloud Run IAM, one invoker | live: unauthenticated = 403 |

Residual risk: the bundled suite is small and written by the author; results on your own suite are what matters. A caller who can run comparisons
spends your Vertex quota up to the guards above.
