# ADR 0002: calibrate against A/A runs and keep infrastructure errors out of the statistics

**Decision**: the eval compares a model with itself to measure the false-alarm rate; provider errors are retried and then excluded rather than counted as failures.
**Why**: the first A/A runs showed false regressions (a single flaky safety failure, uncorrected multiple comparisons) and a rate-limit confound that made a healthy baseline look broken. Each was fixed and test-covered.
**Consequence**: a single critical failure is reported as `changed`, not `regression`; see the README for the measured noise floor.
