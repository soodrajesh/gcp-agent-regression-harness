# ADR 0001: deterministic assertions decide; statistics gate the verdict

**Decision**: pass/fail comes from deterministic checks over the trace; a regression requires a significant pass-rate drop (exact test, corrected across cases). A model judge is informational.
**Why**: a model grading another model is itself noisy and biased; and a single sampled run is not evidence.
**Consequence**: you must write assertions; borderline cases need more repeats.
