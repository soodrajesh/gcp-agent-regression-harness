# Runbook 03: gate a model or prompt change in CI

```bash
agentreg compare suite.json --baseline gemini:gemini-2.5-flash --candidate gemini:gemini-2.5-flash \
    --candidate-system-file new_prompt.txt --repeats 5
```
Exit codes: 0 ok, 1 changed, 2 regression, 3 error. Fail the pipeline on 2; treat 1 as "review the diff". Run an A/A comparison
(same model and prompt on both sides) occasionally to see the noise floor for your suite; `evals/run.py` does this.
