# Runbook 04: investigate a regression

1. Open the case. Read the reason line: pass-rate drop with its p-value, or a critical assertion that held on the baseline.
2. Compare the tool sequences (most common first) and the example traces side by side. Look for argument changes and `novel` calls.
3. Check "failed assertions" on both sides: a baseline that also fails something is a flaky check, not necessarily a regression.
4. Re-run with more repeats (10) to firm up a borderline p-value.
5. If the cause is the prompt, fix it and re-run the same comparison; if it is the model, decide whether to accept, mitigate in the prompt, or stay on the baseline.
