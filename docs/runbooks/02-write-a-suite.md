# Runbook 02: write a suite

A suite is JSON: `name`, `system` (the prompt under test), `tools` (names your toolbox implements) and `cases`.
Each case has a `prompt` and `assertions`. See `agentreg/samples/order_support.json`.

Guidance:
- Assert behavior, not wording: `tool_called` / `tool_not_called` with `args_subset`, `calls_before`, and short `answer_contains` any-of lists.
- Mark the checks that must never break (for example "does not issue a refund for an undelivered order") `"critical": true`.
- Include cases where the right behavior is to do nothing or ask a question (`tool_not_called`, `answer_regex`).
- Use at least 5 runs per case; with fewer, a real regression cannot reach significance.
- To use your own tools, implement the `ToolBox` interface (`decls()` and `invoke(name, args)`) and pass it where `ToolBox` is constructed in `compare.py`.
