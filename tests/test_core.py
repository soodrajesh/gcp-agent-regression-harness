import pytest

from agentreg.assertions import check, check_all
from agentreg.models import Assertion, Call, Trace, Usage
from agentreg.providers import ScriptedProvider
from agentreg.runner import run_case
from agentreg.stats import fisher_one_sided_lower, wilson
from agentreg.tools import ReplayToolBox, ToolBox, safe_eval

from .helpers import CASE, SUITE, good_agent, tool_results, turn


def tr(calls=(), final="", cost=0.0, ms=0, error=""):
    return Trace(
        case_id="c",
        provider="p",
        model="m",
        prompt="x",
        final=final,
        error=error,
        calls=[Call(name=n, args=a) for n, a in calls],
        usage=Usage(cost_usd=cost),
        latency_ms=ms,
    )


# --- tools ----------------------------------------------------------------------------------


def test_calculator_is_arithmetic_only():
    assert safe_eval("120 * 0.15") == 18.0
    for bad in ("__import__('os')", "open('x')", "1 if 1 else 2"):
        with pytest.raises((ValueError, SyntaxError)):
            safe_eval(bad)


def test_toolbox_isolated_side_effects_and_unknown_tool():
    a, b = ToolBox(["refund_order"]), ToolBox(["refund_order"])
    a.invoke("refund_order", {"order_id": "A-1", "reason": "x"})
    assert len(a.refunds) == 1 and b.refunds == []
    with pytest.raises(KeyError):
        a.invoke("get_order", {})


def test_replay_toolbox_serves_recorded_and_flags_novel_calls():
    live = ToolBox(["get_order"])
    rec = [Call(name="get_order", args={"order_id": "A-1001"}, result="RECORDED")]
    rb = ReplayToolBox(rec, live)
    assert rb.invoke("get_order", {"order_id": "A-1001"}) == "RECORDED" and rb.novel == []
    assert "A-1003" in rb.invoke("get_order", {"order_id": "A-1003"}) and len(rb.novel) == 1
    assert "no recorded result" in ReplayToolBox(rec, None).invoke("get_order", {"order_id": "Z"})


# --- assertions -----------------------------------------------------------------------------


def test_assertion_kinds():
    t = tr(
        [("get_order", {"order_id": "A-1001"}), ("refund_order", {"order_id": "A-1001"})],
        "Done. 30 days",
        0.01,
        500,
    )
    cases = [
        (Assertion(type="tool_called", name="get_order", args_subset={"order_id": "a-1001"}), True),
        (Assertion(type="tool_called", name="get_order", args_subset={"order_id": "A-9"}), False),
        (Assertion(type="tool_not_called", name="calculate"), True),
        (Assertion(type="answer_contains", any_of=["30 days"]), True),
        (Assertion(type="answer_not_contains", any_of=["secret"]), True),
        (Assertion(type="answer_regex", pattern=r"\d+ days"), True),
        (Assertion(type="max_tool_calls", n=1), False),
        (Assertion(type="max_cost_usd", n=0.05), True),
        (Assertion(type="max_latency_ms", n=100), False),
        (Assertion(type="calls_before", name="get_order", then="refund_order"), True),
        (Assertion(type="calls_before", name="refund_order", then="get_order"), False),
    ]
    for a, want in cases:
        assert check(a, t).ok is want, a.type


def test_no_error_requires_an_answer():
    assert not check(Assertion(type="no_error"), tr(final="")).ok
    assert not check(Assertion(type="no_error"), tr(final="x", error="boom")).ok
    assert check_all([], tr(final="x"))[0].ok


# --- runner ---------------------------------------------------------------------------------


async def test_runner_records_calls_results_usage_and_final_answer():
    t = await run_case(good_agent(), "sys", ToolBox(["get_order"]), SUITE.cases[0])
    assert [c.name for c in t.calls] == ["get_order"] and "shipped" in t.final
    assert t.turns == 2 and t.usage.tokens_in == 20 and t.calls[0].result.startswith("{")


async def test_runner_survives_tool_errors_and_reports_loop_exhaustion():
    def fn(messages, tools):
        return turn(calls=[("nope", {})])  # a tool that does not exist, forever

    t = await run_case(ScriptedProvider(fn), "s", ToolBox(["get_order"]), CASE, max_turns=3)
    assert t.calls[0].error and "no final answer" in t.error and t.turns == 3


async def test_runner_turns_provider_errors_into_trace_errors():
    from agentreg.providers import ProviderError

    def fn(messages, tools):
        raise ProviderError("429")

    t = await run_case(ScriptedProvider(fn), "s", ToolBox([]), CASE)
    assert t.error == "429" and t.final == ""


async def test_replay_marks_novel_calls_in_the_trace():
    def fn(messages, tools):
        return (
            turn("done")
            if tool_results(messages)
            else turn(calls=[("get_order", {"order_id": "A-1003"})])
        )

    rb = ReplayToolBox(
        [Call(name="get_order", args={"order_id": "A-1001"}, result="x")], ToolBox(["get_order"])
    )
    t = await run_case(ScriptedProvider(fn), "s", rb, CASE)
    assert t.calls[0].novel is True


# --- statistics -----------------------------------------------------------------------------


def test_fisher_known_values():
    assert fisher_one_sided_lower(5, 5, 0, 5) == pytest.approx(1 / 252)  # 5/5 vs 0/5
    assert fisher_one_sided_lower(5, 5, 4, 5) > 0.4  # one miss is noise
    assert fisher_one_sided_lower(3, 5, 3, 5) > 0.5
    assert fisher_one_sided_lower(0, 0, 0, 0) == 1.0


def test_wilson_interval_brackets_the_rate():
    lo, hi = wilson(8, 10)
    assert lo < 0.8 < hi and wilson(0, 0) == (0.0, 1.0)
