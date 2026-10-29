import json

from fastapi.testclient import TestClient

from agentreg.api import create_app
from agentreg.cli import main
from agentreg.compare import compare_suites, verdict_for
from agentreg.config import Settings
from agentreg.models import AssertionResult, Call, Config, Run, Trace
from agentreg.providers import ProviderError, ScriptedProvider

from .helpers import SUITE, careless_agent, good_agent, turn

B, C = Config(label="b", provider="mock", model="b"), Config(label="c", provider="mock", model="c")


def run_(ok=True, critical=False, seq=(), final="x"):
    t = Trace(
        case_id="c",
        provider="p",
        model="m",
        prompt="p",
        final=final,
        calls=[Call(name=n) for n in seq],
    )
    return Run(trace=t, results=[AssertionResult(type="a", ok=ok, critical=critical)])


# --- per-case verdicts --------------------------------------------------------------------------


def test_clear_drop_is_a_regression_with_p_value():
    v, reasons, p, _ = verdict_for([run_()] * 5, [run_(False)] * 5)
    assert v == "regression" and p < 0.01 and "5/5 -> 0/5" in reasons[0]


def test_one_flaky_failure_is_not_a_regression():
    v, *_ = verdict_for([run_()] * 5, [run_()] * 4 + [run_(False)])
    assert v == "same"


def test_flaky_baseline_does_not_make_a_stable_candidate_a_regression_and_vice_versa():
    assert verdict_for([run_()] * 3 + [run_(False)] * 2, [run_()] * 5)[0] in ("same", "improvement")
    assert (
        verdict_for([run_()] * 3 + [run_(False)] * 2, [run_()] * 3 + [run_(False)] * 2)[0] == "same"
    )


def test_one_critical_failure_is_flagged_as_changed_not_a_regression():
    """Calibration finding: a single flaky safety failure appears in same-model comparisons."""
    base = [run_(critical=True)] * 5
    cand = [run_(critical=True)] * 4 + [run_(False, critical=True)]
    v, reasons, *_ = verdict_for(base, cand)
    assert v == "changed" and "critical assertion failed in 1/5" in reasons[0]


def test_repeated_critical_failures_are_a_regression_at_a_looser_bar_than_ordinary_ones():
    base = [run_(critical=True)] * 5
    cand = [run_(critical=True)] * 3 + [run_(False, critical=True)] * 2
    v, reasons, *_ = verdict_for(base, cand)
    assert v == "regression" and "critical assertions held in 5/5" in reasons[0]
    # the same drop on a non-critical assertion is not significant at alpha 0.1
    assert verdict_for([run_()] * 5, [run_()] * 3 + [run_(False)] * 2)[0] == "same"


def test_critical_failure_that_also_failed_on_baseline_is_not_new():
    base = [run_(False, critical=True)] * 3
    assert verdict_for(base, [run_(False, critical=True)] * 3)[0] == "same"


def test_improvement_detected():
    assert verdict_for([run_(False)] * 5, [run_()] * 5)[0] == "improvement"


# --- end to end with scripted agents --------------------------------------------------------------


async def test_same_agent_against_itself_is_ok():
    rep = await compare_suites(
        SUITE, B, C, 3, base_provider=good_agent(), cand_provider=good_agent()
    )
    assert rep.verdict == "ok" and rep.counts["regression"] == 0 and rep.counts["same"] == 4


async def test_careless_candidate_regresses_the_safety_cases():
    rep = await compare_suites(
        SUITE, B, C, 5, base_provider=good_agent(), cand_provider=careless_agent()
    )
    by = {c.case_id: c for c in rep.cases}
    assert rep.verdict == "regression"
    assert by["refund-bad"].verdict == "regression" and by["refund-ok"].verdict == "regression"
    assert by["status"].verdict == "same"
    assert "refund_order" in by["refund-bad"].cand_example.calls[0].name
    assert by["refund-bad"].cand.failed_assertions.get("tool_not_called:refund_order") == 5


async def test_behavior_change_without_assertion_failure_is_reported_as_changed():
    def talkative(messages, tools):
        t = good_agent().fn(messages, tools)
        if t.text and t.text.startswith("Order A-1001"):
            t.text = (
                "Great news! Your order A-1001 has shipped and is on its way to you with tracking."
            )
        return t

    rep = await compare_suites(
        SUITE,
        B,
        C,
        3,
        base_provider=good_agent(),
        cand_provider=ScriptedProvider(talkative, "talk"),
    )
    assert rep.verdict in ("ok", "changed") and rep.counts["regression"] == 0


async def test_argument_changes_are_surfaced():
    def other_order(messages, tools):
        t = good_agent().fn(messages, tools)
        for c in t.tool_calls:
            if c.name == "get_order" and c.args["order_id"] == "A-1001":
                c.args = {"order_id": "A-1002"}
        return t

    rep = await compare_suites(
        SUITE, B, C, 5, base_provider=good_agent(), cand_provider=ScriptedProvider(other_order, "x")
    )
    status = next(c for c in rep.cases if c.case_id == "status")
    assert status.arg_diffs and "A-1001" in status.arg_diffs[0]
    assert status.verdict == "regression"  # it also fails the get_order(A-1001) assertion


async def test_provider_errors_are_reported_not_counted_as_agent_failures():
    def broken(messages, tools):
        raise ProviderError("gemini: 400 INVALID_ARGUMENT")

    rep = await compare_suites(
        SUITE, B, C, 3, base_provider=good_agent(), cand_provider=ScriptedProvider(broken, "x")
    )
    assert rep.verdict == "changed" and rep.counts["regression"] == 0 and rep.degraded


async def test_judge_is_informational_and_position_randomised():
    def judge(messages, tools):
        return turn(json.dumps({"winner": "A", "reason": "r"}))

    rep = await compare_suites(
        SUITE,
        B,
        C,
        2,
        base_provider=good_agent(),
        cand_provider=good_agent(),
        judge_provider=ScriptedProvider(judge, "j"),
    )
    winners = {c.judge["winner"] for c in rep.cases}
    assert (
        winners <= {"baseline", "candidate"} and rep.verdict == "ok"
    )  # judge never changes the verdict


async def test_judge_garbage_is_degraded_not_fatal():
    rep = await compare_suites(
        SUITE,
        B,
        C,
        2,
        base_provider=good_agent(),
        cand_provider=good_agent(),
        judge_provider=ScriptedProvider(lambda m, t: turn("not json"), "j"),
    )
    assert rep.verdict == "ok" and any("judge unavailable" in d for d in rep.degraded)


async def test_candidate_system_prompt_override_is_used():
    seen = []

    def spy(messages, tools):
        return turn("ok")

    class P(ScriptedProvider):
        async def generate(self, messages, tools, system, temperature):
            seen.append(system)
            return await super().generate(messages, tools, system, temperature)

    await compare_suites(
        SUITE,
        B,
        Config(label="c", provider="mock", model="c", system="OTHER"),
        1,
        base_provider=P(spy, "a"),
        cand_provider=P(spy, "b"),
    )
    assert set(seen) == {"sys", "OTHER"}


# --- API and CLI ------------------------------------------------------------------------------


def factory_for(agent):
    return lambda provider, model: agent


async def test_api_job_lifecycle_and_limits():
    import asyncio

    app = create_app(
        Settings(allowed_providers=("mock",)), provider_factory=lambda p, m: good_agent()
    )
    with TestClient(app) as c:
        body = {
            "sample": "order_support",
            "baseline": B.model_dump(),
            "candidate": C.model_dump(),
            "repeats": 1,
        }
        jid = c.post("/api/runs", json=body).json()["id"]
        for _ in range(100):
            j = c.get(f"/api/runs/{jid}").json()
            if j["status"] != "running":
                break
            await asyncio.sleep(0.1)
        assert j["status"] == "done" and j["report"]["verdict"] in ("ok", "changed", "regression")
        assert c.get("/api/runs").json()[0]["id"] == jid
        assert c.post("/api/runs", json={**body, "repeats": 99}).status_code == 422
        assert (
            c.post(
                "/api/runs", json={**body, "baseline": {**B.model_dump(), "provider": "gemini"}}
            ).status_code
            == 403
        )
        assert c.post("/api/runs", json={**body, "candidate_prompt": "nope"}).status_code == 422
        assert c.post("/api/runs", content=b"x").status_code == 422
        assert c.get("/api/runs/none").status_code == 404
        assert c.get("/api/prompts").json().keys() >= {"careful", "careless"}
        assert "Agent Regression" in c.get("/").text


def test_api_hourly_limit():
    app = create_app(
        Settings(allowed_providers=("mock",), max_runs_per_hour=1),
        provider_factory=lambda p, m: good_agent(),
    )
    with TestClient(app) as c:
        body = {"baseline": B.model_dump(), "candidate": C.model_dump(), "repeats": 1}
        assert c.post("/api/runs", json=body).status_code == 202
        assert c.post("/api/runs", json=body).status_code == 429


def test_cli_with_mock_provider_and_error_paths(tmp_path, capsys):
    p = tmp_path / "s.json"
    p.write_text(SUITE.model_dump_json())
    assert main(
        ["compare", str(p), "--baseline", "mock:a", "--candidate", "mock:b", "--repeats", "2"]
    ) in (0, 1, 2)
    assert "VERDICT" in capsys.readouterr().out
    assert main(["compare", str(tmp_path / "missing.json")]) == 3
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert main(["compare", str(bad)]) == 3


def test_benjamini_hochberg_correction():
    from agentreg.stats import bh_significant

    assert bh_significant([0.0079] + [0.9] * 9) == [True] + [False] * 9
    assert (
        bh_significant([0.0476] + [0.9] * 9) == [False] * 10
    )  # marginal call among many cases does not survive
    assert bh_significant([]) == []


# --- infrastructure errors must not look like agent failures --------------------------------------


async def test_transient_provider_errors_are_retried(monkeypatch):
    from agentreg import runner
    from agentreg.models import Case
    from agentreg.tools import ToolBox

    monkeypatch.setattr(runner, "BACKOFF_S", 0.001)
    n = {"calls": 0}

    def flaky(messages, tools):
        n["calls"] += 1
        if n["calls"] < 3:
            raise ProviderError("gemini: 429 RESOURCE_EXHAUSTED")
        return turn("fine")

    t = await runner.run_case(ScriptedProvider(flaky), "s", ToolBox([]), Case(id="c", prompt="p"))
    assert t.final == "fine" and not t.error and n["calls"] == 3


async def test_non_transient_errors_are_not_retried_and_are_marked_infrastructure(monkeypatch):
    from agentreg import runner
    from agentreg.models import Case
    from agentreg.tools import ToolBox

    monkeypatch.setattr(runner, "BACKOFF_S", 0.001)
    n = {"calls": 0}

    def bad(messages, tools):
        n["calls"] += 1
        raise ProviderError("gemini: 400 INVALID_ARGUMENT")

    t = await runner.run_case(ScriptedProvider(bad), "s", ToolBox([]), Case(id="c", prompt="p"))
    assert n["calls"] == 1 and t.infra_error and "400" in t.error


async def test_infrastructure_errors_are_excluded_from_the_statistics(monkeypatch):
    from agentreg import runner

    monkeypatch.setattr(runner, "BACKOFF_S", 0.001)
    state = {"n": 0}
    inner = good_agent()

    def sometimes_down(messages, tools):
        state["n"] += 1
        if state["n"] % 4 == 0:
            raise ProviderError("gemini: 400 INVALID_ARGUMENT")
        return inner.fn(messages, tools)

    rep = await compare_suites(
        SUITE,
        B,
        C,
        6,
        base_provider=ScriptedProvider(sometimes_down, "x"),
        cand_provider=good_agent(),
    )
    assert rep.verdict != "regression"  # outages on one side are not behavior
    assert any("excluded from the statistics" in d for d in rep.degraded)
    assert any(c.base.excluded > 0 for c in rep.cases)


async def test_too_few_valid_runs_gives_no_conclusion():
    def down(messages, tools):
        raise ProviderError("gemini: 400 INVALID_ARGUMENT")

    rep = await compare_suites(
        SUITE, B, C, 3, base_provider=good_agent(), cand_provider=ScriptedProvider(down, "x")
    )
    assert rep.verdict == "changed" and all("too few valid runs" in c.reasons[0] for c in rep.cases)
