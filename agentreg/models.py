from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Call(BaseModel):
    id: str = ""
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    error: str = ""
    novel: bool = False  # replay only: the candidate made a call the recording did not contain


class Usage(BaseModel):
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


class Trace(BaseModel):
    case_id: str
    provider: str
    model: str
    prompt: str
    system_sha: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    calls: list[Call] = Field(default_factory=list)
    final: str = ""
    usage: Usage = Field(default_factory=Usage)
    turns: int = 0
    latency_ms: int = 0
    error: str = ""
    infra_error: bool = (
        False  # provider/network failure (not the agent's behavior); excluded from statistics
    )


class Assertion(BaseModel):
    type: Literal[
        "tool_called",
        "tool_not_called",
        "answer_contains",
        "answer_not_contains",
        "answer_regex",
        "max_tool_calls",
        "max_cost_usd",
        "max_latency_ms",
        "calls_before",
        "no_error",
    ]
    name: str = ""  # tool name
    args_subset: dict[str, Any] = Field(default_factory=dict)
    any_of: list[str] = Field(default_factory=list)
    pattern: str = ""
    n: float = 0
    then: str = ""
    critical: bool = False


class Case(BaseModel):
    id: str
    prompt: str
    assertions: list[Assertion] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class Suite(BaseModel):
    name: str
    system: str
    tools: list[str]
    cases: list[Case]


class Config(BaseModel):
    """One side of a comparison: a model, optionally with a different system prompt."""

    label: str = ""
    provider: Literal["gemini", "openai", "mock"] = "gemini"
    model: str = "gemini-2.5-flash"
    system: str | None = None  # override the suite's system prompt
    temperature: float = 1.0


class AssertionResult(BaseModel):
    type: str
    ok: bool
    detail: str = ""
    critical: bool = False


class Run(BaseModel):
    trace: Trace
    results: list[AssertionResult]

    @property
    def passed(self) -> bool:
        return all(r.ok for r in self.results)


class SideStats(BaseModel):
    runs: int
    passed: int
    mean_calls: float
    mean_cost_usd: float
    mean_latency_ms: float
    mean_tokens: float
    errors: int = 0
    excluded: int = 0  # runs dropped from the statistics because of provider/infrastructure errors
    tool_sequences: dict[str, int] = Field(default_factory=dict)  # "a>b" -> count
    failed_assertions: dict[str, int] = Field(default_factory=dict)


class CaseComparison(BaseModel):
    case_id: str
    tags: list[str] = Field(default_factory=list)
    base: SideStats
    cand: SideStats
    p_regress: float  # one-sided Fisher: probability of a drop this large by chance
    p_improve: float
    verdict: Literal["same", "changed", "regression", "improvement"]
    reasons: list[str] = Field(default_factory=list)
    arg_diffs: list[str] = Field(default_factory=list)
    answer_similarity: float = 1.0
    judge: dict[str, Any] | None = None
    base_example: Trace | None = None
    cand_example: Trace | None = None


class Report(BaseModel):
    id: str
    created_at: str
    suite: str
    baseline: Config
    candidate: Config
    repeats: int
    verdict: Literal["ok", "changed", "regression"]
    summary: str
    cases: list[CaseComparison]
    counts: dict[str, int] = Field(default_factory=dict)
    cost_usd: float = 0.0
    duration_ms: int = 0
    degraded: list[str] = Field(default_factory=list)
    alpha: float = 0.1
