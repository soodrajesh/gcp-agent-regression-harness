from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    allowed_providers: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            x.strip()
            for x in os.environ.get("AGENTREG_ALLOWED_PROVIDERS", "mock").split(",")
            if x.strip()
        )
    )
    max_runs_per_hour: int = field(
        default_factory=lambda: int(os.environ.get("AGENTREG_MAX_RUNS_PER_HOUR", 20))
    )
    max_repeats: int = 10
    max_cases: int = 50
    max_cost_usd: float = field(
        default_factory=lambda: float(os.environ.get("AGENTREG_MAX_COST_USD", 0.5))
    )
    concurrency: int = 6
    max_body_bytes: int = 1_000_000
