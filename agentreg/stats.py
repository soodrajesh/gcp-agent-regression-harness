"""Small exact statistics so one flaky run is not called a regression."""

from __future__ import annotations

from math import comb


def fisher_one_sided_lower(a_pass: int, a_n: int, b_pass: int, b_n: int) -> float:
    """P(candidate passes <= b_pass | margins) for a 2x2 table: how surprising is a pass count this low
    for candidate B if A and B had the same pass rate? Hypergeometric tail, exact."""
    total, passes = a_n + b_n, a_pass + b_pass
    if b_n == 0 or total == 0:
        return 1.0
    denom = comb(total, b_n)
    lo = max(0, b_n - (total - passes))
    p = 0.0
    for k in range(lo, min(b_pass, passes, b_n) + 1):
        p += comb(passes, k) * comb(total - passes, b_n - k) / denom
    return min(1.0, p)


def wilson(passes: int, n: int, z: float = 1.645) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = passes / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return max(0.0, (c - m) / d), min(1.0, (c + m) / d)


def bh_significant(pvals: list[float], q: float = 0.1) -> list[bool]:
    """Benjamini-Hochberg: which hypotheses are rejected at false discovery rate q when many cases are
    tested at once. Without it, 10 cases at alpha 0.1 throw about one false call per comparison."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    cutoff = -1
    for rank, i in enumerate(order, 1):
        if pvals[i] <= q * rank / m:
            cutoff = rank
    keep = {order[r] for r in range(cutoff)} if cutoff > 0 else set()
    return [i in keep for i in range(m)]
