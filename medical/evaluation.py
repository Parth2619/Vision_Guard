"""Phase 9 — reproducible evaluation harness.

Pure Python metrics over pipeline outputs. Takes a list of EvalCase (report text
+ expected per-finding status) and runs them through any audit callable.
Produces per-case verdicts, confusion matrix, precision/recall/F1 per status,
coverage, abstention rate, and routing stats.

Honest limits:
- Metrics compare the pipeline's verdicts to the EXPECTED LABELS SUPPLIED.
- If those labels come from the original report (weak labels), the metric is
  "agreement with report", NOT clinical accuracy.
- Synthetic fixtures bundled with the repo are labeled synthetic.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .claims import AuditReport, Status

STATUSES = ("supported", "contradicted", "indeterminate")


@dataclass
class EvalCase:
    name: str
    report_text: str
    expected: dict[str, str]  # finding (lowercase) -> expected status value
    mode: str = "normal"


@dataclass
class CaseResult:
    name: str
    overall_status: str
    per_claim: list[dict]  # [{finding, expected, actual, matched}]
    recheck_count: int
    latency_ms: float
    needs_human_review: bool


@dataclass
class EvalMetrics:
    n_cases: int
    n_claims: int
    accuracy: float
    coverage: float  # 1 - abstention
    abstention_rate: float
    precision: dict[str, float]
    recall: dict[str, float]
    f1: dict[str, float]
    confusion: dict[str, dict[str, int]]
    recheck_mean: float
    recheck_max: int
    latency_ms_median: float


@dataclass
class EvalReport:
    cases: list[CaseResult] = field(default_factory=list)
    metrics: EvalMetrics | None = None

    def to_dict(self) -> dict:
        return {"cases": [asdict(c) for c in self.cases],
                "metrics": asdict(self.metrics) if self.metrics else None}


def load_cases(path: str | Path) -> list[EvalCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalCase(**row) for row in data]


def _match_verdicts(report: AuditReport, expected: dict[str, str]) -> list[dict]:
    """Match extracted claims to expected labels by finding (lowercased)."""
    by_finding: dict[str, str] = {}
    for v in report.verdicts:
        claim = next(c for c in report.claims if c.claim_id == v.claim_id)
        by_finding.setdefault(claim.finding.lower(), v.status.value)
    rows = []
    for finding, exp in expected.items():
        actual = by_finding.get(finding.lower(), "indeterminate")  # missing → indeterminate (conservative)
        rows.append({"finding": finding, "expected": exp, "actual": actual,
                     "matched": actual == exp})
    return rows


def _compute_metrics(cases: list[CaseResult]) -> EvalMetrics:
    all_rows = [r for c in cases for r in c.per_claim]
    n_claims = len(all_rows)
    accuracy = (sum(1 for r in all_rows if r["matched"]) / n_claims) if n_claims else 0.0
    abstention = sum(1 for r in all_rows if r["actual"] == "indeterminate") / n_claims if n_claims else 0.0

    confusion = {t: {p: 0 for p in STATUSES} for t in STATUSES}
    for r in all_rows:
        if r["expected"] in confusion and r["actual"] in confusion[r["expected"]]:
            confusion[r["expected"]][r["actual"]] += 1

    precision, recall, f1 = {}, {}, {}
    for s in STATUSES:
        tp = confusion[s][s]
        fp = sum(confusion[t][s] for t in STATUSES if t != s)
        fn = sum(confusion[s][p] for p in STATUSES if p != s)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r_ = tp / (tp + fn) if (tp + fn) else 0.0
        precision[s] = p
        recall[s] = r_
        f1[s] = (2 * p * r_ / (p + r_)) if (p + r_) else 0.0

    rechecks = [c.recheck_count for c in cases] or [0]
    lat = [c.latency_ms for c in cases] or [0.0]
    return EvalMetrics(
        n_cases=len(cases), n_claims=n_claims,
        accuracy=accuracy, coverage=1 - abstention, abstention_rate=abstention,
        precision=precision, recall=recall, f1=f1, confusion=confusion,
        recheck_mean=statistics.mean(rechecks), recheck_max=max(rechecks),
        latency_ms_median=statistics.median(lat),
    )


def evaluate(cases: Iterable[EvalCase],
             audit_fn: Callable[[str, str], AuditReport]) -> EvalReport:
    import time
    results: list[CaseResult] = []
    for case in cases:
        t0 = time.perf_counter()
        report = audit_fn(case.report_text, case.mode)
        latency_ms = (time.perf_counter() - t0) * 1000
        results.append(CaseResult(
            name=case.name,
            overall_status=report.overall_hallucination_status,
            per_claim=_match_verdicts(report, case.expected),
            recheck_count=report.recheck_count,
            latency_ms=latency_ms,
            needs_human_review=report.needs_human_review,
        ))
    return EvalReport(cases=results, metrics=_compute_metrics(results))
