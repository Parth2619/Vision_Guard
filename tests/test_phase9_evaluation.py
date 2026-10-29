"""Phase 9 — overall status + evaluation harness."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from graph.build import overall_status, run_audit
from medical.claims import Claim, ClaimVerdict, Status
from medical.evaluation import EvalCase, evaluate, load_cases


# ---- Overall-status derivation ---------------------------------------------

def _v(status, risk=None):
    return ClaimVerdict(claim_id="c", status=status, explanation="", risk_level=risk)


def test_overall_contradicted_wins():
    assert overall_status([_v(Status.SUPPORTED), _v(Status.CONTRADICTED)]) == "contradicted"


def test_overall_all_indeterminate_is_insufficient():
    assert overall_status([_v(Status.INDETERMINATE), _v(Status.INDETERMINATE)]) == "insufficient_evidence"


def test_overall_mixed_indeterminate_needs_review():
    assert overall_status([_v(Status.SUPPORTED), _v(Status.INDETERMINATE)]) == "needs_review"


def test_overall_all_supported_is_clean():
    assert overall_status([_v(Status.SUPPORTED), _v(Status.SUPPORTED)]) == "clean"


def test_overall_empty_defaults_to_insufficient():
    assert overall_status([]) == "insufficient_evidence"


def test_audit_report_populates_overall_status():
    r = run_audit("", mode="synthetic_demo")
    assert r.overall_hallucination_status in {
        "clean", "contradicted", "needs_review", "insufficient_evidence"}


# ---- Evaluation harness -----------------------------------------------------

def test_evaluate_computes_accuracy_and_f1():
    cases = [
        EvalCase(name="demo", report_text="", mode="synthetic_demo",
                 expected={"pleural effusion": "supported", "cardiomegaly": "contradicted"}),
    ]
    report = evaluate(cases, lambda t, m: run_audit(t, mode=m))
    m = report.metrics
    assert m.n_cases == 1 and m.n_claims == 2
    assert 0.0 <= m.accuracy <= 1.0
    for s in ("supported", "contradicted", "indeterminate"):
        assert s in m.precision and s in m.recall and s in m.f1


def test_evaluate_recall_1_when_all_expected_indeterminate_without_evidence():
    cases = [
        EvalCase(name="indet", report_text="FINDINGS: Pneumonia.", mode="normal",
                 expected={"pneumonia": "indeterminate"}),
    ]
    m = evaluate(cases, lambda t, mode: run_audit(t, mode=mode)).metrics
    assert m.accuracy == 1.0
    assert m.abstention_rate == 1.0


def test_bundled_fixture_file_loads_and_runs():
    cases = load_cases(Path(__file__).resolve().parents[1] / "datasets" / "eval_fixtures.json")
    report = evaluate(cases, lambda t, m: run_audit(t, mode=m))
    assert len(report.cases) == len(cases)
    # Should execute without crashing on all bundled cases.
    assert report.metrics.n_claims >= 1
