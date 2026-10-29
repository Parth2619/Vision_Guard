"""CLI: run the evaluation harness over a JSON fixture file.

Default fixture ships at `datasets/eval_fixtures.json`. Each case must list
expected per-finding status. Metrics are printed as JSON on stdout.

    python scripts/evaluate.py                               # synthetic bundled
    python scripts/evaluate.py --cases some/other.json       # user-provided
    python scripts/evaluate.py --baseline                    # compare to baseline

The baseline compares the agentic pipeline against the non-agentic
`baselines.text_only.baseline_audit` (all indeterminate) so routing/coverage
gains are visible.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from baselines.text_only import baseline_audit  # noqa: E402
from graph.build import run_audit  # noqa: E402
from medical.claims import AuditReport  # noqa: E402
from medical.evaluation import evaluate, load_cases  # noqa: E402


def _audit_fn(text: str, mode: str) -> AuditReport:
    return run_audit(text, mode=mode)


def _baseline_fn(text: str, mode: str) -> AuditReport:
    # Non-agentic baseline: extract claims once, mark every one indeterminate.
    from medical import SYNTHETIC_REPORT, aggregate, extract
    from medical.claims import ClaimVerdict, Status, TraceEntry
    source = SYNTHETIC_REPORT if mode == "synthetic_demo" else text
    claims = extract(source)
    verdicts = [ClaimVerdict(claim_id=c.claim_id, status=Status.INDETERMINATE,
                             explanation="Baseline has no independent evidence.")
                for c in claims]
    return AuditReport(
        input_type="text" if mode == "normal" else "synthetic_demo",
        mode=mode, claims=claims, evidence=[], verdicts=verdicts,
        critic_findings=[], trace=[TraceEntry(node="baseline")],
        aggregate=aggregate(verdicts),
        verification_method="baseline: all indeterminate",
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=pathlib.Path,
                   default=_ROOT / "datasets" / "eval_fixtures.json")
    p.add_argument("--baseline", action="store_true",
                   help="also run and print the non-agentic baseline")
    args = p.parse_args()
    cases = load_cases(args.cases)

    out = {"agentic": evaluate(cases, _audit_fn).to_dict()}
    if args.baseline:
        out["baseline"] = evaluate(cases, _baseline_fn).to_dict()
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
