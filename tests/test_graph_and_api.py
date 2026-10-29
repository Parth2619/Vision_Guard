from fastapi.testclient import TestClient

from api.main import app
from graph.build import run_audit

client = TestClient(app)


def test_synthetic_demo_end_to_end():
    report = run_audit("", mode="synthetic_demo")
    assert report.mode == "synthetic_demo"
    assert report.claims, "demo should extract at least one claim"
    assert sum(report.aggregate.values()) == len(report.verdicts)
    # Trace captures every expected stage.
    nodes = [t.node for t in report.trace]
    for required in ("input_validation", "claim_extraction", "evidence_retrieval",
                     "claim_verification", "fusion_calibration", "critic",
                     "risk_gate_human_review"):
        assert required in nodes


def test_graph_bounded_recheck_terminates():
    report = run_audit("", mode="synthetic_demo")
    assert report.recheck_count <= 1  # MAX_RECHECKS


def test_normal_mode_without_independent_evidence_is_all_indeterminate():
    report = run_audit("FINDINGS: There is pneumonia in the right lung.", mode="normal")
    for v in report.verdicts:
        assert v.status.value == "indeterminate"


def test_api_audit_route_runs_demo():
    r = client.post("/audit/json", json={"report_text": "", "mode": "synthetic_demo"})
    assert r.status_code == 200
    body = r.json()
    assert "audit_id" in body and body["mode"] == "synthetic_demo"
    assert "indeterminate" in body["aggregate"]
    assert body["disclaimer"].startswith("VisionGuard")


def test_api_rejects_empty_normal_input():
    r = client.post("/audit/json", json={"report_text": "   ", "mode": "normal"})
    assert r.status_code == 422


def test_api_does_not_leak_stack_traces_on_bad_json():
    r = client.post("/audit/json", json={"nope": 1})
    assert r.status_code == 422
    assert "Traceback" not in r.text


def test_frontend_declares_prototype_still():
    import pathlib
    html = (pathlib.Path(__file__).resolve().parents[1] / "frontend/web/index.html").read_text(encoding="utf-8")
    assert "Research prototype" in html and "Not clinically validated" in html
