"""LangGraph workflow. Real StateGraph with conditional edges and bounded recheck."""
from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from medical import (
    SYNTHETIC_REPORT,
    AuditReport,
    Claim,
    ClaimVerdict,
    CriticFinding,
    Evidence,
    TraceEntry,
    aggregate,
    collect,
    critique,
    extract,
    verify_claim,
)
from medical.risk import claim_risk
from models.calibration import calibrate, features_for
from models.detector import get_backend as get_detector_backend
from models.vlm import get_backend as get_vlm_backend
from models.xray import FINDING_TO_LABEL, get_backend

MAX_RECHECKS = 1  # bounded retry budget


def _append(a: list, b: list) -> list:
    return (a or []) + (b or [])


class GraphState(TypedDict, total=False):
    report_text: str
    mode: str
    image_bytes: bytes | None
    image_probs: dict[str, float] | None
    detections: list | None
    valid: bool
    claims: list[Claim]
    evidence: list[Evidence]
    verdicts: list[ClaimVerdict]
    critic: list[CriticFinding]
    recheck_count: int
    needs_human_review: bool
    trace: Annotated[list[TraceEntry], _append]


def _trace(name: str, note: str | None = None) -> list[TraceEntry]:
    return [TraceEntry(node=name, note=note)]


def node_validate(state: GraphState) -> dict:
    text = (state.get("report_text") or "").strip()
    valid = bool(text) or state.get("mode") == "synthetic_demo"
    note = "ok" if valid else "empty input in normal mode"
    return {"valid": valid, "trace": _trace("input_validation", note)}


def node_extract(state: GraphState) -> dict:
    claims = extract(state["report_text"])
    return {"claims": claims, "trace": _trace("claim_extraction", f"{len(claims)} claim(s)")}


def node_evidence(state: GraphState) -> dict:
    probs = state.get("image_probs")
    detections = state.get("detections")
    img = state.get("image_bytes")
    classifier_err = detector_err = None
    if probs is None and img:
        try:
            probs = get_backend().probabilities(img)
        except Exception as exc:  # noqa: BLE001 - never let the pipeline silently claim success
            classifier_err = f"{type(exc).__name__}: {exc}"
    if detections is None and img and state.get("claims"):
        prompts = sorted({c.finding for c in state["claims"] if c.finding})
        try:
            detections = get_detector_backend().detect(img, prompts)
        except Exception as exc:  # noqa: BLE001
            detector_err = f"{type(exc).__name__}: {exc}"
            detections = []
    ev = collect(
        state.get("claims", []),
        state["report_text"],
        synthetic=state.get("mode") == "synthetic_demo",
        image_probs=probs,
        finding_to_label=FINDING_TO_LABEL if probs else None,
        detections=detections,
    )
    note_parts = [f"{len(ev)} evidence item(s)"]
    if probs:
        note_parts.append(f"classifier_labels={len(probs)}")
    if detections is not None:
        note_parts.append(f"detections={len(detections)}")
    if classifier_err:
        note_parts.append(f"classifier_error={classifier_err}")
    if detector_err:
        note_parts.append(f"detector_error={detector_err}")
    return {"evidence": ev, "image_probs": probs, "detections": detections,
            "trace": _trace("evidence_retrieval", ", ".join(note_parts))}


def node_verify(state: GraphState) -> dict:
    verdicts = [verify_claim(c, state.get("evidence", [])) for c in state.get("claims", [])]
    return {"verdicts": verdicts, "trace": _trace("claim_verification")}


def node_calibrate(state: GraphState) -> dict:
    evidence = state.get("evidence", [])
    claims_by_id = {c.claim_id: c for c in state.get("claims", [])}
    for v in state.get("verdicts", []):
        c = claims_by_id.get(v.claim_id)
        if c is None:
            continue
        rel = [e for e in evidence if e.claim_id == v.claim_id]
        feats = features_for(c, rel)
        _, probs = calibrate(feats)
        v.calibrated_probs = probs
        v.calibrated_confidence = probs.get(v.status.value)
        v.risk_level = claim_risk(c)  # computed here so one pass; used by human_review
    return {"verdicts": state.get("verdicts", []),
            "trace": _trace("fusion_calibration",
                            f"calibrated {len(state.get('verdicts', []))} verdict(s)")}


def node_critic(state: GraphState) -> dict:
    findings = critique(state.get("claims", []), state.get("verdicts", []),
                        state.get("evidence", []))
    blockers = [f for f in findings if f.severity == "block"]
    note = f"{len(findings)} finding(s), {len(blockers)} blocking"
    return {"critic": findings, "trace": _trace("critic", note)}


def node_human_review(state: GraphState) -> dict:
    """Risk-aware decision gate: escalate on blockers, on INDETERMINATE, or when a
    high-risk claim has low calibrated confidence (< 0.75)."""
    blockers = [f for f in state.get("critic", []) if f.severity == "block"]
    reasons: list[str] = []
    if blockers:
        reasons.append(f"{len(blockers)} blocker(s)")
    for v in state.get("verdicts", []):
        if v.status.value == "indeterminate":
            reasons.append(f"{v.claim_id}:indeterminate")
            continue
        conf = v.calibrated_confidence
        if v.risk_level == "high" and (conf is None or conf < 0.75):
            reasons.append(f"{v.claim_id}:high_risk_low_confidence")
    needed = bool(reasons)
    return {"needs_human_review": needed,
            "trace": _trace("risk_gate_human_review",
                            f"needed={needed}; reasons={reasons[:5]}")}


def route_after_critic(state: GraphState) -> str:
    blockers = [f for f in state.get("critic", []) if f.severity == "block"]
    rechecks = state.get("recheck_count", 0)
    if blockers and rechecks < MAX_RECHECKS:
        return "recheck"
    return "finalize"


def node_recheck(state: GraphState) -> dict:
    # Bounded recheck: re-run evidence collection + verification once.
    return {
        "recheck_count": state.get("recheck_count", 0) + 1,
        "trace": _trace("conditional_recheck",
                        f"attempt {state.get('recheck_count', 0) + 1}/{MAX_RECHECKS}"),
    }


def _route_validation(state: GraphState) -> str:
    return "extract" if state.get("valid") else "finalize_invalid"


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("validate", node_validate)
    g.add_node("extract", node_extract)
    g.add_node("evidence", node_evidence)
    g.add_node("verify", node_verify)
    g.add_node("critic", node_critic)
    g.add_node("recheck", node_recheck)
    g.add_node("human_review", node_human_review)

    g.add_edge(START, "validate")
    g.add_conditional_edges("validate", _route_validation,
                            {"extract": "extract", "finalize_invalid": "human_review"})
    g.add_edge("extract", "evidence")
    g.add_edge("evidence", "verify")
    g.add_node("calibrate", node_calibrate)
    g.add_edge("verify", "calibrate")
    g.add_edge("calibrate", "critic")
    g.add_conditional_edges("critic", route_after_critic,
                            {"recheck": "recheck", "finalize": "human_review"})
    g.add_edge("recheck", "evidence")  # loop back once, then critic will route to finalize
    g.add_edge("human_review", END)
    return g.compile()


_COMPILED = None


def get_graph():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_graph()
    return _COMPILED


def overall_status(verdicts: list[ClaimVerdict]) -> str:
    """Derive the top-line audit status from per-claim verdicts."""
    if not verdicts:
        return "insufficient_evidence"
    if any(v.status.value == "contradicted" for v in verdicts):
        return "contradicted"
    if all(v.status.value == "indeterminate" for v in verdicts):
        return "insufficient_evidence"
    if any(v.status.value == "indeterminate" and v.risk_level == "high" for v in verdicts):
        return "needs_review"
    if any(v.status.value == "indeterminate" for v in verdicts):
        return "needs_review"
    return "clean"


def run_audit(report_text: str, mode: str = "normal",
              image_bytes: bytes | None = None) -> AuditReport:
    generated: str | None = None
    if mode == "synthetic_demo":
        report_text = SYNTHETIC_REPORT
    elif mode == "vlm_from_image":
        if not image_bytes:
            raise ValueError("vlm_from_image mode requires an image_bytes payload")
        generated = get_vlm_backend().generate_report(image_bytes)
        report_text = generated
    initial: GraphState = {"report_text": report_text, "mode": mode,
                           "image_bytes": image_bytes, "image_probs": None,
                           "detections": None,
                           "recheck_count": 0, "trace": []}
    final = get_graph().invoke(initial)
    verdicts = final.get("verdicts", [])
    input_type = ("synthetic_demo" if mode == "synthetic_demo"
                  else "vlm_generated" if mode == "vlm_from_image"
                  else "text")
    return AuditReport(
        input_type=input_type,
        mode=mode,
        generated_report_text=generated,
        claims=final.get("claims", []),
        evidence=final.get("evidence", []),
        verdicts=verdicts,
        critic_findings=final.get("critic", []),
        recheck_count=final.get("recheck_count", 0),
        needs_human_review=bool(final.get("needs_human_review")),
        trace=final.get("trace", []),
        aggregate=aggregate(verdicts),
        overall_hallucination_status=overall_status(verdicts),
        verification_method=(
            "Rule-based typed extraction. Report-text evidence always; synthetic KB in "
            "demo mode; pretrained torchxrayvision classifier AND zero-shot "
            "Grounding-DINO detector when an image is uploaded. Classifier evidence "
            "counts only when probability is confident (>=0.6 or <=0.4). Detector "
            "boxes are experimental AI outputs. A fusion MLP (128->64->3) emits a "
            "calibrated_confidence against the rule verdicts (NOT medical ground "
            "truth). Risk-aware gate escalates INDETERMINATE and high-risk-low-"
            "confidence claims to human review."
        ),
    )
