// VisionGuard Medical — web client

const API = "";  // same origin (FastAPI serves this file)

const $  = (q, el=document) => el.querySelector(q);
const $$ = (q, el=document) => Array.from(el.querySelectorAll(q));

const state = {
  lastAudit: null,
  objectUrl: null,  // for the uploaded X-ray preview
};

// ---- Nav --------------------------------------------------
$$(".nav-item").forEach(btn => btn.addEventListener("click", () => {
  const v = btn.dataset.view;
  $$(".nav-item").forEach(b => {
    b.classList.toggle("is-active", b === btn);
    b.setAttribute("aria-selected", String(b === btn));
  });
  $$(".view").forEach(view => {
    view.hidden = view.dataset.view !== v;
  });
  if (v === "review") renderReview();
}));

// ---- File picker ------------------------------------------
const fileInput = $("#image");
const drop      = $("#drop");
const dropName  = $("#drop-filename");

fileInput.addEventListener("change", () => {
  const f = fileInput.files?.[0];
  if (!f) { drop.classList.remove("has-file"); dropName.textContent = ""; return; }
  drop.classList.add("has-file");
  dropName.textContent = `${f.name}  ·  ${Math.round(f.size/1024)} KB`;
  // Preview in viewer immediately
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = URL.createObjectURL(f);
  renderViewer({ src: state.objectUrl, boxes: [] });
});

// ---- Mode-aware report hint -------------------------------
const modeInputs = $$('input[name="mode"]');
const reportHint = $("#report-hint");
const reportText = $("#report_text");
modeInputs.forEach(r => r.addEventListener("change", () => {
  const mode = currentMode();
  if (mode === "vlm_from_image") { reportHint.textContent = "The uploaded X-ray is passed to the VLM, which generates the report. Image upload is required."; }
  else { reportHint.textContent = "Required in Normal mode."; }
}));

function currentMode() {
  return $$('input[name="mode"]').find(r => r.checked)?.value ?? "normal";
}

// ---- Form submit -----------------------------------------
$("#audit-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const mode = currentMode();
  const text = reportText.value.trim();
  const file = fileInput.files?.[0];

  if (mode === "normal" && !text) return toast("Paste a report in Normal mode, or switch mode.", "error");
  if (mode === "vlm_from_image" && !file) return toast("vlm_from_image needs an uploaded X-ray.", "error");

  const fd = new FormData();
  fd.set("mode", mode);
  fd.set("report_text", text);
  if (file) fd.set("image", file);

  const btn = $("#run");
  btn.disabled = true; btn.textContent = "Auditing…";
  document.body.classList.add("loading");

  try {
    const r = await fetch(`${API}/audit`, { method: "POST", body: fd });
    if (!r.ok) {
      const detail = await safeJson(r);
      throw new Error(detail?.detail?.[0]?.msg || detail?.detail || `HTTP ${r.status}`);
    }
    const audit = await r.json();
    state.lastAudit = audit;
    renderAudit(audit);
    toast("Audit complete.", "ok");
  } catch (err) {
    toast(err.message || "Audit failed.", "error");
  } finally {
    btn.disabled = false; btn.textContent = "Run audit";
    document.body.classList.remove("loading");
  }
});

$("#reset").addEventListener("click", () => {
  state.lastAudit = null;
  drop.classList.remove("has-file"); dropName.textContent = "";
  if (state.objectUrl) { URL.revokeObjectURL(state.objectUrl); state.objectUrl = null; }
  renderViewerEmpty();
  renderVerdictsEmpty();
  $("#viewer-foot").hidden = true;
  $("#trace-wrap").hidden = true;
});

// ---- Render: viewer ---------------------------------------
function renderViewerEmpty() {
  $("#viewer").innerHTML = `
    <div class="viewer-empty">
      <div class="viewer-empty-mark" aria-hidden="true"></div>
      <p>Upload a chest X-ray and run the audit. The image appears here with bounding-box evidence overlaid.</p>
    </div>
  `;
}

function renderViewer({ src, boxes }) {
  const viewer = $("#viewer");
  viewer.classList.add("reveal");
  if (!src) { renderViewerEmpty(); return; }
  viewer.innerHTML = `
    <div class="radiograph">
      <img id="xray-img" alt="Chest X-ray" />
      <svg class="overlay" preserveAspectRatio="none" aria-hidden="true"></svg>
    </div>
  `;
  const img = $("#xray-img", viewer);
  const svg = $(".overlay", viewer);
  img.src = src;
  img.onload = () => {
    const W = img.naturalWidth, H = img.naturalHeight;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.style.width  = img.clientWidth + "px";
    svg.style.height = img.clientHeight + "px";
    svg.innerHTML = (boxes ?? []).map(b => {
      const w = b.x2 - b.x1, h = b.y2 - b.y1;
      return `<g>
        <rect class="bbox" data-status="${b.status}" x="${b.x1}" y="${b.y1}" width="${w}" height="${h}" />
        <text class="bbox-label" x="${b.x1 + 4}" y="${b.y1 + 14}">${escapeHtml(b.label)} ${b.score?.toFixed?.(2) ?? ""}</text>
      </g>`;
    }).join("");
    // Keep overlay sized to image on window resize
    const sync = () => {
      svg.style.width  = img.clientWidth + "px";
      svg.style.height = img.clientHeight + "px";
    };
    new ResizeObserver(sync).observe(img);
  };
  img.onerror = () => renderViewerEmpty();
}

// ---- Render: full audit result ---------------------------
function renderAudit(audit) {
  // Status LED + overall label
  const foot = $("#viewer-foot");
  foot.hidden = false;
  const overall = audit.overall_hallucination_status || "insufficient_evidence";
  const dot = foot.querySelector(".dot");
  dot.setAttribute("data-status", statusForOverall(overall));
  $("#overall-status").textContent = overall.replace(/_/g, " ");
  $("#audit-id").textContent = audit.audit_id;
  $("#review-flag").textContent = audit.needs_human_review ? "required" : "not required";

  // Aggregate counts
  const agg = audit.aggregate || {};
  $("#agg-supported").textContent    = agg.supported    ?? 0;
  $("#agg-contradicted").textContent = agg.contradicted ?? 0;
  $("#agg-indeterminate").textContent= agg.indeterminate?? 0;

  // Verdicts list
  const list = $("#verdicts");
  const empty = $("#verdicts-empty");
  const claimsById = Object.fromEntries((audit.claims || []).map(c => [c.claim_id, c]));
  const evByClaim = (audit.evidence || []).reduce((acc, e) => {
    (acc[e.claim_id] ||= []).push(e); return acc;
  }, {});
  list.innerHTML = (audit.verdicts || []).map(v => {
    const claim = claimsById[v.claim_id] || {};
    const conf = v.calibrated_confidence != null ? v.calibrated_confidence.toFixed(2) : "—";
    const risk = v.risk_level || "—";
    const ev = (evByClaim[v.claim_id] || []).map(e =>
      `<div class="ev-item"><em>${escapeHtml(e.source)}</em><span>${e.confidence != null ? e.confidence.toFixed(2) : ""} ${e.agrees === true ? "agrees" : e.agrees === false ? "disagrees" : ""}</span></div>`
    ).join("");
    return `
      <li class="verdict" data-status="${v.status}">
        <span class="marker" aria-hidden="true"></span>
        <div class="verdict-body">
          <h3>${escapeHtml(claim.sentence || v.claim_id)}</h3>
          <div class="verdict-meta">
            <span><span class="k">status</span><span class="status">${v.status}</span></span>
            <span><span class="k">conf</span><span class="v">${conf}</span></span>
            <span><span class="k">risk</span><span class="v">${risk}</span></span>
            <span><span class="k">type</span><span class="v">${claim.claim_type || "—"}</span></span>
          </div>
          <div class="verdict-expand">
            <p>${escapeHtml(v.explanation || "")}</p>
            <div class="ev">${ev || '<em style="color:var(--faint)">no evidence collected</em>'}</div>
          </div>
        </div>
      </li>
    `;
  }).join("");
  empty.style.display = (audit.verdicts || []).length === 0 ? "block" : "none";

  // Click to expand
  $$(".verdict", list).forEach(el => el.addEventListener("click", () => {
    el.toggleAttribute("open");
  }));

  // X-ray with bounding boxes
  const srcPromise = (async () => {
    if (state.objectUrl) return state.objectUrl;
    // synthetic_demo without an uploaded image: no visual
    return null;
  })();

  srcPromise.then(src => {
    const boxes = (audit.evidence || [])
      .filter(e => e.bbox)
      .map(e => {
        const verdict = (audit.verdicts || []).find(v => v.claim_id === e.claim_id);
        return {
          ...e.bbox, status: verdict?.status || "indeterminate",
          label: shortSource(e.source), score: e.confidence,
        };
      });
    if (src) renderViewer({ src, boxes });
  });

  // Trace
  const trace = audit.trace || [];
  $("#trace-wrap").hidden = trace.length === 0;
  $("#trace-method").textContent = audit.verification_method ? "Verification method: see audit JSON." : "";
  $("#trace-list").innerHTML = trace.map(t => `
    <li class="trace-step reveal">
      <span class="node">${escapeHtml(t.node)}</span>
      <span class="note">${escapeHtml(t.note || "")}</span>
    </li>
  `).join("");

  // VLM-generated banner — reuse report_text field for display
  if (audit.generated_report_text) {
    reportText.value = audit.generated_report_text;
  }
}

function renderVerdictsEmpty() {
  $("#verdicts").innerHTML = "";
  $("#verdicts-empty").style.display = "block";
  ["supported","contradicted","indeterminate"].forEach(k => $(`#agg-${k}`).textContent = "—");
}

function statusForOverall(overall) {
  if (overall === "contradicted") return "contradicted";
  if (overall === "clean") return "supported";
  if (overall === "insufficient_evidence") return "indeterminate";
  if (overall === "needs_review") return "indeterminate";
  return "indeterminate";
}

function shortSource(src) {
  // "grounding_dino[pleural effusion]" -> "pleural effusion"
  const m = src.match(/\[(.+?)\]/);
  return m ? m[1] : src;
}

// ---- Review view -----------------------------------------
function renderReview() {
  const list = $("#review-list");
  const audit = state.lastAudit;
  if (!audit) {
    list.innerHTML = `<div class="verdicts-empty">Run an audit first; flagged claims appear here.</div>`;
    return;
  }
  const claimsById = Object.fromEntries((audit.claims || []).map(c => [c.claim_id, c]));
  const flagged = (audit.verdicts || []).filter(v =>
    v.status === "indeterminate" || v.risk_level === "high"
  );
  if (!flagged.length) {
    list.innerHTML = `<div class="verdicts-empty">No flagged claims in the last audit.</div>`;
    return;
  }
  list.innerHTML = flagged.map(v => {
    const c = claimsById[v.claim_id] || {};
    return `
      <div class="review-item">
        <div>
          <p class="review-sentence">${escapeHtml(c.sentence || v.claim_id)}</p>
          <div class="review-meta">
            <span>${v.status}</span>
            <span>finding: ${escapeHtml(c.finding || "—")}</span>
            <span>risk: ${v.risk_level || "—"}</span>
            <span>conf: ${v.calibrated_confidence?.toFixed?.(2) ?? "—"}</span>
          </div>
        </div>
        <div class="review-actions">
          <button class="confirm"      data-decision="confirm"      data-cid="${v.claim_id}">Confirm</button>
          <button class="reject"       data-decision="reject"       data-cid="${v.claim_id}">Reject</button>
          <button class="insufficient" data-decision="insufficient" data-cid="${v.claim_id}">Insufficient</button>
        </div>
      </div>
    `;
  }).join("");

  $$(".review-actions button", list).forEach(btn => btn.addEventListener("click", async () => {
    btn.disabled = true;
    const payload = {
      audit_id: audit.audit_id,
      claim_id: btn.dataset.cid,
      decision: btn.dataset.decision,
      note: "",
    };
    try {
      const r = await fetch(`${API}/review`, {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      toast(`Recorded ${payload.decision}.`, "ok");
      btn.closest(".review-item").style.opacity = 0.5;
    } catch (e) {
      toast(e.message, "error");
      btn.disabled = false;
    }
  }));
}

// ---- Export JSON -----------------------------------------
$("#export-json").addEventListener("click", () => {
  if (!state.lastAudit) return;
  const blob = new Blob([JSON.stringify(state.lastAudit, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = `${state.lastAudit.audit_id}.json`;
  a.click();
  URL.revokeObjectURL(url);
});

// ---- Helpers ---------------------------------------------
function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}
async function safeJson(r) { try { return await r.json(); } catch { return null; } }

let toastTimer = null;
function toast(msg, kind="ok") {
  const el = $("#toast");
  el.textContent = msg;
  el.setAttribute("data-kind", kind);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 3200);
}
