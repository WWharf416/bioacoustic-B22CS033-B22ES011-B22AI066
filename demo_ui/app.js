/* ============================================================
   BioAcoustic Inference Demo – Application Logic
   Connected to real Flask backend at /api/*
   ============================================================ */

const CLASS_LABELS = {
  WMW: "",
  BV: "",
  HT: "",
  JD: "",
  MT: "",
};

// ── State ────────────────────────────────────────────────────
let classesData = {};       // populated from /api/classes
let uploadedFile = null;    // File object if user uploads one

// ── Helpers ──────────────────────────────────────────────────
function $(id) { return document.getElementById(id); }

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }


// ── Init ─────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initSliders();
  loadClasses();

  $("audio-upload").addEventListener("change", (e) => {
    uploadedFile = e.target.files[0] || null;
    $("upload-filename").textContent = uploadedFile ? `✓ ${uploadedFile.name}` : "";
    if (uploadedFile) {
      $("audio-select").value = "__upload__";
    }
    updateCLI();
  });

  $("class-select").addEventListener("change", () => {
    populateAudioFiles();
    updateCLI();
  });

  $("audio-select").addEventListener("change", () => {
    if ($("audio-select").value !== "__upload__") {
      uploadedFile = null;
      $("upload-filename").textContent = "";
    }
    updateCLI();
  });
});


// ── Load real classes from backend ───────────────────────────
async function loadClasses() {
  const badge = $("badge-server");
  try {
    const res = await fetch("/api/classes");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    classesData = await res.json();

    // Populate class select
    const sel = $("class-select");
    sel.innerHTML = "";
    const classNames = Object.keys(classesData).sort();
    classNames.forEach(cls => {
      const opt = document.createElement("option");
      opt.value = cls;
      const label = CLASS_LABELS[cls] || cls;
      const numFiles = classesData[cls].files.length;
      opt.textContent = `${cls} — ${label}`;
      sel.appendChild(opt);
    });

    // Default to WMW if available
    if (classNames.includes("WMW")) sel.value = "WMW";

    populateAudioFiles();
    updateCLI();

    // Server live
    badge.innerHTML = '<span class="live-dot live"></span> Server Live';
    badge.classList.remove("badge-connecting");
    badge.classList.add("badge-live-ok");
  } catch (err) {
    console.error("Failed to load classes:", err);
    badge.innerHTML = '<span class="live-dot dead"></span> Server Offline';
    badge.classList.add("badge-offline");
    $("class-select").innerHTML = '<option value="">Server unavailable</option>';
  }
}


function populateAudioFiles() {
  const cls = $("class-select").value;
  const sel = $("audio-select");
  sel.innerHTML = "";

  if (!cls || !classesData[cls]) {
    sel.innerHTML = '<option value="">Select a class first</option>';
    return;
  }

  // "Auto-pick" option
  const autoOpt = document.createElement("option");
  autoOpt.value = "";
  autoOpt.textContent = "Auto-pick best query file";
  sel.appendChild(autoOpt);

  // Real files from the dataset
  classesData[cls].files.forEach(f => {
    const opt = document.createElement("option");
    opt.value = f.path;
    opt.textContent = f.name;
    sel.appendChild(opt);
  });

  // Upload placeholder
  const upOpt = document.createElement("option");
  upOpt.value = "__upload__";
  upOpt.textContent = "↑ Use uploaded file";
  sel.appendChild(upOpt);
}


// ── Slider binding ───────────────────────────────────────────
function initSliders() {
  const sliders = [
    { id: "n-shot-slider", display: "n-shot-value", fmt: v => v },
    { id: "threshold-slider", display: "threshold-value", fmt: v => v == 0 ? "0.0 (adaptive)" : parseFloat(v).toFixed(2) },
    { id: "kernel-slider", display: "kernel-value", fmt: v => v },
    { id: "maxsec-slider", display: "maxsec-value", fmt: v => v == 0 ? "Full audio" : v + " s" },
    { id: "merge-slider", display: "merge-value", fmt: v => v + " ms" },
  ];

  sliders.forEach(({ id, display, fmt }) => {
    const slider = $(id);
    const disp = $(display);
    slider.addEventListener("input", () => {
      disp.textContent = fmt(slider.value);
      updateCLI();
    });
  });
}


function updateCLI() {
  const cls = $("class-select").value || "WMW";
  const audio = $("audio-select").value;
  const nShot = $("n-shot-slider").value;
  const threshold = $("threshold-slider").value;
  const kernel = $("kernel-slider").value;
  const maxSec = $("maxsec-slider").value;
  const mergeGap = $("merge-slider").value;

  let audioArg = "";
  if (audio && audio !== "__upload__") {
    audioArg = ` --audio ${audio.split("/").pop()}`;
  } else if (uploadedFile) {
    audioArg = ` --audio ${uploadedFile.name}`;
  }

  $("cli-command").textContent =
    `python inference.py${audioArg} --class-name ${cls} --n-shot ${nShot} --threshold ${threshold} --median-kernel ${kernel} --max-seconds ${maxSec === "0" ? "0.0" : maxSec + ".0"} --merge-gap-ms ${mergeGap}`;
}


// ── Run real inference ───────────────────────────────────────
async function runInference() {
  const btn = $("run-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="run-icon">⏳</span> Running…';

  // Gather params
  const className = $("class-select").value;
  const queryFile = $("audio-select").value;
  const nShot = parseInt($("n-shot-slider").value);
  const threshold = parseFloat($("threshold-slider").value);
  const medianKernel = parseInt($("kernel-slider").value);
  const mergeGapMs = parseFloat($("merge-slider").value);
  const maxSeconds = parseFloat($("maxsec-slider").value);

  // Show loading
  $("status-idle").classList.add("hidden");
  $("status-error").classList.add("hidden");
  $("status-loading").classList.remove("hidden");
  $("results-content").classList.add("hidden");
  $("status-area").classList.remove("hidden");

  // Animate loading steps
  const steps = ["lstep-1", "lstep-2", "lstep-3", "lstep-4"];
  let stepIdx = 0;
  const stepTimer = setInterval(() => {
    if (stepIdx < steps.length) {
      steps.forEach((s, idx) => {
        const el = $(s);
        el.classList.remove("active", "done");
        if (idx < stepIdx) el.classList.add("done");
        if (idx === stepIdx) el.classList.add("active");
      });
      stepIdx++;
    }
  }, 2000);

  try {
    let response;

    if (queryFile === "__upload__" && uploadedFile) {
      // Upload mode
      const formData = new FormData();
      formData.append("audio", uploadedFile);
      formData.append("class_name", className);
      formData.append("n_shot", nShot);
      formData.append("threshold", threshold);
      formData.append("median_kernel", medianKernel);
      formData.append("merge_gap_ms", mergeGapMs);
      formData.append("max_seconds", maxSeconds);

      response = await fetch("/api/upload_and_infer", { method: "POST", body: formData });
    } else {
      // JSON mode
      const body = {
        class_name: className,
        n_shot: nShot,
        threshold: threshold,
        median_kernel: medianKernel,
        merge_gap_ms: mergeGapMs,
        max_seconds: maxSeconds,
      };
      if (queryFile && queryFile !== "__upload__") {
        body.query_file = queryFile;
      }
      response = await fetch("/api/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    }

    clearInterval(stepTimer);

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.message || `HTTP ${response.status}`);
    }

    const result = await response.json();
    if (result.status === "error") throw new Error(result.message);

    // All steps done
    steps.forEach(s => $(s).classList.add("done"));
    await sleep(300);

    displayResults(result);

  } catch (err) {
    clearInterval(stepTimer);
    console.error("Inference error:", err);
    $("status-loading").classList.add("hidden");
    $("status-error").classList.remove("hidden");
    $("error-msg").textContent = `Error: ${err.message}`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="run-icon">▶</span> Run Inference';
  }
}


// ── Display results ──────────────────────────────────────────
function displayResults(result) {
  // Hide loading, show results
  $("status-loading").classList.add("hidden");
  $("status-area").classList.add("hidden");
  $("results-content").classList.remove("hidden");

  // Query info
  $("query-info-text").textContent =
    `Class: ${result.class_name} · Query: ${result.query_file} · ${result.support_files.length}-shot support`;

  // Support files
  const supportInfo = $("support-info");
  supportInfo.classList.remove("hidden");
  const list = $("support-files-list");
  list.innerHTML = result.support_files.map(f => `<li>${f}</li>`).join("");

  // Metrics
  $("metric-events-val").textContent = result.pred_events.length;
  $("metric-gt-val").textContent = result.gt_events.length;
  $("metric-f1-val").textContent = result.f1.toFixed(3);
  $("metric-thresh-val").textContent = result.threshold.toFixed(3);

  // F1 border color
  const f1Card = $("metric-f1");
  f1Card.style.borderColor = result.f1 >= 0.7
    ? "rgba(16,185,129,0.4)"
    : result.f1 >= 0.4
      ? "rgba(245,158,11,0.4)"
      : "rgba(239,68,68,0.4)";

  // Draw score chart
  drawScoreChart(
    $("score-chart"),
    result.times,
    result.raw_scores,
    result.smoothed_scores,
    result.threshold,
    result.pred_events,
    result.gt_events,
  );

  // Events table
  const tbody = $("events-tbody");
  tbody.innerHTML = "";
  result.pred_events.forEach((ev, idx) => {
    let matchType = "miss";
    for (const gt of result.gt_events) {
      const inter = Math.max(0, Math.min(gt.offset, ev.offset) - Math.max(gt.onset, ev.onset));
      const union = (gt.offset - gt.onset) + (ev.offset - ev.onset) - inter;
      const iou = union > 0 ? inter / union : 0;
      if (iou >= 0.5) { matchType = "hit"; break; }
      else if (iou >= 0.2 && matchType !== "hit") { matchType = "partial"; }
    }
    const matchLabels = { hit: "✓ Match", miss: "✗ Miss", partial: "~ Partial" };
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${idx + 1}</td>
      <td>${ev.onset.toFixed(3)}</td>
      <td>${ev.offset.toFixed(3)}</td>
      <td>${(ev.offset - ev.onset).toFixed(3)}</td>
      <td><span class="match-badge ${matchType}">${matchLabels[matchType]}</span></td>
    `;
    tbody.appendChild(tr);
  });

  if (result.pred_events.length === 0) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="5" style="text-align:center;color:var(--text-muted);padding:20px;">No events detected</td>`;
    tbody.appendChild(tr);
  }

  // Stats grid
  const s = result.stats;
  $("stats-grid").innerHTML = [
    { label: "Raw Score Min", value: s.raw_min.toFixed(4) },
    { label: "Raw Score Max", value: s.raw_max.toFixed(4) },
    { label: "Raw Score Mean", value: s.raw_mean.toFixed(4) },
    { label: "Raw Score Std", value: s.raw_std.toFixed(4) },
    { label: "Smooth Min", value: s.smooth_min.toFixed(4) },
    { label: "Smooth Max", value: s.smooth_max.toFixed(4) },
    { label: "Smooth Mean", value: s.smooth_mean.toFixed(4) },
    { label: "Frames Above θ", value: `${s.frames_above}/${s.total_frames} (${(100 * s.frames_above / s.total_frames).toFixed(1)}%)` },
    { label: "Total Frames", value: s.total_frames },
    { label: "Hop Size", value: `${s.hop_ms.toFixed(2)} ms` },
    { label: "Duration", value: `${s.duration.toFixed(1)} s` },
    { label: "Kernel Size", value: s.kernel_size },
  ].map(item => `
    <div class="stat-item">
      <div class="stat-label">${item.label}</div>
      <div class="stat-value">${item.value}</div>
    </div>
  `).join("");

  // Scroll to results
  $("results-section").scrollIntoView({ behavior: "smooth", block: "start" });
}


// ── Chart Drawing ────────────────────────────────────────────

function drawScoreChart(canvas, times, rawScores, smoothedScores, threshold, predEvents, gtEvents) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  const W = rect.width - 40;
  const H = 280;

  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + "px";
  canvas.style.height = H + "px";
  ctx.scale(dpr, dpr);

  const pad = { top: 20, right: 20, bottom: 35, left: 50 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const tMax = times[times.length - 1] || 12;
  const allVals = [...rawScores, ...smoothedScores, threshold];
  const yMin = Math.min(...allVals) - 0.02;
  const yMax = Math.max(...allVals) + 0.02;

  function xMap(t) { return pad.left + (t / tMax) * plotW; }
  function yMap(v) { return pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH; }

  // Background
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = "rgba(148,163,184,0.08)";
  ctx.lineWidth = 0.5;
  for (let v = Math.ceil(yMin * 10) / 10; v <= yMax; v += 0.05) {
    ctx.beginPath(); ctx.moveTo(pad.left, yMap(v)); ctx.lineTo(W - pad.right, yMap(v)); ctx.stroke();
  }
  for (let t = 0; t <= tMax; t += 1) {
    ctx.beginPath(); ctx.moveTo(xMap(t), pad.top); ctx.lineTo(xMap(t), H - pad.bottom); ctx.stroke();
  }

  // GT events (orange)
  for (const ev of gtEvents) {
    ctx.fillStyle = "rgba(251, 191, 36, 0.12)";
    const x1 = xMap(ev.onset), x2 = xMap(ev.offset);
    ctx.fillRect(x1, pad.top, x2 - x1, plotH);
    ctx.fillStyle = "rgba(251, 191, 36, 0.5)";
    ctx.fillRect(x1, pad.top, x2 - x1, 3);
  }

  // Predicted events (green)
  for (const ev of predEvents) {
    ctx.fillStyle = "rgba(16, 185, 129, 0.15)";
    const x1 = xMap(ev.onset), x2 = xMap(ev.offset);
    ctx.fillRect(x1, pad.top, x2 - x1, plotH);
    ctx.fillStyle = "rgba(16, 185, 129, 0.5)";
    ctx.fillRect(x1, H - pad.bottom - 3, x2 - x1, 3);
  }

  // Threshold line
  ctx.strokeStyle = "#ef4444";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 4]);
  ctx.beginPath(); ctx.moveTo(pad.left, yMap(threshold)); ctx.lineTo(W - pad.right, yMap(threshold)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#ef4444";
  ctx.font = "600 10px 'Inter'";
  ctx.fillText(`θ = ${threshold.toFixed(3)}`, W - pad.right - 70, yMap(threshold) - 5);

  // Raw scores
  ctx.strokeStyle = "#6366f1";
  ctx.lineWidth = 0.8;
  ctx.globalAlpha = 0.35;
  ctx.beginPath();
  for (let i = 0; i < times.length; i++) {
    ctx[i === 0 ? "moveTo" : "lineTo"](xMap(times[i]), yMap(rawScores[i]));
  }
  ctx.stroke();
  ctx.globalAlpha = 1;

  // Smoothed scores
  ctx.strokeStyle = "#06b6d4";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < times.length; i++) {
    ctx[i === 0 ? "moveTo" : "lineTo"](xMap(times[i]), yMap(smoothedScores[i]));
  }
  ctx.stroke();

  // Axes
  ctx.strokeStyle = "rgba(148,163,184,0.3)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, H - pad.bottom); ctx.lineTo(W - pad.right, H - pad.bottom);
  ctx.stroke();

  // X labels
  ctx.fillStyle = "#64758b";
  ctx.font = "500 10px 'Inter'";
  ctx.textAlign = "center";
  const xStep = tMax > 30 ? 5 : tMax > 15 ? 2 : 1;
  for (let t = 0; t <= tMax; t += xStep) {
    ctx.fillText(t + "s", xMap(t), H - pad.bottom + 16);
  }

  // Y labels
  ctx.textAlign = "right";
  for (let v = Math.ceil(yMin * 10) / 10; v <= yMax; v += 0.1) {
    ctx.fillText(v.toFixed(1), pad.left - 6, yMap(v) + 4);
  }

  // Axis titles
  ctx.fillStyle = "#94a3b8";
  ctx.font = "500 11px 'Inter'";
  ctx.textAlign = "center";
  ctx.fillText("Time (s)", pad.left + plotW / 2, H - 2);

  ctx.save();
  ctx.translate(12, pad.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Score", 0, 0);
  ctx.restore();
}
