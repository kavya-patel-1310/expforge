/* ExpForge dashboard — dependency-free SPA (vanilla JS + Chart.js). */
"use strict";

const API = "/api";
const PALETTE = ["#5b9dff", "#3fb68b", "#e3b341", "#f0616d", "#b083f0", "#3ec1d3", "#f28c38", "#d16ba5", "#8fbf4d", "#a0a8b8"];
const LOWER_IS_BETTER = /(loss|error|mae|mse|rmse|brier|time|latency|size|gap|ece)/i;
const app = document.getElementById("app");
let charts = [];
let pollTimer = null;

Chart.defaults.color = "#8b98a8";
Chart.defaults.borderColor = "#2a3441";
Chart.defaults.font.family = "Inter, system-ui, sans-serif";
Chart.defaults.plugins.legend.labels.boxWidth = 12;
Chart.defaults.animation.duration = 250;

// ------------------------------------------------------------------ utils
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 4) => (typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : Math.abs(v) >= 1000 ? v.toFixed(1) : v.toFixed(d)) : v == null ? "—" : esc(typeof v === "object" ? JSON.stringify(v) : v));
// metric formatter: always show decimals for scores (so 1.0 reads as 1.0000, not "1")
const fm = (v, d = 4) => (typeof v === "number" ? (Number.isInteger(v) && Math.abs(v) > 1 ? v.toLocaleString() : Math.abs(v) >= 1000 ? v.toFixed(1) : v.toFixed(d)) : fmt(v));
const ago = (iso) => {
  const s = (Date.now() - new Date(iso + "Z").getTime()) / 1000;
  if (s < 60) return `${Math.floor(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
};
const dur = (s) => (s == null ? "—" : s < 60 ? `${s.toFixed(1)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`);
const badge = (s) => `<span class="badge ${esc(s)}">${esc(s)}</span>`;
const tags = (t) => Object.entries(t || {}).filter(([k]) => !["sweep_id", "trial", "task", "model_family"].includes(k)).map(([k, v]) => `<span class="tag">${esc(k)}:${esc(v)}</span>`).join("");

async function api(path, opts = {}) {
  const res = await fetch(API + path, { headers: { "Content-Type": "application/json" }, ...opts, body: opts.body ? JSON.stringify(opts.body) : undefined });
  if (!res.ok) throw new Error(`${res.status}: ${(await res.text()).slice(0, 200)}`);
  return res.json();
}
function toast(msg, err = false) {
  const t = document.getElementById("toast");
  t.textContent = msg; t.className = `toast show ${err ? "err" : ""}`;
  clearTimeout(t._h); t._h = setTimeout(() => (t.className = "toast"), 3200);
}
function chart(canvasId, config) {
  const el = document.getElementById(canvasId);
  if (!el) return null;
  const c = new Chart(el, config);
  charts.push(c);
  return c;
}
const lineOpts = (xTitle, yTitle, extra = {}) => ({
  responsive: true, maintainAspectRatio: false, interaction: { mode: "nearest", intersect: false },
  scales: { x: { type: "linear", title: { display: !!xTitle, text: xTitle } }, y: { title: { display: !!yTitle, text: yTitle } } },
  plugins: { legend: { position: "bottom" } }, elements: { point: { radius: 0 } }, ...extra,
});
// Zoom a value axis around the data so close scores are distinguishable (bars no longer start at 0).
const zoomAxis = (vals) => { const v = vals.filter((x) => typeof x === "number"); if (!v.length) return {}; const lo = Math.min(...v), hi = Math.max(...v), pad = (hi - lo || Math.abs(hi) * 0.05 || 1) * 0.6; return { min: lo >= 0 && lo - pad < 0 ? 0 : +(lo - pad).toPrecision(3), max: +(hi + pad * 0.3).toPrecision(4) }; };
const primaryMetric = (summary) => ["test_f1_macro", "test_r2", "test_accuracy", "val_f1_macro"].find((k) => k in (summary || {}));

// ------------------------------------------------------------------ router
const routes = [
  [/^#?\/?$/, overview], [/^#\/experiments$/, experimentsPage], [/^#\/exp\/(\d+)$/, experimentPage],
  [/^#\/run\/(\d+)$/, runPage], [/^#\/compare\?ids=([\d,]+)$/, comparePage], [/^#\/sweeps$/, sweepsPage],
  [/^#\/models$/, modelsPage], [/^#\/launch$/, launchPage],
];
async function render() {
  charts.forEach((c) => c.destroy()); charts = [];
  clearInterval(pollTimer);
  const hash = location.hash || "#/";
  const section = hash.split("/")[1]?.split("?")[0] || "overview";
  const navKey = { exp: "experiments", run: "experiments", compare: "experiments" }[section] || section || "overview";
  document.querySelectorAll("nav a").forEach((a) => a.classList.toggle("active", a.dataset.nav === navKey));
  for (const [re, fn] of routes) {
    const m = hash.match(re);
    if (m) {
      try { await fn(...m.slice(1)); } catch (e) { app.innerHTML = `<div class="card empty">Error: ${esc(e.message)}</div>`; }
      window.scrollTo(0, 0);
      return;
    }
  }
  app.innerHTML = `<div class="card empty">Page not found</div>`;
}
window.addEventListener("hashchange", render);

// ------------------------------------------------------------------ overview
async function overview() {
  const [stats, exps, runs, jobs] = await Promise.all([api("/stats"), api("/experiments"), api("/runs?limit=12"), api("/jobs")]);
  const st = stats.runs_by_status || {};
  app.innerHTML = `
    <h1>Overview</h1><p class="sub">Everything tracked across your ML workspace.</p>
    <div class="tiles">
      <div class="tile"><div class="label">Experiments</div><div class="value">${stats.experiments}</div></div>
      <div class="tile"><div class="label">Runs</div><div class="value">${stats.runs}</div><div class="hint">${st.finished || 0} finished · ${st.running || 0} running · ${st.failed || 0} failed</div></div>
      <div class="tile"><div class="label">Sweeps</div><div class="value">${stats.sweeps}</div></div>
      <div class="tile"><div class="label">Metric points</div><div class="value">${stats.metric_points.toLocaleString()}</div></div>
      <div class="tile"><div class="label">Registered models</div><div class="value">${stats.registered_models}</div></div>
    </div>
    <div class="grid g2">
      <div class="card"><h3>Experiments</h3><div class="grid">${exps.map((e) => `
        <a class="card exp-card" href="#/exp/${e.id}"><div class="row"><span class="name">${esc(e.name)}</span><span class="spacer"></span><span class="muted">${e.run_count} runs</span></div>
        <div class="muted" style="font-size:12.5px">created ${ago(e.created_at)}</div></a>`).join("") || `<div class="empty">No experiments yet — run <span class="mono">expforge demo</span></div>`}</div></div>
      <div class="card"><h3>Runs by model family</h3><div class="chart-box"><canvas id="c-family"></canvas></div></div>
    </div>
    <div class="card tight" style="margin-top:16px"><div style="padding:16px 18px 6px"><h3>Recent runs</h3></div>
      <div class="table-wrap"><table><thead><tr><th>Run</th><th>Experiment</th><th>Status</th><th>Model</th><th class="num">Key metric</th><th class="num">Duration</th><th>Started</th></tr></thead><tbody>
      ${runs.map((r) => { const k = primaryMetric(r.summary); return `<tr><td><a href="#/run/${r.id}">${esc(r.name)}</a></td><td>${esc(r.experiment_name)}</td><td>${badge(r.status)}</td><td>${esc(r.params.model || "")}</td><td class="num">${k ? `<span class="muted">${k}</span> ${fm(r.summary[k])}` : "—"}</td><td class="num">${dur(r.duration_s)}</td><td class="muted">${ago(r.started_at)}</td></tr>`; }).join("")}
      </tbody></table></div></div>
    ${jobs.length ? `<div class="card tight" style="margin-top:16px"><div style="padding:16px 18px 6px"><h3>Training jobs</h3></div><table><thead><tr><th>Job</th><th>Status</th><th>Model</th><th>Dataset</th><th>Runs</th><th>Created</th></tr></thead><tbody>
      ${jobs.slice(0, 8).map((j) => `<tr><td>#${j.id}</td><td>${badge(j.status)}</td><td>${esc(j.config.model?.name)}</td><td>${esc(j.config.dataset?.name)}</td><td>${j.run_ids.map((id) => `<a href="#/run/${id}">${id}</a>`).join(", ") || "—"}</td><td class="muted">${ago(j.created_at)}</td></tr>`).join("")}
      </tbody></table></div>` : ""}`;
  const all = await api("/runs?limit=2000");
  const fam = {};
  all.forEach((r) => (fam[r.params.model || "custom"] = (fam[r.params.model || "custom"] || 0) + 1));
  const entries = Object.entries(fam).sort((a, b) => b[1] - a[1]);
  chart("c-family", { type: "bar", data: { labels: entries.map((e) => e[0]), datasets: [{ data: entries.map((e) => e[1]), backgroundColor: PALETTE[0], borderRadius: 4 }] },
    options: { indexAxis: "y", maintainAspectRatio: false, plugins: { legend: { display: false } } } });
  if (jobs.some((j) => ["queued", "running"].includes(j.status))) pollTimer = setInterval(render, 4000);
}

// ------------------------------------------------------------------ experiments
async function experimentsPage() {
  const exps = await api("/experiments");
  app.innerHTML = `<h1>Experiments</h1><p class="sub">Each experiment groups runs that attack the same problem.</p>
    <div class="card tight"><table><thead><tr><th>Name</th><th class="num">Runs</th><th>Description</th><th>Created</th><th></th></tr></thead><tbody>
    ${exps.map((e) => `<tr><td><a href="#/exp/${e.id}">${esc(e.name)}</a></td><td class="num">${e.run_count}</td><td class="muted">${esc(e.description || "—")}</td><td class="muted">${ago(e.created_at)}</td>
    <td style="text-align:right"><button data-del="${e.id}">Delete</button></td></tr>`).join("") || `<tr><td colspan="5" class="empty">No experiments</td></tr>`}</tbody></table></div>`;
  app.querySelectorAll("[data-del]").forEach((b) => (b.onclick = async () => {
    if (!confirm("Delete experiment and all of its runs and artifacts?")) return;
    await api(`/experiments/${b.dataset.del}`, { method: "DELETE" }); toast("Experiment deleted"); render();
  }));
}

async function experimentPage(id) {
  const [exp, runs] = await Promise.all([api(`/experiments/${id}`), api(`/runs?experiment_id=${id}`)]);
  const metricKeys = [...new Set(runs.flatMap((r) => Object.keys(r.summary).filter((k) => typeof r.summary[k] === "number")))].sort();
  const paramKeys = [...new Set(runs.flatMap((r) => Object.keys(r.params)))].filter((k) => !["dataset", "seed", "test_size", "val_size"].includes(k));
  const defaultMetric = metricKeys.includes("val_f1_macro") ? "val_f1_macro" : metricKeys.includes("val_r2") ? "val_r2" : metricKeys[0];
  const cols = ["val_f1_macro", "test_f1_macro", "test_accuracy", "test_roc_auc", "val_r2", "test_r2", "test_rmse", "test_mae", "train_time_s", "latency_ms_per_sample"].filter((k) => metricKeys.includes(k));
  const hasTags = runs.some((r) => tags(r.tags));
  // default scatter param: the numeric hyperparameter with the most distinct values
  const scatterDefault = paramKeys.map((k) => [k, new Set(runs.map((r) => JSON.stringify(r.params[k]))).size, runs.every((r) => typeof r.params[k] === "number")]).filter(([k]) => k !== "random_state").sort((a, b) => (b[2] - a[2]) || (b[1] - a[1]))[0]?.[0];
  const state = { sort: defaultMetric, dir: LOWER_IS_BETTER.test(defaultMetric || "") ? 1 : -1, selected: new Set(), filter: "" };

  app.innerHTML = `
    <div class="crumbs"><a href="#/experiments">Experiments</a> /</div>
    <div class="row"><h1>${esc(exp.name)}</h1><span class="spacer"></span>
      <button id="btn-compare" class="primary" disabled>Compare selected</button></div>
    <p class="sub">${runs.length} runs · created ${ago(exp.created_at)}</p>
    <div class="grid g2">
      <div class="card"><div class="row" style="margin-bottom:10px"><h3 style="margin:0">Leaderboard</h3><span class="spacer"></span>
        <select id="lb-metric">${metricKeys.map((k) => `<option ${k === defaultMetric ? "selected" : ""}>${k}</option>`).join("")}</select></div>
        <div class="chart-box"><canvas id="c-lb"></canvas></div></div>
      <div class="card"><div class="row" style="margin-bottom:10px"><h3 style="margin:0">Hyperparameter vs metric</h3><span class="spacer"></span>
        <select id="sc-param">${paramKeys.map((k) => `<option ${k === scatterDefault ? "selected" : ""}>${k}</option>`).join("")}</select></div>
        <div class="chart-box"><canvas id="c-scatter"></canvas></div></div>
    </div>
    <div class="card" style="margin-top:16px"><h3>Parallel coordinates <span class="muted" style="text-transform:none;letter-spacing:0">— each line is a run, coloured by <span id="pc-metric"></span></span></h3><div id="pcoords"></div></div>
    <div class="card tight" style="margin-top:16px">
      <div class="row" style="padding:14px 16px"><h3 style="margin:0">Runs</h3><span class="spacer"></span><input id="filter" placeholder="Filter by name, model, tag…" style="width:260px"></div>
      <div class="table-wrap"><table id="runs-table"></table></div></div>`;

  const sortVal = (r, k) => (k in r.summary ? r.summary[k] : k === "name" ? r.name : k === "model" ? r.params.model : k === "started" ? r.started_at : null);
  function drawTable() {
    const f = state.filter.toLowerCase();
    const rows = runs.filter((r) => !f || JSON.stringify([r.name, r.params, r.tags]).toLowerCase().includes(f))
      .sort((a, b) => { const x = sortVal(a, state.sort), y = sortVal(b, state.sort); if (x == null) return 1; if (y == null) return -1; return (x > y ? 1 : x < y ? -1 : 0) * state.dir; });
    const best = {};
    cols.forEach((k) => { const v = runs.map((r) => r.summary[k]).filter((x) => typeof x === "number"); if (v.length) best[k] = LOWER_IS_BETTER.test(k) ? Math.min(...v) : Math.max(...v); });
    const th = (k, label, num) => `<th class="sortable ${num ? "num" : ""}" data-sort="${k}">${label}${state.sort === k ? (state.dir > 0 ? " ▲" : " ▼") : ""}</th>`;
    document.getElementById("runs-table").innerHTML = `<thead><tr><th><input type="checkbox" id="sel-all"></th>${th("name", "Run")}<th>Status</th>${th("model", "Model")}${hasTags ? "<th>Tags</th>" : ""}
      ${cols.map((k) => th(k, k.replace(/_/g, " "), true)).join("")}${th("started", "Started")}</tr></thead><tbody>
      ${rows.map((r) => `<tr class="${state.selected.has(r.id) ? "selected" : ""}"><td><input type="checkbox" data-sel="${r.id}" ${state.selected.has(r.id) ? "checked" : ""}></td>
        <td><a href="#/run/${r.id}">${esc(r.name)}</a></td><td>${badge(r.status)}</td><td>${esc(r.params.model || "")}</td>${hasTags ? `<td>${tags(r.tags)}</td>` : ""}
        ${cols.map((k) => `<td class="num ${r.summary[k] === best[k] ? "best" : ""}">${fm(r.summary[k])}</td>`).join("")}<td class="muted">${ago(r.started_at)}</td></tr>`).join("")}</tbody>`;
    document.querySelectorAll("[data-sort]").forEach((el) => (el.onclick = () => { const k = el.dataset.sort; state.dir = state.sort === k ? -state.dir : LOWER_IS_BETTER.test(k) ? 1 : -1; state.sort = k; drawTable(); }));
    document.querySelectorAll("[data-sel]").forEach((el) => (el.onchange = () => { const id = +el.dataset.sel; el.checked ? state.selected.add(id) : state.selected.delete(id); drawTable(); }));
    document.getElementById("sel-all").onchange = (e) => { rows.forEach((r) => (e.target.checked ? state.selected.add(r.id) : state.selected.delete(r.id))); drawTable(); };
    const btn = document.getElementById("btn-compare");
    btn.disabled = state.selected.size < 2 || state.selected.size > 20;
    btn.textContent = `Compare selected (${state.selected.size})`;
  }
  document.getElementById("filter").oninput = (e) => { state.filter = e.target.value; drawTable(); };
  document.getElementById("btn-compare").onclick = () => (location.hash = `#/compare?ids=${[...state.selected].join(",")}`);
  drawTable();

  let lbChart, scChart;
  async function drawLeaderboard() {
    const m = document.getElementById("lb-metric").value;
    const lb = await api(`/experiments/${id}/leaderboard?metric=${encodeURIComponent(m)}&direction=${LOWER_IS_BETTER.test(m) ? "minimize" : "maximize"}&limit=12`);
    lbChart?.destroy();
    lbChart = chart("c-lb", { type: "bar", data: { labels: lb.map((r) => r.name), datasets: [{ label: m, data: lb.map((r) => r.value), backgroundColor: lb.map((_, i) => (i === 0 ? PALETTE[1] : PALETTE[0])), borderRadius: 4 }] },
      options: { indexAxis: "y", maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: zoomAxis(lb.map((r) => r.value)) }, onClick: (_, els) => els.length && (location.hash = `#/run/${lb[els[0].index].run_id}`) } });
    drawScatter(); drawPcoords(m);
  }
  function drawScatter() {
    const m = document.getElementById("lb-metric").value, p = document.getElementById("sc-param").value;
    const pts = runs.filter((r) => typeof r.summary[m] === "number" && r.params[p] != null);
    const numeric = pts.every((r) => typeof r.params[p] === "number");
    const cats = numeric ? null : [...new Set(pts.map((r) => String(r.params[p])))];
    scChart?.destroy();
    scChart = chart("c-scatter", { type: "scatter", data: { datasets: [{ data: pts.map((r) => ({ x: numeric ? r.params[p] : cats.indexOf(String(r.params[p])), y: r.summary[m], name: r.name, id: r.id })), backgroundColor: PALETTE[0], pointRadius: 6, pointHoverRadius: 9 }] },
      options: { maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${c.raw.name}: ${p}=${numeric ? c.raw.x : cats[c.raw.x]}, ${m}=${fmt(c.raw.y)}` } } },
        scales: { x: { title: { display: true, text: p }, type: numeric && pts.length && Math.max(...pts.map((r) => r.params[p])) / Math.max(1e-12, Math.min(...pts.map((r) => r.params[p]))) > 50 ? "logarithmic" : "linear", ticks: cats ? { stepSize: 1, callback: (v) => cats[v] ?? "" } : {} }, y: { title: { display: true, text: m } } },
        onClick: (_, els) => els.length && (location.hash = `#/run/${pts[els[0].index].id}`) } });
  }
  function drawPcoords(metric) {
    document.getElementById("pc-metric").textContent = metric;
    parallelCoords(document.getElementById("pcoords"), runs, paramKeys.filter((k) => k !== "model" || new Set(runs.map((r) => r.params.model)).size > 1), metric);
  }
  document.getElementById("lb-metric").onchange = drawLeaderboard;
  document.getElementById("sc-param").onchange = drawScatter;
  if (metricKeys.length) drawLeaderboard();
}

/* Parallel-coordinates plot in raw SVG: hyperparameter axes + the target metric as the last axis. */
function parallelCoords(el, runs, params, metric) {
  const usable = params.filter((p) => p !== "random_state" && runs.every((r) => r.params[p] != null) && new Set(runs.map((r) => JSON.stringify(r.params[p]))).size > 1).slice(0, 8);
  const rows = runs.filter((r) => typeof r.summary[metric] === "number");
  if (!usable.length || rows.length < 2) { el.innerHTML = `<div class="empty">Need at least two runs that share a differing hyperparameter (sweeps are ideal).</div>`; return; }
  const axes = [...usable, metric];
  const W = Math.max(el.clientWidth, 600), H = 300, pad = { l: 50, r: 70, t: 28, b: 24 };
  const x = (i) => pad.l + (i * (W - pad.l - pad.r)) / (axes.length - 1);
  const scales = axes.map((a, i) => {
    const vals = rows.map((r) => (i === axes.length - 1 ? r.summary[a] : r.params[a]));
    if (vals.every((v) => typeof v === "number" || v == null)) {
      const nums = vals.filter((v) => v != null), lo = Math.min(...nums), hi = Math.max(...nums);
      const log = lo > 0 && hi / lo > 50;
      const f = (v) => (log ? Math.log(v) : v), span = f(hi) - f(lo) || 1;
      return { pos: (v) => (v == null ? H - pad.b : H - pad.b - ((f(v) - f(lo)) / span) * (H - pad.t - pad.b)), ticks: [lo, hi].map((v) => [v, fmt(v, 3)]) };
    }
    const cats = [...new Set(vals.map(String))];
    return { pos: (v) => H - pad.b - ((cats.indexOf(String(v)) + 0.5) / cats.length) * (H - pad.t - pad.b), ticks: cats.map((c) => [c, c]) };
  });
  const ms = rows.map((r) => r.summary[metric]), mlo = Math.min(...ms), mhi = Math.max(...ms);
  const lowerBetter = LOWER_IS_BETTER.test(metric);
  const color = (v) => { let t = (v - mlo) / (mhi - mlo || 1); if (lowerBetter) t = 1 - t; return `hsl(${Math.round(355 + t * 150) % 360}, 70%, ${45 + t * 12}%)`; };
  const lines = rows.map((r) => {
    const d = axes.map((a, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${scales[i].pos(i === axes.length - 1 ? r.summary[a] : r.params[a]).toFixed(1)}`).join(" ");
    return `<a href="#/run/${r.id}"><path class="line" d="${d}" stroke="${color(r.summary[metric])}"><title>${esc(r.name)} — ${metric}=${fmt(r.summary[metric])}</title></path></a>`;
  }).join("");
  const axesSvg = axes.map((a, i) => {
    const s = scales[i];
    const ticks = s.ticks.map(([v, label]) => `<text x="${x(i) + 6}" y="${s.pos(v) + 4}">${esc(String(label).slice(0, 12))}</text>`).join("");
    return `<line class="axis-line" x1="${x(i)}" x2="${x(i)}" y1="${pad.t}" y2="${H - pad.b}"/><text x="${x(i)}" y="${pad.t - 10}" text-anchor="middle" style="fill:${i === axes.length - 1 ? "#e6edf3" : ""};font-weight:600">${esc(a)}</text>${ticks}`;
  }).join("");
  el.innerHTML = `<svg class="pcoords" width="100%" viewBox="0 0 ${W} ${H}">${lines}${axesSvg}</svg>`;
}

// ------------------------------------------------------------------ run detail
async function runPage(id) {
  const [run, series] = await Promise.all([api(`/runs/${id}`), api(`/runs/${id}/metrics`)]);
  const s = run.summary, ev = run.evaluation || {};
  const isClf = ev.type === "classification";
  const headline = isClf ? ["test_accuracy", "test_f1_macro", "test_roc_auc", "test_log_loss", "cv_mean", "train_time_s", "latency_ms_per_sample", "overfit_gap"]
    : ["test_r2", "test_rmse", "test_mae", "test_mape", "cv_mean", "train_time_s", "latency_ms_per_sample", "overfit_gap"];
  app.innerHTML = `
    <div class="crumbs"><a href="#/experiments">Experiments</a> / <a href="#/exp/${run.experiment_id}">${esc(run.experiment_name)}</a> /</div>
    <div class="row"><h1>${esc(run.name)}</h1>${badge(run.status)}<span class="spacer"></span>
      ${run.artifacts.some((a) => a.kind === "model") ? `<button id="btn-register" class="primary">Register model</button>` : ""}</div>
    <p class="sub">Run #${run.id} · ${esc(run.params.model || "")} on ${esc(run.params.dataset || "")} · ${dur(run.duration_s)} · started ${ago(run.started_at)} ${tags(run.tags)}</p>
    ${run.error ? `<div class="card" style="border-color:var(--bad);margin-bottom:16px"><b style="color:var(--bad)">Error</b><pre class="code">${esc(run.error)}</pre></div>` : ""}
    <div class="tiles">${headline.filter((k) => k in s).map((k) => `<div class="tile"><div class="label">${k.replace(/_/g, " ")}</div><div class="value">${fm(s[k], 4)}</div>${k === "cv_mean" ? `<div class="hint">± ${fmt(s.cv_std, 4)} (${s.cv_folds}-fold ${esc(s.cv_metric)})</div>` : ""}</div>`).join("")}</div>
    <div class="tabs">${["Curves", "Evaluation", "Features", "Metrics", "Params & System", "Artifacts"].map((t, i) => `<button data-tab="${i}" class="${i === 0 ? "active" : ""}">${t}</button>`).join("")}</div>
    <div id="tab-body"></div>`;

  const tabs = [
    () => {  // learning curves
      const keys = Object.keys(series);
      if (!keys.length) return `<div class="card empty">This model was trained in a single fit — no per-epoch curves. See CV folds under Metrics.</div>`;
      const groups = {};
      keys.forEach((k) => { const base = k.replace(/^(train|val)_/, ""); (groups[base] = groups[base] || []).push(k); });
      setTimeout(() => Object.entries(groups).forEach(([g, ks], gi) => chart(`c-curve-${gi}`, {
        type: "line", data: { datasets: ks.map((k, i) => ({ label: k, data: series[k].map((p) => ({ x: p.step, y: p.value })), borderColor: PALETTE[i], backgroundColor: PALETTE[i], borderWidth: 2, tension: .2, pointRadius: series[k].length < 15 ? 3 : 0 })) },
        options: lineOpts(s.epochs_trained ? "epoch / iteration" : "step", g, { plugins: { legend: { position: "bottom" }, annotation: {} } }) })));
      return `${s.best_epoch ? `<p class="muted">Best checkpoint at step <b style="color:var(--text)">${s.best_epoch}</b> of ${s.epochs_trained}${s.early_stopped ? " · early stopped" : ""} · best val loss ${fmt(s.best_val_loss)}</p>` : ""}
        <div class="grid g2">${Object.keys(groups).map((g, gi) => `<div class="card"><h3>${esc(g)}</h3><div class="chart-box"><canvas id="c-curve-${gi}"></canvas></div></div>`).join("")}</div>`;
    },
    () => (isClf ? evalClassification(ev) : ev.type === "regression" ? evalRegression(ev) : `<div class="card empty">No evaluation report logged.</div>`),
    () => {
      const fi = ev.feature_importance;
      if (!fi) return `<div class="card empty">No feature importance available.</div>`;
      setTimeout(() => chart("c-fi", { type: "bar", data: { labels: fi.features, datasets: [{ data: fi.importance, backgroundColor: PALETTE[0], borderRadius: 3 }] }, options: { indexAxis: "y", maintainAspectRatio: false, plugins: { legend: { display: false } } } }));
      return `<div class="card"><h3>Top features · ${esc(fi.method)}</h3><div class="chart-box" style="height:${Math.max(260, fi.features.length * 24)}px"><canvas id="c-fi"></canvas></div></div>`;
    },
    () => {
      const keys = Object.keys(s).filter((k) => typeof s[k] === "number").sort();
      const split = (p) => keys.filter((k) => k.startsWith(p));
      const metricNames = [...new Set(keys.filter((k) => /^(train|val|test)_/.test(k)).map((k) => k.replace(/^(train|val|test)_/, "")))];
      const other = keys.filter((k) => !/^(train|val|test)_/.test(k));
      return `<div class="grid g2"><div class="card tight"><table><thead><tr><th>Metric</th><th class="num">Train</th><th class="num">Val</th><th class="num">Test</th></tr></thead><tbody>
        ${metricNames.map((m) => `<tr><td>${m}</td>${["train", "val", "test"].map((p) => `<td class="num">${fm(s[`${p}_${m}`])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
        <div class="card tight"><table><thead><tr><th>Other</th><th class="num">Value</th></tr></thead><tbody>${other.map((k) => `<tr><td>${k}</td><td class="num">${fmt(s[k])}</td></tr>`).join("")}</tbody></table></div></div>`;
    },
    () => `<div class="grid g2"><div class="card"><h3>Hyperparameters</h3><dl class="kv">${Object.entries(run.params).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${fmt(v)}</dd>`).join("")}</dl></div>
      <div class="card"><h3>System & data lineage</h3><dl class="kv">${Object.entries(run.system_info).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${fmt(v)}</dd>`).join("")}</dl></div></div>
      ${ev.config ? `<div class="card" style="margin-top:16px"><h3>Resolved training config (reproducible)</h3><pre class="code">${esc(JSON.stringify(ev.config, null, 2))}</pre></div>` : ""}`,
    () => `<div class="card tight"><table><thead><tr><th>Name</th><th>Kind</th><th class="num">Size</th><th>SHA-256</th><th></th></tr></thead><tbody>
      ${run.artifacts.map((a) => `<tr><td>${esc(a.name)}</td><td><span class="badge neutral">${esc(a.kind)}</span></td><td class="num">${(a.size_bytes / 1024).toFixed(1)} KB</td><td class="mono muted">${a.sha256.slice(0, 16)}…</td><td style="text-align:right"><a class="btn" href="${API}/artifacts/${a.id}/download">Download</a></td></tr>`).join("") || `<tr><td colspan="5" class="empty">No artifacts</td></tr>`}</tbody></table></div>`,
  ];
  const showTab = (i) => { charts.forEach((c) => c.destroy()); charts = []; document.getElementById("tab-body").innerHTML = tabs[i](); document.querySelectorAll("[data-tab]").forEach((b) => b.classList.toggle("active", +b.dataset.tab === i)); };
  document.querySelectorAll("[data-tab]").forEach((b) => (b.onclick = () => showTab(+b.dataset.tab)));
  showTab(0);
  const reg = document.getElementById("btn-register");
  if (reg) reg.onclick = async () => {
    const name = prompt("Registered model name:", run.experiment_name.replace(/[^a-z0-9]+/gi, "-") + "-model");
    if (!name) return;
    try { const m = await api("/models", { method: "POST", body: { name, run_id: run.id } }); toast(`Registered ${name} v${m.versions.at(-1).version}`); }
    catch (e) { toast(e.message, true); }
  };
  if (run.status === "running") pollTimer = setInterval(render, 3000);
}

function evalClassification(ev) {
  const cm = ev.confusion_matrix, n = cm.length, max = Math.max(...cm.flat());
  const rowSums = cm.map((r) => r.reduce((a, b) => a + b, 0));
  const cells = cm.map((row, i) => row.map((v, j) => { const t = rowSums[i] ? v / rowSums[i] : 0; const diag = i === j;
    return `<div class="cell" title="true ${esc(ev.class_names[i])} → pred ${esc(ev.class_names[j])}: ${v} (${(t * 100).toFixed(1)}%)" style="background:${diag ? `rgba(63,182,139,${0.12 + t * 0.8})` : v ? `rgba(240,97,109,${0.15 + (v / max) * 0.75})` : "var(--panel-2)"};color:${t > .5 || (!diag && v / max > .5) ? "#fff" : "var(--text)"}">${v}</div>`; }).join(""));
  const short = (c) => esc(String(c).slice(0, 8));
  setTimeout(() => {
    if (ev.roc_curves?.length) chart("c-roc", { type: "line", data: { datasets: [...ev.roc_curves.map((c, i) => ({ label: `${c.label} (AUC ${c.auc.toFixed(3)})`, data: c.fpr.map((x, k) => ({ x, y: c.tpr[k] })), borderColor: PALETTE[i % 10], borderWidth: 2 })), { label: "chance", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: "#444c56", borderDash: [5, 5], borderWidth: 1 }] }, options: lineOpts("false positive rate", "true positive rate", { scales: { x: { type: "linear", min: 0, max: 1, title: { display: true, text: "false positive rate" } }, y: { min: 0, max: 1.02, title: { display: true, text: "true positive rate" } } } }) });
    if (ev.pr_curves?.length) chart("c-pr", { type: "line", data: { datasets: ev.pr_curves.map((c, i) => ({ label: `${c.label} (AP ${c.ap.toFixed(3)})`, data: c.recall.map((x, k) => ({ x, y: c.precision[k] })), borderColor: PALETTE[i % 10], borderWidth: 2, stepped: true })) }, options: lineOpts("recall", "precision", { scales: { x: { type: "linear", min: 0, max: 1, title: { display: true, text: "recall" } }, y: { min: 0, max: 1.02, title: { display: true, text: "precision" } } } }) });
    if (ev.calibration) chart("c-cal", { type: "line", data: { datasets: [{ label: "model", data: ev.calibration.mean_predicted.map((x, i) => ({ x, y: ev.calibration.fraction_positive[i] })), borderColor: PALETTE[0], pointRadius: 4, borderWidth: 2 }, { label: "perfectly calibrated", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: "#444c56", borderDash: [5, 5], borderWidth: 1 }] }, options: lineOpts("mean predicted probability", "fraction of positives", { scales: { x: { type: "linear", min: 0, max: 1, title: { display: true, text: "mean predicted probability" } }, y: { min: 0, max: 1, title: { display: true, text: "fraction of positives" } } } }) });
    if (ev.thresholds) chart("c-th", { type: "line", data: { datasets: ["precision", "recall", "f1"].map((k, i) => ({ label: k, data: ev.thresholds.map((t) => ({ x: t.threshold, y: t[k] })), borderColor: PALETTE[i], borderWidth: 2, pointRadius: 2 })) }, options: lineOpts("decision threshold", "score") });
  });
  return `<div class="grid g2">
      <div class="card"><h3>Confusion matrix (test)</h3><div style="overflow:auto"><div class="heatmap" style="grid-template-columns: 70px repeat(${n}, minmax(40px, 1fr))">
        <div class="axis">true ↓ pred →</div>${ev.class_names.map((c) => `<div class="axis" title="${esc(c)}">${short(c)}</div>`).join("")}
        ${cm.map((row, i) => `<div class="axis" title="${esc(ev.class_names[i])}">${short(ev.class_names[i])}</div>${cells[i]}`).join("")}
      </div></div></div>
      <div class="card tight"><table><thead><tr><th>Class</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th><th class="num">Support</th></tr></thead><tbody>
        ${ev.per_class.map((c) => `<tr><td>${esc(c.class)}</td><td class="num">${fmt(c.precision, 3)}</td><td class="num">${fmt(c.recall, 3)}</td><td class="num">${fmt(c.f1, 3)}</td><td class="num">${c.support}</td></tr>`).join("")}</tbody></table></div>
      ${ev.roc_curves?.length ? `<div class="card"><h3>ROC curve${ev.roc_curves.length > 1 ? "s (one-vs-rest)" : ""}</h3><div class="chart-box tall"><canvas id="c-roc"></canvas></div></div>` : ""}
      ${ev.pr_curves?.length ? `<div class="card"><h3>Precision–recall</h3><div class="chart-box tall"><canvas id="c-pr"></canvas></div></div>` : ""}
      ${ev.calibration ? `<div class="card"><h3>Calibration · ECE ${ev.calibration.ece.toFixed(4)}</h3><div class="chart-box"><canvas id="c-cal"></canvas></div></div>` : ""}
      ${ev.thresholds ? `<div class="card"><h3>Threshold analysis</h3><div class="chart-box"><canvas id="c-th"></canvas></div></div>` : ""}
    </div>`;
}

function evalRegression(ev) {
  setTimeout(() => {
    const a = ev.scatter.actual, p = ev.scatter.predicted, lo = Math.min(...a, ...p), hi = Math.max(...a, ...p);
    chart("c-avp", { type: "scatter", data: { datasets: [{ label: "test samples", data: a.map((x, i) => ({ x, y: p[i] })), backgroundColor: "rgba(91,157,255,.6)", pointRadius: 3.5 }, { type: "line", label: "ideal", data: [{ x: lo, y: lo }, { x: hi, y: hi }], borderColor: "#444c56", borderDash: [5, 5], borderWidth: 1, pointRadius: 0 }] }, options: lineOpts("actual", "predicted") });
    const e = ev.residual_hist.edges;
    chart("c-res", { type: "bar", data: { labels: e.slice(0, -1).map((v, i) => ((v + e[i + 1]) / 2).toFixed(1)), datasets: [{ data: ev.residual_hist.counts, backgroundColor: PALETTE[4], borderRadius: 2, barPercentage: 1, categoryPercentage: 1 }] }, options: { maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: "residual (actual − predicted)" } } } } });
  });
  return `<div class="grid g2"><div class="card"><h3>Predicted vs actual</h3><div class="chart-box tall"><canvas id="c-avp"></canvas></div></div>
    <div class="card"><h3>Residual distribution · mean ${fmt(ev.residual_mean, 3)} · std ${fmt(ev.residual_std, 3)}</h3><div class="chart-box tall"><canvas id="c-res"></canvas></div></div></div>`;
}

// ------------------------------------------------------------------ compare
async function comparePage(ids) {
  const data = await api("/compare", { method: "POST", body: { run_ids: ids.split(",").map(Number) } });
  const runs = data.runs;
  const head = `<tr><th></th>${runs.map((r, i) => `<th><span style="color:${PALETTE[i % 10]}">●</span> <a href="#/run/${r.id}">${esc(r.name)}</a></th>`).join("")}</tr>`;
  const metricRows = Object.entries(data.metrics).filter(([k]) => /^(val|test)_|cv_mean|train_time|latency|model_size|overfit/.test(k));
  const seriesKeys = [...new Set(Object.values(data.series).flatMap((s) => Object.keys(s)))].filter((k) => /^val_/.test(k));
  const barMetrics = metricRows.map(([k]) => k).filter((k) => /^test_(accuracy|f1_macro|roc_auc|precision_macro|recall_macro|r2|balanced_accuracy)$/.test(k));
  app.innerHTML = `
    <div class="crumbs"><a href="#/exp/${runs[0].experiment_id}">${esc(runs[0].experiment_name)}</a> /</div>
    <h1>Compare ${runs.length} runs</h1><p class="sub">Differing hyperparameters are highlighted; the best value of each metric is green.</p>
    <div class="grid g2">
      ${barMetrics.length ? `<div class="card"><h3>Test metrics</h3><div class="chart-box tall"><canvas id="c-bars"></canvas></div></div>` : ""}
      ${seriesKeys.map((k, i) => `<div class="card"><h3>${esc(k)} over training</h3><div class="chart-box tall"><canvas id="c-ov-${i}"></canvas></div></div>`).join("")}
    </div>
    <div class="grid" style="margin-top:16px">
      <div class="card tight"><div style="padding:14px 16px 4px"><h3>Hyperparameters · ${data.differing_params.length} differ</h3></div><div class="table-wrap"><table><thead>${head}</thead><tbody>
        ${Object.entries(data.params).sort(([a], [b]) => data.differing_params.includes(b) - data.differing_params.includes(a)).map(([k, vals]) => `<tr class="${data.differing_params.includes(k) ? "diff" : "same"}"><td>${esc(k)}</td>${vals.map((v) => `<td class="mono">${fmt(v)}</td>`).join("")}</tr>`).join("")}
      </tbody></table></div></div>
      <div class="card tight"><div style="padding:14px 16px 4px"><h3>Metrics</h3></div><div class="table-wrap"><table><thead>${head}</thead><tbody>
        ${metricRows.map(([k, vals]) => `<tr><td>${esc(k)}</td>${vals.map((v, i) => `<td class="num ${data.best_run_per_metric[k] === runs[i].id ? "best" : ""}">${fm(v)}</td>`).join("")}</tr>`).join("")}
      </tbody></table></div></div>
    </div>`;
  if (barMetrics.length) chart("c-bars", { type: "bar", data: { labels: barMetrics.map((k) => k.replace("test_", "")), datasets: runs.map((r, i) => ({ label: r.name, data: barMetrics.map((k) => data.metrics[k][i]), backgroundColor: PALETTE[i % 10], borderRadius: 3 })) },
    options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom" } }, scales: { y: zoomAxis(barMetrics.flatMap((k) => data.metrics[k])) } } });
  seriesKeys.forEach((k, idx) => chart(`c-ov-${idx}`, { type: "line", data: { datasets: runs.map((r, i) => ({ label: r.name, data: (data.series[r.id][k] || []).map((p) => ({ x: p.step, y: p.value })), borderColor: PALETTE[i % 10], borderWidth: 2, tension: .2 })).filter((d) => d.data.length) }, options: lineOpts("step", k) }));
}

// ------------------------------------------------------------------ sweeps
async function sweepsPage() {
  const [sweeps, exps] = await Promise.all([api("/sweeps"), api("/experiments")]);
  const expName = Object.fromEntries(exps.map((e) => [e.id, e.name]));
  app.innerHTML = `<h1>Hyperparameter sweeps</h1><p class="sub">Grid and random searches; each trial is a full tracked run.</p><div id="sweeps" class="grid"></div>`;
  const host = document.getElementById("sweeps");
  if (!sweeps.length) { host.innerHTML = `<div class="card empty">No sweeps yet. Add a <span class="mono">sweep:</span> block to a training config.</div>`; return; }
  for (const [si, sw] of sweeps.entries()) {
    const runs = (await api(`/runs?sweep_id=${sw.id}`)).filter((r) => typeof r.summary[sw.objective] === "number").sort((a, b) => a.id - b.id);
    const best = runs.find((r) => r.id === sw.best_run_id);
    host.insertAdjacentHTML("beforeend", `<div class="card"><div class="row"><h2 style="margin:0">${esc(sw.name)}</h2>${badge(sw.status)}<span class="badge neutral">${esc(sw.strategy)}</span><span class="spacer"></span>
      <span class="muted">${esc(expName[sw.experiment_id] || "")} · ${runs.length} trials · objective <b style="color:var(--text)">${esc(sw.objective)}</b> (${esc(sw.direction)})</span></div>
      <div class="grid g2" style="margin-top:14px"><div><div class="chart-box"><canvas id="c-sw-${si}"></canvas></div></div>
      <div><h3>Best trial</h3>${best ? `<p><a href="#/run/${best.id}">${esc(best.name)}</a> — ${esc(sw.objective)} = <b>${fm(best.summary[sw.objective])}</b></p><dl class="kv">${Object.keys(sw.search_space).map((k) => { const p = k.split(".").pop(); return `<dt>${esc(p)}</dt><dd>${fmt(best.params[p])}</dd>`; }).join("")}</dl>` : `<p class="muted">—</p>`}
      <h3 style="margin-top:14px">Search space</h3><pre class="code">${esc(JSON.stringify(sw.search_space, null, 2))}</pre>
      <div class="row" style="margin-top:12px"><a class="btn" href="#/compare?ids=${runs.slice().sort((a, b) => (sw.direction === "maximize" ? b.summary[sw.objective] - a.summary[sw.objective] : a.summary[sw.objective] - b.summary[sw.objective])).slice(0, 5).map((r) => r.id).join(",")}">Compare top 5</a><a class="btn" href="#/exp/${sw.experiment_id}">Open experiment</a></div></div></div></div>`);
    let running = sw.direction === "maximize" ? -Infinity : Infinity;
    const bestSoFar = runs.map((r) => (running = sw.direction === "maximize" ? Math.max(running, r.summary[sw.objective]) : Math.min(running, r.summary[sw.objective])));
    chart(`c-sw-${si}`, { data: { labels: runs.map((_, i) => i + 1), datasets: [
      { type: "scatter", label: "trial", data: runs.map((r, i) => ({ x: i + 1, y: r.summary[sw.objective] })), backgroundColor: PALETTE[0], pointBackgroundColor: runs.map((r) => (r.id === sw.best_run_id ? PALETTE[1] : PALETTE[0])), pointRadius: runs.map((r) => (r.id === sw.best_run_id ? 8 : 5)) },
      { type: "line", label: "best so far", data: bestSoFar.map((y, i) => ({ x: i + 1, y })), borderColor: PALETTE[1], stepped: true, borderWidth: 2, pointRadius: 0 }] },
      options: lineOpts("trial", sw.objective, { onClick: (_, els) => els.length && (location.hash = `#/run/${runs[els[0].index].id}`) }) });
  }
}

// ------------------------------------------------------------------ model registry
async function modelsPage() {
  const models = await api("/models");
  app.innerHTML = `<h1>Model registry</h1><p class="sub">Versioned models with lifecycle stages. The <b>Production</b> version is served at <span class="mono">POST /api/models/{name}/predict</span>.</p>
    <div class="grid">${models.map((m) => `<div class="card"><div class="row"><h2 style="margin:0">${esc(m.name)}</h2><span class="muted">${m.versions.length} version(s)</span></div>
      <table style="margin-top:12px"><thead><tr><th>Version</th><th>Stage</th><th>Source run</th><th class="num">Key metric</th><th>Registered</th><th>Set stage</th></tr></thead><tbody>
      ${m.versions.slice().reverse().map((v) => { const k = primaryMetric(v.metrics_snapshot); return `<tr><td>v${v.version}</td><td>${badge(v.stage)}</td><td><a href="#/run/${v.run_id}">run #${v.run_id}</a></td><td class="num">${k ? `<span class="muted">${k}</span> ${fm(v.metrics_snapshot[k])}` : "—"}</td><td class="muted">${ago(v.created_at)}</td>
        <td><select data-model="${esc(m.name)}" data-version="${v.version}">${["None", "Staging", "Production", "Archived"].map((s) => `<option ${s === v.stage ? "selected" : ""}>${s}</option>`).join("")}</select></td></tr>`; }).join("")}</tbody></table>
      <details style="margin-top:12px"><summary class="muted" style="cursor:pointer">Try the prediction API</summary>
        <div style="margin-top:10px"><textarea data-predict-input="${esc(m.name)}" placeholder="JSON list of feature rows, e.g. [[1.2, 3.4, ...]]"></textarea>
        <div class="row" style="margin-top:8px"><button data-predict="${esc(m.name)}" class="primary">Predict (Production)</button><button data-sample="${esc(m.name)}" data-run="${m.versions.at(-1)?.run_id}">Fill with random sample</button></div>
        <pre class="code" data-predict-out="${esc(m.name)}" style="margin-top:8px">—</pre></div></details></div>`).join("") || `<div class="card empty">No registered models yet. Open a run and click <b>Register model</b>.</div>`}</div>`;
  app.querySelectorAll("select[data-model]").forEach((sel) => (sel.onchange = async () => {
    try { await api(`/models/${encodeURIComponent(sel.dataset.model)}/versions/${sel.dataset.version}`, { method: "PATCH", body: { stage: sel.value } }); toast(`${sel.dataset.model} v${sel.dataset.version} → ${sel.value}`); render(); }
    catch (e) { toast(e.message, true); }
  }));
  app.querySelectorAll("[data-sample]").forEach((b) => (b.onclick = async () => {
    const run = await api(`/runs/${b.dataset.run}`);
    const n = run.system_info.n_features || 4;
    document.querySelector(`[data-predict-input="${CSS.escape(b.dataset.sample)}"]`).value = JSON.stringify([Array.from({ length: n }, () => +(Math.random() * 2).toFixed(3))]);
  }));
  app.querySelectorAll("[data-predict]").forEach((b) => (b.onclick = async () => {
    const name = b.dataset.predict, out = document.querySelector(`[data-predict-out="${CSS.escape(name)}"]`);
    try { const instances = JSON.parse(document.querySelector(`[data-predict-input="${CSS.escape(name)}"]`).value); out.textContent = JSON.stringify(await api(`/models/${encodeURIComponent(name)}/predict`, { method: "POST", body: { instances, return_proba: true } }), null, 2); }
    catch (e) { out.textContent = e.message; }
  }));
}

// ------------------------------------------------------------------ launch
async function launchPage() {
  const cat = await api("/catalog");
  app.innerHTML = `<h1>Launch training</h1><p class="sub">Submit a job to the server's worker pool. Results stream into the experiment as they're logged.</p>
    <div class="grid g2"><div class="card"><div class="form-grid">
      <label class="field">Experiment<input id="f-exp" value="playground"></label>
      <label class="field">Run name (optional)<input id="f-name" placeholder="auto"></label>
      <label class="field">Dataset<select id="f-ds">${Object.entries(cat.datasets).filter(([k]) => k !== "csv").map(([k, v]) => `<option value="${k}">${k} (${v.task})</option>`).join("")}</select></label>
      <label class="field">Model<select id="f-model"></select></label>
      <label class="field">Feature scaling<select id="f-scale"><option value="true">StandardScaler</option><option value="false">none</option></select></label>
      <label class="field">CV folds (single-fit models)<input id="f-cv" type="number" value="5" min="0" max="10"></label>
      <label class="field">Max epochs (iterative models)<input id="f-epochs" type="number" value="80"></label>
      <label class="field">Early-stopping patience<input id="f-pat" type="number" value="8"></label>
    </div>
    <label class="field" style="margin-top:14px">Model hyperparameters (JSON)<textarea id="f-params"></textarea></label>
    <label class="field" style="margin-top:14px"><span><input type="checkbox" id="f-sweep-on"> Run as a hyperparameter sweep</span></label>
    <div id="sweep-box" style="display:none;margin-top:10px"><div class="form-grid">
      <label class="field">Strategy<select id="f-strategy"><option>random</option><option>grid</option></select></label>
      <label class="field">Trials (random)<input id="f-trials" type="number" value="10"></label>
      <label class="field">Objective<input id="f-obj" value="val_f1_macro"></label>
      <label class="field">Direction<select id="f-dir"><option>maximize</option><option>minimize</option></select></label></div>
      <label class="field" style="margin-top:10px">Search space (JSON, dotted config paths)<textarea id="f-space"></textarea></label></div>
    <div class="row" style="margin-top:16px"><button id="f-submit" class="primary">Launch job</button><span id="f-status" class="muted"></span></div></div>
    <div class="card"><h3>Equivalent YAML config</h3><pre class="code" id="f-yaml"></pre><p class="muted" style="font-size:12.5px">Save it as <span class="mono">my.yaml</span> and run <span class="mono">expforge train my.yaml</span> to reproduce from the CLI.</p></div></div>`;
  const $ = (id) => document.getElementById(id);
  const fillModels = () => {
    const task = cat.datasets[$("f-ds").value].task;
    $("f-model").innerHTML = Object.entries(cat.models).filter(([, m]) => m.tasks.includes(task)).map(([k, m]) => `<option value="${k}">${k}${m.iterative ? " · curves" : ""}</option>`).join("");
    $("f-obj").value = task === "classification" ? "val_f1_macro" : "val_rmse";
    $("f-dir").value = task === "classification" ? "maximize" : "minimize";
    fillParams();
  };
  const fillParams = () => {
    const d = { ...cat.models[$("f-model").value].defaults }; delete d.n_jobs;
    $("f-params").value = JSON.stringify(d, null, 2);
    const numeric = Object.entries(d).filter(([, v]) => typeof v === "number" && !/random_state|max_iter/.test(String(v))).slice(0, 2);
    $("f-space").value = JSON.stringify(Object.fromEntries(numeric.filter(([k]) => k !== "random_state").map(([k, v]) => [`model.params.${k}`, Number.isInteger(v) ? [Math.max(1, Math.round(v / 2)), v, v * 2] : { type: "loguniform", low: v / 10, high: v * 10 }])), null, 2);
    preview();
  };
  const build = () => {
    const cfg = { experiment: $("f-exp").value || "playground", dataset: { name: $("f-ds").value }, model: { name: $("f-model").value, params: JSON.parse($("f-params").value || "{}") },
      training: { scale: $("f-scale").value === "true", cv_folds: +$("f-cv").value, epochs: +$("f-epochs").value, early_stopping_patience: +$("f-pat").value } };
    if ($("f-name").value) cfg.name = $("f-name").value;
    if ($("f-sweep-on").checked) cfg.sweep = { strategy: $("f-strategy").value, n_trials: +$("f-trials").value, objective: $("f-obj").value, direction: $("f-dir").value, search_space: JSON.parse($("f-space").value || "{}") };
    return cfg;
  };
  const toYaml = (o, ind = "") => Object.entries(o).map(([k, v]) => (v && typeof v === "object" && !Array.isArray(v) ? `${ind}${k}:\n${toYaml(v, ind + "  ")}` : `${ind}${k}: ${JSON.stringify(v)}`)).join("\n");
  const preview = () => { try { $("f-yaml").textContent = toYaml(build()); } catch { $("f-yaml").textContent = "invalid JSON"; } };
  $("f-ds").onchange = fillModels; $("f-model").onchange = fillParams;
  $("f-sweep-on").onchange = (e) => { $("sweep-box").style.display = e.target.checked ? "block" : "none"; preview(); };
  app.querySelectorAll("input, select, textarea").forEach((el) => el.addEventListener("input", preview));
  fillModels();
  $("f-submit").onclick = async () => {
    let cfg; try { cfg = build(); } catch { return toast("Hyperparameters / search space must be valid JSON", true); }
    try {
      const job = await api("/jobs", { method: "POST", body: { config: cfg } });
      toast(`Job #${job.id} queued`);
      $("f-submit").disabled = true;
      const poll = async () => {
        const j = await api(`/jobs/${job.id}`);
        $("f-status").innerHTML = `Job #${j.id}: ${badge(j.status)} ${j.run_ids.length ? `→ ${j.run_ids.map((r) => `<a href="#/run/${r}">run ${r}</a>`).join(", ")}` : ""} ${j.error ? `<span style="color:var(--bad)">${esc(j.error)}</span>` : ""}`;
        if (["finished", "failed"].includes(j.status)) { clearInterval(pollTimer); $("f-submit").disabled = false; }
      };
      poll(); pollTimer = setInterval(poll, 1500);
    } catch (e) { toast(e.message, true); }
  };
}

api("/health").then((h) => (document.getElementById("version").textContent = `v${h.version}`)).catch(() => {});
render();
