from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from .io import read_json, write_json


def build_site(args) -> None:
    runs_root = Path(args.runs)
    site_dir = Path(args.site_dir)
    video_mode = getattr(args, "video_mode", "copy")
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "assets").mkdir(exist_ok=True)
    (site_dir / "assets" / "videos").mkdir(exist_ok=True)
    items = collect_runs(runs_root, site_dir, video_mode)
    write_json(site_dir / "data.json", {"title": args.title, "items": items})
    (site_dir / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (site_dir / "styles.css").write_text(STYLES_CSS, encoding="utf-8")
    (site_dir / "app.js").write_text(APP_JS, encoding="utf-8")
    (site_dir / ".nojekyll").write_text("", encoding="utf-8")


def collect_runs(runs_root: Path, site_dir: Path, video_mode: str) -> list[dict[str, Any]]:
    items = []
    for summary_path in runs_root.rglob("run_summary.json"):
        run_dir = summary_path.parent
        summary = _maybe_json(run_dir / "run_summary.json")
        metadata = _maybe_json(run_dir / "metadata.json")
        qa = _maybe_json(run_dir / "qa.json")
        plan = _maybe_json(run_dir / "plan.json")
        grouped = _maybe_json(run_dir / "grouped_steps.json")
        alignment = _maybe_json(run_dir / "alignment.json")
        dataset_row = _maybe_json(run_dir / "dataset_row.json")
        thumbnail = copy_thumbnail(run_dir, site_dir)
        video = prepare_video(summary.get("video"), run_dir, site_dir, video_mode)
        items.append(
            {
                "id": run_dir.name,
                "config": run_dir.parent.name,
                "activity": metadata.get("activity", qa.get("inferred_activity", run_dir.name)),
                "reference_steps": metadata.get("steps", []),
                "score": qa.get("score"),
                "coverage_score": qa.get("coverage_score"),
                "order_score": qa.get("order_score"),
                "relevance_score": qa.get("relevance_score"),
                "issues": qa.get("issues", []),
                "reasoning": qa.get("reasoning", ""),
                "plan": plan,
                "local_checks": qa.get("local_checks", {}),
                "segments": grouped,
                "alignment": alignment,
                "dataset_row": compact_dataset_row(dataset_row),
                "thumbnail": thumbnail,
                "video": video,
            }
        )
    return sorted(items, key=lambda item: (item.get("config", ""), item.get("id", "")))


def copy_thumbnail(run_dir: Path, site_dir: Path) -> str | None:
    frames = sorted((run_dir / "frames").glob("*.jpg"))
    if not frames:
        return None
    out_name = f"{safe_name(run_dir.parent.name)}_{safe_name(run_dir.name)}.jpg"
    out_path = site_dir / "assets" / out_name
    shutil.copyfile(frames[len(frames) // 2], out_path)
    return f"assets/{out_name}"


def prepare_video(video_path_value: str | None, run_dir: Path, site_dir: Path, video_mode: str) -> dict[str, Any] | None:
    if not video_path_value or video_mode == "none":
        return None
    video_path = Path(video_path_value)
    video = {
        "source_path": str(video_path),
        "filename": video_path.name,
        "size_mb": round(video_path.stat().st_size / (1024 * 1024), 2) if video_path.exists() else None,
        "src": None,
    }
    if not video_path.exists():
        return video
    if video_mode == "link":
        video["src"] = str(video_path)
        return video

    out_name = f"{safe_name(run_dir.parent.name)}_{safe_name(run_dir.name)}{video_path.suffix.lower()}"
    out_path = site_dir / "assets" / "videos" / out_name
    if not out_path.exists() or out_path.stat().st_size != video_path.stat().st_size:
        shutil.copyfile(video_path, out_path)
    video["src"] = f"assets/videos/{out_name}"
    return video


def compact_dataset_row(row: dict[str, Any]) -> dict[str, Any]:
    keep = ["video_path", "duration_in_sec", "task", "domain", "query", "category", "question", "answer", "mcq_answer"]
    return {key: row[key] for key in keep if key in row}


def safe_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value)).strip("-")
    return value or "item"


def _maybe_json(path: Path):
    if not path.exists():
        return {} if path.suffix == ".json" else []
    return read_json(path)


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Step Alignment Results</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header class="topbar">
    <div>
      <h1>Step Alignment Results</h1>
      <p>Review generated procedural transcripts, QA scores, and the corresponding source videos.</p>
    </div>
    <div class="controls">
      <input id="search" type="search" placeholder="Search activity, id, issue">
      <select id="configFilter" aria-label="Filter by config"></select>
    </div>
  </header>
  <main>
    <section class="metrics" id="metrics"></section>
    <section class="layout">
      <nav class="results" id="results"></nav>
      <article class="detail" id="detail"></article>
    </section>
  </main>
  <script src="app.js"></script>
</body>
</html>
"""


STYLES_CSS = """:root {
  color-scheme: dark;
  --bg: #07090d;
  --panel: #10151d;
  --panel-2: #151b25;
  --text: #eef3f8;
  --muted: #93a4b7;
  --line: #273243;
  --accent: #5eead4;
  --accent-2: #f8c471;
  --bad: #fb7185;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.topbar {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  padding: 28px 32px 20px;
  border-bottom: 1px solid var(--line);
  background: #0b0f15;
}
h1 { margin: 0 0 6px; font-size: 24px; font-weight: 700; }
p { margin: 0; color: var(--muted); line-height: 1.5; }
.controls { display: flex; gap: 10px; align-items: end; }
input, select {
  height: 38px;
  min-width: 210px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel);
  color: var(--text);
  padding: 0 12px;
}
main { padding: 20px 32px 36px; }
.metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(120px, 1fr));
  gap: 12px;
  margin-bottom: 18px;
}
.metric {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
}
.metric span { display: block; color: var(--muted); font-size: 12px; }
.metric strong { display: block; margin-top: 4px; font-size: 24px; }
.layout {
  display: grid;
  grid-template-columns: minmax(260px, 390px) 1fr;
  gap: 18px;
  align-items: start;
}
.results {
  display: grid;
  gap: 10px;
  max-height: calc(100vh - 210px);
  overflow: auto;
  padding-right: 4px;
}
.result {
  display: grid;
  grid-template-columns: 72px 1fr;
  gap: 12px;
  width: 100%;
  text-align: left;
  color: var(--text);
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px;
  cursor: pointer;
}
.result[aria-selected="true"] { border-color: var(--accent); background: var(--panel-2); }
.thumb {
  width: 72px;
  height: 54px;
  object-fit: cover;
  border-radius: 6px;
  background: #05070a;
}
.result h2 { margin: 0; font-size: 14px; line-height: 1.35; }
.result small { color: var(--muted); }
.score {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 36px;
  height: 24px;
  border-radius: 999px;
  background: rgba(94, 234, 212, 0.12);
  color: var(--accent);
  font-weight: 700;
  margin-top: 8px;
}
.detail {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  min-height: 520px;
  padding: 18px;
}
.detail h2 { margin: 0 0 8px; font-size: 22px; }
.video-wrap {
  margin: 14px 0 16px;
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
  background: #05070a;
}
video {
  display: block;
  width: 100%;
  max-height: 58vh;
  background: #05070a;
}
.video-meta {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 8px 10px;
  color: var(--muted);
  font-size: 12px;
  border-top: 1px solid var(--line);
}
.chips { display: flex; gap: 8px; flex-wrap: wrap; margin: 14px 0; }
.chip {
  border: 1px solid var(--line);
  background: #0b0f15;
  border-radius: 999px;
  padding: 5px 9px;
  color: var(--muted);
  font-size: 12px;
}
.grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
  margin: 16px 0;
}
.subscore {
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 10px;
}
.subscore span { color: var(--muted); font-size: 12px; }
.subscore strong { display: block; margin-top: 3px; font-size: 20px; }
.steps { display: grid; gap: 8px; margin-top: 12px; }
.step {
  border-left: 3px solid var(--accent);
  background: #0b0f15;
  padding: 10px 12px;
  border-radius: 0 6px 6px 0;
}
.step time { color: var(--accent-2); font-size: 12px; }
.issues { color: var(--bad); }
.plan {
  display: grid;
  gap: 10px;
  margin: 16px 0;
}
.plan-panel {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #0b0f15;
  padding: 12px;
}
.plan-panel h3 {
  margin-top: 0;
}
.plan-list {
  display: grid;
  gap: 8px;
  padding-left: 0;
  list-style: none;
}
.plan-list li {
  line-height: 1.45;
  border-left: 3px solid var(--accent);
  padding: 8px 10px;
  background: var(--panel);
  border-radius: 0 6px 6px 0;
}
pre {
  overflow: auto;
  background: #07090d;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 12px;
}
@media (max-width: 860px) {
  .topbar, .controls { flex-direction: column; align-items: stretch; }
  .layout { grid-template-columns: 1fr; }
  .metrics, .grid { grid-template-columns: repeat(2, 1fr); }
  .results { max-height: none; }
}
"""


APP_JS = """let state = { items: [], filtered: [], selected: 0 };

fetch("data.json").then(r => r.json()).then(data => {
  state.items = data.items || [];
  setupFilters();
  render();
});

function setupFilters() {
  const configs = ["all", ...new Set(state.items.map(item => item.config).filter(Boolean))];
  const select = document.getElementById("configFilter");
  select.innerHTML = configs.map(value => `<option value="${value}">${value}</option>`).join("");
  document.getElementById("search").addEventListener("input", render);
  select.addEventListener("change", render);
}

function applyFilters() {
  const query = document.getElementById("search").value.toLowerCase();
  const config = document.getElementById("configFilter").value;
  state.filtered = state.items.filter(item => {
    const haystack = JSON.stringify(item).toLowerCase();
    return (config === "all" || item.config === config) && haystack.includes(query);
  });
  if (state.selected >= state.filtered.length) state.selected = 0;
}

function render() {
  applyFilters();
  renderMetrics();
  renderList();
  renderDetail();
}

function renderMetrics() {
  const scored = state.filtered.filter(item => typeof item.score === "number");
  const avg = scored.length ? (scored.reduce((sum, item) => sum + item.score, 0) / scored.length).toFixed(1) : "n/a";
  const pass = scored.filter(item => item.score >= 5).length;
  document.getElementById("metrics").innerHTML = [
    metric("Runs", state.filtered.length),
    metric("Avg score", avg),
    metric("Score >= 5", pass),
    metric("With video", state.filtered.filter(item => item.video?.src).length)
  ].join("");
}

function metric(label, value) {
  return `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
}

function renderList() {
  document.getElementById("results").innerHTML = state.filtered.map((item, index) => `
    <button class="result" aria-selected="${index === state.selected}" onclick="selectItem(${index})">
      ${item.thumbnail ? `<img class="thumb" src="${item.thumbnail}" alt="">` : `<div class="thumb"></div>`}
      <span>
        <h2>${escapeHtml(item.activity)}</h2>
        <small>${escapeHtml(item.config)} / ${escapeHtml(item.id)}</small><br>
        <span class="score">${item.score ?? "n/a"}</span>
      </span>
    </button>
  `).join("");
}

function selectItem(index) {
  state.selected = index;
  renderList();
  renderDetail();
}

function renderDetail() {
  const item = state.filtered[state.selected];
  if (!item) {
    document.getElementById("detail").innerHTML = "<p>No runs match the current filters.</p>";
    return;
  }
  const segments = item.segments || [];
  document.getElementById("detail").innerHTML = `
    <h2>${escapeHtml(item.activity)}</h2>
    ${renderVideo(item)}
    <p>${escapeHtml(item.reasoning || "No QA reasoning recorded.")}</p>
    <div class="chips">
      <span class="chip">${escapeHtml(item.config)}</span>
      <span class="chip">${escapeHtml(item.id)}</span>
      <span class="chip">${segments.length} grouped segments</span>
    </div>
    <div class="grid">
      ${subscore("Overall", item.score)}
      ${subscore("Coverage", item.coverage_score)}
      ${subscore("Order", item.order_score)}
      ${subscore("Relevance", item.relevance_score)}
    </div>
    ${renderIssues(item.issues)}
    ${renderPlan(item.plan)}
    <h3>Grouped Transcript</h3>
    <div class="steps">
      ${segments.map((segment, index) => `
        <div class="step">
          <time>${fmt(segment.start_ts)}s-${fmt(segment.end_ts)}s · step ${item.alignment?.[index] ?? "n/a"}</time>
          <div>${escapeHtml(segment.caption || "")}</div>
        </div>
      `).join("")}
    </div>
    <h3>Source</h3>
    <pre>${escapeHtml(JSON.stringify(item.dataset_row || item.video || {}, null, 2))}</pre>
  `;
}

function renderVideo(item) {
  if (!item.video?.src) {
    return `<p>No playable video was copied for this run.</p>`;
  }
  return `
    <div class="video-wrap">
      <video controls preload="metadata" src="${escapeHtml(item.video.src)}"></video>
      <div class="video-meta">
        <span>${escapeHtml(item.video.filename || "source video")}</span>
        <span>${item.video.size_mb ?? "?"} MB</span>
      </div>
    </div>
  `;
}

function renderPlan(plan) {
  const outline = plan?.outline || plan?.plan_steps?.map(step => step.instruction) || [];
  if (!plan || !outline.length) return "";
  const materials = plan.materials?.length
    ? `<div class="chips">${plan.materials.map(item => `<span class="chip">${escapeHtml(item)}</span>`).join("")}</div>`
    : "";
  const cautions = plan.cautions?.length
    ? `<p class="issues">${plan.cautions.map(escapeHtml).join(" · ")}</p>`
    : "";
  return `
    <section class="plan">
      <div class="plan-panel">
        <h3>${escapeHtml(plan.title || "Plan")}</h3>
        <p>${escapeHtml(plan.overview || "")}</p>
        ${materials}
        ${cautions}
        <ul class="plan-list">
          ${outline.map(item => `
            <li>${escapeHtml(item || "")}</li>
          `).join("")}
        </ul>
      </div>
    </section>
  `;
}

function subscore(label, value) {
  return `<div class="subscore"><span>${label}</span><strong>${value ?? "n/a"}</strong></div>`;
}

function renderIssues(issues) {
  if (!issues || !issues.length) return "";
  return `<p class="issues">${issues.map(escapeHtml).join(" · ")}</p>`;
}

function fmt(value) {
  return Number(value || 0).toFixed(1);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
"""
