let state = { items: [], filtered: [], selected: 0 };

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
