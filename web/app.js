const state = {
  currentRunId: null,
  currentJobId: null,
  pollingTimer: null,
};

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data;
}

function setMessage(text, type = "muted") {
  const element = document.getElementById("submission-result");
  element.textContent = text;
  element.className = `message ${type}`;
}

function getRunPayload() {
  const userInput = document.getElementById("user-input").value.trim();
  const maxResults = Number(document.getElementById("max-results").value || 8);
  const liveLlm = document.getElementById("live-llm").checked;
  return {
    user_input: userInput,
    max_results: maxResults,
    live_llm: liveLlm,
  };
}

function formatDate(value) {
  if (!value) return "未知时间";
  return new Date(value).toLocaleString();
}

function renderRuns(runs) {
  const container = document.getElementById("runs-list");
  container.innerHTML = "";
  if (!runs.length) {
    container.textContent = "还没有历史运行。";
    container.className = "list-state";
    return;
  }

  container.className = "runs-grid";
  const template = document.getElementById("run-item-template");
  runs.forEach((run) => {
    const node = template.content.firstElementChild.cloneNode(true);
    node.querySelector(".run-status").textContent = run.status;
    node.querySelector(".run-status").dataset.status = run.status;
    node.querySelector(".run-stage").textContent = run.current_stage;
    node.querySelector(".run-title").textContent = run.user_input;
    node.querySelector(".run-time").textContent = `更新于 ${formatDate(run.updated_at)}`;
    node.querySelector(".view-run").addEventListener("click", () => loadRunDetail(run.run_id));
    container.appendChild(node);
  });
}

function renderTopic(topic) {
  if (!topic) return "<p class='muted'>当前没有选中的候选方向。</p>";
  const differentiation = (topic.differentiation || []).map((item) => `<li>${item}</li>`).join("");
  const failures = (topic.failure_modes || []).map((item) => `<li>${item}</li>`).join("");
  const papers = (topic.source_papers || []).map((item) => `<li>${item}</li>`).join("");
  return `
    <div class="topic-card">
      <h3>${topic.title}</h3>
      <p>${topic.rationale}</p>
      <div class="two-col">
        <div>
          <h4>可切入的差异点</h4>
          <ul>${differentiation || "<li>暂无</li>"}</ul>
        </div>
        <div>
          <h4>潜在失败点</h4>
          <ul>${failures || "<li>暂无</li>"}</ul>
        </div>
      </div>
      <div>
        <h4>来源论文</h4>
        <ul>${papers || "<li>暂无</li>"}</ul>
      </div>
    </div>
  `;
}

function renderArtifacts(artifacts) {
  if (!artifacts.length) return "<p class='muted'>当前没有 artifact。</p>";
  return `
    <ul class="artifact-list">
      ${artifacts
        .slice(-10)
        .reverse()
        .map(
          (artifact) => `
            <li>
              <strong>${artifact.kind}</strong>
              <span>${artifact.stage}</span>
              <code>${artifact.file_path || "inline payload"}</code>
            </li>
          `
        )
        .join("")}
    </ul>
  `;
}

function renderEvents(events) {
  if (!events.length) return "<p class='muted'>当前没有事件记录。</p>";
  return `
    <ul class="event-list">
      ${events
        .slice(-12)
        .reverse()
        .map(
          (event) => `
            <li>
              <span class="event-level">${event.level}</span>
              <div>
                <strong>${event.stage || "system"}</strong>
                <p>${event.message}</p>
              </div>
            </li>
          `
        )
        .join("")}
    </ul>
  `;
}

function renderGeneratedFiles(files) {
  if (!files.length) return "<p class='muted'>当前没有生成文件。</p>";
  return `
    <ul class="file-list">
      ${files
        .map(
          (file) => `
            <li>
              <strong>${file.path}</strong>
              <p>${file.description}</p>
            </li>
          `
        )
        .join("")}
    </ul>
  `;
}

async function loadRunDetail(runId) {
  state.currentRunId = runId;
  const [run, artifacts, events] = await Promise.all([
    fetchJson(`/runs/${runId}`),
    fetchJson(`/runs/${runId}/artifacts`),
    fetchJson(`/runs/${runId}/events`),
  ]);
  const detail = document.getElementById("run-detail");
  detail.className = "detail-grid";
  detail.innerHTML = `
    <section class="detail-card">
      <div class="detail-head">
        <span class="pill" data-status="${run.status}">${run.status}</span>
        <span class="muted">${run.current_stage}</span>
      </div>
      <h3>${run.user_input}</h3>
      <p class="muted">Run ID: ${run.run_id}</p>
      <p class="muted">已完成阶段：${(run.completed_stages || []).join(" -> ") || "暂无"}</p>
      ${renderTopic(run.selected_topic)}
    </section>

    <section class="detail-card">
      <h3>研究计划预览</h3>
      <pre class="preview">${run.plan_markdown || "尚未生成研究计划。"}</pre>
    </section>

    <section class="detail-card">
      <h3>生成文件</h3>
      ${renderGeneratedFiles(run.generated_files || [])}
    </section>

    <section class="detail-card">
      <h3>Artifacts</h3>
      ${renderArtifacts(artifacts.artifacts || [])}
    </section>

    <section class="detail-card">
      <h3>事件流</h3>
      ${renderEvents(events.events || [])}
    </section>

    <section class="detail-card">
      <h3>候选方向</h3>
      <ul class="topic-list">
        ${(run.candidate_topics || [])
          .map(
            (topic) => `
              <li>
                <strong>${topic.title}</strong>
                <p>${topic.rationale}</p>
              </li>
            `
          )
          .join("")}
      </ul>
    </section>
  `;
}

async function refreshRuns() {
  const data = await fetchJson("/runs");
  renderRuns(data.runs || []);
  if (state.currentRunId) {
    const exists = (data.runs || []).some((run) => run.run_id === state.currentRunId);
    if (exists) {
      await loadRunDetail(state.currentRunId);
    }
  }
}

async function runSync() {
  const payload = getRunPayload();
  if (!payload.user_input) {
    setMessage("先输入一个研究想法。", "error");
    return;
  }
  setMessage("正在同步执行，通常几秒内会返回结果。", "info");
  try {
    const run = await fetchJson("/runs", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    setMessage(`同步运行完成，run_id=${run.run_id}`, "success");
    await refreshRuns();
    await loadRunDetail(run.run_id);
  } catch (error) {
    setMessage(`同步运行失败：${error.message}`, "error");
  }
}

function stopPolling() {
  if (state.pollingTimer) {
    window.clearInterval(state.pollingTimer);
    state.pollingTimer = null;
  }
}

async function startAsyncRun() {
  const payload = getRunPayload();
  if (!payload.user_input) {
    setMessage("先输入一个研究想法。", "error");
    return;
  }
  setMessage("后台任务已提交，正在轮询状态。", "info");
  try {
    const job = await fetchJson("/jobs", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.currentJobId = job.job_id;
    stopPolling();
    state.pollingTimer = window.setInterval(async () => {
      try {
        const snapshot = await fetchJson(`/jobs/${state.currentJobId}`);
        setMessage(`后台任务状态：${snapshot.status}`, snapshot.status === "failed" ? "error" : "info");
        if (snapshot.status === "completed" && snapshot.run_id) {
          stopPolling();
          setMessage(`后台任务完成，run_id=${snapshot.run_id}`, "success");
          await refreshRuns();
          await loadRunDetail(snapshot.run_id);
        }
        if (snapshot.status === "failed") {
          stopPolling();
          setMessage(`后台任务失败：${snapshot.error || "unknown error"}`, "error");
        }
      } catch (error) {
        stopPolling();
        setMessage(`轮询失败：${error.message}`, "error");
      }
    }, 1200);
  } catch (error) {
    setMessage(`后台任务提交失败：${error.message}`, "error");
  }
}

function bootstrap() {
  document.getElementById("run-sync").addEventListener("click", runSync);
  document.getElementById("run-async").addEventListener("click", startAsyncRun);
  document.getElementById("refresh-runs").addEventListener("click", refreshRuns);
  refreshRuns().catch((error) => {
    setMessage(`加载最近运行失败：${error.message}`, "error");
  });
}

window.addEventListener("DOMContentLoaded", bootstrap);
