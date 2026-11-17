/* MovieMatch website logic.
   Reads window.MOVIEMATCH (written by src/run_pipeline.py into data/results.js).
   Nothing here computes or invents results; it only displays exported values. */
(function () {
  "use strict";
  const D = window.MOVIEMATCH;
  if (!D) {
    document.querySelector("main").insertAdjacentHTML("afterbegin",
      '<p class="callout">Results file <code>data/results.js</code> was not found. Run <code>python -m src.run_pipeline</code> to generate it.</p>');
    return;
  }

  // ------------------------------------------------------------ helpers
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const get = (path) => path.split("/").reduce((o, k) => (o == null ? undefined : o[k]), D);
  const fmt = {
    int: (v) => Number(v).toLocaleString("en-US"),
    pct: (v) => (v * 100).toFixed(1) + "%",
    pct0: (v) => Math.round(v * 100) + "%",
    f2: (v) => Number(v).toFixed(2),
    f3: (v) => Number(v).toFixed(3),
    signed: (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(3),
    ci: (v) => "[" + fmt.signed(v[0]) + ", " + fmt.signed(v[1]) + "]",
    date: (v) => new Date(v.replace(" ", "T").replace("+00:00", "Z")).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" }),
    raw: (v) => String(v),
  };
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const MODEL_VAR = {
    "Majority baseline": "--maj", "Popularity baseline": "--pop", "Logistic Regression": "--lr",
    "Random Forest": "--rf", "LR (movie features only)": "--lrm",
  };
  const MODELS = Object.keys(MODEL_VAR);
  const SHORT = { "Majority baseline": "Majority", "Popularity baseline": "Popularity", "Logistic Regression": "Logistic Regression",
    "Random Forest": "Random Forest", "LR (movie features only)": "LR, movie only" };
  const svgNS = "http://www.w3.org/2000/svg";
  function svgEl(tag, attrs, text) {
    const e = document.createElementNS(svgNS, tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    return e;
  }
  function seg(container, options, current, onPick) {
    container.innerHTML = "";
    options.forEach(([value, label]) => {
      const b = document.createElement("button");
      b.type = "button"; b.dataset.v = value; b.textContent = label;
      b.setAttribute("aria-pressed", String(value === current));
      b.addEventListener("click", () => {
        $$("button", container).forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
        onPick(value);
      });
      container.appendChild(b);
    });
  }
  function bindSeg(container, onPick) {
    $$("button", container).forEach((b) => b.addEventListener("click", () => {
      $$("button", container).forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      onPick(b.dataset.v);
    }));
  }

  // ------------------------------------------------------------ bound numbers
  $$("[data-k]").forEach((el) => {
    const v = get(el.dataset.k);
    if (v === undefined || v === null) return;          // keep fallback text
    el.textContent = (fmt[el.dataset.f] || fmt.raw)(v);
  });

  // ------------------------------------------------------------ repository links
  // Set once in index.html: <meta name="repo-url" content="https://github.com/...">
  const REPO = ((document.querySelector('meta[name="repo-url"]') || {}).content || "").replace(/\/+$/, "");
  $$("a[data-repo]").forEach((a) => {
    const path = a.dataset.repo;
    if (REPO) {
      a.href = path ? `${REPO}/${path.endsWith("/") ? "tree" : "blob"}/main/${path.replace(/\/$/, "")}` : REPO;
      a.rel = "noopener";
    } else {
      a.removeAttribute("href");            // no URL configured yet: plain text, not a dead link
    }
  });
  if (REPO) $$(".repo-text").forEach((e) => { e.textContent = REPO + ".git"; });

  // ------------------------------------------------------------ data section
  const q = D.quality;
  const qRows = [
    ["Missing values in ratings", q.missing_values_ratings],
    ["Duplicate user–movie pairs", q.duplicate_user_movie_pairs],
    ["Ratings outside 1–5", q.ratings_outside_1_5],
    ["Movie IDs unmatched after join", q.unmatched_movie_ids_after_join],
    ["Movies missing a release date", q.missing_release_date],
    ["Movies flagged genre “unknown”", q.movies_with_unknown_genre_flag],
    ["Duplicate titles (different IDs)", q.duplicate_titles_in_catalogue],
  ];
  $("#qualityTable").innerHTML = "<thead><tr><th>Check</th><th>Result</th></tr></thead><tbody>" +
    qRows.map(([a, b]) => `<tr><td>${a}</td><td>${fmt.int(b)}</td></tr>`).join("") + "</tbody>";

  const S = D.split;
  const NAMES = { train: "Train", val: "Validation", test: "Test" };
  $("#splitSteps").innerHTML = ["train", "val", "test"].map((k) =>
    `<div class="${k}"><b>${NAMES[k]}: ${fmt.int(S[k].rows)} ratings</b>${fmt.date(S[k].first)} to ${fmt.date(S[k].last)}<br>${fmt.int(S[k].users)} users, ${fmt.pct(S[k].positive_rate)} positive</div>`).join("");
  $("#splitTable").innerHTML = "<thead><tr><th>Split</th><th>Rows</th><th>Share</th><th>First rating</th><th>Last rating</th><th>Users</th><th>Movies</th><th>Positive</th></tr></thead><tbody>" +
    ["train", "val", "test"].map((k) => `<tr><td>${NAMES[k]}</td><td>${fmt.int(S[k].rows)}</td><td>${fmt.pct(S[k].share)}</td><td>${S[k].first.slice(0, 19)}</td><td>${S[k].last.slice(0, 19)}</td><td>${S[k].users}</td><td>${fmt.int(S[k].movies)}</td><td>${fmt.pct(S[k].positive_rate)}</td></tr>`).join("") + "</tbody>";

  // ------------------------------------------------------------ charts
  const charts = [];
  const onResize = (fn) => { charts.push(fn); fn(); };
  let rt;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => charts.forEach((f) => f()), 120); });
  const widthOf = (svg) => Math.max(280, svg.parentElement.clientWidth - 2);

  // timeline of weekly ratings coloured by split
  onResize(function timeline() {
    const svg = $("#timelineChart"), W = widthOf(svg), H = 170, m = { l: 44, r: 8, t: 12, b: 26 };
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.innerHTML = "";
    const wk = D.eda_weekly, max = Math.max(...wk.map((d) => d.n));
    const bw = (W - m.l - m.r) / wk.length;
    const tv = new Date(S.val.first), tt = new Date(S.test.first);
    wk.forEach((d, i) => {
      const end = new Date(d.week + "T00:00:00Z");
      const start = new Date(end.getTime() - 6 * 864e5);
      const col = end < tv ? "--pos" : start >= tt ? "--lr" : start >= tv && end < tt ? "--pop" : "--maj";
      const h = (d.n / max) * (H - m.t - m.b);
      svg.appendChild(svgEl("rect", { x: m.l + i * bw + 1, y: H - m.b - h, width: Math.max(1, bw - 2), height: h, rx: 1.5, style: `fill:var(${col})` }));
      if (i % 4 === 0) svg.appendChild(svgEl("text", { x: m.l + i * bw, y: H - 8, "font-size": 10, class: "muted" }, d.week.slice(0, 7)));
    });
    [0, max].forEach((v) => svg.appendChild(svgEl("text", { x: m.l - 6, y: H - m.b - (v / max) * (H - m.t - m.b) + 4, "text-anchor": "end", "font-size": 10, class: "muted" }, fmt.int(v))));
    const lg = [["--pos", "train"], ["--pop", "validation"], ["--lr", "test"], ["--maj", "week spans a cutoff"]];
    let x = m.l + 8;
    lg.forEach(([c, t]) => {
      svg.appendChild(svgEl("rect", { x, y: m.t, width: 9, height: 9, rx: 2, style: `fill:var(${c})` }));
      const tx = svgEl("text", { x: x + 13, y: m.t + 8.5, "font-size": 11 }, t); svg.appendChild(tx);
      x += 22 + t.length * 6.2;
    });
  });

  // metric comparison chart + results table
  const METRICS = [["pr_auc", "PR-AUC"], ["f1", "F1"], ["precision", "Precision"], ["recall", "Recall"], ["accuracy", "Accuracy"]];
  let proto = "fixed", metric = "pr_auc";
  const metricsFor = (p) => (p === "fixed" ? D.test : D.sequential.test);
  function drawMetric() {
    const svg = $("#metricChart"), W = widthOf(svg), m = { l: 140, r: 56, t: 6, b: 6 }, rowH = 34;
    const H = m.t + m.b + rowH * MODELS.length;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.innerHTML = "";
    const data = metricsFor(proto), best = Math.max(...MODELS.map((n) => data[n][metric]));
    const pr = metricsFor(proto)["Majority baseline"].positive_rate;
    const sx = (v) => m.l + v * (W - m.l - m.r);
    MODELS.forEach((n, i) => {
      const v = data[n][metric], y = m.t + i * rowH;
      svg.appendChild(svgEl("text", { x: m.l - 10, y: y + rowH / 2 + 4, "text-anchor": "end", "font-size": 12.5, "font-weight": n === "Logistic Regression" ? 700 : 500 }, SHORT[n]));
      svg.appendChild(svgEl("rect", { x: m.l, y: y + 7, width: sx(1) - m.l, height: rowH - 14, rx: 3, style: "fill:var(--paper)" }));
      svg.appendChild(svgEl("rect", { x: m.l, y: y + 7, width: Math.max(1, sx(v) - m.l), height: rowH - 14, rx: 3, style: `fill:var(${MODEL_VAR[n]})` }));
      svg.appendChild(svgEl("text", { x: sx(v) + 6, y: y + rowH / 2 + 4, "font-size": 12.5, "font-weight": v === best ? 800 : 500 }, v.toFixed(3)));
    });
    if (metric === "pr_auc" || metric === "accuracy" || metric === "precision") {
      svg.appendChild(svgEl("line", { x1: sx(pr), x2: sx(pr), y1: 0, y2: H, style: "stroke:var(--ink)", "stroke-dasharray": "3 3", "stroke-width": 1 }));
    }
    svg.setAttribute("aria-label", `Test ${metric} by model, ${proto === "fixed" ? "fixed snapshot" : "sequential"} protocol. Dashed line: positive rate.`);
    // table
    const keys = ["threshold", "accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"];
    const bestOf = Object.fromEntries(keys.map((k) => [k, Math.max(...MODELS.slice(1).map((n) => data[n][k]))]));
    $("#resultsTable").innerHTML = `<thead><tr><th>${proto === "fixed" ? "Fixed snapshot" : "Sequential"}, test</th><th>Threshold</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>PR-AUC</th><th>ROC-AUC</th></tr></thead><tbody>` +
      MODELS.map((n) => `<tr><td><span class="swatch" style="background:var(${MODEL_VAR[n]})"></span>${n}</td>` +
        keys.map((k) => `<td class="${k !== "threshold" && k !== "recall" && data[n][k] === bestOf[k] ? "best" : ""}">${data[n][k].toFixed(k === "threshold" ? 3 : 4)}</td>`).join("") + "</tr>").join("") + "</tbody>";
  }
  seg($("#metricSeg"), METRICS, metric, (v) => { metric = v; drawMetric(); });
  bindSeg($("#protoSeg"), (v) => { proto = v; drawMetric(); });
  onResize(drawMetric);

  // bootstrap + ranking tables
  const B = D.bootstrap_pr_auc_diff;
  $("#bootTable").innerHTML = "<thead><tr><th>PR-AUC difference</th><th>Fixed snapshot</th><th>95% interval</th><th>Sequential</th><th>95% interval</th></tr></thead><tbody>" +
    Object.keys(B.fixed_snapshot).map((k) => `<tr><td>${k}</td><td>${fmt.signed(B.fixed_snapshot[k].diff)}</td><td>${fmt.ci(B.fixed_snapshot[k].ci95)}</td><td>${fmt.signed(B.sequential[k].diff)}</td><td>${fmt.ci(B.sequential[k].ci95)}</td></tr>`).join("") + "</tbody>";
  const RF = D.ranking_test, RS = D.sequential.ranking_test;
  $("#rankTable").innerHTML = "<thead><tr><th>Within-user ranking</th><th>P@5 fixed</th><th>P@10 fixed</th><th>P@5 sequential</th><th>P@10 sequential</th></tr></thead><tbody>" +
    `<tr><td>Random order (expected)</td><td>${fmt.f3(RF.random_order["p@5"])}</td><td>${fmt.f3(RF.random_order["p@10"])}</td><td>${fmt.f3(RS.random_order["p@5"])}</td><td>${fmt.f3(RS.random_order["p@10"])}</td></tr>` +
    Object.keys(RF.models).map((n) => `<tr><td><span class="swatch" style="background:var(${MODEL_VAR[n]})"></span>${n}</td><td>${fmt.f3(RF.models[n]["p@5"])}</td><td>${fmt.f3(RF.models[n]["p@10"])}</td><td>${fmt.f3(RS.models[n]["p@5"])}</td><td>${fmt.f3(RS.models[n]["p@10"])}</td></tr>`).join("") + "</tbody>";

  // precision-recall curves
  const prShown = new Set(["Popularity baseline", "Logistic Regression", "Random Forest", "LR (movie features only)"]);
  const prCont = $("#prSeg");
  Object.keys(D.pr_curves_test).forEach((n) => {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = SHORT[n]; b.setAttribute("aria-pressed", "true");
    b.addEventListener("click", () => { prShown.has(n) ? prShown.delete(n) : prShown.add(n); b.setAttribute("aria-pressed", String(prShown.has(n))); drawPR(); });
    prCont.appendChild(b);
  });
  function drawPR() {
    const svg = $("#prChart"), W = Math.min(widthOf(svg), 720), H = Math.round(W * 0.58), m = { l: 46, r: 12, t: 10, b: 36 };
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.style.maxWidth = W + "px"; svg.innerHTML = "";
    const y0 = 0.4, sx = (r) => m.l + r * (W - m.l - m.r), sy = (p) => H - m.b - ((p - y0) / (1 - y0)) * (H - m.t - m.b);
    for (let i = 0; i <= 5; i++) {
      const r = i / 5, p = y0 + (i / 6) * (1 - y0);
      svg.appendChild(svgEl("line", { x1: sx(r), x2: sx(r), y1: m.t, y2: H - m.b, class: "axis" }));
      svg.appendChild(svgEl("text", { x: sx(r), y: H - m.b + 16, "text-anchor": "middle", "font-size": 10.5, class: "muted" }, r.toFixed(1)));
    }
    for (let i = 0; i <= 6; i++) {
      const p = y0 + i * 0.1;
      svg.appendChild(svgEl("line", { x1: m.l, x2: W - m.r, y1: sy(p), y2: sy(p), class: "axis" }));
      svg.appendChild(svgEl("text", { x: m.l - 6, y: sy(p) + 3.5, "text-anchor": "end", "font-size": 10.5, class: "muted" }, p.toFixed(1)));
    }
    svg.appendChild(svgEl("text", { x: (W + m.l) / 2, y: H - 4, "text-anchor": "middle", "font-size": 11 }, "Recall"));
    svg.appendChild(svgEl("text", { x: 12, y: (H - m.b) / 2, "font-size": 11, transform: `rotate(-90 12 ${(H - m.b) / 2})`, "text-anchor": "middle" }, "Precision"));
    const pos = D.test["Majority baseline"].positive_rate;
    svg.appendChild(svgEl("line", { x1: m.l, x2: W - m.r, y1: sy(pos), y2: sy(pos), style: "stroke:var(--ink)", "stroke-dasharray": "2 4" }));
    Object.entries(D.pr_curves_test).forEach(([n, pts]) => {
      if (!prShown.has(n)) return;
      const d = pts.map(([r, p], i) => (i ? "L" : "M") + sx(r).toFixed(1) + " " + sy(Math.max(p, y0)).toFixed(1)).join(" ");
      svg.appendChild(svgEl("path", { d, fill: "none", "stroke-width": n === "Logistic Regression" ? 2.4 : 1.6, style: `stroke:var(${MODEL_VAR[n]})` }));
    });
  }
  onResize(drawPR);

  // threshold slider
  const sweep = D.threshold_sweep_test_lr, thr = $("#thr");
  thr.value = (Math.round(D.thresholds["Logistic Regression"] * 100) / 100).toFixed(2);
  function drawThr() {
    const t = Number(thr.value), row = sweep.reduce((a, b) => (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a));
    $("#thrVal").textContent = row.t.toFixed(2);
    const c = row.confusion;
    $("#thrReadouts").innerHTML = [["Precision", row.precision], ["Recall", row.recall], ["F1", row.f1], ["Accuracy", row.accuracy]]
      .map(([k, v]) => `<div><b>${v.toFixed(3)}</b><span>${k}</span></div>`).join("") +
      `<div style="grid-column:1/-1"><span>Predicted positive: ${fmt.int(c.tp + c.fp)} of ${fmt.int(c.tp + c.fp + c.tn + c.fn)} pairs. True positives ${fmt.int(c.tp)}, false positives ${fmt.int(c.fp)}, false negatives ${fmt.int(c.fn)}, true negatives ${fmt.int(c.tn)}.</span></div>`;
  }
  thr.addEventListener("input", drawThr); drawThr();

  // subgroup chart
  let gProto = "fixed";
  function drawGroups() {
    const rows = gProto === "fixed" ? D.test_by_user_history : D.sequential.test_by_user_history;
    const models = ["Popularity baseline", "Logistic Regression", "Random Forest"];
    const svg = $("#groupChart"), W = widthOf(svg), H = 250, m = { l: 40, r: 10, t: 26, b: 42 };
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.innerHTML = "";
    const y0 = 0.5, sy = (v) => H - m.b - ((Math.max(v, y0) - y0) / (1 - y0)) * (H - m.t - m.b);
    [0.5, 0.6, 0.7, 0.8, 0.9, 1.0].forEach((v) => {
      svg.appendChild(svgEl("line", { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), class: "axis" }));
      svg.appendChild(svgEl("text", { x: m.l - 6, y: sy(v) + 3.5, "text-anchor": "end", "font-size": 10.5, class: "muted" }, v.toFixed(1)));
    });
    const gw = (W - m.l - m.r) / rows.length, bw = Math.min(34, (gw - 24) / models.length);
    rows.forEach((r, i) => {
      const cx = m.l + gw * i + gw / 2;
      models.forEach((n, j) => {
        const v = r[n + "|pr_auc"]; if (v == null) return;
        const x = cx - (models.length * bw) / 2 + j * bw;
        svg.appendChild(svgEl("rect", { x: x + 1, y: sy(v), width: bw - 2, height: H - m.b - sy(v), rx: 2, style: `fill:var(${MODEL_VAR[n]})` }));
        svg.appendChild(svgEl("text", { x: x + bw / 2, y: sy(v) - 4, "text-anchor": "middle", "font-size": 9.5 }, v.toFixed(2)));
      });
      svg.appendChild(svgEl("text", { x: cx, y: H - m.b + 15, "text-anchor": "middle", "font-size": 11, "font-weight": 600 }, r.group));
      svg.appendChild(svgEl("text", { x: cx, y: H - m.b + 29, "text-anchor": "middle", "font-size": 10, class: "muted" }, "n = " + fmt.int(r.n)));
    });
    let x = m.l;
    models.forEach((n) => {
      svg.appendChild(svgEl("rect", { x, y: 4, width: 9, height: 9, rx: 2, style: `fill:var(${MODEL_VAR[n]})` }));
      svg.appendChild(svgEl("text", { x: x + 13, y: 12.5, "font-size": 11 }, SHORT[n]));
      x += 30 + SHORT[n].length * 6.3;
    });
    svg.appendChild(svgEl("text", { x: W - m.r, y: 12.5, "font-size": 10.5, "text-anchor": "end", class: "muted" }, "PR-AUC (axis starts at 0.5)"));
  }
  bindSeg($("#groupSeg"), (v) => { gProto = v; drawGroups(); });
  onResize(drawGroups);

  // confusion matrix
  const CM_TEXT = {
    tp: (c) => `<strong>True positives (${fmt.int(c.tp)}).</strong> Predicted a 4–5 star rating and the user did give 4–5 stars. These are the useful recommendations.`,
    fp: (c) => `<strong>False positives (${fmt.int(c.fp)}).</strong> Predicted 4–5 stars but the user gave 1–3. In a recommender this is a suggestion the user did not enjoy as much as promised. Many are 3-star ratings, the ambiguous middle.`,
    fn: (c) => `<strong>False negatives (${fmt.int(c.fn)}).</strong> Predicted 1–3 stars but the user gave 4–5. A movie the user would have liked is left out of the recommendations.`,
    tn: (c) => `<strong>True negatives (${fmt.int(c.tn)}).</strong> Predicted 1–3 stars and the user gave 1–3. Correctly not recommended.`,
  };
  let cmModel = "Logistic Regression", cmCell = "fp";
  function drawCM() {
    const c = D.test[cmModel].confusion, n = c.tp + c.fp + c.tn + c.fn;
    const cell = (k, label) => `<button type="button" class="${k}" data-c="${k}" aria-pressed="${k === cmCell}">${fmt.int(c[k])}<small>${label}, ${fmt.pct0(c[k] / n)}</small></button>`;
    $("#cmGrid").innerHTML = `<span></span><span class="lab">Predicted 1–3</span><span class="lab">Predicted 4–5</span>
      <span class="lab">Rated 1–3</span>${cell("tn", "true negative")}${cell("fp", "false positive")}
      <span class="lab">Rated 4–5</span>${cell("fn", "false negative")}${cell("tp", "true positive")}`;
    $$("#cmGrid button").forEach((b) => b.addEventListener("click", () => { cmCell = b.dataset.c; drawCM(); }));
    $("#cmExplain").innerHTML = CM_TEXT[cmCell](c) + `<br><span class="ui" style="font-size:.8rem;color:var(--muted)">${cmModel}, threshold ${D.test[cmModel].threshold.toFixed(3)}.</span>`;
  }
  seg($("#cmSeg"), ["Majority baseline", "Popularity baseline", "Logistic Regression", "Random Forest"].map((n) => [n, SHORT[n]]), cmModel, (v) => { cmModel = v; drawCM(); });
  drawCM();

  // error tables
  function errTable(rows) {
    return "<thead><tr><th>User</th><th>Movie</th><th>Actual rating</th><th>LR score</th><th>User’s earlier ratings</th><th>User positive rate</th><th>Movie positive rate</th><th>Movie’s training ratings</th></tr></thead><tbody>" +
      rows.map((r) => `<tr><td>${r.user_id}</td><td class="wrap">${esc(r.title)}</td><td>${"★".repeat(r.rating)}</td><td>${r.p.toFixed(3)}</td><td>${r.user_hist_count ? fmt.int(r.user_hist_count) : "none (cold start)"}</td><td>${r.user_hist_count ? r.user_pos_rate.toFixed(2) : "prior " + r.user_pos_rate.toFixed(2)}</td><td>${r.movie_pos_rate.toFixed(2)}</td><td>${fmt.int(r.movie_hist_count)}</td></tr>`).join("") + "</tbody>";
  }
  $("#fpTable").innerHTML = errTable(D.error_examples.false_positives);
  $("#fnTable").innerHTML = errTable(D.error_examples.false_negatives);
  $("#genreErrTable").innerHTML = "<thead><tr><th>Genre</th><th>Test ratings</th><th>Error rate</th><th>False-positive share</th><th>False-negative share</th></tr></thead><tbody>" +
    D.test_errors_by_genre.map((g) => `<tr><td>${g.genre}</td><td>${fmt.int(g.n)}</td><td>${fmt.pct(g.error_rate)}</td><td>${fmt.pct(g.fp_rate_of_rows)}</td><td>${fmt.pct(g.fn_rate_of_rows)}</td></tr>`).join("") + "</tbody>";

  // ------------------------------------------------------------ explorer
  const THR = D.thresholds["Logistic Regression"];
  const users = D.demo_users;
  let cur = users[0], tab = "later", sel = null;
  const outcomeTag = (later) => later === 1 ? '<span class="tag pos">Rated 4–5</span>'
    : later === 0 ? '<span class="tag neg">Rated 1–3</span>' : '<span class="tag unk">Not rated</span>';

  const chips = $("#userChips");
  users.forEach((u, i) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "chip"; b.setAttribute("aria-pressed", String(i === 0));
    b.innerHTML = `User ${u.user_id} <small>${u.profile.n_ratings ? u.profile.n_ratings + " earlier" : "no history"}</small>`;
    b.addEventListener("click", () => { $$(".chip", chips).forEach((x) => x.setAttribute("aria-pressed", String(x === b))); cur = u; sel = null; renderExplorer(); });
    chips.appendChild(b);
  });
  $$(".tab").forEach((t) => t.addEventListener("click", () => {
    tab = t.dataset.tab; $$(".tab").forEach((x) => x.setAttribute("aria-selected", String(x === t))); renderList();
  }));

  function renderProfile() {
    const p = cur.profile;
    if (!p.n_ratings) {
      $("#profilePanel").innerHTML = `<h3>User ${cur.user_id}</h3><p class="empty">This user has no ratings before the training cutoff. The model falls back to movie features and training-wide averages, so every cold-start user gets the same scores, much like the popularity baseline.</p>`;
      return;
    }
    $("#profilePanel").innerHTML = `<h3>User ${cur.user_id}: what the model knows</h3>
      <p class="hint">Training-period ratings only, before ${fmt.date(S.val.first)}.</p>
      <div class="profile-stats"><div><b>${p.n_ratings}</b><span>earlier ratings</span></div><div><b>${p.mean_rating.toFixed(2)}</b><span>mean stars</span></div>
      <div style="grid-column:1/-1"><b>${fmt.pct0(p.positive_rate)}</b><span>rated 4–5 stars</span></div></div>
      <p class="hint" style="margin-bottom:.3rem">Share rated 4–5, by genre (count)</p>` +
      p.genres.slice(0, 8).map((g) => `<div class="gbar"><span>${esc(g.genre)}</span><span class="track"><span class="fill" style="width:${g.positive_rate * 100}%;display:block"></span></span><span class="n">${g.n}</span></div>`).join("");
  }
  const itemRow = (it, cols) => `<li><button type="button" class="movie-btn" data-id="${it.movie_id}" aria-current="${sel && sel.movie_id === it.movie_id}">
      <span class="t" title="${esc(it.title)}">${esc(it.title)}</span>${cols(it)}</button></li>`;
  const scoreCell = (v, cls) => `<span><span class="scorebar ${cls || ""}"><i style="width:${v * 100}%"></i></span><span class="num" style="font-size:.74rem">${v.toFixed(2)}</span></span>`;
  function hitLine(list, k, label) {
    const top = list.slice(0, k), rated = top.filter((x) => x.later != null), pos = rated.filter((x) => x.later === 1);
    return `<p class="hits">${label}: ${rated.length} of top ${k} were rated later, ${pos.length} of them 4–5 stars.</p>`;
  }
  function renderList() {
    const body = $("#listBody");
    if (tab === "later") {
      const items = cur.test_items, shown = items.slice(0, 40);
      const top5 = items.slice(0, 5), p5 = top5.filter((x) => x.later === 1).length;
      const total = cur.n_test_items || items.length, totalPos = cur.n_test_items_positive != null ? cur.n_test_items_positive : items.filter((x) => x.later === 1).length;
      body.innerHTML = `<p class="hint">Movies this user rated during the test period, sorted by the model’s score. The model never saw these ratings. Of the top 5, ${p5} were rated 4–5 stars; across all ${total} test ratings, ${totalPos} were.</p>
        <div class="list-head"><span>Movie</span><span>LR score</span><span>Outcome</span></div>
        <ul class="movie-list">${shown.map((it) => itemRow(it, (x) => scoreCell(x.lr) + outcomeTag(x.later))).join("")}</ul>` +
        (total > shown.length ? `<p class="hits">Showing the ${shown.length} highest-scored of ${total}.</p>` : "");
    } else {
      body.innerHTML = `<p class="hint">The 10 highest-scoring movies this user had not rated by the training cutoff. Most were never rated, so their outcome is unknown, not negative.</p>
        <div class="compare-cols"><div><h4 style="color:var(--lr)">Logistic Regression</h4>${hitLine(cur.top_lr, 10, "Personalised")}
        <ul class="movie-list">${cur.top_lr.map((it) => itemRow(it, (x) => outcomeTag(x.later))).join("")}</ul></div>
        <div><h4 style="color:var(--pop)">Popularity baseline</h4>${hitLine(cur.top_pop, 10, "Same for everyone")}
        <ul class="movie-list">${cur.top_pop.map((it) => itemRow(it, (x) => outcomeTag(x.later))).join("")}</ul></div></div>`;
    }
    $$(".movie-btn", body).forEach((b) => b.addEventListener("click", () => {
      const id = Number(b.dataset.id);
      sel = [...cur.test_items, ...cur.top_lr, ...cur.top_pop].find((x) => x.movie_id === id);
      $$(".movie-btn", body).forEach((x) => x.setAttribute("aria-current", String(Number(x.dataset.id) === id)));
      renderVerdict();
    }));
  }
  function renderVerdict() {
    const it = sel;
    if (!it) { $("#verdictPanel").innerHTML = '<p class="empty">Select a movie to see its prediction.</p>'; return; }
    const positive = it.lr >= THR;
    $("#verdictPanel").innerHTML = `<p class="movie-title">${esc(it.title)}</p>
      <p class="genres">${it.genres.map(esc).join(", ")}</p>
      <div class="gauge"><div class="gauge-num">${Math.round(it.lr * 100)}<small>% chance of 4–5 stars</small></div>
        <div class="gauge-track"><div class="gauge-fill" style="width:${it.lr * 100}%"></div>
        <div class="gauge-thr" style="left:${THR * 100}%"><span>threshold ${THR.toFixed(2)}</span></div></div></div>
      <dl class="kv">
        <dt>Model decision</dt><dd style="color:var(${positive ? "--pos" : "--muted"})">${positive ? "Recommend" : "Do not recommend"}</dd>
        <dt>Popularity score</dt><dd>${it.pop.toFixed(2)}</dd>
        <dt>Movie’s training ratings</dt><dd>${fmt.int(it.movie_hist_count)}</dd>
        <dt>What the user did later</dt><dd>${it.later === 1 ? "Rated 4–5" : it.later === 0 ? "Rated 1–3" : "Not rated (unknown)"}</dd>
      </dl>
      <p class="note">Precomputed by the trained Logistic Regression from training-period data (fixed snapshot). This page does not run a live model.</p>`;
  }
  function renderExplorer() {
    renderProfile();
    sel = sel || cur.test_items[0] || cur.top_lr[0];
    renderList(); renderVerdict();
  }
  renderExplorer();

  // ------------------------------------------------------------ rail + topnav highlight
  const links = $$(".rail a[href^='#']");
  const navLinks = $$(".topnav-links a[href^='#']");
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => entries.forEach((e) => {
      if (e.isIntersecting) {
        const id = "#" + e.target.id;
        links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === id));
        navLinks.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === id));
      }
    }), { rootMargin: "-20% 0px -70% 0px" });
    $$("main section").forEach((s) => io.observe(s));
  }

  // ------------------------------------------------------------ topnav repo link
  const navRepo = $(".topnav-repo a");
  if (navRepo && REPO) { navRepo.href = REPO; navRepo.rel = "noopener"; }
  else if (navRepo && !REPO) { navRepo.removeAttribute("href"); }
})();
