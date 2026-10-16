// FocusAI content script: extracts the page, asks the backend for a decision, shows the overlay.
(() => {
  if (window.__focusaiLoaded) return;
  window.__focusaiLoaded = true;

  const BLOCK_SELECTOR = "p, h1, h2, h3, h4, li, blockquote";
  const SKIP_ANCESTORS = "nav, header, footer, aside, form, [aria-hidden='true'], #focusai-host";
  let lastUrl = null;
  let pauseTimer = null;
  let heatmapOn = false;
  const heatBackup = new Map();

  // ---------- extraction ----------
  const isYouTube = () => /(^|\.)youtube\.com$/.test(location.hostname);
  const isYouTubeVideo = () => isYouTube() && (location.pathname === "/watch" || location.pathname.startsWith("/shorts/"));

  // ---------- YouTube: extract the individual video (title, description, channel, hashtags) ----------
  // The transcript is fetched by the backend from the video id: caption URLs in the page are signed and
  // short-lived, so reading them from a content script is unreliable. Every step here is best-effort.
  const normalizeText = (s) =>
    (s || "")
      .replace(/https?:\/\/\S+|www\.\S+/g, " ")
      .replace(/\b\d{1,2}:\d{2}(?::\d{2})?\b/g, " ")
      .replace(/(\.\.\.more|Show more|Show less|…more)\s*/gi, " ")
      .replace(/\s+/g, " ")
      .trim();

  const textOf = (selectors) => {
    for (const sel of selectors) {
      const el = document.querySelector(sel);
      const t = el && (el.textContent || "").replace(/\s+/g, " ").trim();
      if (t) return t;
    }
    return "";
  };

  function youTubeVideoId() {
    if (location.pathname === "/watch") return new URLSearchParams(location.search).get("v") || "";
    const m = location.pathname.match(/^\/shorts\/([^/?]+)/);
    return m ? m[1] : "";
  }

  // <meta> tags are NOT refreshed on YouTube's in-page navigation, so only trust them when the
  // canonical URL still points at the current video.
  function metaMatchesVideo(id) {
    const canon = document.querySelector("link[rel='canonical']");
    return !!(canon && id && canon.href.includes(id));
  }

  function youTubeTitle(id) {
    const dom = textOf(["ytd-watch-metadata h1", "h1.ytd-watch-metadata", "h1.slim-video-information-title", "ytd-reel-video-renderer[is-active] h2"]);
    if (dom) return dom;
    if (metaMatchesVideo(id)) {
      const m = document.querySelector("meta[name='title'], meta[property='og:title']");
      if (m && m.content) return m.content;
    }
    return document.title.replace(/^\(\d+\)\s*/, "").replace(/ - YouTube$/, "");
  }

  function youTubeDescription(id) {
    const dom = textOf(["ytd-watch-metadata #description-inner", "#description-inline-expander", "ytd-text-inline-expander", "#description"]);
    if (dom) return dom;
    if (metaMatchesVideo(id)) {
      const m = document.querySelector("meta[name='description'], meta[property='og:description']");
      if (m && m.content) return m.content;
    }
    return "";
  }

  function youTubeDomReady(id) {
    const flexy = document.querySelector("ytd-watch-flexy");
    const shown = flexy && flexy.getAttribute("video-id");
    if (shown && shown !== id) return false; // still showing the previous video
    const t = youTubeTitle(id).replace(/^\(\d+\)\s*/, "").trim().toLowerCase();
    return !!t && t !== "youtube";
  }

  async function waitFor(predicate, timeoutMs, stepMs, cancelled) {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      if (cancelled()) return false;
      try { if (predicate()) return true; } catch (_) { /* keep waiting */ }
      await new Promise((r) => setTimeout(r, stepMs));
    }
    return false; // proceed anyway: the backend knows the authoritative title
  }

  async function buildYouTubePayload(url, cancelled) {
    const id = youTubeVideoId();
    await waitFor(() => youTubeDomReady(id), 3500, 250, cancelled);
    let metadata = { video_id: id };
    let title = "";
    try {
      title = normalizeText(youTubeTitle(id));
      const hashtags = Array.from(document.querySelectorAll("ytd-watch-metadata a[href^='/hashtag/']"))
        .map((a) => (a.textContent || "").replace(/^#/, "").trim()).filter(Boolean).slice(0, 10);
      metadata = {
        video_id: id,
        description: normalizeText(youTubeDescription(id)).slice(0, 2000),
        channel: textOf(["ytd-watch-metadata ytd-channel-name a", "#owner ytd-channel-name a", "ytd-video-owner-renderer a"]).slice(0, 200),
        keywords: hashtags,
      };
    } catch (_) { /* extraction must never throw: fall back to whatever we have */ }
    return { url, title, text: "", metadata };
  }

  function extractText() {
    const root = document.querySelector("article") || document.querySelector("main") || document.body;
    const parts = [];
    root.querySelectorAll(BLOCK_SELECTOR).forEach((el) => {
      if (el.closest(SKIP_ANCESTORS) || (["LI", "BLOCKQUOTE"].includes(el.tagName) && el.querySelector("p, li"))) return;
      const t = (el.innerText || "").replace(/\s+/g, " ").trim();
      if (!t || (el.tagName !== "H1" && t.split(" ").length < 3)) return;
      parts.push(/[.!?:;"\u201d)]$/.test(t) ? t : t + ".");
    });
    let text = parts.join(" ");
    if (text.length < 200) text = (document.body.innerText || "").replace(/\s+/g, " ").trim();
    return text.slice(0, 30000);
  }

  function pageTitle() {
    return isYouTube() ? document.title.replace(/^\(\d+\)\s*/, "").replace(/ - YouTube$/, "") : document.title;
  }

  // ---------- tiny DOM helper (all text via textContent: page/backend text is never parsed as HTML) ----------
  function h(tag, props = {}, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (k === "class") el.className = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v);
    }
    kids.flat().forEach((c) => c != null && el.append(c));
    return el;
  }

  const STYLE = `
    :host { all: initial; }
    * { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }

    /* Overlay Backdrop */
    .overlay {
      position: fixed;
      inset: 0;
      z-index: 2147483647;
      background: rgba(15, 23, 42, 0.82);
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      display: flex;
      align-items: center;
      justify-content: center;
      color: #0f172a;
      animation: focusai-fade-in 0.2s ease-out;
    }

    @keyframes focusai-fade-in {
      from { opacity: 0; }
      to { opacity: 1; }
    }

    @keyframes focusai-slide-up {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Block Card Modal */
    .card {
      background: #ffffff;
      border-radius: 20px;
      padding: 32px;
      width: min(540px, 92vw);
      max-height: 88vh;
      overflow-y: auto;
      box-shadow: 0 25px 50px -12px rgba(15, 23, 42, 0.35);
      border-top: 5px solid #dc2626;
      animation: focusai-slide-up 0.25s ease-out;
    }

    /* Toast Container (Warn / Allow Details) */
    .toast {
      position: fixed;
      right: 20px;
      bottom: 20px;
      z-index: 2147483647;
      width: min(450px, 92vw);
      max-height: 88vh;
      overflow-y: auto;
      background: #ffffff;
      color: #0f172a;
      border-radius: 16px;
      padding: 20px 22px;
      box-shadow: 0 16px 36px -4px rgba(15, 23, 42, 0.2), 0 6px 16px -2px rgba(15, 23, 42, 0.1);
      border: 1px solid #e2e8f0;
      border-left: 5px solid var(--c);
      animation: focusai-slide-up 0.25s ease-out;
    }

    /* Floating Pill (Allow Notification) */
    .pill {
      position: fixed;
      right: 20px;
      bottom: 20px;
      z-index: 2147483647;
      background: #16a34a;
      color: #ffffff;
      border-radius: 9999px;
      padding: 9px 18px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      box-shadow: 0 6px 20px rgba(22, 163, 74, 0.35);
      display: flex;
      align-items: center;
      gap: 8px;
      animation: focusai-slide-up 0.25s ease-out;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .pill:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(22, 163, 74, 0.45);
    }

    /* Card & Toast Header Typography */
    .brand-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;
    }
    .brand-name {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: #64748b;
    }
    .status-badge {
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.04em;
      padding: 3px 8px;
      border-radius: 9999px;
      text-transform: uppercase;
    }
    .status-badge.block { background: #fee2e2; color: #991b1b; }
    .status-badge.warn { background: #fef3c7; color: #92400e; }
    .status-badge.allow { background: #dcfce7; color: #166534; }

    h2 {
      margin: 0 0 8px;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.01em;
      color: #0f172a;
      line-height: 1.3;
    }
    .toast h2 {
      font-size: 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }

    .score-callout {
      display: inline-flex;
      align-items: baseline;
      gap: 8px;
      font-size: 14px;
      color: #475569;
      margin: 4px 0 12px;
    }
    .score-value {
      font-size: 26px;
      font-weight: 800;
      color: var(--c);
      font-variant-numeric: tabular-nums;
    }
    .score-pill {
      background: #f1f5f9;
      color: var(--c);
      padding: 3px 9px;
      border-radius: 9999px;
      font-weight: 700;
      font-size: 12px;
      border: 1px solid #e2e8f0;
      font-variant-numeric: tabular-nums;
    }

    p {
      margin: 8px 0;
      font-size: 14px;
      line-height: 1.55;
      color: #334155;
    }
    .toast p {
      font-size: 13px;
    }

    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 10px 0;
    }
    .chip {
      background: #eef2ff;
      color: #4338ca;
      border: 1px solid #c7d2fe;
      border-radius: 9999px;
      padding: 2px 10px;
      font-size: 12px;
      font-weight: 500;
    }

    ul {
      margin: 8px 0 0 18px;
      padding: 0;
      font-size: 13px;
      color: #475569;
      line-height: 1.5;
    }

    .row {
      display: flex;
      gap: 10px;
      margin-top: 16px;
      flex-wrap: wrap;
    }

    button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      border: 1px solid transparent;
      border-radius: 10px;
      padding: 9px 16px;
      font-size: 13px;
      font-weight: 600;
      transition: all 0.15s ease;
    }
    button:focus-visible {
      outline: 2px solid #4f46e5;
      outline-offset: 2px;
    }
    .primary {
      background: #4f46e5;
      color: #ffffff;
      box-shadow: 0 1px 3px rgba(79, 70, 229, 0.3);
    }
    .primary:hover:not(:disabled) {
      background: #4338ca;
      box-shadow: 0 3px 8px rgba(79, 70, 229, 0.4);
    }
    .ghost {
      background: #f1f5f9;
      border-color: #e2e8f0;
      color: #1e293b;
    }
    .ghost:hover:not(:disabled) {
      background: #e2e8f0;
      color: #0f172a;
    }
    button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .legend {
      font-size: 11px;
      color: #64748b;
      margin-top: 8px;
    }

    /* Integrated Gradients (IG) Redesign */
    .ig-container {
      margin-top: 14px;
      border-top: 1px solid #e2e8f0;
      padding-top: 14px;
    }
    .ig-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 8px;
    }
    .ig-header-title {
      font-size: 13px;
      font-weight: 700;
      color: #0f172a;
    }
    .ig-badge {
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      background: #eef2ff;
      color: #4f46e5;
      padding: 2px 7px;
      border-radius: 9999px;
      border: 1px solid #c7d2fe;
    }
    .ig-selector-label {
      font-size: 11px;
      font-weight: 600;
      color: #64748b;
      margin-bottom: 6px;
    }
    .ig-list {
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-bottom: 8px;
    }
    .ig-passage-btn {
      text-align: left;
      font-weight: 500;
      font-size: 12px;
      padding: 8px 12px;
      border-radius: 8px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      color: #334155;
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      align-items: baseline;
      gap: 6px;
    }
    .ig-passage-btn:hover {
      background: #f1f5f9;
      border-color: #cbd5e1;
      color: #0f172a;
    }
    .ig-passage-btn.active {
      background: #eef2ff;
      border-color: #4f46e5;
      color: #3730a3;
      font-weight: 600;
    }
    .ig-passage-num {
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      color: #64748b;
      flex-shrink: 0;
    }
    .ig-passage-text {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .ig-panel {
      margin-top: 10px;
      padding: 12px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      max-height: 44vh;
      overflow-y: auto;
    }
    .ig-words-box {
      margin-bottom: 10px;
    }
    .ig-words {
      font-size: 13.5px;
      line-height: 2.1;
      margin: 0;
      word-wrap: break-word;
    }
    .ig-words .w {
      display: inline-block;
      border-radius: 4px;
      padding: 1px 4px;
      margin: 1px 1px;
      cursor: pointer;
      outline-offset: 1px;
      transition: transform 0.1s ease, box-shadow 0.1s ease;
    }
    .ig-words .w:hover, .ig-words .w:focus-visible {
      outline: 2px solid #4f46e5;
      transform: translateY(-1px);
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* Detail Card */
    .ig-detail-card {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      padding: 10px 12px;
      margin: 10px 0;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
      min-height: 44px;
    }
    .ig-detail-content {
      font-size: 12px;
      color: #0f172a;
      line-height: 1.5;
    }

    /* Legend */
    .ig-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 6px 14px;
      font-size: 11px;
      color: #475569;
      margin: 10px 0 6px;
      padding-top: 8px;
      border-top: 1px solid #e2e8f0;
      align-items: center;
    }
    .ig-legend-item {
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }
    .ig-legend i {
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 3px;
      border: 1px solid rgba(0, 0, 0, 0.15);
      flex-shrink: 0;
    }

    /* Collapsible Technical Details */
    .ig-about {
      margin-top: 8px;
      font-size: 11px;
      color: #64748b;
    }
    .ig-about summary {
      cursor: pointer;
      font-weight: 600;
      color: #475569;
      padding: 4px 0;
      user-select: none;
    }
    .ig-about summary:hover {
      color: #0f172a;
    }
    .ig-about-content {
      padding: 8px 10px;
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      margin-top: 6px;
    }
    .ig-consistency-badge {
      display: inline-block;
      padding: 1px 6px;
      border-radius: 4px;
      font-weight: 600;
      font-size: 10px;
      text-transform: uppercase;
      margin-left: 6px;
    }
    .ig-consistency-badge.ok {
      background: #dcfce7;
      color: #166534;
    }
    .ig-consistency-badge.approx {
      background: #fef3c7;
      color: #92400e;
    }
    .ig-note {
      font-size: 11px;
      color: #64748b;
      margin: 4px 0;
      line-height: 1.4;
    }
  `;

  const COLORS = { ALLOW: "#16a34a", WARN: "#d97706", BLOCK: "#dc2626" };
  let hostEl = null;

  function mount(...nodes) {
    unmount();
    hostEl = h("div", { id: "focusai-host" });
    const root = hostEl.attachShadow({ mode: "closed" });
    root.append(h("style", {}, STYLE), ...nodes);
    document.documentElement.append(hostEl);
  }

  function unmount() {
    if (hostEl) hostEl.remove();
    hostEl = null;
    clearInterval(pauseTimer);
    document.documentElement.style.removeProperty("overflow");
  }

  function details(r, includeHeatmapButton) {
    const nodes = [];
    nodes.push(h("p", {}, r.explanation));
    if (r.matching_concepts.length) {
      nodes.push(h("div", { class: "chips" }, r.matching_concepts.slice(0, 6).map((c) => h("span", { class: "chip" }, c))));
    }
    if (r.reasons.length) nodes.push(h("ul", {}, r.reasons.slice(0, 4).map((x) => h("li", {}, x))));
    if (includeHeatmapButton && r.heatmap.length) {
      nodes.push(h("div", { class: "legend" }, "Heatmap: green = relevant, yellow = partly, red = unrelated passages"));
    }
    return nodes;
  }

  // ---------- UI states ----------
  function showBlock(r, payload) {
    const color = COLORS.BLOCK;
    const card = h("div", { class: "card", style: `--c:${color}` },
      h("div", { class: "brand-bar" },
        h("span", { class: "brand-name" }, "FocusAI Study Shield"),
        h("span", { class: "status-badge block" }, "Distraction Blocked"),
      ),
      h("h2", {}, "This page looks like a distraction"),
      h("div", { class: "score-callout" },
        h("span", { class: "score-value" }, `${Math.round(r.relevance_score)}%`),
        h("span", {}, "relevant to your active study goal"),
      ),
      ...details(r, false),
      h("div", { class: "row" },
        h("button", { class: "primary", onclick: goBack }, "Go back to studying"),
        h("button", { class: "ghost", onclick: () => proceed(payload) }, "Continue anyway (logged)"),
      ),
    );
    mount(h("div", { class: "overlay" }, card));
    document.documentElement.style.setProperty("overflow", "hidden");
    const pause = () => document.querySelectorAll("video, audio").forEach((m) => { try { m.pause(); } catch (_) {} });
    pause();
    pauseTimer = setInterval(pause, 500);
  }

  // ---------- optional word-level explanation (Integrated Gradients) ----------
  // Rendered inside FocusAI's own closed shadow root: the page's DOM, links and inputs are never touched.
  // Diverging blue (+) / orange (-) scale; sign is also given in the tooltip/aria-label and detail line.
  const igResults = new Map(); // sentence -> backend response (cleared on navigation)
  const igBusy = new Set();    // sentences with a request in flight: no duplicate requests
  const IG_POS = "37,99,235";
  const IG_NEG = "234,88,12";
  const igColor = (n) => (Math.abs(n) < 0.05 ? "transparent" : `rgba(${n > 0 ? IG_POS : IG_NEG},${(0.12 + 0.6 * Math.min(1, Math.abs(n))).toFixed(2)})`);
  const signed = (v) => (v > 0 ? "+" : v < 0 ? "\u2212" : "") + Math.abs(v).toFixed(4);
  const clip = (t, n) => (t.length > n ? t.slice(0, n - 1).trimEnd() + "\u2026" : t);

  function igMessage(panel, res) {
    const r = res && res.result;
    let msg = "Word-level explanation is unavailable right now. The sentence heatmap is unaffected.";
    if (r && r.available === false && r.message) msg = r.message + " The sentence heatmap is unaffected.";
    else if (res && res.skipped) msg = "Word-level explanation is only available during an active study session.";
    panel.replaceChildren(h("p", { class: "ig-note", role: "status" }, msg));
  }

  function renderExplanation(panel, r) {
    const detailCard = h("div", { class: "ig-detail-card", "aria-live": "polite" });
    const detailContent = h("div", { class: "ig-detail-content" }, "Hover, focus or tap a word to inspect its attribution.");
    detailCard.append(detailContent);

    const wordsBox = h("div", { class: "ig-words-box" });
    const words = h("p", { class: "ig-words", role: "group", "aria-label": "Word-level attribution for the selected sentence" });
    wordsBox.append(words);

    let prevEnd = null;
    r.words.forEach((w) => {
      if (prevEnd != null && w.start > prevEnd) words.append(" ");
      prevEnd = w.end;
      const effect = w.normalized > 0 ? "raises" : w.normalized < 0 ? "lowers" : "does not change";
      const effectSymbol = w.normalized > 0 ? "+ raises" : w.normalized < 0 ? "− lowers" : "≈ no effect";
      const label = `${w.word}: ${signed(w.attribution)} (${effect} similarity to your study goal)`;
      const span = h("span", { class: "w", tabindex: "0", title: label, "aria-label": label }, w.word);
      span.style.backgroundColor = igColor(w.normalized);
      const show = () => {
        const subs = w.tokens && w.tokens.length > 1
          ? ` · sub-words: ${w.tokens.map((t) => `${t.token} (${signed(t.attribution)})`).join(", ")}`
          : "";
        detailContent.textContent = `\u201c${w.word}\u201d [${effectSymbol}] · raw IG: ${signed(w.attribution)} · scaled: ${w.normalized.toFixed(2)}${subs}`;
      };
      ["mouseenter", "focus", "click"].forEach((ev) => span.addEventListener(ev, show));
      words.append(span);
    });

    const swatch = (color, text) => h("span", { class: "ig-legend-item" },
      (() => { const i = h("i"); i.style.backgroundColor = color; return i; })(),
      text,
    );
    const legend = h("div", { class: "ig-legend" },
      swatch(igColor(1), "+ raises similarity (positive)"),
      swatch("transparent", "≈ negligible effect"),
      swatch(igColor(-1), "− lowers similarity (negative)"),
    );

    const expected = (r.target_value ?? 0) - (r.baseline_value ?? 0);
    const about = h("details", { class: "ig-about" },
      h("summary", {}, "About this explanation & consistency verification"),
      h("div", { class: "ig-about-content" },
        h("p", { class: "ig-note" },
          "Completeness check: attributions sum to ",
          h("strong", {}, r.sum_attributions.toFixed(3)),
          ` (expected ${expected.toFixed(3)}) \u2014 `,
          h("span", { class: `ig-consistency-badge ${r.converged ? "ok" : "approx"}` }, r.converged ? "Consistent" : "Approximate"),
        ),
        h("p", { class: "ig-note" },
          "Target: Cosine similarity of this sentence embedding to your study goal embedding. " +
          "Tooltip and detail values show raw Integrated Gradients attributions. " +
          "Color intensity reflects normalized |score| relative to largest |score|. " +
          "This explains similarity only, not the full FocusAI decision.",
        ),
        r.truncated ? h("p", { class: "ig-note", style: "color:#b45309;" },
          `Long sentence: only the first ${r.n_tokens}${r.n_tokens_total ? ` of ${r.n_tokens_total}` : ""} tokens were explained.`,
        ) : null,
      ),
    );

    panel.replaceChildren(wordsBox, detailCard, legend, about);
  }

  async function explainSentence(sentence, panel, btn) {
    if (igBusy.has(sentence)) return;
    const cached = igResults.get(sentence);
    if (cached) { renderExplanation(panel, cached); return; }
    igBusy.add(sentence);
    btn.disabled = true;
    panel.replaceChildren(h("p", { class: "ig-note", role: "status" }, "Calculating word-level attributions\u2026"));
    let res;
    try { res = await chrome.runtime.sendMessage({ type: "EXPLAIN_TOKENS", sentence }); } catch (_) { res = { error: "unreachable" }; }
    igBusy.delete(sentence);
    btn.disabled = false;
    if (!panel.isConnected) return; // dismissed or navigated away meanwhile
    if (res && res.result && res.result.available && Array.isArray(res.result.words) && res.result.words.length) {
      igResults.set(sentence, res.result);
      renderExplanation(panel, res.result);
    } else {
      igMessage(panel, res);
    }
  }

  function explainSection(r) {
    const source = r.highlighted_sentences && r.highlighted_sentences.length
      ? r.highlighted_sentences
      : (r.heatmap || []).slice().sort((a, b) => b.score - a.score);
    const picks = source.filter((s) => s && typeof s.text === "string" && s.text.trim()).slice(0, 3);
    if (!picks.length) return [];
    const panel = h("div", { class: "ig-panel" });
    panel.style.display = "none";

    let activeBtn = null;
    const buttons = picks.map((s, idx) => {
      const btn = h("button", {
        class: "ig-passage-btn",
        "aria-label": `Explain words in passage ${idx + 1}: ${clip(s.text, 80)}`,
      },
        h("span", { class: "ig-passage-num" }, `Passage ${idx + 1}`),
        h("span", { class: "ig-passage-text" }, `\u201c${clip(s.text, 44)}\u201d`),
      );
      btn.addEventListener("click", () => {
        if (activeBtn) activeBtn.classList.remove("active");
        btn.classList.add("active");
        activeBtn = btn;
        panel.style.display = "block";
        explainSentence(s.text, panel, btn);
      });
      return btn;
    });

    const container = h("div", { class: "ig-container" },
      h("div", { class: "ig-header" },
        h("span", { class: "ig-header-title" }, "Word-level explanation"),
        h("span", { class: "ig-badge" }, "Integrated Gradients"),
      ),
      h("div", { class: "ig-selector-label" }, "Select a passage to explain with token attributions:"),
      h("div", { class: "ig-list" }, ...buttons),
      panel,
    );

    return [container];
  }

  function showToast(r, decision) {
    const color = COLORS[decision];
    const isWarn = decision === "WARN";
    const title = isWarn ? "Only partly related to your goal" : "On topic";
    const badgeText = isWarn ? "Partially Relevant" : "Relevant";
    const badgeClass = isWarn ? "warn" : "allow";

    const toast = h("div", { class: "toast", style: `--c:${color}` },
      h("div", { class: "brand-bar" },
        h("span", { class: "brand-name" }, "FocusAI Assistant"),
        h("span", { class: `status-badge ${badgeClass}` }, badgeText),
      ),
      h("h2", {},
        h("span", {}, title),
        h("span", { class: "score-pill" }, `${Math.round(r.relevance_score)}%`),
      ),
      ...details(r, true),
      ...explainSection(r),
      h("div", { class: "row" },
        r.heatmap.length ? h("button", { class: "primary", onclick: () => toggleHeatmap(r.heatmap) }, "Toggle heatmap") : "",
        h("button", { class: "ghost", onclick: unmount }, "Dismiss"),
      ),
    );
    mount(toast);
  }

  function showPill(r) {
    const pill = h("div", { class: "pill", onclick: () => showToast(r, "ALLOW") },
      h("span", {}, "\u2713"),
      h("span", {}, `FocusAI \u00b7 ${Math.round(r.relevance_score)}% relevant`),
    );
    mount(pill);
    setTimeout(() => { if (hostEl && pill.isConnected) unmount(); }, 4500);
  }

  function goBack() {
    unmount();
    if (history.length > 1) history.back();
    else location.href = "about:blank";
  }

  async function proceed(payload) {
    try { await chrome.runtime.sendMessage({ type: "OVERRIDE", payload }); } catch (_) { /* logging is best-effort */ }
    unmount();
  }

  // ---------- in-page relevance heatmap (element-level shading from sentence scores) ----------
  const norm = (s) => s.toLowerCase().replace(/\s+/g, " ").trim();

  function toggleHeatmap(heatmap) {
    if (heatmapOn) {
      heatBackup.forEach((prev, el) => el.style.setProperty("background-color", prev || "", prev ? "important" : ""));
      heatBackup.clear();
      heatmapOn = false;
      return;
    }
    const snippets = heatmap.filter((s) => s.text.length > 20).map((s) => ({ key: norm(s.text).slice(0, 60), score: s.score }));
    document.querySelectorAll(BLOCK_SELECTOR).forEach((el) => {
      if (el.closest(SKIP_ANCESTORS)) return;
      const text = norm(el.innerText || "");
      let best = null;
      for (const s of snippets) if (text.includes(s.key)) best = Math.max(best ?? 0, s.score);
      if (best == null) return;
      const color = best >= 60 ? "rgba(34,197,94,.30)" : best >= 30 ? "rgba(250,204,21,.34)" : "rgba(239,68,68,.16)";
      heatBackup.set(el, el.style.getPropertyValue("background-color"));
      el.style.setProperty("background-color", color, "important");
    });
    heatmapOn = true;
  }

  // ---------- Phase 7: activity tracking (hostname only, never full URLs or page text) ----------
  let trackDecision = null;
  let trackHost = "";
  let visibleSecs = 0;
  let lastClick = 0;
  let lastScroll = 0;

  function track(event_type, metadata = {}) {
    try {
      const p = chrome.runtime.sendMessage({ type: "TRACK", events: [{ event_type, metadata }] });
      if (p && p.catch) p.catch(() => {});
    } catch (_) { /* extension reloaded: ignore */ }
  }

  function flushDwell() {
    if (trackDecision && visibleSecs >= 1) {
      track("dwell_time", { seconds: visibleSecs, decision: trackDecision, host: trackHost });
    }
    visibleSecs = 0;
  }

  setInterval(() => {
    if (trackDecision && document.visibilityState === "visible" && document.hasFocus()) visibleSecs += 1;
  }, 1000);
  setInterval(flushDwell, 15000);
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden") flushDwell(); });
  window.addEventListener("pagehide", flushDwell);
  document.addEventListener("click", () => {
    const n = Date.now();
    if (trackDecision && n - lastClick > 5000) { lastClick = n; track("click"); }
  }, true);
  document.addEventListener("scroll", () => {
    const n = Date.now();
    if (trackDecision && n - lastScroll > 5000) { lastScroll = n; track("scroll"); }
  }, { capture: true, passive: true });
  // ---------- main flow ----------
  let runId = 0;
  async function run() {
    const url = location.href.split("#")[0];
    if (url === lastUrl) return;
    lastUrl = url;
    flushDwell();
    trackDecision = null;
    const myRun = ++runId; // a newer navigation invalidates this run
    const stale = () => myRun !== runId || location.href.split("#")[0] !== url;
    unmount();
    igResults.clear();
    igBusy.clear();
    heatBackup.forEach((prev, el) => el.style.removeProperty("background-color"));
    heatBackup.clear();
    heatmapOn = false;

    // YouTube home/search/channel pages are navigation hubs: judge the individual videos, not the site.
    if (isYouTube() && !isYouTubeVideo()) return;

    let payload;
    try {
      payload = isYouTube() ? await buildYouTubePayload(url, stale) : { url, title: pageTitle(), text: extractText() };
    } catch (_) {
      return; // extraction failed: stay out of the way
    }
    if (stale()) return;

    let res;
    try {
      res = await chrome.runtime.sendMessage({ type: "ANALYZE_PAGE", payload });
    } catch (_) {
      return; // extension reloaded or service worker unavailable: fail open
    }
    if (!res || !res.result || stale()) return; // skipped, failed (fail-open) or navigated away

    const r = res.result;
    trackDecision = r.decision;
    trackHost = location.hostname;
    if (r.decision === "BLOCK") showBlock(r, payload);
    else if (r.decision === "WARN") showToast(r, "WARN");
    else showPill(r);
  }

  // Single-page apps (YouTube etc.) change the URL without reloading.
  // The poller only schedules a URL that is not already scheduled: otherwise a delay longer than the
  // poll interval (YouTube waits 1.5s for its DOM) would be pushed back forever and never run.
  const currentUrl = () => location.href.split("#")[0];
  let debounce = null;
  let scheduledUrl = null;
  const schedule = (delay) => {
    scheduledUrl = currentUrl();
    clearTimeout(debounce);
    debounce = setTimeout(run, delay);
  };
  schedule(isYouTube() ? 1500 : 300);
  window.addEventListener("yt-navigate-finish", () => schedule(1200));
  document.addEventListener("yt-navigate-finish", () => schedule(1200));
  window.addEventListener("popstate", () => schedule(800));
  setInterval(() => {
    const cur = currentUrl();
    if (cur !== lastUrl && cur !== scheduledUrl) schedule(isYouTube() ? 1500 : 600);
  }, 1000);
})();
