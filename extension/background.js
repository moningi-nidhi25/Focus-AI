// FocusAI service worker: talks to the FastAPI backend on behalf of content scripts.
const DEFAULTS = { backendUrl: "http://127.0.0.1:8000", enabled: true, sessionId: "", topic: "" };
const CACHE_TTL_MS = 10 * 60 * 1000;
const OVERRIDE_TTL_MS = 15 * 60 * 1000;
const MAX_TEXT_CHARS = 30000;

const getConfig = () => chrome.storage.local.get(DEFAULTS);
const sessionGet = async (key) => (await chrome.storage.session.get(key))[key];

// The login token lives in chrome.storage.session, which web pages and content scripts cannot read.
async function authHeaders(extra = {}) {
  const token = await sessionGet("token");
  return token ? { ...extra, Authorization: `Bearer ${token}` } : extra;
}

async function setBadge(tabId, decision) {
  if (tabId == null) return;

  const map = {
    ALLOW: ["\u2713", "#16a34a"],
    WARN: ["!", "#d97706"],
    BLOCK: ["\u00d7", "#dc2626"]
  };

  const [text, color] = map[decision] || ["", "#64748b"];

  try {
    await chrome.action.setBadgeText({ tabId, text });
    await chrome.action.setBadgeBackgroundColor({ tabId, color });
  } catch (_) {
    // Tab may have been closed or navigated away while analysis was running.
    // Never let a badge failure break page analysis.
  }
}

const ANALYZE_TIMEOUT_MS = 35000; // backend may fetch a YouTube transcript and call the LLM
const MAX_CACHE_ENTRIES = 150;

// Prune stale/oversized cache so chrome.storage.session never hits its quota.
function pruneCache(cache) {
  const now = Date.now();
  const entries = Object.entries(cache).filter(([, v]) => now - v.at < CACHE_TTL_MS);
  entries.sort((a, b) => b[1].at - a[1].at);
  return Object.fromEntries(entries.slice(0, MAX_CACHE_ENTRIES));
}

// Every failure path returns {error} (never throws): the content script then leaves the page alone.
async function analyze(payload, tabId) {
  try {
    const cfg = await getConfig();
    if (!cfg.enabled) return { skipped: "disabled" };
    if (!cfg.sessionId) return { skipped: "no_session" };

    const overrides = (await sessionGet("overrides")) || {};
    if (overrides[payload.url] && overrides[payload.url] > Date.now()) return { skipped: "override" };

    const cacheKey = `${cfg.sessionId}|${payload.url}`;
    let cache = (await sessionGet("cache")) || {};
    const hit = cache[cacheKey];
    if (hit && Date.now() - hit.at < CACHE_TTL_MS) {
      await setBadge(tabId, hit.result.decision);
      trackDecision(payload, hit.result.decision);
      return { result: hit.result, cached: true };
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), ANALYZE_TIMEOUT_MS);
    let result;
    try {
      const resp = await fetch(`${cfg.backendUrl}/api/relevance/analyze`, {
        method: "POST",
        headers: await authHeaders({ "Content-Type": "application/json" }),
        signal: controller.signal,
        body: JSON.stringify({
          session_id: cfg.sessionId,
          url: payload.url,
          title: payload.title,
          text: (payload.text || "").slice(0, MAX_TEXT_CHARS),
          metadata: payload.metadata || undefined, // YouTube: description, channel, hashtags
        }),
      });
      if (!resp.ok) throw new Error(`Backend returned ${resp.status}`);
      result = await resp.json();
    } finally {
      clearTimeout(timer);
    }

    // Storing state is best-effort: a closed tab or full storage must not turn a good result into an error.
    try {
      cache = pruneCache(cache);
      cache[cacheKey] = { at: Date.now(), result };
      const update = { cache };
      if (tabId != null) update[`tab:${tabId}`] = { url: payload.url, result };
      await chrome.storage.session.set(update);
    } catch (_) { /* ignore */ }
    await setBadge(tabId, result.decision);
    trackDecision(payload, result.decision);
    return { result };
  } catch (err) {
    // Fail open: never break browsing because the backend is down, slow or returned an error.
    await setBadge(tabId, null);
    return { error: String((err && err.message) || err) };
  }
}

// ---------- optional word-level explanation (Integrated Gradients) ----------
// Never part of the core analysis: failures just return {error}/{unavailable} and the sentence heatmap stays.
const EXPLAIN_TIMEOUT_MS = 30000;
const MAX_SENTENCE_CHARS = 2000;
const explainPending = new Map(); // one in-flight request per sentence

function hashText(s) { // small non-cryptographic key so no page text is used as a storage key
  let h1 = 5381, h2 = 52711;
  for (let i = 0; i < s.length; i++) { const c = s.charCodeAt(i); h1 = ((h1 * 33) ^ c) >>> 0; h2 = ((h2 * 31) + c) >>> 0; }
  return `${s.length}:${h1.toString(36)}${h2.toString(36)}`;
}

async function explainTokens(sentence) {
  try {
    const text = String(sentence || "").replace(/\s+/g, " ").trim().slice(0, MAX_SENTENCE_CHARS);
    if (!text) return { error: "empty" };
    const cfg = await getConfig();
    if (!cfg.enabled) return { skipped: "disabled" };
    if (!cfg.sessionId) return { skipped: "no_session" };

    const key = `${cfg.sessionId}|${hashText(text)}`; // study context changes -> new session id -> new key
    const cache = (await sessionGet("igCache")) || {};
    const hit = cache[key];
    if (hit && Date.now() - hit.at < CACHE_TTL_MS) return { result: hit.result, cached: true };
    if (explainPending.has(key)) return explainPending.get(key);

    const job = (async () => {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), EXPLAIN_TIMEOUT_MS);
      try {
        const resp = await fetch(`${cfg.backendUrl}/api/relevance/explain-tokens`, {
          method: "POST",
          headers: await authHeaders({ "Content-Type": "application/json" }),
          signal: controller.signal,
          body: JSON.stringify({ session_id: cfg.sessionId, sentence: text }),
        });
        if (!resp.ok) throw new Error(`Backend returned ${resp.status}`);
        const result = await resp.json();
        if (result && result.available) { // only real attributions are cached
          try {
            const next = pruneCache((await sessionGet("igCache")) || {});
            next[key] = { at: Date.now(), result };
            await chrome.storage.session.set({ igCache: next });
          } catch (_) { /* storage is best-effort */ }
        }
        return { result };
      } finally {
        clearTimeout(timer);
      }
    })().catch((err) => ({ error: String((err && err.message) || err) }));
    explainPending.set(key, job);
    try { return await job; } finally { explainPending.delete(key); }
  } catch (err) {
    return { error: String((err && err.message) || err) };
  }
}

async function override(payload) {
  const cfg = await getConfig();
  try {
    const overrides = (await sessionGet("overrides")) || {};
    overrides[payload.url] = Date.now() + OVERRIDE_TTL_MS;
    await chrome.storage.session.set({ overrides });
  } catch (_) { /* the page is still released; only the 15-minute memory is lost */ }
  try {
    await fetch(`${cfg.backendUrl}/api/relevance/override`, {
      method: "POST",
      headers: await authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ session_id: cfg.sessionId, url: payload.url, title: payload.title }),
    });
  } catch (_) { /* logging is best-effort */ }
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  const tabId = sender.tab ? sender.tab.id : null;
  switch (msg.type) {
    case "ANALYZE_PAGE":
      analyze(msg.payload, tabId).then(sendResponse, (e) => sendResponse({ error: String(e) }));
      return true;
    case "EXPLAIN_TOKENS":
      explainTokens(msg.sentence).then(sendResponse, (e) => sendResponse({ error: String(e) }));
      return true;
    case "OVERRIDE":
      override(msg.payload).then(() => sendResponse({ ok: true }), () => sendResponse({ ok: false }));
      return true;
    case "SET_SESSION":
      setSession(msg).then(() => sendResponse({ ok: true }), () => sendResponse({ ok: false }));
      return true;
    case "TRACK":
      enqueue(msg.events || []).then(() => sendResponse({ ok: true }), () => sendResponse({ ok: false }));
      return true;
    case "SET_TOKEN": {
      // Only the FocusAI web app may hand the extension a login token.
      const fromApp = sender.url && /^http:\/\/(localhost|127\.0\.0\.1):5173\//.test(sender.url);
      if (!fromApp) { sendResponse({ ok: false }); return false; }
      chrome.storage.session.set({ token: msg.token || "" })
        .then(() => sendResponse({ ok: true }), () => sendResponse({ ok: false }));
      return true;
    }
    case "GET_TAB_STATE":
      sessionGet(`tab:${msg.tabId}`).then((state) => sendResponse(state || null));
      return true;
  }
});

chrome.tabs.onRemoved.addListener((tabId) => { chrome.storage.session.remove(`tab:${tabId}`).catch(() => {}); });

// ---------- Phase 7: activity tracking (batched, best-effort, never breaks browsing) ----------
const FLUSH_ALARM = "focusai-flush";
const FLUSH_BATCH = 10;
const MAX_QUEUE = 400;
let chain = Promise.resolve();
const serial = (fn) => (chain = chain.then(fn, fn)); // one queue operation at a time

async function postEvents(sessionId, events) {
  const cfg = await getConfig();
  for (let i = 0; i < events.length; i += 200) {
    const resp = await fetch(`${cfg.backendUrl}/api/activity/events`, {
      method: "POST",
      headers: await authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ session_id: sessionId, events: events.slice(i, i + 200) }),
    });
    if (!resp.ok) throw new Error(`events ${resp.status}`);
  }
}

function flushNow() {
  return serial(async () => {
    const q = await sessionGet("queue");
    if (!q || !q.events || !q.events.length) return;
    await chrome.storage.session.set({ queue: { sessionId: q.sessionId, events: [] } });
    try {
      await postEvents(q.sessionId, q.events);
    } catch (_) {
      const cur = (await sessionGet("queue")) || { sessionId: q.sessionId, events: [] };
      const merged = q.sessionId === cur.sessionId ? q.events.concat(cur.events) : q.events;
      await chrome.storage.session.set({ queue: { sessionId: q.sessionId, events: merged.slice(-MAX_QUEUE) } });
    }
  });
}

function enqueue(events) {
  return serial(async () => {
    const cfg = await getConfig();
    if (!cfg.enabled || !cfg.sessionId || !events.length) return;
    let q = (await sessionGet("queue")) || { sessionId: cfg.sessionId, events: [] };
    if (q.sessionId !== cfg.sessionId) q = { sessionId: cfg.sessionId, events: [] };
    q.events = q.events.concat(events.map((e) => ({ event_type: e.event_type, metadata: e.metadata || {} })));
    await chrome.storage.session.set({ queue: q });
    if (q.events.length >= FLUSH_BATCH) flushNow();
  });
}

function trackDecision(payload, decision) {
  let host = "";
  try { host = new URL(payload.url).hostname; } catch (_) { /* ignore */ }
  const type = { ALLOW: "content_allowed", WARN: "warning_shown", BLOCK: "content_blocked" }[decision];
  const events = [{ event_type: "page_open", metadata: { host } }];
  if (type) events.push({ event_type: type, metadata: { host } });
  enqueue(events).catch(() => {});
}

async function setSession(msg) {
  const prev = (await getConfig()).sessionId;
  const next = msg.sessionId || "";
  await flushNow();
  if (prev && prev !== next) await postEvents(prev, [{ event_type: "session_end", metadata: {} }]).catch(() => {});
  await chrome.storage.local.set({ sessionId: next, topic: msg.topic || "" });
  await chrome.storage.session.set({ cache: {}, overrides: {} });
  if (next && next !== prev) await postEvents(next, [{ event_type: "session_start", metadata: {} }]).catch(() => {});
}

chrome.tabs.onActivated.addListener(() => { enqueue([{ event_type: "tab_switch", metadata: {} }]).catch(() => {}); });
chrome.alarms.create(FLUSH_ALARM, { periodInMinutes: 0.5 });
chrome.alarms.onAlarm.addListener((a) => { if (a.name === FLUSH_ALARM) flushNow().catch(() => {}); });

// ---------- Phase 8: proactive notification when the LSTM predicts a focus dip ----------
const PREDICT_ALARM = "focusai-predict";
const NOTIFY_COOLDOWN_MS = 15 * 60 * 1000;

async function checkFocusForecast() {
  const cfg = await getConfig();
  if (!cfg.enabled || !cfg.sessionId) return;
  const last = await sessionGet("lastForecastAlert");
  if (last && Date.now() - last < NOTIFY_COOLDOWN_MS) return;
  const resp = await fetch(`${cfg.backendUrl}/api/prediction/sessions/${cfg.sessionId}`, {
    method: "POST",
    headers: await authHeaders(),
  });
  if (!resp.ok) return; // e.g. not enough activity yet
  const data = await resp.json();
  if (!data.alert) return;
  await chrome.storage.session.set({ lastForecastAlert: Date.now() });
  chrome.notifications.create({
    type: "basic",
    iconUrl: "icons/icon128.png",
    title: "FocusAI: focus dip ahead",
    message: data.message,
  });
}

chrome.alarms.create(PREDICT_ALARM, { periodInMinutes: 3 });
chrome.alarms.onAlarm.addListener((a) => { if (a.name === PREDICT_ALARM) checkFocusForecast().catch(() => {}); });