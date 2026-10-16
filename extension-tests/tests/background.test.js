// background.js: fail-open behaviour. Run with: npm install && npm test
const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const CODE = fs.readFileSync(path.join(__dirname, "../../extension/background.js"), "utf8");
const GOOD = { decision: "BLOCK", relevance_score: 4, explanation: "x", reasons: [], matching_concepts: [], heatmap: [] };

function load({ fetchImpl, badgeImpl, config = { enabled: true, sessionId: "S1", backendUrl: "http://b" } } = {}) {
  const store = { local: { ...config }, session: {} };
  const badges = [];
  let listener;
  const chrome = {
    storage: {
      local: { get: async (d) => ({ ...d, ...store.local }), set: async (o) => Object.assign(store.local, o) },
      session: {
        get: async (k) => (typeof k === "string" ? { [k]: store.session[k] } : {}),
        set: async (o) => Object.assign(store.session, o),
        remove: async (k) => delete store.session[k],
      },
    },
    action: {
      setBadgeText: async (a) => { badges.push(a); if (badgeImpl) return badgeImpl(a); },
      setBadgeBackgroundColor: async () => {},
    },
    runtime: { onMessage: { addListener: (fn) => { listener = fn; } } },
    tabs: { onRemoved: { addListener: () => {} }, onActivated: { addListener: () => {} } },
    alarms: { create: () => {}, onAlarm: { addListener: () => {} } },
    notifications: { create: () => {} },
  };
  const calls = [];
  const fetch = async (url, init) => { calls.push({ url, body: init && init.body && JSON.parse(init.body) }); return fetchImpl(url, init); };
  vm.runInNewContext(CODE, { chrome, fetch, setTimeout, clearTimeout, AbortController, Date, JSON, Promise, Object, String, Number, Math, Error });
  const send = (msg, tabId = 7) => new Promise((resolve) => { listener(msg, { tab: tabId == null ? undefined : { id: tabId } }, resolve); });
  return { send, calls, badges, store };
}
const ok = (body) => async () => ({ ok: true, status: 200, json: async () => body });
const payload = { url: "https://www.youtube.com/watch?v=a", title: "T", text: "", metadata: { video_id: "a", description: "d", channel: "c", keywords: ["k"] } };

test("backend unavailable -> {error}, never throws (fail open)", async () => {
  const { send, badges } = load({ fetchImpl: async () => { throw new TypeError("Failed to fetch"); } });
  const res = await send({ type: "ANALYZE_PAGE", payload });
  assert.ok(res.error && !res.result);
  assert.equal(badges.at(-1).text, ""); // neutral badge
});

test("backend 500 -> {error}", async () => {
  const { send } = load({ fetchImpl: async () => ({ ok: false, status: 500, json: async () => ({}) }) });
  assert.match((await send({ type: "ANALYZE_PAGE", payload })).error, /500/);
});

test("malformed JSON from backend -> {error}", async () => {
  const { send } = load({ fetchImpl: async () => ({ ok: true, status: 200, json: async () => { throw new SyntaxError("bad"); } }) });
  assert.ok((await send({ type: "ANALYZE_PAGE", payload })).error);
});

test("closed tab ('No tab with id') does not break analysis", async () => {
  const { send } = load({ fetchImpl: ok(GOOD), badgeImpl: async () => { throw new Error("No tab with id: 7"); } });
  const res = await send({ type: "ANALYZE_PAGE", payload });
  assert.equal(res.result.decision, "BLOCK");
});

test("missing sender tab does not break analysis", async () => {
  const { send } = load({ fetchImpl: ok(GOOD) });
  assert.equal((await send({ type: "ANALYZE_PAGE", payload }, null)).result.decision, "BLOCK");
});

test("forwards YouTube metadata and caches the decision", async () => {
  const { send, calls } = load({ fetchImpl: ok(GOOD) });
  await send({ type: "ANALYZE_PAGE", payload });
  const second = await send({ type: "ANALYZE_PAGE", payload });
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0].body.metadata, payload.metadata);
  assert.equal(calls[0].body.session_id, "S1");
  assert.equal(second.cached, true);
});

test("disabled or no session -> skipped without network", async () => {
  const a = load({ fetchImpl: ok(GOOD), config: { enabled: false, sessionId: "S1" } });
  assert.equal((await a.send({ type: "ANALYZE_PAGE", payload })).skipped, "disabled");
  const b = load({ fetchImpl: ok(GOOD), config: { enabled: true, sessionId: "" } });
  assert.equal((await b.send({ type: "ANALYZE_PAGE", payload })).skipped, "no_session");
  assert.equal(a.calls.length + b.calls.length, 0);
});

test("override survives a failing backend", async () => {
  const { send, store } = load({ fetchImpl: async () => { throw new Error("down"); } });
  assert.deepEqual(await send({ type: "OVERRIDE", payload: { url: "https://x" } }), { ok: true });
  assert.ok(store.session.overrides["https://x"] > Date.now());
});

// ---------- EXPLAIN_TOKENS (optional Integrated Gradients explanation) ----------
const IG_OK = { available: true, method: "integrated-gradients", words: [{ word: "deadlock", start: 0, end: 8, attribution: 0.02, normalized: 1, tokens: [] }], converged: true };

test("EXPLAIN_TOKENS posts only session + sentence, caches real results", async () => {
  const { send, calls } = load({ fetchImpl: ok(IG_OK) });
  const a = await send({ type: "EXPLAIN_TOKENS", sentence: "Deadlock   prevention\n works." });
  assert.equal(a.result.available, true);
  assert.equal(calls.length, 1);
  assert.match(calls[0].url, /\/api\/relevance\/explain-tokens$/);
  assert.deepEqual(calls[0].body, { session_id: "S1", sentence: "Deadlock prevention works." }); // whitespace normalised, nothing else sent
  const b = await send({ type: "EXPLAIN_TOKENS", sentence: "Deadlock prevention works." });
  assert.equal(b.cached, true);
  assert.equal(calls.length, 1); // no second request
});

test("EXPLAIN_TOKENS: different sentence or different session is a cache miss", async () => {
  const { send, calls, store } = load({ fetchImpl: ok(IG_OK) });
  await send({ type: "EXPLAIN_TOKENS", sentence: "First sentence here." });
  await send({ type: "EXPLAIN_TOKENS", sentence: "Second sentence here." });
  assert.equal(calls.length, 2);
  store.local.sessionId = "S2";
  await send({ type: "EXPLAIN_TOKENS", sentence: "First sentence here." });
  assert.equal(calls.length, 3);
});

test("EXPLAIN_TOKENS: unavailable results are returned but not cached", async () => {
  const { send, calls } = load({ fetchImpl: ok({ available: false, reason: "unsupported_backend", message: "m", words: [] }) });
  const a = await send({ type: "EXPLAIN_TOKENS", sentence: "Some sentence." });
  assert.equal(a.result.available, false);
  await send({ type: "EXPLAIN_TOKENS", sentence: "Some sentence." });
  assert.equal(calls.length, 2);
});

test("EXPLAIN_TOKENS: concurrent identical requests share one network call", async () => {
  let release;
  const gate = new Promise((r) => { release = r; });
  const { send, calls } = load({ fetchImpl: async () => { await gate; return { ok: true, status: 200, json: async () => IG_OK }; } });
  const p1 = send({ type: "EXPLAIN_TOKENS", sentence: "Same sentence." });
  const p2 = send({ type: "EXPLAIN_TOKENS", sentence: "Same sentence." });
  await new Promise((r) => setTimeout(r, 20));
  release();
  const [a, b] = await Promise.all([p1, p2]);
  assert.equal(calls.length, 1);
  assert.equal(a.result.available, true);
  assert.equal(b.result.available, true);
});

test("EXPLAIN_TOKENS: failures return {error}, never throw; disabled/no session skip the network", async () => {
  const down = load({ fetchImpl: async () => { throw new TypeError("Failed to fetch"); } });
  assert.ok((await down.send({ type: "EXPLAIN_TOKENS", sentence: "A sentence." })).error);
  const s500 = load({ fetchImpl: async () => ({ ok: false, status: 500, json: async () => ({}) }) });
  assert.match((await s500.send({ type: "EXPLAIN_TOKENS", sentence: "A sentence." })).error, /500/);
  const off = load({ fetchImpl: ok(IG_OK), config: { enabled: false, sessionId: "S1", backendUrl: "http://b" } });
  assert.equal((await off.send({ type: "EXPLAIN_TOKENS", sentence: "A sentence." })).skipped, "disabled");
  assert.equal(off.calls.length, 0);
  const none = load({ fetchImpl: ok(IG_OK), config: { enabled: true, sessionId: "", backendUrl: "http://b" } });
  assert.equal((await none.send({ type: "EXPLAIN_TOKENS", sentence: "A sentence." })).skipped, "no_session");
  assert.equal((await none.send({ type: "EXPLAIN_TOKENS", sentence: "   " })).error, "empty");
});

test("EXPLAIN_TOKENS: very long sentences are cut to 2000 characters", async () => {
  const { send, calls } = load({ fetchImpl: ok(IG_OK) });
  await send({ type: "EXPLAIN_TOKENS", sentence: "word ".repeat(1000) });
  assert.ok(calls[0].body.sentence.length <= 2000);
});
