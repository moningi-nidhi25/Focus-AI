// content.js: YouTube extraction, navigation races and fail-open behaviour (jsdom).
const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const { JSDOM } = require("jsdom");

const CODE = fs.readFileSync(path.join(__dirname, "../../extension/content.js"), "utf8");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const unhandled = [];
process.on("unhandledRejection", (e) => unhandled.push(e));

const WATCH_HTML = (id, { title = "Deadlock prevention explained", desc = "Banker's algorithm https://x.io/a 12:30 intro", stale = false } = {}) => `
  <ytd-watch-flexy video-id="${stale ? "OTHER" : id}">
    <ytd-watch-metadata>
      <h1>${title}</h1>
      <ytd-channel-name><a>CS Channel</a></ytd-channel-name>
      <a href="/hashtag/os">#os</a><a href="/hashtag/deadlock">#deadlock</a>
      <div id="description-inner">${desc} ...more</div>
    </ytd-watch-metadata>
  </ytd-watch-flexy>
  <ytd-app><div id="guide">Home Shorts Subscriptions Trending Gaming</div></ytd-app>`;

function page({ url, html = "", reply = async () => ({ result: { decision: "ALLOW", relevance_score: 90, explanation: "ok", reasons: [], matching_concepts: [], heatmap: [] } }) }) {
  const dom = new JSDOM(`<html><head><title>${url.includes("youtube") ? "YouTube" : "Doc"}</title></head><body>${html}</body></html>`, { url, runScripts: "outside-only", pretendToBeVisual: true });
  const w = dom.window;
  Object.defineProperty(w.HTMLElement.prototype, "innerText", { get() { return this.textContent; } });
  const sent = [];
  w.chrome = { runtime: { sendMessage: async (m) => { sent.push(m); return reply(m); } } };
  w.eval(CODE);
  const analyze = () => sent.filter((m) => m.type === "ANALYZE_PAGE").map((m) => m.payload);
  return { w, sent, analyze, host: () => !!w.document.getElementById("focusai-host"), close: () => w.close() };
}

test("YouTube watch page: extracts title, cleaned description, channel and hashtags (no navigation text)", async () => {
  const p = page({ url: "https://www.youtube.com/watch?v=vidA", html: WATCH_HTML("vidA") });
  await sleep(2400);
  const [pl] = p.analyze();
  assert.equal(pl.title, "Deadlock prevention explained");
  assert.equal(pl.text, "");
  assert.equal(pl.metadata.video_id, "vidA");
  assert.equal(pl.metadata.description, "Banker's algorithm intro"); // URL, timestamp, "...more" removed
  assert.equal(pl.metadata.channel, "CS Channel");
  assert.deepEqual(pl.metadata.keywords, ["os", "deadlock"]);
  assert.ok(!JSON.stringify(pl).includes("Subscriptions")); // unrelated navigation text is never sent
  assert.ok(p.host()); // ALLOW pill shown
  p.close();
});

test("YouTube home / search pages are not analysed as a single piece of content", async () => {
  for (const url of ["https://www.youtube.com/", "https://www.youtube.com/results?search_query=os", "https://www.youtube.com/@channel"]) {
    const p = page({ url, html: "<div>Recommended videos</div>" });
    await sleep(2000);
    assert.equal(p.analyze().length, 0, url);
    p.close();
  }
});

test("stale DOM from the previous video: waits, then still sends without throwing", async () => {
  const p = page({ url: "https://www.youtube.com/watch?v=vidB", html: WATCH_HTML("vidB", { stale: true }) });
  await sleep(6200);
  assert.equal(p.analyze().length, 1);
  assert.equal(p.analyze()[0].metadata.video_id, "vidB");
  p.close();
});

test("page with no recognisable YouTube markup does not throw and still reports the video id", async () => {
  const p = page({ url: "https://www.youtube.com/watch?v=vidC" });
  await sleep(6200);
  const [pl] = p.analyze();
  assert.equal(pl.metadata.video_id, "vidC");
  assert.equal(pl.metadata.description, "");
  p.close();
});

test("navigating between videos: the stale analysis result is not shown", async () => {
  let resolveFirst;
  let n = 0;
  const p = page({
    url: "https://www.youtube.com/watch?v=one",
    html: WATCH_HTML("one"),
    reply: () => (++n === 1
      ? new Promise((r) => { resolveFirst = () => r({ result: { decision: "BLOCK", relevance_score: 1, explanation: "x", reasons: [], matching_concepts: [], heatmap: [] } }); })
      : Promise.resolve({ result: { decision: "ALLOW", relevance_score: 95, explanation: "ok", reasons: [], matching_concepts: [], heatmap: [] } })),
  });
  await sleep(2400); // first analysis is now in flight
  p.w.document.body.innerHTML = WATCH_HTML("two", { title: "Banker's algorithm walkthrough" });
  p.w.history.pushState({}, "", "/watch?v=two");
  await sleep(3500);
  resolveFirst(); // late BLOCK for video one arrives after navigation
  await sleep(300);
  assert.equal(p.analyze().length, 2);
  assert.equal(p.analyze()[1].metadata.video_id, "two");
  assert.ok(p.host()); // shows video two's ALLOW pill, not the stale BLOCK overlay
  p.close();
});

test("extension unavailable (sendMessage rejects): no overlay, no unhandled rejection", async () => {
  const p = page({ url: "https://example.com/a", html: "<article><h1>Wedding gowns</h1><p>Fans shared their favourite red carpet looks online today.</p></article>", reply: async () => { throw new Error("Extension context invalidated."); } });
  await sleep(1200);
  assert.equal(p.analyze().length, 1);
  assert.equal(p.host(), false);
  assert.deepEqual(unhandled, []);
  p.close();
});

test("backend error / skipped responses leave the page alone (fail open)", async () => {
  for (const reply of [async () => ({ error: "Backend returned 500" }), async () => ({ skipped: "no_session" }), async () => undefined]) {
    const p = page({ url: "https://example.com/a", html: "<article><p>Some ordinary page text with enough words in it.</p></article>", reply });
    await sleep(1000);
    assert.equal(p.host(), false);
    p.close();
  }
});

test("BLOCK shows the overlay on a normal page", async () => {
  const p = page({ url: "https://example.com/a", html: "<article><p>Celebrity wedding gowns gallery with many red carpet looks.</p></article>", reply: async () => ({ result: { decision: "BLOCK", relevance_score: 3, explanation: "no", reasons: [], matching_concepts: [], heatmap: [] } }) });
  await sleep(1000);
  assert.ok(p.host());
  p.close();
});
