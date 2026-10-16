// Real-browser (Chromium) test of the content-script UI, incl. the word-level Integrated Gradients panel.
// Needs Playwright (not a repo dependency):  NODE_PATH=$(npm root -g) node browser/ig-ui.browser.js
// Optional: SHOT_DIR=/some/dir to save screenshots, CHROMIUM=/path/to/chromium.
const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert");
const { chromium } = require("playwright");

const CODE = fs.readFileSync(path.join(__dirname, "../../extension/content.js"), "utf8");
const SHOT_DIR = process.env.SHOT_DIR || "";
const URL_ = "http://focusai-test.local/article";
const SENT1 = "The Banker's algorithm avoids deadlock by simulating resource allocation.";
const SENT2 = "Celebrity weddings featured stunning gowns.";
const PAGE = `<html><head><title>Deadlocks</title></head><body>
  <main><article>
   <h1>Operating systems notes</h1>
   <p id="p1">${SENT1} It tests safety first.</p>
   <p id="p2">${SENT2} Fans shared looks online.</p>
   <p><a id="lnk" href="#x">a real link</a> <button id="btn">page button</button> <input id="inp" value="keep me"></p>
  </article></main></body></html>`;

const ANALYZE = (decision) => ({
  decision, relevance_score: decision === "BLOCK" ? 12 : 55, explanation: "Test explanation", reasons: ["r1"], matching_concepts: ["deadlock"],
  heatmap: [{ text: SENT1, score: 82, level: "high" }, { text: SENT2, score: 8, level: "low" }],
  highlighted_sentences: [{ text: SENT1, score: 82, level: "high" }],
});
const IG = {
  available: true, method: "integrated-gradients", n_tokens: 14, truncated: false, steps: 32, converged: true,
  target_value: 0.61, baseline_value: 0.12, sum_attributions: 0.489,
  words: [
    { word: "The", start: 0, end: 3, attribution: 0.001, normalized: 0.02, tokens: [{ token: "the", attribution: 0.001 }] },
    { word: "Banker's", start: 4, end: 12, attribution: 0.12, normalized: 0.8, tokens: [{ token: "banker", attribution: 0.09 }, { token: "'", attribution: 0.01 }, { token: "s", attribution: 0.02 }] },
    { word: "algorithm", start: 13, end: 22, attribution: 0.15, normalized: 1, tokens: [] },
    { word: "avoids", start: 23, end: 29, attribution: -0.06, normalized: -0.4, tokens: [] },
    { word: "deadlock", start: 30, end: 38, attribution: 0.2, normalized: 1, tokens: [] },
  ],
};

async function open(browser, { decision, igReply }) {
  const ctx = await browser.newContext({ viewport: { width: 1000, height: 800 } });
  const page = await ctx.newPage();
  await page.route(URL_, (r) => r.fulfill({ contentType: "text/html", body: PAGE }));
  await page.addInitScript(({ decision, analyze, igReply }) => {
    // expose the closed shadow root for inspection only
    const orig = Element.prototype.attachShadow;
    Element.prototype.attachShadow = function (o) { const r = orig.call(this, { ...o, mode: "open" }); window.__root = r; return r; };
    window.__sent = [];
    window.chrome = { runtime: { sendMessage: async (m) => {
      window.__sent.push(m.type);
      if (m.type === "ANALYZE_PAGE") return { result: analyze };
      if (m.type === "EXPLAIN_TOKENS") { await new Promise((r) => setTimeout(r, 150)); return igReply; }
      return { ok: true };
    } } };
  }, { decision, analyze: ANALYZE(decision), igReply });
  await page.goto(URL_);
  await page.addScriptTag({ content: CODE });
  return { ctx, page };
}
const q = (page, sel) => page.evaluate((s) => (window.__root ? window.__root.querySelector(s) !== null : false), sel);
const bodyWithoutHost = (page) => page.evaluate(() => { const c = document.body.cloneNode(true); return c.innerHTML; });

(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROMIUM || undefined, args: ["--no-sandbox"] });
  const results = [];
  const check = async (name, fn) => { try { await fn(); results.push(["PASS", name]); } catch (e) { results.push(["FAIL", name + " :: " + e.message.split("\n")[0]]); } };

  // ---------- WARN toast with an available explanation
  {
    const { ctx, page } = await open(browser, { decision: "WARN", igReply: { result: IG } });
    await page.waitForFunction(() => !!window.__root && !!window.__root.querySelector(".toast"), null, { timeout: 5000 });
    const before = await bodyWithoutHost(page);
    await check("toast shows sentence-level legend + Toggle heatmap + Explain button", async () => {
      assert.ok(await q(page, ".toast .legend"));
      const labels = await page.evaluate(() => [...window.__root.querySelectorAll("button")].map((b) => b.textContent));
      assert.ok(labels.some((l) => /Toggle heatmap/.test(l)) && labels.some((l) => /Explain words/.test(l)) && labels.some((l) => /Dismiss/.test(l)), labels.join("|"));
    });
    await check("analysis did not wait for / trigger an explanation request", async () => {
      assert.deepEqual(await page.evaluate(() => window.__sent), ["ANALYZE_PAGE"]);
    });
    await page.evaluate(() => [...window.__root.querySelectorAll("button")].find((b) => /Explain words/.test(b.textContent)).click());
    await check("loading state is announced, button disabled while pending; duplicate click sends one request", async () => {
      assert.match(await page.evaluate(() => window.__root.querySelector(".ig-panel [role=status]").textContent), /Calculating/);
      await page.evaluate(() => { const b = [...window.__root.querySelectorAll("button")].find((x) => /Explain words/.test(x.textContent)); b.disabled = false; b.click(); });
    });
    await page.waitForFunction(() => window.__root.querySelectorAll(".ig-words .w").length > 0);
    await check("one EXPLAIN_TOKENS request despite two clicks", async () => {
      assert.equal((await page.evaluate(() => window.__sent)).filter((t) => t === "EXPLAIN_TOKENS").length, 1);
    });
    await check("words rendered with spacing/punctuation preserved and correct colours by sign", async () => {
      const out = await page.evaluate(() => {
        const w = [...window.__root.querySelectorAll(".ig-words .w")];
        return { text: window.__root.querySelector(".ig-words").textContent, cols: w.map((x) => x.style.backgroundColor), labels: w.map((x) => x.getAttribute("aria-label")) };
      });
      assert.equal(out.text, "The Banker's algorithm avoids deadlock");
      assert.match(out.cols[2], /rgba\(37, 99, 235/);       // positive -> blue
      assert.match(out.cols[3], /rgba\(234, 88, 12/);       // negative -> orange
      assert.ok(out.cols[0] === "transparent" || /rgba\(0, 0, 0, 0\)|transparent/.test(out.cols[0]));  // ~0 -> none
      assert.match(out.labels[3], /−0\.0600 \(lowers similarity/);   // sign is in text, not colour only
      assert.match(out.labels[2], /\+0\.1500 \(raises similarity/);
    });
    await check("words are keyboard-focusable and focus shows raw + scaled score and sub-words", async () => {
      await page.evaluate(() => window.__root.querySelectorAll(".ig-words .w")[1].focus());
      const d = await page.evaluate(() => window.__root.querySelector(".ig-detail").textContent);
      assert.match(d, /Banker's.*raises similarity.*raw IG \+0\.1200.*scaled 0\.80.*sub-words: banker \+0\.0900/);
      assert.equal(await page.evaluate(() => window.__root.querySelector(".ig-words .w").getAttribute("tabindex")), "0");
    });
    await check("legend, scope statement and completeness check are shown", async () => {
      const t = await page.evaluate(() => window.__root.querySelector(".ig-panel").textContent);
      assert.match(t, /lowers similarity \(negative\)/); assert.match(t, /raises similarity \(positive\)/);
      assert.match(t, /raw attributions/); assert.match(t, /not the full FocusAI decision/);
      assert.match(t, /Check: attributions sum to 0\.489 \(expected 0\.490\) — consistent/);
    });
    await check("page DOM is untouched (outside #focusai-host) and page controls still work", async () => {
      const after = await page.evaluate(() => document.body.innerHTML.replace(/<div id="focusai-host"><\/div>/, ""));
      assert.equal(after.replace(/\s+/g, ""), before.replace(/<div id="focusai-host"><\/div>/, "").replace(/\s+/g, ""));
      assert.equal(await page.evaluate(() => document.getElementById("inp").value), "keep me");
      await page.click("#btn"); await page.click("#lnk");
    });
    await check("existing sentence heatmap toggle still shades page paragraphs and restores them", async () => {
      await page.evaluate(() => [...window.__root.querySelectorAll("button")].find((b) => /Toggle heatmap/.test(b.textContent)).click());
      const on = await page.evaluate(() => [document.getElementById("p1").style.backgroundColor, document.getElementById("p2").style.backgroundColor]);
      assert.match(on[0], /rgba\(34, 197, 94/); assert.match(on[1], /rgba\(239, 68, 68/);
      await page.evaluate(() => [...window.__root.querySelectorAll("button")].find((b) => /Toggle heatmap/.test(b.textContent)).click());
      assert.equal(await page.evaluate(() => document.getElementById("p1").style.backgroundColor), "");
    });
    if (SHOT_DIR) await page.screenshot({ path: path.join(SHOT_DIR, "ig-panel.png") });
    await check("re-opening the same sentence uses the in-page result (no second request)", async () => {
      await page.evaluate(() => [...window.__root.querySelectorAll("button")].find((b) => /Explain words/.test(b.textContent)).click());
      await page.waitForTimeout(250);
      assert.equal((await page.evaluate(() => window.__sent)).filter((t) => t === "EXPLAIN_TOKENS").length, 1);
      assert.equal(await page.evaluate(() => window.__root.querySelectorAll(".ig-words .w").length), 5); // replaced, not duplicated
    });
    await ctx.close();
  }

  // ---------- backend says unavailable (lexical fallback) / backend error
  for (const [label, reply, expect] of [
    ["unavailable result", { result: { available: false, reason: "unsupported_backend", message: "Word-level explanations need the sentence-transformers embedding model.", words: [] } }, /need the sentence-transformers/],
    ["transport error", { error: "Failed to fetch" }, /unavailable right now/],
    ["no active session", { skipped: "no_session" }, /active study session/],
  ]) {
    const { ctx, page } = await open(browser, { decision: "WARN", igReply: reply });
    await page.waitForFunction(() => !!window.__root && !!window.__root.querySelector(".toast"));
    await page.evaluate(() => [...window.__root.querySelectorAll("button")].find((b) => /Explain words/.test(b.textContent)).click());
    await page.waitForFunction(() => !window.__root.querySelector(".ig-panel [role=status]")?.textContent.includes("Calculating"));
    await check(`${label}: friendly message, sentence heatmap still available`, async () => {
      assert.match(await page.evaluate(() => window.__root.querySelector(".ig-panel").textContent), expect);
      assert.equal(await page.evaluate(() => window.__root.querySelectorAll(".ig-words .w").length), 0);
      assert.ok(await page.evaluate(() => [...window.__root.querySelectorAll("button")].some((b) => /Toggle heatmap/.test(b.textContent))));
    });
    await ctx.close();
  }

  // ---------- ALLOW pill -> toast, BLOCK overlay unchanged
  {
    const { ctx, page } = await open(browser, { decision: "ALLOW", igReply: { result: IG } });
    await page.waitForFunction(() => !!window.__root && !!window.__root.querySelector(".pill"));
    await check("ALLOW pill opens the toast, which offers explanations", async () => {
      await page.evaluate(() => window.__root.querySelector(".pill").click());
      await page.waitForFunction(() => !!window.__root.querySelector(".toast .ig-list"));
    });
    await ctx.close();
  }
  {
    const { ctx, page } = await open(browser, { decision: "BLOCK", igReply: { result: IG } });
    await page.waitForFunction(() => !!window.__root && !!window.__root.querySelector(".overlay"));
    await check("BLOCK overlay is unchanged (no explain controls, both buttons present)", async () => {
      const labels = await page.evaluate(() => [...window.__root.querySelectorAll("button")].map((b) => b.textContent));
      assert.deepEqual(labels, ["Go back to studying", "Continue anyway (logged)"]);
    });
    await ctx.close();
  }

  // ---------- older backend response without highlighted_sentences / empty heatmap
  {
    const { ctx, page } = await open(browser, { decision: "WARN", igReply: { result: IG } });
    await ctx.close();
    const c2 = await browser.newContext(); const p2 = await c2.newPage();
    await p2.route(URL_, (r) => r.fulfill({ contentType: "text/html", body: PAGE }));
    await p2.addInitScript(() => {
      const orig = Element.prototype.attachShadow;
      Element.prototype.attachShadow = function (o) { const r = orig.call(this, { ...o, mode: "open" }); window.__root = r; return r; };
      window.chrome = { runtime: { sendMessage: async (m) => m.type === "ANALYZE_PAGE" ? { result: { decision: "WARN", relevance_score: 40, explanation: "e", reasons: [], matching_concepts: [], heatmap: [] } } : { ok: true } } };
    });
    await p2.goto(URL_); await p2.addScriptTag({ content: CODE });
    await p2.waitForFunction(() => !!window.__root && !!window.__root.querySelector(".toast"));
    await check("response without highlighted sentences / heatmap: toast renders, no explain controls, no errors", async () => {
      assert.equal(await p2.evaluate(() => window.__root.querySelectorAll(".ig-list").length), 0);
    });
    await c2.close();
  }

  await browser.close();
  let failed = 0;
  for (const [s, n] of results) { console.log(s, "-", n); if (s === "FAIL") failed++; }
  console.log(`\n${results.length - failed}/${results.length} browser checks passed`);
  process.exit(failed ? 1 : 0);
})();
