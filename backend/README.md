# FocusAI backend

## How a page/video becomes ALLOW / WARN / BLOCK

Thresholds (unchanged): **ALLOW ≥ 75**, **WARN ≥ 30**, **BLOCK < 30**
(`RELEVANCE_ALLOW_THRESHOLD`, `RELEVANCE_WARN_THRESHOLD`).

`POST /api/relevance/analyze` returns, besides the decision, `evaluation_path`, `llm_status`,
`content_quality` and `content_sources` (which of title / description / metadata / transcript were available)
so you can always see *why* a result looks the way it does.

### Evaluation paths (`app/ai/relevance_engine.py`)

| Situation | Path | Final score |
|---|---|---|
| Normal content (score outside the ambiguity band) | `embedding` | embedding/rule score only. LLM not called. |
| Ambiguous (score within 25–75, i.e. 5 below WARN up to ALLOW) | `embedding+llm` | `0.6 × embedding + 0.4 × LLM` |
| Low content (< 25 words of body, e.g. a video with no transcript) | `llm+embedding` | `0.3 × embedding + 0.7 × LLM` |
| Fewer than 3 meaningful words in title+description+text | `insufficient` | neutral 50 / ALLOW (fail-open); LLM not called |

Why these weights:
* Embeddings are the validated primary signal (calibrated on the labelled set, deterministic, free,
  not prompt-injectable) → they keep the majority (0.6) in ambiguous cases. 0.4 is enough for the LLM to move a
  score across a threshold when it strongly disagrees (embedding 60 + LLM 100 → 76 ALLOW; embedding 40 + LLM 0 → 24 BLOCK)
  but not to override clear embedding evidence alone.
* With almost no text, one short title embedding is high-variance and key-term coverage is nearly binary, whereas the
  LLM can use world knowledge about a title/description → it becomes the main reviewer (0.7). The embedding stays as
  an anchor (0.3) so a wrong or manipulated LLM answer cannot decide alone.
* These weights are design choices, not fitted values. Re-check them with `scripts/eval_relevance.py` (below) when
  you have an LLM key.

LLM unavailable or malformed → the engine uses the embedding score alone (`llm_status = failed:<kind>`).
With little text and no LLM result, the stricter rule applies (block only when score < 15), so a failed LLM call can
never block something merely because it could not be read. When the LLM *was* used, the normal WARN threshold (30) applies.

### LLM reviewer (`app/ai/llm.py`): Gemini or Anthropic
Select the provider in `backend/.env` (loaded by `main.py`; the eval script loads it too):
```
LLM_PROVIDER=gemini               # gemini | anthropic (default when unset, i.e. previous behaviour)
LLM_API_KEY=<secret>              # empty = LLM disabled; MiniLM embeddings keep working on their own
LLM_MODEL=gemini-3.5-flash-lite   # default; anthropic default is claude-haiku-5-5
LLM_TIMEOUT_SECONDS=15
```
Gemini uses the official REST API through httpx: `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`,
key in the `x-goog-api-key` header only (never in the URL), JSON mode (`responseMimeType` + `responseJsonSchema`) and the same
`validate_judgement` as Anthropic. Default model `gemini-3.5-flash-lite` is the stable low-cost model listed on
ai.google.dev (GA July 2026); Google restricts the older 2.5 models for new projects and 2.0 is shut down, so avoid them.
`gemini-3.6-flash` is the stable, more capable alternative. As Google recommends for Gemini 3.x, no `temperature`/thinking
parameters are sent. If the API rejects the schema field, the call is retried once in plain JSON mode. Keys are redacted from
logs and error details. `GET /api/relevance/engine` shows the active provider/model (never the key).

Both providers receive STUDY TOPIC, LEARNING OBJECTIVE, KEYWORDS/CONCEPTS, PAGE TITLE,
PAGE DESCRIPTION, METADATA and EXTRACTED CONTENT/TRANSCRIPT and return
`{classification, relevance_score, reason, concepts}` (Anthropic via a forced tool call, Gemini via JSON mode). The answer is validated again in code
(`validate_judgement`): enum label, finite number in 0–100, label/score must not contradict by more than one band,
bounded reason/concepts. Page text is wrapped in `<content>` tags and treated as untrusted.

### YouTube extraction
* **Extension (`extension/content.js`)** analyses only individual videos (`/watch`, `/shorts`), never the home/search/channel
  pages. It waits (≤3.5 s) until the page shows the *current* video, then sends title, description (URLs/timestamps stripped),
  channel and hashtags from the video's own DOM, never the surrounding navigation. `<meta>` tags are used only when the
  canonical URL matches the video (YouTube does not refresh them on in-page navigation).
* **Backend (`app/services/content_extractor.py`)** adds, in parallel with a shared deadline (`YT_FETCH_TIMEOUT_SECONDS`, 9 s):
  the oEmbed title (authoritative for the video id), the watch page's player response (full description, tags, channel,
  category) and the transcript via `youtube-transcript-api`. Results are cached per video for an hour. A failed or slow part is
  skipped; `content_sources.transcript_status` says why (`unavailable:<reason>` / `timeout`).
* A generic tab title such as "YouTube" (shown while a new video loads) is never used as the title.

### Limitations
* **Transcripts are best-effort.** Captions can be disabled, age/region-restricted, or the backend IP can be rate-limited
  or blocked by YouTube (common from cloud servers; usually fine from a home connection). Non-English transcripts are used
  as-is: MiniLM is English-centric, so the LLM matters more there.
* Reading the watch page is unofficial scraping of YouTube's embedded player data and can break when YouTube changes it;
  the code then falls back to oEmbed + what the extension saw.
* The first analysis of a video can take a few seconds (transcript + optional LLM call); later ones are cached.
* `m.youtube.com` was not changed or tested.

## Word-level explanation (Integrated Gradients)

An **optional** extra on top of the sentence-level heatmap. It never changes a score or decision, and the app works
without it.

* **Endpoint:** `POST /api/relevance/explain-tokens` (login required; `session_id` is ownership-checked).
  Body: `{"session_id": "...", "sentence": "..."}` or `{"topic": "...", "learning_objectives": "...", "sentence": "..."}`;
  optional `steps` (8-64, default 32). `sentence` is 1-2000 chars.
* **Method:** genuine Integrated Gradients (`app/ai/integrated_gradients.py`) on the pretrained all-MiniLM-L6-v2 weights
  (no training/fine-tuning). Gradients are taken w.r.t. the **word-embedding** vectors (`inputs_embeds`) through the
  transformer, masked mean pooling and L2 normalisation.
* **Target explained:** `cosine(sentence embedding, study-context embedding)`; the context embedding is the one the existing
  pipeline produces and is held fixed. It does **not** explain the keyword term, the calibration/blend, the thresholds or the
  final ALLOW/WARN/BLOCK decision.
* **Baseline:** `[PAD]` embedding at every non-special position, `[CLS]`/`[SEP]` kept (so they get exactly 0). Other
  baselines would give different numbers; the choice is part of the explanation.
* **Integration:** trapezoid rule, 32 steps by default; the completeness gap `|sum(attr) - (F(x) - F(baseline))|` is
  returned (`completeness_error`, `converged`). If it exceeds max(5 % of the difference, 0.002) the step count is doubled once.
* **Scores:** `attribution` is the **raw signed** IG value in cosine units (positive = pulls the sentence toward the study
  context, negative = away). `normalized` = attribution / max|attribution| in [-1, 1] and is **for colouring only**.
  Sub-word tokens are **summed** into words (keeps completeness and sign); the per-token values are included in `tokens`.
* **Limits:** sentences are truncated to 128 tokens (`truncated: true`); `/explain-tokens` is serialised by a lock and capped
  at 64 steps. Results are cached in memory (LRU, 256) keyed by a digest of model, sentence, study-context text and steps.
* **Diagnostics:** `uv run python scripts/diagnose_ig.py` prints the installed model structure (modules, pooling mode, tokenizer) and runs IG with the pipeline-equivalence numbers. IG refuses to answer if its forward pass deviates from the embedding pipeline by more than 0.01.
* **Unavailable cases** return HTTP 200 with `available: false` and a `reason` (`unsupported_backend` for the lexical
  fallback, `unsupported_model`, `empty_input`, `torch_missing`, `attribution_failed`). Requires `uv sync --extra ml`;
  the first run downloads the model.
* **Extension:** the toast offers "Explain words" for the top passages; the words are drawn in FocusAI's own shadow-DOM panel
  (blue = raises similarity, orange = lowers it; sign also in tooltips/aria-labels). The host page DOM is never modified.

## Tests and evaluation
```bash
uv run pytest                                   # offline tests (HTTP mocked, no API key needed); see PROJECT_PROGRESS.md for counts
uv run pytest tests/test_integrated_gradients.py -v -s   # IG: API + tiny-BERT + real all-MiniLM-L6-v2 (skips if weights unavailable)
PYTHONPATH=. python -m unittest tests.test_ig_core       # torch-free IG kernel/aggregation/cache tests
uv run python scripts/eval_relevance.py         # original labelled set
uv run python scripts/eval_relevance.py --file relevance_eval_lowcontent.json   # title/description-only pages
cd ../extension-tests && npm install && npm test   # extension tests (jsdom + vm)
NODE_PATH=$(npm root -g) node browser/ig-ui.browser.js   # optional real-Chromium UI test (needs Playwright)
```
