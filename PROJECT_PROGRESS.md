# FocusAI – Project Progress

## Session 2: fixes after the first real-environment test run

**Results reported by the user (their machine, real torch + sentence-transformers):**
`tests/test_ig_core.py` 18 passed · `tests/test_integrated_gradients.py` 11 passed, 3 failed.
**This sandbox still has no PyPI/HF access**, so the fixes below were NOT executed against torch or MiniLM here.

| Failure | Root cause | Certainty |
|---|---|---|
| 1, 2: real MiniLM -> "Only mean-pooling ... supported" (`_components`) | `_components` decided mean pooling from ONE attribute: `getattr(pooling, "pooling_mode_mean_tokens", False)`. The error was raised on exactly that branch, so on your install that attribute was missing or falsy; the `False` default silently turned "attribute not exposed" into "not mean pooling". | The faulty logic is certain (read from the code path). *Why* your Pooling does not expose it (renamed / moved to a `pooling_mode` value in newer sentence-transformers) is **not confirmed** - I cannot inspect your install. |
| 3: tiny fixture gives `["[UNK]"]` for "deadlocks" | The fixture built the tokenizer with `BertTokenizerFast(vocab_file=...)`. Output of a single `[UNK]` for a word whose pieces are all in the vocab means the vocabulary I wrote was not the one in use. Most likely cause: your transformers release ignores/handles `vocab_file=` differently. The earlier tiny test still passed because it only checked word-level output, which [UNK] tokens also satisfy. | Mechanism **not confirmed** (could not run it). The fix does not depend on it. |
| 4 (long-input test) | Same as 1, 2 (never reached the truncation logic). | - |

**Changes**
* `backend/app/ai/integrated_gradients.py`
  * New `pooling_modes()` reads the pooling mode from every place releases have used (`get_pooling_mode_str()`, a `pooling_mode` str/list, boolean `pooling_mode_*` flags), normalises names, and accepts only exactly `{"mean"}`. If undeterminable it refuses and prints what it saw. Pooling validation is NOT removed.
  * `_components()` now also requires that anything after pooling is only `Normalize` (a Dense/other layer would make our forward pass differ from the real embedding).
  * New runtime equivalence guard: after IG, F(x) is compared with `cos(embedder.embed(sentence), context)`; if they differ by more than 0.01 the request fails with `AttributionFailed` instead of returning attributions of a different function (skipped when truncated because the pipeline encodes up to 256 tokens and IG caps at 128; reported as `pipeline_deviation=None`).
  * `TokenExplanation` / API response gained `n_tokens_total` (additive); `n_tokens` is now exactly the number of non-special tokens the model processed.
  * Unchanged: the differentiable forward pass (inputs_embeds -> encoder -> attention-masked mean pool -> L2 norm -> cosine with the fixed context vector), baseline, integration, relevance formula, thresholds.
* `backend/tests/test_integrated_gradients.py`
  * Tiny tokenizer is now built with the `tokenizers` library directly (WordPiece + BertNormalizer + BertPreTokenizer + TemplateProcessing). Fixture asserts `tokenize("deadlocks") == ["dead","##lock","##s"]` and `tokenize("unhappy") == ["un","##happy"]` so it fails loudly if sub-words are not exercised. The original assertion was kept and strengthened.
  * Truncation/alignment checks (tiny and real MiniLM) now compare against the tokenizer's own truncated output: same tokens in the same order, `n_tokens` equals the processed count, `n_tokens_total` larger, words are an exact prefix of the input words, offsets monotonic, word sums == token sums == `sum_attributions` (special tokens 0), nothing attributed beyond the cut.
  * New: real-MiniLM module-structure test, pipeline-equivalence assertions, deviation-refusal test, punctuation/word-sum test. I removed two assertions of mine I judged weak/flaky on a random network (a vacuous `or`, and a "both signs must appear" check); sign preservation is covered deterministically in `test_ig_core.py`.
* `backend/tests/test_ig_core.py`: +8 torch-free tests for pooling detection across attribute styles, Normalize allowed, Dense/CLS/undeterminable rejected with informative messages.
* `backend/scripts/diagnose_ig.py` (new): prints versions, `st`, `st[0]`, `st[1]`, tokenizer, auto_model, pooling attributes, `_components()` verdict and a full IG run with the equivalence numbers. **Run it and paste the output if anything still fails.**
* `extension/content.js`: truncation note now says "first N of M tokens". `backend/app/schemas/relevance.py`, `backend/app/services/relevance_service.py`: `n_tokens_total` plumbing only.

**Executed in this sandbox after the changes (exact):**
`PYTHONPATH=. python -m unittest tests.test_ig_core` -> **26/26 passed** · `node --test tests/background.test.js` -> **14/14** · `node browser/ig-ui.browser.js` -> **16/16** · all backend files compile.
**NOT executed here:** `tests/test_integrated_gradients.py`, the full `uv run pytest -q` (needs your machine), jsdom tests.

**Please run, in this order:**
```
cd backend
uv run pytest tests/test_ig_core.py -v
uv run python scripts/diagnose_ig.py
uv run pytest tests/test_integrated_gradients.py -v -s
uv run pytest -q
```
**Remaining uncertainty:** whether real MiniLM IG now works is **unverified by me**. The new structure test and equivalence guard are designed to either pass or fail with a precise message. Any failure in the full `pytest -q` must be compared with a baseline on the original ZIP to separate pre-existing failures from mine.

---

# (Session 1 report follows; its "NOT RUN" entries for the torch/MiniLM tests are superseded by the user's run above)

## Status in one paragraph
The IG feature is **implemented end to end**, but the sandbox had **no access to PyPI / npm / Hugging Face**
(egress proxy returned 403 for pypi.org; DNS fails otherwise), so torch, sentence-transformers, FastAPI, SQLAlchemy, pytest
and jsdom could not be installed. Therefore the **backend test-suite and the real MiniLM run were NOT executed**. Everything
that could be executed with what was present (numpy, pydantic, Node, Chromium/Playwright) was executed and is listed below.
**You must run the commands in "Verify on your machine" before relying on the backend/model path.**

## Test matrix (exact results observed in this session)
| Suite | Result |
|---|---|
| `PYTHONPATH=. python -m unittest tests.test_ig_core` (IG kernel, aggregation, cache, unavailable cases) | **18/18 passed** |
| `node --test tests/background.test.js` | **14/14 passed** (8 original + 6 new). Baseline BEFORE my change: 0/8 – pre-existing: the test's `chrome` mock lacked `alarms`, `tabs.onActivated`, `notifications`; I added those stubs, no assertion touched |
| `node browser/ig-ui.browser.js` (real Chromium, stubbed chrome API/backend) | **16/16 passed** (panel, colours by sign, a11y labels, no duplicate requests, DOM untouched, existing heatmap toggle, BLOCK overlay, unavailable/error states) |
| Request-schema validation (pydantic, 7 invalid payloads rejected, valid accepted) | passed |
| `python -m compileall backend` | all files compile |
| `node --test tests/content.test.js` (jsdom) | **NOT RUN** – jsdom not installable (baseline and now) |
| Backend `pytest` (all existing tests) | **NOT RUN** – no FastAPI/SQLAlchemy/pytest. No baseline recorded |
| `tests/test_integrated_gradients.py` (API/cache/validation/auth; tiny-BERT real-torch IG; real all-MiniLM-L6-v2) | **WRITTEN, NOT RUN** |

Equivalence evidence for "existing behaviour unchanged": file-level diff against the original ZIP (see IMPLEMENTATION_PLAN.md).
That is static evidence, not a passing regression run.

## Verify on your machine (needs network)
```
cd backend && uv sync --extra ml --group dev
uv run pytest -q                                          # existing suite + new tests; record the numbers
uv run pytest tests/test_integrated_gradients.py -v -s    # tiny-BERT torch IG + real all-MiniLM-L6-v2 (prints word attributions)
cd ../extension-tests && npm ci && npm test               # includes jsdom content tests
NODE_PATH=$(npm root -g) node browser/ig-ui.browser.js    # optional, needs Playwright
```
Things most likely to need a tweak if something fails: (1) sentence-transformers/transformers version differences in how
`SentenceTransformer[0]` exposes `.tokenizer/.auto_model` (the code then reports `unsupported_model` instead of crashing);
(2) the real-MiniLM test's completeness tolerance (5 % / 0.002) – if `converged` is False raise steps or inspect `completeness_error`;
(3) the existing README test counts (128 / 142 / 16) look stale: repo contains ~95 backend `def test_` functions and 8 original
extension tests – confirm at baseline.

## Definition of done (IG feature)
- [x] Genuine IG implemented (path integral, completeness check) – kernel verified on analytic functions
- [x] Target defined and documented (module docstring, API `target`/`scope`, UI text)
- [ ] Real MiniLM executes the attribution path – **not verified (environment)**
- [~] Finite values / shape – verified for kernel; torch path test written, not run
- [x] Sub-words → words by summation, offsets-based, punctuation separate (verified, unit tests)
- [x] Signs preserved; raw vs normalised documented and labelled in API + UI
- [~] Backend endpoint exposes real attributions – written, not run
- [x] Extension shows word heatmap, interactive (hover/focus/click), legend, scope note – verified in Chromium
- [x] Sentence heatmap still works – verified in Chromium
- [x] Existing relevance code unchanged – verified by diff; regression run pending (needs your machine)
- [ ] Real-model verification result – attempted, blocked by network (recorded here)
- [~] End-to-end (real backend ↔ extension) – not tested; each side tested against a stub of the other (schema shared via ExplainTokens* models / content.js)
- [x] Docs updated

## Known limitations / honest caveats
* IG explains only cos(sentence, study context). Baseline = [PAD] embeddings; other baselines give different numbers.
  Attributions are over word-embedding vectors only (position/segment embeddings not attributed).
* Positive attribution means "pushes the embedding toward the study context", which is the direction of the explained target;
  it is not the same as "FocusAI relevance %" (keyword term/calibration not included).
* Words are drawn in the extension's panel, not painted on the host page, to avoid modifying page DOM. In-page painting
  is a possible follow-up.
* The dashboard (`frontend/`) Heatmap component was not extended.
* LRU cache is per-process memory.

## Project-wide status (static inspection only; nothing here was executed)
* Implemented per code: auth (JWT/bcrypt), sessions, relevance engine (embeddings + keywords + optional Gemini), sentence heatmap,
  extension overlay/override/activity tracking, dashboards, content extraction (incl. YouTube).
* **Synthetic / not real-user models:** the focus-decay LSTM was trained on simulated data (`trained_on: "synthetic"`); the README's
  LSTM-vs-baseline numbers are on synthetic data. The relevance-eval table in the README was not reproduced here.
* **Missing:** rate limiting, consent flow (no code found), LRP (not required).
* The shipped `focusai.db` contains data and `focus_lstm.pt` is a synthetic-data model; both are inside the ZIP as in the original.
